#include "Orchestrator.h"
#include "CFRGame.h"
#include "CFRUtils.h"
#include "DeepCFR.h"

Orchestrator::Orchestrator(int numThreads, Reservoir *advRes0,
                           Reservoir *advRes1, Reservoir *polRes, uint64_t seed)
    : numThreads_(numThreads), seed_(seed), advReservoirs_{advRes0, advRes1},
      polReservoir_(polRes) {
    // g_indexer is initialised by loadTables(), which callers must invoke
    // before constructing an Orchestrator.

    schedulerPtrs.resize(numThreads, nullptr);
    threadPool_.reserve(numThreads);
    for (int i = 0; i < numThreads; i++)
        threadPool_.emplace_back(&Orchestrator::runWorker, this, i);

    // Block until every thread has registered its Scheduler pointer.
    std::unique_lock lock(mtx_);
    doneCV_.wait(lock, [this] { return registeredCount_ == numThreads_; });
}

Orchestrator::~Orchestrator() { drainPool(); }

void Orchestrator::drainPool() {
    {
        std::unique_lock lock(mtx_);
        shutdown_ = true;
    }
    startCV_.notify_all();
    for (auto &t : threadPool_)
        if (t.joinable())
            t.join();
}

void Orchestrator::startIteration(bool hero, int t, int totalSamples) {
    {
        std::unique_lock lock(mtx_);
        currentHero_ = hero;
        currentT_ = t;
        totalSamples_ = totalSamples;
        nSeenAtStart_ =
            advReservoirs_[hero]->nSeen.load(std::memory_order_relaxed);
        threadsActive_ = numThreads_;
        iterReady_ = true;
    }
    startCV_.notify_all();
}

void Orchestrator::waitIteration() {
    {
        std::unique_lock lock(mtx_);
        doneCV_.wait(lock, [this] { return threadsActive_ == 0; });
        iterReady_ = false;
    }
    // Release workers waiting on !iterReady_ so they can loop back to sleep.
    startCV_.notify_all();

    // Re-throw any exception captured from a worker thread. Done after
    // releasing workers so they are not left blocked on startCV_.
    if (workerException_)
        std::rethrow_exception(std::exchange(workerException_, nullptr));
}

void Orchestrator::clearBuffers() {
    for (Scheduler *s : schedulerPtrs)
        s->clearBuffers();
}

void Orchestrator::runWorker(int threadIdx) {
    Scheduler sched(iq, threadIdx);

    // Register this thread's Scheduler so Python can access training data.
    {
        std::unique_lock lock(mtx_);
        schedulerPtrs[threadIdx] = &sched;
        if (++registeredCount_ == numThreads_)
            doneCV_.notify_one();
    }

    // Seed this thread's RNG uniquely. Adding threadIdx to the base seed gives
    // each thread a distinct stream while keeping runs reproducible for a
    // given seed.
    rng.seed(seed_ + static_cast<uint64_t>(threadIdx));

    while (true) {
        // Wait for startIteration() or drainPool().
        {
            std::unique_lock lock(mtx_);
            startCV_.wait(lock, [this] { return iterReady_ || shutdown_; });
            if (shutdown_)
                break;
            sched.advReservoir = advReservoirs_[currentHero_];
            sched.polReservoir = (currentT_ > 50) ? polReservoir_ : nullptr;
        }

        sched.clearBuffers();

        try {
            // Fill the initial pool.
            for (int i = 0; i < POOL_SIZE; i++) {
                CFRGame g;
                g.begin(STARTING_STACK, STARTING_STACK, currentHero_);
                sched.spawn(DeepCFR::rollout(std::move(g), currentHero_,
                                             currentT_, sched));
            }

            // Run until the sample quota is met and all active rollouts have
            // drained.
            while (sched.activeTasks() > 0) {
                sched.runOneBatch();
                int completed = sched.purgeCompleted();
                if (sched.advReservoir->nSeen - nSeenAtStart_ <
                    static_cast<std::size_t>(totalSamples_))
                    for (int i = 0; i < completed; i++) {
                        CFRGame g;
                        g.begin(STARTING_STACK, STARTING_STACK, currentHero_);
                        sched.spawn(DeepCFR::rollout(std::move(g), currentHero_,
                                                     currentT_, sched));
                    }
            }
        } catch (...) {
            // Store the first exception; subsequent ones are silently dropped
            // since the iteration is already failed.
            {
                std::unique_lock lock(mtx_);
                if (!workerException_)
                    workerException_ = std::current_exception();
            }
        }

        // Flush any data accumulated after the last flushBatch() — rollouts
        // that completed without triggering another inference round.
        if (sched.advReservoir)
            sched.advReservoir->insert(threadIdx, sched.advantageInputs,
                                       sched.advantageOutputs);
        if (sched.polReservoir)
            sched.polReservoir->insert(threadIdx, sched.policyInputs,
                                       sched.policyOutputs,
                                       &sched.policyWeights);

        // Always push the sentinel so Python's pop() loop is not left hanging,
        // regardless of whether the work block succeeded or threw.
        iq.push(nullptr);

        {
            std::unique_lock lock(mtx_);
            if (--threadsActive_ == 0)
                doneCV_.notify_one();
        }

        // Wait until waitIteration() resets iterReady_ before looping back,
        // so we don't re-enter the work block for the same iteration.
        {
            std::unique_lock lock(mtx_);
            startCV_.wait(lock, [this] { return !iterReady_ || shutdown_; });
        }
    }
}
