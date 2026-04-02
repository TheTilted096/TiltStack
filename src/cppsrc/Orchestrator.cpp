#include "Orchestrator.h"
#include "CFRGame.h"
#include "CFRUtils.h"
#include "DeepCFR.h"

Orchestrator::Orchestrator(int numThreads, uint64_t seed)
    : numThreads_(numThreads), seed_(seed) {
    // hand_index.c initialises its global lookup tables behind a plain
    // `static bool tables_ready` guard (no mutex, not atomic). Initialising
    // g_indexer here — on the main thread, before any workers are spawned —
    // runs hand_index_ctor() once while single-threaded, so every subsequent
    // worker call sees tables_ready == true and skips it.
    uint8_t rounds[] = {2, 3, 1, 1};
    hand_indexer_init(4, rounds, &g_indexer);

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
        samplesPerThread_ = totalSamples / numThreads_;
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
    Scheduler sched(iq);

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
        bool hero;
        int t, target;
        {
            std::unique_lock lock(mtx_);
            startCV_.wait(lock, [this] { return iterReady_ || shutdown_; });
            if (shutdown_)
                break;
            hero = currentHero_;
            t = currentT_;
            target = samplesPerThread_;
        }

        sched.clearBuffers();

        try {
            // Fill the initial pool.
            for (int i = 0; i < POOL_SIZE; i++) {
                CFRGame g;
                g.begin(STARTING_STACK, STARTING_STACK, hero);
                sched.spawn(DeepCFR::rollout(std::move(g), hero, t, sched));
            }

            // Run until the sample quota is met and all active rollouts have
            // drained.
            while (sched.activeTasks() > 0) {
                sched.runOneBatch();
                int completed = sched.purgeCompleted();
                if (sched.advantageSize() < target)
                    for (int i = 0; i < completed; i++) {
                        CFRGame g;
                        g.begin(STARTING_STACK, STARTING_STACK, hero);
                        sched.spawn(
                            DeepCFR::rollout(std::move(g), hero, t, sched));
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
