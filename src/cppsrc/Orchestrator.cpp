#include "Orchestrator.h"
#include "CFRUtils.h"
#include "DeepCFR.h"

Orchestrator::Orchestrator(int numThreads, uint64_t seed)
    : numThreads_(numThreads), seed_(seed) {
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
        if (t.joinable()) t.join();
}

void Orchestrator::startIteration(bool hero, int t, int totalSamples) {
    {
        std::unique_lock lock(mtx_);
        currentHero_      = hero;
        currentT_         = t;
        samplesPerThread_ = totalSamples / numThreads_;
        threadsActive_    = numThreads_;
        iterReady_        = true;
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
        int  t, target;
        {
            std::unique_lock lock(mtx_);
            startCV_.wait(lock, [this] { return iterReady_ || shutdown_; });
            if (shutdown_) break;
            hero   = currentHero_;
            t      = currentT_;
            target = samplesPerThread_;
        }

        sched.clearBuffers();

        // Fill the initial pool.
        for (int i = 0; i < POOL_SIZE; i++)
            sched.spawn(DeepCFR::rollout(CFRGame{}, hero, t, sched));

        // Run until the sample quota is met and all active rollouts have drained.
        while (sched.activeTasks() > 0) {
            sched.runOneBatch();
            int completed = sched.purgeCompleted();
            if (sched.advantageSize() < target)
                for (int i = 0; i < completed; i++)
                    sched.spawn(DeepCFR::rollout(CFRGame{}, hero, t, sched));
        }

        // Push a nullptr sentinel. Python's pop() loop counts one per thread
        // to know when all workers are finished for this iteration.
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
