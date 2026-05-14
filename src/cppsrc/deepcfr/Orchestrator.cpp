#include "Orchestrator.h"
#include "CFRUtils.h"

Orchestrator::Orchestrator(int numThreads, uint64_t seed)
    : numThreads_(numThreads), seed_(seed) {
    schedulerPtrs.resize(numThreads, nullptr);
    threadPool_.reserve(numThreads);
    for (int i = 0; i < numThreads; i++)
        threadPool_.emplace_back(&Orchestrator::runWorker, this, i);

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

void Orchestrator::startIteration(TaskFactory factory, StopPredicate stop,
                                  WorkerSetup setup) {
    {
        std::unique_lock lock(mtx_);
        currentFactory_ = std::move(factory);
        currentStop_ = std::move(stop);
        currentSetup_ = setup ? std::move(setup) : [](Scheduler &) {};
        iterReady_ = true;
    }
    startCV_.notify_all();
}

void Orchestrator::waitIteration() {
    std::unique_lock lock(mtx_);
    doneCV_.wait(lock, [this] { return threadsParked_ == numThreads_; });
    iterReady_ = false;
    startCV_.notify_all(); // workers are provably in bottom wait — safe to
                           // notify under lock
}

void Orchestrator::clearBuffers() {
    for (Scheduler *s : schedulerPtrs)
        s->clearBuffers();
}

std::vector<CoroFramePool::Stats> Orchestrator::getPoolStats() const {
    std::unique_lock lock(const_cast<std::mutex &>(mtx_));
    return poolStats_;
}

void Orchestrator::clearPoolStats() {
    std::unique_lock lock(mtx_);
    poolStats_.clear();
}

void Orchestrator::runWorker(int threadIdx) {
    Scheduler sched(iq, threadIdx);

    {
        std::unique_lock lock(mtx_);
        schedulerPtrs[threadIdx] = &sched;
        if (++registeredCount_ == numThreads_)
            doneCV_.notify_one();
    }

    rng.seed(seed_ + static_cast<uint64_t>(threadIdx));

    while (true) {
        TaskFactory factory;
        StopPredicate shouldStop;

        {
            std::unique_lock lock(mtx_);
            startCV_.wait(lock, [this] { return iterReady_ || shutdown_; });
            if (shutdown_)
                break;
            currentSetup_(sched);
            factory = currentFactory_;
            shouldStop = currentStop_;
        }

        sched.clearBuffers();
        coroPool.resetStats();

        for (int i = 0; i < POOL_SIZE; i++) {
            if (shouldStop())
                break;
            sched.spawn(factory(sched));
        }

        while (sched.activeTasks() > 0) {
            sched.runOneBatch();
            int completed = sched.purgeCompleted();
            for (int i = 0; i < completed; i++) {
                if (shouldStop())
                    break;
                sched.spawn(factory(sched));
            }
        }

        // Flush any training data accumulated after the last flushBatch().
        if (sched.advReservoir)
            sched.advReservoir->insert(threadIdx, sched.advantageInputs,
                                       sched.advantageOutputs);
        if (sched.polReservoir)
            sched.polReservoir->insert(threadIdx, sched.policyInputs,
                                       sched.policyOutputs,
                                       &sched.policyWeights);

        {
            std::unique_lock lock(mtx_);
            poolStats_.push_back(coroPool.getStats());
        }

        iq.push(nullptr);

        {
            std::unique_lock lock(mtx_);
            if (++threadsParked_ == numThreads_)
                doneCV_.notify_one();
            startCV_.wait(lock, [this] { return !iterReady_ || shutdown_; });
            --threadsParked_;
        }
    }
}
