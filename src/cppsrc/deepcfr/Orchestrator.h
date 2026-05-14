#pragma once

#include "Coroutine.h"
#include "Scheduler.h"

#include <condition_variable>
#include <cstdint>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

// ---------------------------------------------------------------------------
// Orchestrator — generic coroutine thread pool.
//
// Workers maintain a fixed pool of Task<float> coroutines.  Callers inject
// behaviour entirely through three closures passed to startIteration():
//
//   TaskFactory   — called once per new spawn; returns a ready-to-run task.
//   StopPredicate — called before each spawn; returning true drains the pool
//                   without spawning replacements.
//   WorkerSetup   — called once per worker at iteration start (while holding
//                   the iteration mutex) to configure per-Scheduler state.
//
// Purpose-specific logic (DeepCFR, TPO, Match) lives entirely in the callers
// of startIteration(), not here.
// ---------------------------------------------------------------------------

class Orchestrator {
  public:
    static constexpr int POOL_SIZE = 4096;

    using TaskFactory = std::function<Task<float>(Scheduler &)>;
    using StopPredicate = std::function<bool()>;
    using WorkerSetup = std::function<void(Scheduler &)>;

    InferenceQueue iq;
    std::vector<Scheduler *> schedulerPtrs; // valid for lifetime of threadPool_

    explicit Orchestrator(int numThreads,
                          uint64_t seed = 0xdeadbeefcafe1234ULL);
    ~Orchestrator();

    // Signal all threads to exit and join them.
    void drainPool();

    // Wake all workers with the given factory/stop/setup.  Non-blocking;
    // call waitIteration() to synchronise.
    void startIteration(TaskFactory factory, StopPredicate stop,
                        WorkerSetup setup = {});

    // Block until all workers have finished and pushed their sentinels.
    void waitIteration();

    Scheduler *pop() { return iq.pop(); }
    std::vector<Scheduler *> drain() { return iq.drain(); }
    int numThreads() const { return numThreads_; }

    void clearBuffers();
    std::vector<CoroFramePool::Stats> getPoolStats() const;
    void clearPoolStats();

  private:
    int numThreads_;
    uint64_t seed_;

    TaskFactory currentFactory_;
    StopPredicate currentStop_;
    WorkerSetup currentSetup_;

    int threadsParked_ = 0; // workers blocked in bottom wait
    int registeredCount_ = 0;

    std::mutex mtx_;
    std::condition_variable startCV_;
    std::condition_variable doneCV_;
    bool iterReady_ = false;
    bool shutdown_ = false;

    std::vector<std::thread> threadPool_;
    std::vector<CoroFramePool::Stats> poolStats_;

    void runWorker(int threadIdx);
};
