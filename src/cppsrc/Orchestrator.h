#pragma once

#include "Scheduler.h"

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <thread>
#include <vector>

class Orchestrator {
  public:
    // Number of concurrent rollouts each worker maintains.
    static constexpr int POOL_SIZE = 4096;

    InferenceQueue           iq;
    std::vector<Scheduler *> schedulerPtrs; // valid for the lifetime of threadPool_

    explicit Orchestrator(int numThreads,
                          uint64_t seed = 0xdeadbeefcafe1234ULL);
    ~Orchestrator();

    // Signal all threads to exit and join them.
    void drainPool();

    // Begin an iteration. Non-blocking: wakes all worker threads and returns
    // immediately so the caller can enter the inference service loop.
    // totalSamples is divided evenly across threads.
    void startIteration(bool hero, int t, int totalSamples = 10000000);

    // Block until all workers have finished and pushed their sentinels.
    // Resets internal state so the next startIteration() can be called.
    void waitIteration();

    // Proxy for iq.pop(). Returns nullptr as a per-thread sentinel — Python's
    // service loop should count numThreads nullptrs before calling waitIteration().
    Scheduler *pop() { return iq.pop(); }

    int numThreads() const { return numThreads_; }

    // Reset all per-thread replay buffers. Call after waitIteration() and
    // before harvesting training data for the next iteration, if needed.
    void clearBuffers();

  private:
    int      numThreads_;
    uint64_t seed_;

    // Per-iteration parameters written by startIteration(), read by workers
    // while holding mtx_ at wakeup.
    bool currentHero_      = false;
    int  currentT_         = 0;
    int  samplesPerThread_ = 0;

    std::atomic<int>        threadsActive_{0};
    int                     registeredCount_ = 0;

    std::mutex              mtx_;
    std::condition_variable startCV_; // workers wait here between iterations
    std::condition_variable doneCV_;  // constructor + waitIteration() wait here
    bool                    iterReady_ = false;
    bool                    shutdown_  = false;

    std::vector<std::thread> threadPool_;

    void runWorker(int threadIdx);
};
