#pragma once

#include <condition_variable>
#include <mutex>
#include <queue>

// Forward declaration — avoids a circular include with Scheduler.h.
class Scheduler;

// ---------------------------------------------------------------------------
// InferenceQueue
// Shared across all worker threads. A worker pushes itself when its inference
// batch is full; the Python main thread pops and services one worker at a time.
// ---------------------------------------------------------------------------

class InferenceQueue {
    std::mutex mtx;
    std::condition_variable cv;
    std::queue<Scheduler *> ready;

  public:
    // Called by a worker thread inside flushBatch() when its batch is full.
    void push(Scheduler *s) {
        {
            std::unique_lock lock(mtx);
            ready.push(s);
        }
        cv.notify_one();
    }

    // Called by the Python main thread. Sleeps until any worker is ready,
    // then returns a pointer to that worker's Scheduler.
    Scheduler *pop() {
        std::unique_lock lock(mtx);
        cv.wait(lock, [this] { return !ready.empty(); });
        Scheduler *s = ready.front();
        ready.pop();
        return s;
    }
};
