#include "Scheduler.h"

void Scheduler::spawn(Task<float> task) {
    ready.push(task.handle());
    tasks.push_back(std::move(task));
}

void Scheduler::runOneBatch() {
    while (!ready.empty()) {
        Handle h = ready.front();
        ready.pop();
        h.resume();
    }
    if (!pendingHandles.empty())
        flushBatch();
}

int Scheduler::purgeCompleted() {
    std::vector<Task<float>> alive;
    alive.reserve(tasks.size());
    int completed = 0;
    for (auto &task : tasks) {
        if (task.done()) {
            ++completed;
        } else {
            alive.push_back(std::move(task));
        }
    }
    tasks = std::move(alive);
    return completed;
}

void Scheduler::run() {
    while (true) {
        while (!ready.empty()) {
            Handle h = ready.front();
            ready.pop();
            h.resume();
        }

        if (pendingHandles.empty())
            break;

        flushBatch();
    }
}

void Scheduler::clearBuffers() {
    advantageInputs.clear();
    advantageOutputs.clear();
    policyInputs.clear();
    policyOutputs.clear();
    policyWeights.clear();
}

std::size_t Scheduler::enqueueInference(InfoSet input, Handle handle) {
    std::size_t index = pendingHandles.size();
    pendingInputs.push_back(input);
    pendingHandles.push_back(handle);
    return index;
}

void Scheduler::flushBatch() {
    // Allocate fresh output slots for this batch. Previous batch results have
    // already been consumed by await_resume() before we get here.
    pendingOutputs.assign(pendingHandles.size(), Regrets{});

    // Tell the main thread this worker is ready, then sleep until it's done.
    {
        std::unique_lock lock(batchMutex);
        batchComplete = false;
    }
    inferenceQueue.push(this);

    {
        std::unique_lock lock(batchMutex);
        cv.wait(lock, [this] { return batchComplete; });
    }

    // pendingOutputs is left intact until the next flushBatch() — resumed
    // coroutines read their result via the index stored in their awaitable.
    for (Handle h : pendingHandles)
        ready.push(h);

    pendingInputs.clear();
    pendingHandles.clear();
}

void Scheduler::submitBatch() {
    {
        std::unique_lock lock(batchMutex);
        batchComplete = true;
    }
    cv.notify_one();
}
