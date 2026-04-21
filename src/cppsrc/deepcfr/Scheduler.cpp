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
            // Re-throw any exception the coroutine stored in its promise.
            // Root tasks are never co_await-ed, so await_resume() — which
            // normally propagates the exception — is never called for them.
            // Checking here ensures coroutine failures are not silently
            // dropped.
            if (task.handle().promise().exception_)
                std::rethrow_exception(task.handle().promise().exception_);
        } else {
            alive.push_back(std::move(task));
        }
    }
    tasks = std::move(alive);
    completedRollouts += completed;
    return completed;
}

void Scheduler::clearBuffers() {
    advantageInputs.clear();
    advantageOutputs.clear();
    policyInputs.clear();
    policyOutputs.clear();
    policyWeights.clear();
    completedRollouts = 0;
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

    // While Python is running GPU inference, write accumulated data to the
    // global reservoirs. This overlaps reservoir writes with GPU computation,
    // eliminating the serial post-iteration collection phase entirely.
    if (advReservoir)
        advReservoir->insert(threadId, advantageInputs, advantageOutputs);
    if (polReservoir)
        polReservoir->insert(threadId, policyInputs, policyOutputs,
                             &policyWeights);

    advantageInputs.clear();
    advantageOutputs.clear();
    policyInputs.clear();
    policyOutputs.clear();
    policyWeights.clear();

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
