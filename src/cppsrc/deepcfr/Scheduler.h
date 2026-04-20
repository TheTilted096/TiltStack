#pragma once

#include "CFRTypes.h"
#include "Coroutine.h"
#include "InferenceQueue.h"
#include "Reservoir.h"

#include <condition_variable>
#include <coroutine>
#include <mutex>
#include <queue>
#include <vector>

using Handle = std::coroutine_handle<>;

// ---------------------------------------------------------------------------
// Scheduler
// Owns all per-thread state for one batch of DeepCFR rollouts on one thread.
//
// Synchronisation with the Python main thread:
//   C++ calls flushBatch()  → pushes this to inferenceQueue, then blocks
//   Python calls pop()      → wakes, wraps pendingInputs/pendingOutputs as
//   tensors, runs network Python calls submitBatch() → results written into
//   pendingOutputs, notifies cv C++ wakes, pushes all pending handles to ready,
//   coroutines resume and read pendingOutputs[index]
// ---------------------------------------------------------------------------

class Scheduler {
    // ---- Shared inference notification channel ------------------------------
    InferenceQueue &inferenceQueue;

    // ---- Coroutine queues ---------------------------------------------------
    std::queue<Handle> ready;

    // Parallel arrays — one slot per coroutine suspended at an inference point.
    // pendingInputs / pendingOutputs are contiguous so they can be wrapped as
    // flat PyTorch tensors without copying. Each awaitable stores its index and
    // reads pendingOutputs[index] directly in await_resume() via friend access.
    std::vector<InfoSet> pendingInputs;
    std::vector<Regrets> pendingOutputs;
    std::vector<Handle> pendingHandles;

    // ---- Root task ownership ------------------------------------------------
    std::vector<Task<float>> tasks;

    // ---- Batch synchronisation with Python ----------------------------------
    std::mutex batchMutex;
    std::condition_variable cv;
    bool batchComplete =
        false; // Python → C++: results written to pendingOutputs

    // ---- Internal -----------------------------------------------------------
    void flushBatch();

    friend struct InferenceAwaitable;

  public:
    int threadId;
    Reservoir *advReservoir = nullptr;
    Reservoir *polReservoir = nullptr;

    explicit Scheduler(InferenceQueue &q, int threadId)
        : inferenceQueue(q), threadId(threadId) {}

    // ---- Replay buffers — appended to by DeepCFR::rollout() -----------------
    std::vector<InfoSet> advantageInputs;
    std::vector<Regrets> advantageOutputs;

    std::vector<InfoSet> policyInputs;
    std::vector<Strategy> policyOutputs;
    std::vector<int> policyWeights;

    int completedRollouts = 0;
    int rolloutCount() const { return completedRollouts; }

    // ---- C++ interface ------------------------------------------------------

    void spawn(Task<float> task);

    // Run the ready queue until empty, then flush to Python if any coroutines
    // are suspended at inference. Returns once ready is empty again.
    void runOneBatch();

    // Remove completed root tasks from the tasks vector and return the count.
    // Call after runOneBatch() to free finished coroutine frames and learn how
    // many slots opened up for new spawns.
    int purgeCompleted();

    // Original one-shot drain — spawns nothing new, runs until all tasks done.
    // Kept for unit tests.
    void run();

    std::size_t enqueueInference(InfoSet input, Handle handle);

    // ---- Python interface ---------------------------------------------------

    // Called by Python after writing inference results into pendingOutputs.
    void submitBatch();

    // Raw pointers for zero-copy numpy / torch tensor construction (inference).
    InfoSet *inputData() { return pendingInputs.data(); }
    Regrets *outputData() { return pendingOutputs.data(); }
    int batchSize() { return static_cast<int>(pendingHandles.size()); }

    // Number of root tasks currently alive (spawned but not yet purged).
    int activeTasks() { return static_cast<int>(tasks.size()); }

    // Raw pointers for zero-copy numpy / torch tensor construction (training).
    InfoSet *advantageInputData() { return advantageInputs.data(); }
    Regrets *advantageOutputData() { return advantageOutputs.data(); }
    int advantageSize() { return static_cast<int>(advantageInputs.size()); }

    InfoSet *policyInputData() { return policyInputs.data(); }
    Strategy *policyOutputData() { return policyOutputs.data(); }
    int *policyWeightData() { return policyWeights.data(); }
    int policySize() { return static_cast<int>(policyInputs.size()); }

    // Reset all replay buffers. Called by the worker at the start of each
    // iteration so Python can safely harvest data between iterations.
    void clearBuffers();
};
