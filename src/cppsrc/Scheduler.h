#pragma once

#include "Coroutine.h"
#include "CFRTypes.h"
#include "InferenceQueue.h"

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
//   Python calls pop()      → wakes, wraps pendingInputs/pendingOutputs as tensors, runs network
//   Python calls submitBatch() → results written into pendingOutputs, notifies cv
//   C++ wakes, pushes all pending handles to ready, coroutines resume and read pendingOutputs[index]
// ---------------------------------------------------------------------------

class Scheduler {
    // ---- Shared inference notification channel ------------------------------
    InferenceQueue& inferenceQueue;

    // ---- Coroutine queues ---------------------------------------------------
    std::queue<Handle> ready;

    // Parallel arrays — one slot per coroutine suspended at an inference point.
    // pendingInputs / pendingOutputs are contiguous so they can be wrapped as
    // flat PyTorch tensors without copying. Each awaitable stores its index and
    // reads pendingOutputs[index] directly in await_resume() via friend access.
    std::vector<InfoSet>  pendingInputs;
    std::vector<Regrets>  pendingOutputs;
    std::vector<Handle>   pendingHandles;

    // ---- Root task ownership ------------------------------------------------
    std::vector<Task<float>> tasks;

    // ---- Batch synchronisation with Python ----------------------------------
    std::mutex              batchMutex;
    std::condition_variable cv;
    bool batchComplete = false;   // Python → C++: results written to pendingOutputs

    // ---- Internal -----------------------------------------------------------
    void flushBatch();

    friend struct InferenceAwaitable;

public:
    explicit Scheduler(InferenceQueue& q) : inferenceQueue(q) {}

    // ---- Replay buffers — appended to by DeepCFR::rollout() -----------------
    std::vector<InfoSet>  advantageInputs;
    std::vector<Regrets>  advantageOutputs;

    std::vector<InfoSet>   policyInputs;
    std::vector<Strategy>  policyOutputs;
    std::vector<int>  policyWeights;

    // ---- C++ interface ------------------------------------------------------

    void spawn(Task<float> task);
    void run();
    std::size_t enqueueInference(InfoSet input, Handle handle);

    // ---- Python interface ---------------------------------------------------

    // Called by Python after writing inference results into pendingOutputs.
    void submitBatch();

    // Raw pointers for zero-copy numpy / torch tensor construction.
    InfoSet* inputData()  { return pendingInputs.data(); }
    Regrets* outputData() { return pendingOutputs.data(); }
    int      batchSize()  { return static_cast<int>(pendingHandles.size()); }
};
