#pragma once

#include "CFRTypes.h"
#include "CFRUtils.h"
#include <atomic>
#include <cstddef>
#include <vector>

// ---------------------------------------------------------------------------
// Reservoir
//
// Thread-safe, lock-free fixed-capacity reservoir sampler (Algorithm R).
// The underlying memory buffers are owned by the caller (Python, for
// PyTorch pinned-memory / zero-copy access) and must remain valid for the
// lifetime of this object.
//
// Parallelism strategy — disjoint slices:
//   The capacity is divided into numThreads equal slices. Thread i owns
//   slots [i*sliceSize, (i+1)*sliceSize) and operates on them entirely
//   independently. Because each slice is itself a uniform reservoir sample
//   of the items that thread contributed, the combined reservoir is a
//   uniform sample of the global stream (assuming items are distributed
//   roughly uniformly across threads, which they are for fixed-pool rollouts).
//   No locking is needed at any point.
//
// seenPerThread_ tracks how many items each thread has contributed. It is
// indexed exclusively by that thread, so it does not need to be atomic.
// alignas(64) causes sizeof(PaddedCounter) to be rounded up to 64 by the
// compiler (required for arrays to preserve each element's alignment), so
// each entry occupies its own cache line without an explicit pad array.
//
// Intended calling sequence in Scheduler::flushBatch(), after
// inferenceQueue.push() unblocks Python and before cv.wait():
//
//   advReservoir->insert(threadId, advantageInputs, advantageOutputs);
//   polReservoir->insert(threadId, policyInputs, policyOutputs,
//   &policyWeights); advantageInputs.clear();  advantageOutputs.clear();
//   policyInputs.clear();     policyOutputs.clear();  policyWeights.clear();
// ---------------------------------------------------------------------------

struct alignas(64) PaddedCounter {
    std::size_t value = 0;
};

class Reservoir {
  public:
    // capacity   : maximum samples to retain
    // numThreads : number of worker threads; determines slice boundaries
    // inputBuf   : caller-owned buffer of capacity × sizeof(InfoSet) bytes
    // (uint8) targetBuf  : caller-owned buffer of capacity × NUM_ACTIONS floats
    // weightBuf  : caller-owned buffer of capacity int32s; nullptr if unused
    Reservoir(std::size_t capacity, int numThreads, uint8_t *inputBuf,
              float *targetBuf, int32_t *weightBuf = nullptr);

    // Lock-free batch insert. threadID must be in [0, numThreads).
    //
    // Regrets and Strategy are both array<float, NUM_ACTIONS> — the same
    // concrete type — so this single overload serves both reservoir kinds.
    // Pass &policyWeights for policy reservoirs; nullptr (default) for adv.
    void insert(int threadID, const std::vector<InfoSet> &inputs,
                const std::vector<Regrets> &targets,
                const std::vector<int32_t> *weights = nullptr);

    // Total items ever offered across all threads (not clamped to capacity).
    std::atomic<std::size_t> nSeen{0};

    // Number of valid entries in the buffer: min(nSeen, capacity).
    std::size_t size() const;

    // Reset to initial state for reuse across iterations.
    // Must not be called while worker threads are active.
    void reset();

  private:
    std::size_t capacity_;
    int numThreads_;
    std::size_t
        sliceSize_; // capacity_ / numThreads_ (last thread may be larger)

    uint8_t *inputBuf_;  // Python-owned
    float *targetBuf_;   // Python-owned
    int32_t *weightBuf_; // Python-owned (nullptr if no weights)

    // Per-thread local item counts for Algorithm R stream positions.
    // Only thread i ever touches seenPerThread_[i], so no atomic needed.
    std::vector<PaddedCounter> seenPerThread_;

    void writeSlot(std::size_t dst, const uint8_t *inputBase,
                   const float *targetBase, const int32_t *weightBase,
                   std::size_t bIdx);
};
