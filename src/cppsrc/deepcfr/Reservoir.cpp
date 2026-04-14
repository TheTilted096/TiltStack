#include "Reservoir.h"
#include <algorithm>
#include <cstring>

Reservoir::Reservoir(std::size_t capacity, int numThreads,
                     uint8_t* inputBuf, float* targetBuf, int32_t* weightBuf)
    : capacity_(capacity), numThreads_(numThreads),
      sliceSize_(capacity / static_cast<std::size_t>(numThreads)),
      inputBuf_(inputBuf), targetBuf_(targetBuf), weightBuf_(weightBuf),
      seenPerThread_(numThreads) {
    // Each thread's phase-2 stream position starts at its slice size,
    // reflecting that phase 1 already filled every slot in the reservoir.
    // This ensures the first phase-2 item has accept probability
    // sliceSize/(sliceSize+1) rather than sliceSize/1 >> 1.
    for (int i = 0; i < numThreads; ++i) {
        const std::size_t sliceStart = static_cast<std::size_t>(i) * sliceSize_;
        seenPerThread_[i].value = (i == numThreads - 1)
            ? capacity_ - sliceStart
            : sliceSize_;
    }
}

std::size_t Reservoir::size() const {
    return std::min(nSeen.load(std::memory_order_relaxed), capacity_);
}

void Reservoir::writeSlot(std::size_t dst,
                           const uint8_t*  inputBase,
                           const float*    targetBase,
                           const int32_t*  weightBase,
                           std::size_t     bIdx) {
    std::memcpy(inputBuf_  + dst  * sizeof(InfoSet),
                inputBase  + bIdx * sizeof(InfoSet),
                sizeof(InfoSet));
    std::memcpy(targetBuf_ + dst  * NUM_ACTIONS,
                targetBase + bIdx * NUM_ACTIONS,
                NUM_ACTIONS * sizeof(float));
    if (weightBuf_ && weightBase)
        weightBuf_[dst] = weightBase[bIdx];
}

void Reservoir::insert(int threadID,
                        const std::vector<InfoSet>& inputs,
                        const std::vector<Regrets>& targets,
                        const std::vector<int32_t>* weights) {
    const std::size_t B = inputs.size();
    if (B == 0) return;

    // Atomically claim a contiguous range of global stream positions.
    // This is the sole shared operation; all writes below are either to
    // uniquely owned fill slots (phase 1) or to this thread's disjoint slice
    // (phase 2), so no further synchronisation is needed.
    const std::size_t base = nSeen.fetch_add(B, std::memory_order_relaxed);

    const uint8_t*  inData  = reinterpret_cast<const uint8_t*>(inputs.data());
    const float*    tgtData = reinterpret_cast<const float*>(targets.data());
    const int32_t*  wgtData = weights ? weights->data() : nullptr;

    // ---- Phase 1: fill empty slots directly --------------------------------
    // Slots [base, fillEnd) are uniquely owned by this call — no other
    // fetch_add can return an overlapping range.
    if (base < capacity_) {
        const std::size_t fillEnd = std::min(base + B, capacity_);
        for (std::size_t pos = base; pos < fillEnd; ++pos)
            writeSlot(pos, inData, tgtData, wgtData, pos - base);

        // If this batch straddles the fill boundary, discard the excess rather
        // than starting Algorithm R mid-batch. The next batch from this thread
        // will have base >= capacity_ and fully enter phase 2.
        return;
    }

    // ---- Phase 2: Algorithm R within this thread's slice (lock-free) -------
    // The reservoir is full. Thread i exclusively manages slots
    // [i*sliceSize, (i+1)*sliceSize). seenPerThread_[threadID] counts how
    // many items this thread has contributed since saturation, giving the
    // local stream position for the acceptance probability calculation.
    // Uses the thread_local xorshift64* RNG from CFRUtils.h, already seeded
    // per-thread by the Orchestrator.
    
    const std::size_t sliceStart = static_cast<std::size_t>(threadID) * sliceSize_;
    const std::size_t sliceSize  = (threadID == numThreads_ - 1)
        ? capacity_ - sliceStart : sliceSize_;

    const std::size_t localBase = seenPerThread_[threadID].value;
    seenPerThread_[threadID].value += B;

    for (std::size_t pos = localBase; pos < localBase + B; ++pos) {
        if (rng.nextFloat() * static_cast<double>(pos + 1) < sliceSize) {
            const std::size_t slot = sliceStart + rng.next() % sliceSize;
            writeSlot(slot, inData, tgtData, wgtData, pos - localBase);
        }
    }
}
