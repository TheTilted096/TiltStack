#pragma once

#include "CFRTypes.h"

#include <cstdint>
#include <vector>

// ---------------------------------------------------------------------------
// PRNG — xorshift64*.
// One instance per thread, shared by CFRGame and DeepCFR.
// Seed rng before training begins.
// ---------------------------------------------------------------------------

class RNG {
    uint64_t state_;
public:
    explicit RNG(uint64_t seed = 0xdeadbeefcafe1234ULL) : state_(seed) {}

    void seed(uint64_t s) { state_ = s; }

    uint64_t next() {
        state_ ^= state_ >> 12;
        state_ ^= state_ << 25;
        state_ ^= state_ >> 27;
        return state_ * 0x2545F4914F6CDD1DULL;
    }

    // Returns a float in [0, 1).
    float nextFloat() { return (next() >> 40) * 0x1.0p-24f; }
};

inline thread_local RNG rng;

// ---------------------------------------------------------------------------
// Cluster tables — populate before constructing any CFRGame.
// gEHS[r][idx] / 65535.0f  →  EHS for round r.
// gLabels[r][idx]           →  cluster label for round r (index 0 unused).
// ---------------------------------------------------------------------------

inline std::vector<uint16_t> gEHS[NUM_ROUNDS];    // TODO: load from *_ehs_fine.bin
inline std::vector<uint16_t> gLabels[NUM_ROUNDS]; // TODO: load from *_labels.bin
