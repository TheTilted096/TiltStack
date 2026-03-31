#pragma once

#include "CFRTypes.h"

#include <cstdint>
#include <vector>

// ---------------------------------------------------------------------------
// PRNG — xorshift64*, shared across all CFRGame instances.
// Seed rngState before training begins.
// ---------------------------------------------------------------------------

inline uint64_t rngState = 0xdeadbeefcafe1234ULL;

inline uint64_t fastRand() {
    rngState ^= rngState >> 12;
    rngState ^= rngState << 25;
    rngState ^= rngState >> 27;
    return rngState * 0x2545F4914F6CDD1DULL;
}

// ---------------------------------------------------------------------------
// Cluster tables — populate before constructing any CFRGame.
// gEHS[r][idx] / 65535.0f  →  EHS for round r.
// gLabels[r][idx]           →  cluster label for round r (index 0 unused).
// ---------------------------------------------------------------------------

inline std::vector<uint16_t> gEHS[NUM_ROUNDS];    // TODO: load from *_ehs_fine.bin
inline std::vector<uint16_t> gLabels[NUM_ROUNDS]; // TODO: load from *_labels.bin
