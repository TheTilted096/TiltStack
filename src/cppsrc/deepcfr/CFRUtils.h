#pragma once

#include "CFRTypes.h"

#include <cstdint>
#include <fstream>
#include <numeric>
#include <stdexcept>
#include <string>
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
// gLabels[r][idx]  →  cluster label for round r (index 0 unused).
// ---------------------------------------------------------------------------

inline std::vector<uint16_t> gLabels[NUM_ROUNDS];

// ---------------------------------------------------------------------------
// loadTables — populate gLabels from the cluster pipeline output.
//
// clusters_dir should contain:
//   flop_labels.bin    turn_labels.bin    river_labels.bin
//
// Preflop has no clustering: gLabels[0] is an identity mapping over the
// 169 canonical hand indices.
//
// Call once from the main thread before constructing any CFRGame.
// ---------------------------------------------------------------------------

inline void readU16File(const std::string &path, std::vector<uint16_t> &out) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f)
        throw std::runtime_error("Cannot open: " + path);
    auto bytes = static_cast<std::size_t>(f.tellg());
    if (bytes % 2 != 0)
        throw std::runtime_error("Odd file size (corrupt uint16 file?): " +
                                 path);
    f.seekg(0);
    out.resize(bytes / 2);
    f.read(reinterpret_cast<char *>(out.data()),
           static_cast<std::streamsize>(bytes));
    if (!f)
        throw std::runtime_error("Read failed: " + path);
}

// Defined in CFRGame.cpp — needs hand_indexer_init which is only available
// after OMPEval/hand-isomorphism includes are resolved in that TU.
// Also initialises g_indexer, so all callers get a ready-to-use indexer
// without needing a separate init step.
void loadTables(const std::string &clusters_dir);
