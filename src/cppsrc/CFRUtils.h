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
// gEHS[r][idx] / 65535.0f  →  EHS for round r.
// gLabels[r][idx]           →  cluster label for round r (index 0 unused).
// ---------------------------------------------------------------------------

inline std::vector<uint16_t> gEHS[NUM_ROUNDS];
inline std::vector<uint16_t> gLabels[NUM_ROUNDS];

// ---------------------------------------------------------------------------
// loadTables — populate gEHS and gLabels from the cluster pipeline output.
//
// clusters_dir should contain:
//   preflop_ehs_fine.bin  (169 uint16)
//   flop_ehs_fine.bin     (1,286,792 uint16)   flop_labels.bin
//   turn_ehs_fine.bin     (~13.5M uint16)       turn_labels.bin
//   river_ehs_fine.bin    (~2.4B uint16)        river_labels.bin
//
// Preflop has no clustering: gLabels[0] is populated with an identity mapping
// (label == canonical hand index), covering all 169 hands directly.
//
// Call once from the main thread before constructing any CFRGame.
// ---------------------------------------------------------------------------

inline void readU16File(const std::string& path, std::vector<uint16_t>& out) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) throw std::runtime_error("Cannot open: " + path);
    auto bytes = static_cast<std::size_t>(f.tellg());
    if (bytes % 2 != 0)
        throw std::runtime_error("Odd file size (corrupt uint16 file?): " + path);
    f.seekg(0);
    out.resize(bytes / 2);
    f.read(reinterpret_cast<char*>(out.data()), static_cast<std::streamsize>(bytes));
    if (!f)
        throw std::runtime_error("Read failed: " + path);
}

inline void loadTables(const std::string& clusters_dir) {
    const std::string d = clusters_dir + "/";

    readU16File(d + "preflop_ehs_fine.bin", gEHS[0]);
    readU16File(d + "flop_ehs_fine.bin",    gEHS[1]);
    readU16File(d + "turn_ehs_fine.bin",    gEHS[2]);
    readU16File(d + "river_ehs_fine.bin",   gEHS[3]);

    // Preflop: no cluster labels — use identity (bucket == canonical index).
    gLabels[0].resize(gEHS[0].size());
    std::iota(gLabels[0].begin(), gLabels[0].end(), uint16_t{0});

    readU16File(d + "flop_labels.bin",  gLabels[1]);
    readU16File(d + "turn_labels.bin",  gLabels[2]);
    readU16File(d + "river_labels.bin", gLabels[3]);
}
