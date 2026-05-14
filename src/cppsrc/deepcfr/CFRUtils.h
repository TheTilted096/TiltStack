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

// ---------------------------------------------------------------------------
// Card dealing — partial Fisher-Yates over a 52-card deck using the
// thread-local RNG.  Writes 9 unique card indices into out[0..8].
// ---------------------------------------------------------------------------

inline void dealCards(std::array<Card, 9> &out) {
    Card deck[52];
    for (int i = 0; i < 52; i++)
        deck[i] = static_cast<Card>(i);
    for (int i = 0; i < 9; i++) {
        int j =
            i + static_cast<int>(rng.next() % static_cast<uint64_t>(52 - i));
        std::swap(deck[i], deck[j]);
        out[i] = deck[i];
    }
}

// ---------------------------------------------------------------------------
// Legal-action strategy helpers
// ---------------------------------------------------------------------------

// Select the legal action with the highest value in logits.
inline Action argmaxLegal(const Regrets &logits, const ActionList &moves,
                          int numMoves) {
    Action best = moves[0];
    float bestVal = logits[static_cast<int>(moves[0])];
    for (int i = 1; i < numMoves; i++) {
        float v = logits[static_cast<int>(moves[i])];
        if (v > bestVal) {
            bestVal = v;
            best = moves[i];
        }
    }
    return best;
}

// Renormalize softmax probability mass to the legal action subset.
// Falls back to uniform if all legal probabilities are zero.
inline Strategy normalizeLegal(const Regrets &probs, const ActionList &moves,
                               int numMoves) {
    float sum = 0.0f;
    for (int i = 0; i < numMoves; i++)
        sum += probs[static_cast<int>(moves[i])];
    Strategy s{};
    if (sum > 1e-8f) {
        for (int i = 0; i < numMoves; i++) {
            int a = static_cast<int>(moves[i]);
            s[a] = probs[a] / sum;
        }
    } else {
        float u = 1.0f / static_cast<float>(numMoves);
        for (int i = 0; i < numMoves; i++)
            s[static_cast<int>(moves[i])] = u;
    }
    return s;
}

// Set illegal action slots to NaN so Python loss masking can detect them via
// isnan().  Call after normalizeLegal() to combine normalization and masking.
inline void nanMaskIllegal(Strategy &s, const ActionList &moves, int numMoves) {
    Strategy masked;
    masked.fill(std::numeric_limits<float>::quiet_NaN());
    for (int i = 0; i < numMoves; i++) {
        int a = static_cast<int>(moves[i]);
        masked[a] = s[a];
    }
    s = masked;
}

// Sample an action from a strategy distribution using the thread-local RNG.
inline Action sampleAction(const Strategy &strat, const ActionList &moves,
                           int numMoves) {
    float sample = rng.nextFloat();
    float cumulative = 0.0f;
    for (int i = 0; i < numMoves; i++) {
        cumulative += strat[static_cast<int>(moves[i])];
        if (sample < cumulative)
            return moves[i];
    }
    return moves[numMoves - 1];
}
