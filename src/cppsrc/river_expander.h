/*
    RiverExpander — compute 169-dim uint8 equity vectors for river states.

    The equity vector for a river state encodes, for each of the 169 canonical
    preflop opponent hand classes, the hero's equity against that class on this
    board/hand combination.  Values are quantised: byte / 255.0 = equity.

    Two computation modes:
      compute_rows(indices, n, out)  — arbitrary indexed states (for sampling)
      compute_range(start, n, out)   — sequential block of states (for full expansion)

    Both methods are parallelised with OpenMP and safe to call with the Python
    GIL released.
*/

#pragma once

// Include OMPEval before hand-isomorphism: deck.h defines CARDS/RANKS/SUITS
// macros that collide with OMPEval identifiers.
#include "../third_party/OMPEval/omp/HandEvaluator.h"

extern "C" {
#include "hand_index.h"
}

#include <array>
#include <cstdint>

class RiverExpander {
    static constexpr int NUM_BUCKETS = 169;
    static constexpr int NUM_CARDS   = 52;

    hand_indexer_t river_indexer_;
    omp::HandEvaluator evaluator_;
    std::array<omp::Hand, NUM_CARDS> card_hands_;
    std::array<std::array<hand_index_t, NUM_CARDS>, NUM_CARDS> bucket_of_;

    void computeRow(hand_index_t idx, uint8_t* row) const;

public:
    static constexpr uint64_t NUM_STATES = 2428287420ULL;
    static constexpr int      DIMS       = NUM_BUCKETS;

    RiverExpander();
    ~RiverExpander();

    uint64_t num_states() const { return NUM_STATES; }

    // Compute equity vectors for n arbitrary state indices.
    // out must point to a buffer of n * DIMS uint8 bytes.
    void compute_rows(const uint64_t* indices, size_t n, uint8_t* out) const;

    // Compute equity vectors for the sequential range [start, start+n).
    // out must point to a buffer of n * DIMS uint8 bytes.
    void compute_range(uint64_t start, int n, uint8_t* out) const;
};
