/*
    RiverExpander — compute 169-dim uint8 equity vectors, per-state EHS values,
    and suit-isomorphism multiplicities for river states.

    The equity vector for a river state encodes, for each of the 169 canonical
    preflop opponent hand classes, the hero's equity against that class.
    Values are quantised: byte / 255.0 = equity.

    Per-state EHS is the multiplicity-weighted equity over all concrete opponent
    hands (totalEqSum / totalCount), returned as a float in [0, 1].

    All compute methods are parallelised with OpenMP and safe to call with the
    GIL released.

    Nathaniel Potter, 03-08-2026
*/

#pragma once

// Include OMPEval before hand-isomorphism: deck.h defines CARDS/RANKS/SUITS
// macros that collide with OMPEval identifiers.
#include "../../third_party/OMPEval/HandEvaluator.h"

extern "C" {
#include "../../third_party/hand-isomorphism/hand_index.h"
}

#include <array>
#include <cstdint>

class RiverExpander {
    static constexpr int NUM_BUCKETS = 169;
    static constexpr int NUM_CARDS = 52;

    hand_indexer_t river_indexer_;
    omp::HandEvaluator evaluator_;
    std::array<omp::Hand, NUM_CARDS> card_hands_;
    std::array<std::array<hand_index_t, NUM_CARDS>, NUM_CARDS> bucket_of_;

    void computeRow(hand_index_t idx, uint8_t *row) const;
    void computeRowEhsMult(hand_index_t idx, uint8_t *row, float *ehs_out,
                           uint8_t *mult_out) const;
    uint8_t computeMult(hand_index_t idx) const;

  public:
    static constexpr uint64_t NUM_STATES = 2428287420ULL;
    static constexpr int DIMS = NUM_BUCKETS;

    RiverExpander();
    ~RiverExpander();

    uint64_t num_states() const { return NUM_STATES; }

    // Compute equity vectors for n arbitrary state indices (sampling step).
    // out must point to a buffer of n * DIMS uint8 bytes.
    void compute_rows(const uint64_t *indices, size_t n, uint8_t *out) const;

    // Compute equity vectors, EHS, and multiplicities for [start, start+n)
    // in a single parallel pass (streaming assignment step).
    // row_out:  n * DIMS uint8 bytes
    // ehs_out:  n floats  (decode directly; already in [0, 1])
    // mult_out: n uint8 bytes  (suit-isomorphism multiplicities in [1, 24])
    void compute_range_ehs_mult(uint64_t start, int n, uint8_t *row_out,
                                float *ehs_out, uint8_t *mult_out) const;
};
