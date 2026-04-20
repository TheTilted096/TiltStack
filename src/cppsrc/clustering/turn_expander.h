/*
    TurnExpander — compute 256-dim uint8 wide-bucket histograms, per-state EHS
    values, and suit-isomorphism multiplicities for turn states.

    Both river files must be loaded at construction time:
      river_labels.bin   ~4.9 GB  (uint16, 2.4 B entries) — for histogram
   computation river_ehs_fine.bin ~4.9 GB  (uint16, 2.4 B entries) — for
   per-state EHS (decode: value / 65535.0)

    This enables a single streaming pass over all turn states that
   simultaneously produces cluster labels and per-state EHS (as done for
   flop/river).

    All compute methods are parallelised with OpenMP and safe to call with the
    GIL released.

    Nathaniel Potter, 03-15-2026
*/

#pragma once

// Include OMPEval before hand-isomorphism: deck.h defines CARDS/RANKS/SUITS
// macros that collide with OMPEval identifiers.
#include "HandEvaluator.h"

extern "C" {
#define _Bool bool
#include "hand_index.h"
#undef _Bool
}

#include <cstdint>
#include <string>
#include <vector>

class TurnExpander {
    static constexpr int NUM_WIDE_BUCKETS = 256; // 8192 fine buckets / 32
    static constexpr int FINE_PER_WIDE = 32;
    static constexpr int NUM_CARDS = 52;

    hand_indexer_t
        turn_indexer_; // rounds [2, 3, 1]    → 6-card canonical states
    hand_indexer_t
        river_indexer_; // rounds [2, 3, 1, 1] → 7-card canonical states

    std::vector<uint16_t> river_labels_;   // loaded from river_labels.bin
    std::vector<uint16_t> river_ehs_fine_; // loaded from river_ehs_fine.bin;
                                           // decode: value / 65535.0f

    void computeRow(hand_index_t turn_idx, uint8_t *row) const;
    void computeRowEhsMult(hand_index_t turn_idx, uint8_t *row, float *ehs_out,
                           uint8_t *mult_out) const;
    uint8_t computeMult(hand_index_t turn_idx) const;

  public:
    static constexpr int DIMS = NUM_WIDE_BUCKETS;

    // Both paths are required; both files are loaded fully into RAM.
    explicit TurnExpander(const std::string &river_labels_path,
                          const std::string &river_ehs_fine_path);
    ~TurnExpander();

    // Total number of canonical turn states (hand_indexer_size at round 2).
    uint64_t num_states() const { return hand_indexer_size(&turn_indexer_, 2); }

    // Compute wide-bucket histograms for n arbitrary turn state indices.
    // out must point to a caller-allocated buffer of n * DIMS uint8 bytes.
    void compute_rows(const uint64_t *indices, size_t n, uint8_t *out) const;

    // Compute histograms, EHS, and multiplicities for [start, start+n)
    // in a single parallel pass (streaming assignment step).
    // row_out:  n * DIMS uint8 bytes
    // ehs_out:  n floats  (already in [0, 1])
    // mult_out: n uint8 bytes  (suit-isomorphism multiplicities in [1, 24])
    void compute_range_ehs_mult(uint64_t start, int n, uint8_t *row_out,
                                float *ehs_out, uint8_t *mult_out) const;
};
