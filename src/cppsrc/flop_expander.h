/*
    FlopExpander — compute 256-dim uint8 wide-bucket histograms and per-state
    EHS values for flop states in a single pass.

    Histogram mode:
      For each 5-card flop state, the 47 possible turn cards are enumerated.
      Each is mapped through the turn hand indexer to a canonical turn state,
      then to a fine turn cluster label in [0, 8192), then to a wide bucket in
      [0, 256) by integer division by 32.  The result is a uint8[256] count
      histogram summing to 47.

    EHS mode:
      For each flop state, the 47 turn EHS values are averaged to give the
      exact per-state flop EHS in [0, 1].

    Both files are required at construction; both computation modes are always
    available.  Multiplicity computation uses only the flop_indexer_ internals.

    Files loaded fully into RAM:
      turn_labels.bin   ~110 MB (uint16, ~55M entries)
      turn_ehs_fine.bin ~110 MB (uint16, ~55M entries; decode: value / 65535.0)

    All compute methods are parallelised with OpenMP and safe to call with the
    GIL released.

    Nathaniel Potter, 03-15-2026
*/

#pragma once

// Include OMPEval before hand-isomorphism: deck.h defines CARDS/RANKS/SUITS
// macros that collide with OMPEval identifiers.
#include "../third_party/OMPEval/HandEvaluator.h"

extern "C" {
#include "hand_index.h"
}

#include <cstdint>
#include <string>
#include <vector>

class FlopExpander {
    static constexpr int NUM_WIDE_BUCKETS = 256; // 8192 fine turn buckets / 32
    static constexpr int FINE_PER_WIDE = 32;
    static constexpr int NUM_CARDS = 52;

    hand_indexer_t flop_indexer_; // rounds [2, 3]    → 5-card canonical states
    hand_indexer_t turn_indexer_; // rounds [2, 3, 1] → 6-card canonical states

    std::vector<uint16_t> turn_labels_;   // loaded from turn_labels.bin
    std::vector<uint16_t> turn_ehs_fine_; // loaded from turn_ehs_fine.bin;
                                          // decode: value / 65535.0f

    void computeRowEhsMult(hand_index_t flop_idx, uint8_t *row, float *ehs_out,
                           uint8_t *mult_out) const;
    uint8_t computeMult(hand_index_t flop_idx) const;

  public:
    static constexpr int DIMS = NUM_WIDE_BUCKETS;

    // Both files are required and loaded fully into RAM on construction
    // (~110 MB for turn_labels, ~110 MB for turn_ehs_fine).
    explicit FlopExpander(const std::string &turn_labels_path,
                          const std::string &turn_ehs_fine_path);
    ~FlopExpander();

    // Total number of canonical flop states (hand_indexer_size at round 1).
    uint64_t num_states() const { return hand_indexer_size(&flop_indexer_, 1); }

    // Compute histograms, EHS, and multiplicities for n arbitrary state indices
    // in a single parallel pass (used for the full flop dataset in one shot).
    // row_out:  n * DIMS uint8 bytes
    // ehs_out:  n floats  (already in [0, 1])
    // mult_out: n uint8 bytes  (suit-isomorphism multiplicities in [1, 24])
    void compute_rows_ehs_mult(const uint64_t *indices, size_t n,
                               uint8_t *row_out, float *ehs_out,
                               uint8_t *mult_out) const;
};
