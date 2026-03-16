/*
    FlopExpander — compute 256-dim uint8 wide-bucket histograms for flop states.

    For each 5-card flop state (2 hole + 3 flop), all 47 remaining deck cards are
    tried as the turn card.  Each candidate is passed through the turn hand indexer
    to get its canonical turn state index, which is looked up in the pre-loaded
    turn_labels array to get a fine turn bucket in [0, 8192).  The fine bucket is
    mapped to a wide bucket in [0, 256) by integer division by 32.

    The result is a uint8[256] histogram where entry i holds the count of turn
    cards that land in wide bucket i.  All 256 counts sum to 47.  The Python
    pipeline computes the cumulative sum to obtain a float32 CDF vector before
    K-means training with L1 distance (equivalent to Earth Mover's Distance).

    turn_labels.bin is loaded fully into RAM at construction (~110 MB for the
    55M turn states).

    Two computation modes:
      compute_rows(indices, n, out)  — arbitrary indexed states (for sampling)
      compute_range(start, n, out)   — sequential block of states (full expansion)

    Both are parallelised with OpenMP and safe to call with the GIL released.

    Nathaniel Potter, 03-15-2026
*/

#pragma once

extern "C" {
#include "hand_index.h"
}

#include <cstdint>
#include <string>
#include <vector>

class FlopExpander {
    static constexpr int NUM_WIDE_BUCKETS = 256;  // 8192 fine turn buckets / 32
    static constexpr int FINE_PER_WIDE    = 32;
    static constexpr int NUM_CARDS        = 52;

    hand_indexer_t flop_indexer_;    // rounds [2, 3]    → 5-card canonical states
    hand_indexer_t turn_indexer_;    // rounds [2, 3, 1] → 6-card canonical states

    std::vector<uint16_t> turn_labels_;  // fully loaded from turn_labels.bin

    void computeRow(hand_index_t flop_idx, uint8_t* row) const;

public:
    static constexpr int DIMS = NUM_WIDE_BUCKETS;

    // Loads turn_labels_path fully into RAM on construction.
    explicit FlopExpander(const std::string& turn_labels_path);
    ~FlopExpander();

    // Total number of canonical flop states (hand_indexer_size at round 1).
    uint64_t num_states() const { return hand_indexer_size(&flop_indexer_, 1); }

    // Compute wide-bucket histograms for n arbitrary flop state indices.
    // out must point to a caller-allocated buffer of n * DIMS uint8 bytes.
    void compute_rows(const uint64_t* indices, size_t n, uint8_t* out) const;

    // Compute wide-bucket histograms for the sequential range [start, start+n).
    // out must point to a caller-allocated buffer of n * DIMS uint8 bytes.
    void compute_range(uint64_t start, int n, uint8_t* out) const;
};
