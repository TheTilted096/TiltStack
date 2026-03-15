/*
    TurnExpander — compute 256-dim uint8 wide-bucket histograms for turn states.

    For each 6-card turn state (2 hole + 3 flop + 1 turn), all 46 remaining
    deck cards are tried as the river card.  Each candidate is passed through
    the river hand indexer to get its canonical river state index, which is
    looked up in the pre-loaded river_labels array to get a fine river bucket
    in [0, 8192).  The fine bucket is mapped to a wide bucket in [0, 256) by
    integer division by 32.

    The result is a uint8[256] histogram where entry i holds the count of river
    cards that land in wide bucket i.  All 256 counts sum to 46.  The Python
    pipeline normalises by 46.0 to obtain a float32 probability vector before
    K-means training with L1 distance.

    river_labels.bin is loaded fully into RAM at construction (~4.9 GB for the
    2.4 B river states).  This eliminates platform-specific mmap code and avoids
    random page faults during the scattered label lookups.

    Two computation modes (same interface as RiverExpander):
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

class TurnExpander {
    static constexpr int NUM_WIDE_BUCKETS = 256;  // 8192 fine buckets / 32
    static constexpr int FINE_PER_WIDE    = 32;
    static constexpr int NUM_CARDS        = 52;

    hand_indexer_t turn_indexer_;    // rounds [2, 3, 1]    → 6-card canonical states
    hand_indexer_t river_indexer_;   // rounds [2, 3, 1, 1] → 7-card canonical states

    std::vector<uint16_t> river_labels_;  // fully loaded from river_labels.bin

    void computeRow(hand_index_t turn_idx, uint8_t* row) const;

public:
    static constexpr int DIMS = NUM_WIDE_BUCKETS;

    // Loads river_labels_path fully into RAM on construction.
    explicit TurnExpander(const std::string& river_labels_path);
    ~TurnExpander();

    // Total number of canonical turn states (hand_indexer_size at round 2).
    uint64_t num_states() const { return hand_indexer_size(&turn_indexer_, 2); }

    // Compute wide-bucket histograms for n arbitrary turn state indices.
    // out must point to a caller-allocated buffer of n * DIMS uint8 bytes.
    void compute_rows(const uint64_t* indices, size_t n, uint8_t* out) const;

    // Compute wide-bucket histograms for the sequential range [start, start+n).
    // out must point to a caller-allocated buffer of n * DIMS uint8 bytes.
    void compute_range(uint64_t start, int n, uint8_t* out) const;
};
