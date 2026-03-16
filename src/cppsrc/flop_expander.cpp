/*
    FlopExpander implementation.

    See flop_expander.h for the public API.

    Nathaniel Potter, 03-15-2026
*/

#include "flop_expander.h"

#include <cstring>
#include <fstream>
#include <stdexcept>

// ---------------------------------------------------------------------------
// Construction / destruction
// ---------------------------------------------------------------------------

FlopExpander::FlopExpander(const std::string& turn_labels_path) {
    uint8_t flopRounds[] = {2, 3};
    uint8_t turnRounds[] = {2, 3, 1};
    hand_indexer_init(2, flopRounds, &flop_indexer_);
    hand_indexer_init(3, turnRounds, &turn_indexer_);

    std::ifstream f(turn_labels_path, std::ios::binary);
    if (!f)
        throw std::runtime_error("FlopExpander: cannot open " + turn_labels_path);

    f.seekg(0, std::ios::end);
    const std::streamsize bytes = f.tellg();
    f.seekg(0, std::ios::beg);

    if (bytes % sizeof(uint16_t) != 0)
        throw std::runtime_error("FlopExpander: label file size not a multiple of 2");

    turn_labels_.resize(static_cast<size_t>(bytes) / sizeof(uint16_t));
    f.read(reinterpret_cast<char*>(turn_labels_.data()), bytes);

    if (!f)
        throw std::runtime_error("FlopExpander: read failed for " + turn_labels_path);
}

FlopExpander::~FlopExpander() {
    hand_indexer_free(&flop_indexer_);
    hand_indexer_free(&turn_indexer_);
}

// ---------------------------------------------------------------------------
// Core computation
// ---------------------------------------------------------------------------

void FlopExpander::computeRow(hand_index_t flop_idx, uint8_t* row) const {
    // Unindex the 5-card flop state: hole[0..1], flop[2..4]
    uint8_t cards[6];
    hand_unindex(&flop_indexer_, 1, flop_idx, cards);

    uint64_t usedMask = 0;
    for (int c = 0; c < 5; c++)
        usedMask |= (1ULL << cards[c]);

    std::memset(row, 0, NUM_WIDE_BUCKETS * sizeof(uint8_t));

    for (uint8_t tc = 0; tc < NUM_CARDS; tc++) {
        if ((usedMask >> tc) & 1) continue;
        cards[5] = tc;
        hand_index_t turn_idx   = hand_index_last(&turn_indexer_, cards);
        uint16_t    fine_bucket = turn_labels_[turn_idx];
        uint8_t     wide_bucket = static_cast<uint8_t>(fine_bucket / FINE_PER_WIDE);
        row[wide_bucket]++;
    }
    // Postcondition: sum(row) == 47
}

// ---------------------------------------------------------------------------
// Public compute methods
// ---------------------------------------------------------------------------

void FlopExpander::compute_rows(const uint64_t* indices, size_t n,
                                uint8_t* out) const {
    #pragma omp parallel for schedule(dynamic, 512)
    for (ptrdiff_t i = 0; i < (ptrdiff_t)n; i++)
        computeRow(indices[i], out + i * DIMS);
}

void FlopExpander::compute_range(uint64_t start, int n, uint8_t* out) const {
    #pragma omp parallel for schedule(dynamic, 512)
    for (int i = 0; i < n; i++)
        computeRow(start + i, out + i * DIMS);
}
