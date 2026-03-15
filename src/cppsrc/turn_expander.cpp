/*
    TurnExpander implementation.

    See turn_expander.h for the public API.

    Nathaniel Potter, 03-15-2026
*/

#include "turn_expander.h"

#include <cstring>
#include <fstream>
#include <stdexcept>

// ---------------------------------------------------------------------------
// Construction / destruction
// ---------------------------------------------------------------------------

TurnExpander::TurnExpander(const std::string& river_labels_path) {
    uint8_t turnRounds[]  = {2, 3, 1};
    uint8_t riverRounds[] = {2, 3, 1, 1};
    hand_indexer_init(3, turnRounds,  &turn_indexer_);
    hand_indexer_init(4, riverRounds, &river_indexer_);

    std::ifstream f(river_labels_path, std::ios::binary);
    if (!f)
        throw std::runtime_error("TurnExpander: cannot open " + river_labels_path);

    f.seekg(0, std::ios::end);
    const std::streamsize bytes = f.tellg();
    f.seekg(0, std::ios::beg);

    if (bytes % sizeof(uint16_t) != 0)
        throw std::runtime_error("TurnExpander: label file size not a multiple of 2");

    river_labels_.resize(static_cast<size_t>(bytes) / sizeof(uint16_t));
    f.read(reinterpret_cast<char*>(river_labels_.data()), bytes);

    if (!f)
        throw std::runtime_error("TurnExpander: read failed for " + river_labels_path);
}

TurnExpander::~TurnExpander() {
    hand_indexer_free(&turn_indexer_);
    hand_indexer_free(&river_indexer_);
}

// ---------------------------------------------------------------------------
// Core computation
// ---------------------------------------------------------------------------

void TurnExpander::computeRow(hand_index_t turn_idx, uint8_t* row) const {
    // Unindex the 6-card turn state: hole[0..1], flop[2..4], turn[5]
    uint8_t cards[7];
    hand_unindex(&turn_indexer_, 2, turn_idx, cards);

    uint64_t usedMask = 0;
    for (int c = 0; c < 6; c++)
        usedMask |= (1ULL << cards[c]);

    std::memset(row, 0, NUM_WIDE_BUCKETS * sizeof(uint8_t));

    for (uint8_t rc = 0; rc < NUM_CARDS; rc++) {
        if ((usedMask >> rc) & 1) continue;
        cards[6] = rc;
        hand_index_t river_idx  = hand_index_last(&river_indexer_, cards);
        uint16_t     fine_bucket = river_labels_[river_idx];
        uint8_t      wide_bucket = static_cast<uint8_t>(fine_bucket / FINE_PER_WIDE);
        row[wide_bucket]++;
    }
    // Postcondition: sum(row) == 46
}

// ---------------------------------------------------------------------------
// Public compute methods
// ---------------------------------------------------------------------------

void TurnExpander::compute_rows(const uint64_t* indices, size_t n,
                                uint8_t* out) const {
    #pragma omp parallel for schedule(dynamic, 512)
    for (ptrdiff_t i = 0; i < (ptrdiff_t)n; i++)
        computeRow(indices[i], out + i * DIMS);
}

void TurnExpander::compute_range(uint64_t start, int n, uint8_t* out) const {
    #pragma omp parallel for schedule(dynamic, 512)
    for (int i = 0; i < n; i++)
        computeRow(start + i, out + i * DIMS);
}
