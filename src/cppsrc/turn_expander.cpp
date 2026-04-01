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

TurnExpander::TurnExpander(const std::string &river_labels_path,
                           const std::string &river_ehs_fine_path) {
    uint8_t turnRounds[] = {2, 3, 1};
    uint8_t riverRounds[] = {2, 3, 1, 1};
    hand_indexer_init(3, turnRounds, &turn_indexer_);
    hand_indexer_init(4, riverRounds, &river_indexer_);

    if (!river_labels_path.empty()) {
        std::ifstream f(river_labels_path, std::ios::binary);
        if (!f)
            throw std::runtime_error("TurnExpander: cannot open " +
                                     river_labels_path);
        f.seekg(0, std::ios::end);
        const std::streamsize bytes = f.tellg();
        f.seekg(0, std::ios::beg);
        if (bytes % sizeof(uint16_t) != 0)
            throw std::runtime_error(
                "TurnExpander: label file size not a multiple of 2");
        river_labels_.resize(static_cast<size_t>(bytes) / sizeof(uint16_t));
        f.read(reinterpret_cast<char *>(river_labels_.data()), bytes);
        if (!f)
            throw std::runtime_error("TurnExpander: read failed for " +
                                     river_labels_path);
    }

    if (!river_ehs_fine_path.empty()) {
        std::ifstream f(river_ehs_fine_path, std::ios::binary);
        if (!f)
            throw std::runtime_error("TurnExpander: cannot open " +
                                     river_ehs_fine_path);
        f.seekg(0, std::ios::end);
        const std::streamsize bytes = f.tellg();
        f.seekg(0, std::ios::beg);
        if (bytes % sizeof(uint16_t) != 0)
            throw std::runtime_error(
                "TurnExpander: EHS file size not a multiple of 2");
        river_ehs_fine_.resize(static_cast<size_t>(bytes) / sizeof(uint16_t));
        f.read(reinterpret_cast<char *>(river_ehs_fine_.data()), bytes);
        if (!f)
            throw std::runtime_error("TurnExpander: read failed for " +
                                     river_ehs_fine_path);
    }

    if (river_labels_.empty() || river_ehs_fine_.empty())
        throw std::runtime_error("TurnExpander: both river_labels_path and "
                                 "river_ehs_fine_path must be non-empty");
}

TurnExpander::~TurnExpander() {
    hand_indexer_free(&turn_indexer_);
    hand_indexer_free(&river_indexer_);
}

// ---------------------------------------------------------------------------
// Core computation
// ---------------------------------------------------------------------------

void TurnExpander::computeRow(hand_index_t turn_idx, uint8_t *row) const {
    // Unindex the 6-card turn state: hole[0..1], flop[2..4], turn[5]
    uint8_t cards[7];
    hand_unindex(&turn_indexer_, 2, turn_idx, cards);

    uint64_t usedMask = 0;
    for (int c = 0; c < 6; c++)
        usedMask |= (1ULL << cards[c]);

    std::memset(row, 0, NUM_WIDE_BUCKETS * sizeof(uint8_t));

    for (uint8_t rc = 0; rc < NUM_CARDS; rc++) {
        if ((usedMask >> rc) & 1)
            continue;
        cards[6] = rc;
        hand_index_t river_idx = hand_index_last(&river_indexer_, cards);
        uint16_t fine_bucket = river_labels_[river_idx];
        uint8_t wide_bucket = static_cast<uint8_t>(fine_bucket / FINE_PER_WIDE);
        row[wide_bucket]++;
    }
    // Postcondition: sum(row) == 46
}

void TurnExpander::computeRowEhsMult(hand_index_t turn_idx, uint8_t *row,
                                     float *ehs_out, uint8_t *mult_out) const {
    uint8_t cards[7];
    hand_unindex(&turn_indexer_, 2, turn_idx, cards);

    uint64_t usedMask = 0;
    for (int c = 0; c < 6; c++)
        usedMask |= (1ULL << cards[c]);

    std::memset(row, 0, NUM_WIDE_BUCKETS * sizeof(uint8_t));

    float sum = 0.0f;
    int cnt = 0;
    for (uint8_t rc = 0; rc < NUM_CARDS; rc++) {
        if ((usedMask >> rc) & 1)
            continue;
        cards[6] = rc;
        hand_index_t river_idx = hand_index_last(&river_indexer_, cards);
        uint16_t fine_bucket = river_labels_[river_idx];
        row[static_cast<uint8_t>(fine_bucket / FINE_PER_WIDE)]++;
        sum += river_ehs_fine_[river_idx] * (1.0f / 65535.0f);
        cnt++;
    }
    // cnt == 46 always
    *ehs_out = sum / static_cast<float>(cnt);
    *mult_out = computeMult(turn_idx);
}

// ---------------------------------------------------------------------------
// Public compute methods
// ---------------------------------------------------------------------------

void TurnExpander::compute_rows(const uint64_t *indices, size_t n,
                                uint8_t *out) const {
#pragma omp parallel for schedule(dynamic, 512)
    for (ptrdiff_t i = 0; i < (ptrdiff_t)n; i++)
        computeRow(indices[i], out + i * DIMS);
}

void TurnExpander::compute_range_ehs_mult(uint64_t start, int n,
                                          uint8_t *row_out, float *ehs_out,
                                          uint8_t *mult_out) const {
#pragma omp parallel for schedule(dynamic, 512)
    for (int i = 0; i < n; i++)
        computeRowEhsMult(start + i, row_out + i * DIMS, ehs_out + i,
                          mult_out + i);
}

// ---------------------------------------------------------------------------
// Multiplicity computation
// ---------------------------------------------------------------------------

uint8_t TurnExpander::computeMult(hand_index_t turn_idx) const {
    // Binary-search for the configuration that owns this index.
    const uint_fast32_t round = 2; // turn is round index 2 ([2,3,1])
    uint_fast32_t low = 0, high = turn_indexer_.configurations[round];
    uint_fast32_t cfg = 0;
    while (low < high) {
        uint_fast32_t mid = (low + high) / 2;
        if (turn_indexer_.configuration_to_offset[round][mid] <= turn_idx) {
            cfg = mid;
            low = mid + 1;
        } else {
            high = mid;
        }
    }

    const uint_fast32_t equal_mask =
        turn_indexer_.configuration_to_equal[round][cfg];

    static const uint8_t fact[5] = {1, 1, 2, 6, 24};
    int denom = 1, gsz = 1;
    for (int j = 1; j < SUITS; ++j) {
        if (equal_mask & (1u << (j - 1))) {
            ++gsz;
        } else {
            denom *= fact[gsz];
            gsz = 1;
        }
    }
    denom *= fact[gsz];
    return static_cast<uint8_t>(24 / denom);
}
