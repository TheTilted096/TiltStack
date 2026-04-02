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

FlopExpander::FlopExpander(const std::string &turn_labels_path,
                           const std::string &turn_ehs_fine_path) {
    uint8_t flopRounds[] = {2, 3};
    uint8_t turnRounds[] = {2, 3, 1};
    hand_indexer_init(2, flopRounds, &flop_indexer_);
    hand_indexer_init(3, turnRounds, &turn_indexer_);

    {
        std::ifstream f(turn_labels_path, std::ios::binary);
        if (!f)
            throw std::runtime_error("FlopExpander: cannot open " +
                                     turn_labels_path);
        f.seekg(0, std::ios::end);
        const std::streamsize bytes = f.tellg();
        f.seekg(0, std::ios::beg);
        if (bytes % sizeof(uint16_t) != 0)
            throw std::runtime_error(
                "FlopExpander: label file size not a multiple of 2");
        turn_labels_.resize(static_cast<size_t>(bytes) / sizeof(uint16_t));
        f.read(reinterpret_cast<char *>(turn_labels_.data()), bytes);
        if (!f)
            throw std::runtime_error("FlopExpander: read failed for " +
                                     turn_labels_path);
    }

    {
        std::ifstream f(turn_ehs_fine_path, std::ios::binary);
        if (!f)
            throw std::runtime_error("FlopExpander: cannot open " +
                                     turn_ehs_fine_path);
        f.seekg(0, std::ios::end);
        const std::streamsize bytes = f.tellg();
        f.seekg(0, std::ios::beg);
        if (bytes % sizeof(uint16_t) != 0)
            throw std::runtime_error(
                "FlopExpander: EHS file size not a multiple of 2");
        turn_ehs_fine_.resize(static_cast<size_t>(bytes) / sizeof(uint16_t));
        f.read(reinterpret_cast<char *>(turn_ehs_fine_.data()), bytes);
        if (!f)
            throw std::runtime_error("FlopExpander: read failed for " +
                                     turn_ehs_fine_path);
    }
}

FlopExpander::~FlopExpander() {
    hand_indexer_free(&flop_indexer_);
    hand_indexer_free(&turn_indexer_);
}

// ---------------------------------------------------------------------------
// Core computation
// ---------------------------------------------------------------------------

void FlopExpander::computeRowEhsMult(hand_index_t flop_idx, uint8_t *row,
                                     float *ehs_out, uint8_t *mult_out) const {
    uint8_t cards[6];
    hand_unindex(&flop_indexer_, 1, flop_idx, cards);

    uint64_t usedMask = 0;
    for (int c = 0; c < 5; c++)
        usedMask |= (1ULL << cards[c]);

    std::memset(row, 0, NUM_WIDE_BUCKETS * sizeof(uint8_t));

    float sum = 0.0f;
    int cnt = 0;
    for (uint8_t tc = 0; tc < NUM_CARDS; tc++) {
        if ((usedMask >> tc) & 1)
            continue;
        cards[5] = tc;
        hand_index_t turn_idx = hand_index_last(&turn_indexer_, cards);
        uint16_t fine_bucket = turn_labels_[turn_idx];
        row[static_cast<uint8_t>(fine_bucket / FINE_PER_WIDE)]++;
        sum += turn_ehs_fine_[turn_idx] * (1.0f / 65535.0f);
        cnt++;
    }
    // cnt == 47 always
    *ehs_out = sum / static_cast<float>(cnt);
    *mult_out = computeMult(flop_idx);
}

// ---------------------------------------------------------------------------
// Public compute methods
// ---------------------------------------------------------------------------

void FlopExpander::compute_rows_ehs_mult(const uint64_t *indices, size_t n,
                                         uint8_t *row_out, float *ehs_out,
                                         uint8_t *mult_out) const {
#pragma omp parallel for schedule(dynamic, 512)
    for (ptrdiff_t i = 0; i < (ptrdiff_t)n; i++)
        computeRowEhsMult(indices[i], row_out + i * DIMS, ehs_out + i,
                          mult_out + i);
}

// ---------------------------------------------------------------------------
// Multiplicity computation
// ---------------------------------------------------------------------------

uint8_t FlopExpander::computeMult(hand_index_t flop_idx) const {
    // Binary-search for the configuration that owns this index.
    const uint_fast32_t round = 1; // flop is round index 1 ([2,3])
    uint_fast32_t low = 0, high = flop_indexer_.configurations[round];
    uint_fast32_t cfg = 0;
    while (low < high) {
        uint_fast32_t mid = (low + high) / 2;
        if (flop_indexer_.configuration_to_offset[round][mid] <= flop_idx) {
            cfg = mid;
            low = mid + 1;
        } else {
            high = mid;
        }
    }

    const uint_fast32_t equal_mask =
        flop_indexer_.configuration_to_equal[round][cfg];

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
