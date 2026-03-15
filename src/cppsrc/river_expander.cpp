/*
    RiverExpander implementation.

    See river_expander.h for the public API.

    Nathaniel Potter, 03-08-2026
*/

#include "river_expander.h"

#include <algorithm>
#include <array>

// ---------------------------------------------------------------------------
// Construction / destruction
// ---------------------------------------------------------------------------

RiverExpander::RiverExpander() {
    for (unsigned c = 0; c < NUM_CARDS; c++)
        card_hands_[c] = omp::Hand(c);

    uint8_t riverRounds[] = {2, 3, 1, 1};
    hand_indexer_init(4, riverRounds, &river_indexer_);

    hand_indexer_t preflop;
    uint8_t preflopRounds[] = {2};
    hand_indexer_init(1, preflopRounds, &preflop);

    for (uint8_t c0 = 0; c0 < NUM_CARDS - 1; c0++) {
        for (uint8_t c1 = c0 + 1; c1 < NUM_CARDS; c1++) {
            uint8_t cards[2] = {c0, c1};
            bucket_of_[c0][c1] = hand_index_last(&preflop, cards);
        }
    }

    hand_indexer_free(&preflop);
}

RiverExpander::~RiverExpander() {
    hand_indexer_free(&river_indexer_);
}

// ---------------------------------------------------------------------------
// Core computation
// ---------------------------------------------------------------------------

void RiverExpander::computeRow(hand_index_t idx, uint8_t* row) const {
    uint8_t cards[7];
    hand_unindex(&river_indexer_, 3, idx, cards);

    uint64_t usedMask = 0;
    for (int c = 0; c < 7; c++)
        usedMask |= (1ULL << cards[c]);

    omp::Hand heroHand = omp::Hand::empty();
    for (int c = 0; c < 7; c++)
        heroHand += card_hands_[cards[c]];
    uint16_t heroRank = evaluator_.evaluate(heroHand);

    omp::Hand board = omp::Hand::empty();
    for (int c = 2; c < 7; c++)
        board += card_hands_[cards[c]];

    std::array<float, NUM_BUCKETS> eqSum = {};
    std::array<int,   NUM_BUCKETS> count = {};

    for (uint8_t c0 = 0; c0 < NUM_CARDS - 1; c0++) {
        if ((usedMask >> c0) & 1) continue;
        for (uint8_t c1 = c0 + 1; c1 < NUM_CARDS; c1++) {
            if ((usedMask >> c1) & 1) continue;

            hand_index_t bucket = bucket_of_[c0][c1];
            omp::Hand oppHand = board + card_hands_[c0] + card_hands_[c1];
            uint16_t oppRank = evaluator_.evaluate(oppHand);

            if (heroRank > oppRank)       eqSum[bucket] += 1.0f;
            else if (heroRank == oppRank) eqSum[bucket] += 0.5f;
            count[bucket]++;
        }
    }

    float totalEqSum = 0.0f;
    int   totalCount = 0;
    for (int bi = 0; bi < NUM_BUCKETS; bi++) {
        totalEqSum += eqSum[bi];
        totalCount += count[bi];
    }
    float avg = totalCount > 0 ? totalEqSum / totalCount : 0.5f;

    for (int bi = 0; bi < NUM_BUCKETS; bi++) {
        float eq = count[bi] > 0 ? eqSum[bi] / count[bi] : avg;
        row[bi] = static_cast<uint8_t>(eq * 255.0f);
    }
}

// ---------------------------------------------------------------------------
// Public compute methods
// ---------------------------------------------------------------------------

void RiverExpander::compute_rows(const uint64_t* indices, size_t n,
                                 uint8_t* out) const {
    #pragma omp parallel for schedule(dynamic, 1024)
    for (ptrdiff_t i = 0; i < (ptrdiff_t)n; i++)
        computeRow(indices[i], out + i * DIMS);
}

void RiverExpander::compute_range(uint64_t start, int n, uint8_t* out) const {
    #pragma omp parallel for schedule(dynamic, 1024)
    for (int i = 0; i < n; i++)
        computeRow(start + i, out + i * DIMS);
}
