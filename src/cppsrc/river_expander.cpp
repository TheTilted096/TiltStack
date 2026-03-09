/*

Expands river states (2 hole cards + 5 board cards) into 169-dimensional equity
vectors against each opponent starting hand bucket (up to suit isomorphism).

Bucket assignment uses the hand-isomorphism library's native preflop indexer,
which maps any 2-card combo to one of 169 canonical classes. The bucket ordering
is determined by the library's internal canonicalization — consumers of the
output file must use the same preflop indexer to interpret bucket indices.

For each river state, we iterate over all C(45,2) = 990 remaining card pairs,
evaluate the opponent's hand, and accumulate results into the appropriate bucket.

Card index = 4 * rank + suit (both libraries share this encoding).
Ranks: 0 = deuce, 12 = ace. Suits: 0-3.

Each byte is a quantized equity: actual_equity = byte_value / 255.0f.
Buckets with no legal combos are filled with the combo-weighted average
equity across all legal matchups (total wins+ties / total combos).

Modes:
  river_expander <outfile|->               Full expansion (2.4B states, ~410 GB)
  river_expander --sample <indices.bin> <outfile|->
    Selective expansion: reads a sorted list of uint64 indices from indices.bin
    and only computes those hands. Used for K-means training samples.

Nathaniel Potter, 03-08-2026

*/

#include <array>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <string>
#include <vector>

// Include OMPEval first: deck.h defines CARDS/RANKS/SUITS macros that
// collide with OMPEval identifiers, so OMPEval must be parsed before them.
#include "../third_party/OMPEval/omp/HandEvaluator.h"

extern "C" {
#define _Bool bool
#include "../third_party/hand-isomorphism/hand_index.h"
#undef _Bool
}

static constexpr int NUM_BUCKETS = 169;
static constexpr int NUM_CARDS = 52;
static constexpr hand_index_t NUM_STATES = 2428287420ULL;
static constexpr int BATCH_SIZE = 1000000;
static constexpr size_t STREAM_BUF_SIZE = 1 << 24;  // 16 MB I/O buffer

// Precomputed lookup tables, built once at startup.
static std::array<omp::Hand, NUM_CARDS> CARD_HANDS;
static std::array<std::array<hand_index_t, NUM_CARDS>, NUM_CARDS> BUCKET_OF;

static void initCardHands() {
    for (unsigned c = 0; c < NUM_CARDS; c++)
        CARD_HANDS[c] = omp::Hand(c);
}

static void initBucketLookup(hand_indexer_t& preflop) {
    for (uint8_t c0 = 0; c0 < NUM_CARDS - 1; c0++) {
        for (uint8_t c1 = c0 + 1; c1 < NUM_CARDS; c1++) {
            uint8_t cards[2] = {c0, c1};
            BUCKET_OF[c0][c1] = hand_index_last(&preflop, cards);
        }
    }
}

// Compute the equity vector for a single river state index.
// Writes NUM_BUCKETS bytes into `row`.
static void computeRow(hand_index_t idx,
                       const hand_indexer_t& river_indexer,
                       const omp::HandEvaluator& evaluator,
                       uint8_t* row) {
    uint8_t cards[7];
    hand_unindex(&river_indexer, 3, idx, cards);

    uint64_t usedMask = 0;
    for (int c = 0; c < 7; c++)
        usedMask |= (1ULL << cards[c]);

    omp::Hand heroHand = omp::Hand::empty();
    for (int c = 0; c < 7; c++)
        heroHand += CARD_HANDS[cards[c]];
    uint16_t heroRank = evaluator.evaluate(heroHand);

    omp::Hand board = omp::Hand::empty();
    for (int c = 2; c < 7; c++)
        board += CARD_HANDS[cards[c]];

    std::array<float, NUM_BUCKETS> eqSum = {};
    std::array<int, NUM_BUCKETS> count = {};

    for (uint8_t c0 = 0; c0 < NUM_CARDS - 1; c0++) {
        if ((usedMask >> c0) & 1) continue;
        for (uint8_t c1 = c0 + 1; c1 < NUM_CARDS; c1++) {
            if ((usedMask >> c1) & 1) continue;

            hand_index_t bucket = BUCKET_OF[c0][c1];
            omp::Hand oppHand = board + CARD_HANDS[c0] + CARD_HANDS[c1];
            uint16_t oppRank = evaluator.evaluate(oppHand);

            if (heroRank > oppRank)      eqSum[bucket] += 1.0f;
            else if (heroRank == oppRank) eqSum[bucket] += 0.5f;
            count[bucket]++;
        }
    }

    float totalEqSum = 0.0f;
    int totalCount = 0;
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

int main(int argc, char* argv[]) {
    // Parse args: [--sample <indices.bin>] [--quiet] <outfile|->
    std::string outPath = "river_equities.bin";
    std::string samplePath;
    bool sampleMode = false;
    bool quiet = false;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--sample" && i + 1 < argc) {
            sampleMode = true;
            samplePath = argv[++i];
        } else if (arg == "--quiet") {
            quiet = true;
        } else {
            outPath = arg;
        }
    }

    auto t_init = std::chrono::steady_clock::now();
    std::cerr << "Initializing card tables..." << std::flush;
    initCardHands();
    omp::HandEvaluator evaluator;
    std::cerr << " done." << std::endl;

    std::cerr << "Building hand indexers..." << std::flush;
    hand_indexer_t river_indexer;
    uint8_t riverRounds[] = {2, 3, 1, 1};
    hand_indexer_init(4, riverRounds, &river_indexer);

    hand_indexer_t preflop_indexer;
    uint8_t preflopRounds[] = {2};
    hand_indexer_init(1, preflopRounds, &preflop_indexer);
    std::cerr << " done." << std::endl;

    std::cerr << "  River indexer size:   " << hand_indexer_size(&river_indexer, 3)
              << " (expected " << NUM_STATES << ")" << std::endl;
    std::cerr << "  Preflop indexer size: " << hand_indexer_size(&preflop_indexer, 0)
              << " (expected 169)" << std::endl;

    std::cerr << "Building bucket lookup table..." << std::flush;
    initBucketLookup(preflop_indexer);
    hand_indexer_free(&preflop_indexer);
    {
        auto dt = std::chrono::steady_clock::now() - t_init;
        double sec = std::chrono::duration<double>(dt).count();
        std::cerr << " done. (init took " << std::fixed << std::setprecision(1)
                  << sec << "s)" << std::endl;
    }

    // Output stream: "-" means stdout (for piping), otherwise open a file.
    bool useStdout = (outPath == "-");
    std::ofstream fileOut;
    std::vector<char> streamBuf(STREAM_BUF_SIZE);
    std::ostream* out;

    if (useStdout) {
        // Put stdout in binary mode and attach a large buffer.
        std::cout.rdbuf()->pubsetbuf(streamBuf.data(), STREAM_BUF_SIZE);
        out = &std::cout;
    } else {
        fileOut.rdbuf()->pubsetbuf(streamBuf.data(), STREAM_BUF_SIZE);
        fileOut.open(outPath, std::ios::binary);
        if (!fileOut) {
            std::cerr << "Error: cannot open " << outPath << " for writing" << std::endl;
            return 1;
        }
        out = &fileOut;
    }

    // --- Load sample indices if in sample mode ---
    std::vector<uint64_t> indices;
    if (sampleMode) {
        std::cerr << "Loading sample indices from " << samplePath << "..." << std::flush;
        std::ifstream idxFile(samplePath, std::ios::binary | std::ios::ate);
        if (!idxFile) {
            std::cerr << "\nError: cannot open " << samplePath << std::endl;
            hand_indexer_free(&river_indexer);
            return 1;
        }
        size_t fileSize = idxFile.tellg();
        idxFile.seekg(0);
        indices.resize(fileSize / sizeof(uint64_t));
        idxFile.read(reinterpret_cast<char*>(indices.data()), fileSize);
        idxFile.close();
        std::cerr << " " << indices.size() << " indices ("
                  << std::fixed << std::setprecision(1)
                  << fileSize / 1e6 << " MB)" << std::endl;
    }

    hand_index_t totalStates = sampleMode ? indices.size() : NUM_STATES;
    std::cerr << (sampleMode ? "Sample" : "Full") << " mode: "
              << totalStates << " states" << std::endl;

    // --- Unified batch loop ---
    std::vector<uint8_t> buffer(static_cast<size_t>(BATCH_SIZE) * NUM_BUCKETS);
    auto t_expand = std::chrono::steady_clock::now();

    for (hand_index_t batchStart = 0; batchStart < totalStates; batchStart += BATCH_SIZE) {
        hand_index_t batchEnd = std::min(batchStart + (hand_index_t)BATCH_SIZE, totalStates);
        int batchSize = static_cast<int>(batchEnd - batchStart);

        #pragma omp parallel for schedule(dynamic, 1024)
        for (int b = 0; b < batchSize; b++) {
            hand_index_t idx = sampleMode ? indices[batchStart + b]
                                          : batchStart + b;
            computeRow(idx, river_indexer, evaluator,
                       &buffer[static_cast<size_t>(b) * NUM_BUCKETS]);
        }

        out->write(reinterpret_cast<const char*>(buffer.data()),
                   static_cast<std::streamsize>(batchSize) * NUM_BUCKETS);
        if (!*out) {
            std::cerr << "\nFatal: write failed at state " << batchEnd
                      << " / " << totalStates << std::endl;
            hand_indexer_free(&river_indexer);
            return 1;
        }

        if (!quiet) {
            auto dt = std::chrono::steady_clock::now() - t_expand;
            double elapsed = std::chrono::duration<double>(dt).count();
            double frac = static_cast<double>(batchEnd) / totalStates;
            double eta = frac > 0 ? elapsed / frac * (1 - frac) : 0;
            double rate = batchEnd / elapsed;
            std::cerr << "\r  " << std::fixed << std::setprecision(2)
                      << (100.0 * frac) << "%  ("
                      << batchEnd << " / " << totalStates << ")  "
                      << std::setprecision(0) << rate / 1e6 << "M states/s  "
                      << "ETA " << static_cast<int>(eta / 60) << "m"
                      << std::setprecision(0) << static_cast<int>(eta) % 60 << "s"
                      << "   " << std::flush;
        }
    }

    out->flush();
    if (!useStdout) {
        fileOut.close();
        if (!fileOut) {
            std::cerr << "\nFatal: final flush/close failed — output may be truncated."
                      << std::endl;
            hand_indexer_free(&river_indexer);
            return 1;
        }
    }
    {
        auto dt = std::chrono::steady_clock::now() - t_expand;
        double sec = std::chrono::duration<double>(dt).count();
        double rate = totalStates / sec;
        std::cerr << "\nDone: " << totalStates << " states in "
                  << std::fixed << std::setprecision(1) << sec << "s ("
                  << std::setprecision(1) << rate / 1e6 << "M states/s)."
                  << (useStdout ? "" : " Output: " + outPath)
                  << std::endl;
    }
    hand_indexer_free(&river_indexer);
    return 0;
}
