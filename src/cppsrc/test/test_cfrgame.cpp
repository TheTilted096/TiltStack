#include "CFRGame.h"
#include <gtest/gtest.h>

static constexpr uint64_t TEST_SEED = 42ULL;

// Resize gEHS[r] / gLabels[r] to accommodate the canonical indices that
// begin() will access when the RNG is seeded with TEST_SEED.  Values are
// left at zero — EHS and bucket fields are irrelevant to these tests.
static void primeGlobals() {
    static bool done = false;
    if (done)
        return;
    done = true;

    RNG tmp(TEST_SEED);
    Card deck[52];
    for (int i = 0; i < 52; i++)
        deck[i] = static_cast<Card>(i);
    for (int i = 0; i < 9; i++) {
        int j =
            i + static_cast<int>(tmp.next() % static_cast<uint64_t>(52 - i));
        std::swap(deck[i], deck[j]);
    }

    uint8_t rounds[] = {2, 3, 1, 1};
    hand_indexer_init(4, rounds, &g_indexer);
    for (int p = 0; p < 2; p++) {
        uint8_t cards[7] = {deck[p * 2], deck[p * 2 + 1], deck[4], deck[5],
                            deck[6],     deck[7],         deck[8]};
        hand_index_t indices[4];
        hand_index_all(&g_indexer, cards, indices);
        for (int r = 0; r < 4; r++) {
            auto i = static_cast<std::size_t>(indices[r]);
            if (i >= gEHS[r].size())
                gEHS[r].resize(i + 1, 0);
            if (i >= gLabels[r].size())
                gLabels[r].resize(i + 1, 0);
        }
    }
}

// Construct a fresh game with a deterministic deal.
#define MAKE_GAME()                                                            \
    primeGlobals();                                                            \
    rng.seed(TEST_SEED);                                                       \
    CFRGame g

// ---------------------------------------------------------------------------

// Verify starting pot, blinds, stacks, and round.
TEST(CFRGame, InitialState) {
    MAKE_GAME();
    EXPECT_EQ(g.history[0].pot, SMALL_BLIND + BIG_BLIND);
    EXPECT_EQ(g.history[0].toCall, BIG_BLIND - SMALL_BLIND);
    EXPECT_EQ(g.stacks[0], STARTING_STACK - SMALL_BLIND);
    EXPECT_EQ(g.stacks[1], STARTING_STACK - BIG_BLIND);
    EXPECT_EQ(g.currentRound, Round::PREFLOP);
}

// stacks[0] + stacks[1] + pot must equal 2 * STARTING_STACK throughout.
TEST(CFRGame, ChipsConserved) {
    MAKE_GAME();
    auto chips = [&] {
        return g.stacks[0] + g.stacks[1] + g.history[g.ply].pot;
    };
    EXPECT_EQ(chips(), STARTING_STACK * 2);
    g.makeMove(Action::BET50);
    EXPECT_EQ(chips(), STARTING_STACK * 2);
    g.makeMove(Action::CALL); // ends preflop
    EXPECT_EQ(chips(), STARTING_STACK * 2);
}

// BET50 = call + half-pot; BET100 = call + full-pot.  Verify pot and toCall.
TEST(CFRGame, BetSizing) {
    MAKE_GAME();
    int pot = g.history[0].pot, tc = g.history[0].toCall;

    g.makeMove(Action::BET50);
    EXPECT_EQ(g.history[g.ply].pot, pot + tc + (pot + tc) / 2);
    EXPECT_EQ(g.history[g.ply].toCall, (pot + tc) / 2);

    g.unmakeMove();
    g.makeMove(Action::BET100);
    EXPECT_EQ(g.history[g.ply].pot, pot + tc + (pot + tc));
    EXPECT_EQ(g.history[g.ply].toCall, (pot + tc));
}

// makeMove then unmakeMove must leave stacks, pot, round, and betHist
// unchanged.
TEST(CFRGame, MakeUnmakeRoundtrip) {
    MAKE_GAME();
    int s0 = g.stacks[0], s1 = g.stacks[1], pot = g.history[0].pot;

    g.makeMove(Action::BET50);
    g.makeMove(Action::CALL); // crosses street boundary to flop
    g.unmakeMove();           // back to preflop
    g.unmakeMove();           // back to initial ply

    EXPECT_EQ(g.stacks[0], s0);
    EXPECT_EQ(g.stacks[1], s1);
    EXPECT_EQ(g.history[0].pot, pot);
    EXPECT_EQ(g.currentRound, Round::PREFLOP);
    EXPECT_FLOAT_EQ(g.betHist[0][0], 0.0f); // cleared by unmakeMove
    EXPECT_FLOAT_EQ(g.betHist[0][1], 0.0f);
}

// SB's first call must NOT end the street (BB still has option).
// BB's subsequent check must end it.
TEST(CFRGame, PreflopBBOption) {
    MAKE_GAME();
    EXPECT_FALSE(g.endsStreet(Action::CALL)); // SB calls, numActs == 0
    g.makeMove(Action::CALL);
    EXPECT_TRUE(g.endsStreet(Action::CHECK)); // BB checks option, numActs == 1
}

// Postflop: first check must not end street; second must.
TEST(CFRGame, PostflopCheckBack) {
    MAKE_GAME();
    g.makeMove(Action::CALL);
    g.makeMove(Action::CHECK);                 // ends preflop → flop
    EXPECT_FALSE(g.endsStreet(Action::CHECK)); // first postflop check
    g.makeMove(Action::CHECK);
    EXPECT_TRUE(g.endsStreet(Action::CHECK)); // check-back
}

// CHECK with toCall > 0 is a fold (terminal 1).
// CALL mid-street when a stack equals toCall is an all-in showdown (terminal
// 2). Completing the action on the river is a showdown (terminal 2).
TEST(CFRGame, TerminalDetection) {
    MAKE_GAME();
    EXPECT_EQ(g.isTerminalState(Action::CHECK), 1); // immediate fold

    rng.seed(TEST_SEED);
    CFRGame g2;
    g2.makeMove(Action::ALLIN);
    EXPECT_EQ(g2.isTerminalState(Action::CALL), 2); // all-in call

    rng.seed(TEST_SEED);
    CFRGame g3;
    g3.makeMove(Action::CALL);
    g3.makeMove(Action::CHECK); // → flop
    g3.makeMove(Action::CHECK);
    g3.makeMove(Action::CHECK); // → turn
    g3.makeMove(Action::CHECK);
    g3.makeMove(Action::CHECK); // → river
    g3.makeMove(Action::CHECK); // first river check
    EXPECT_EQ(g3.isTerminalState(Action::CHECK), 2);
}

// Initial state: callAmt=1000, effPot=4000, minRaise=3000.
// BET33 (→2333) is below minRaise and is skipped; the remaining 9 actions
// are emitted in ascending fraction order.
// CALL is absent after SB calls (toCall == 0).
TEST(CFRGame, GenerateActions) {
    MAKE_GAME();
    ActionList al;
    EXPECT_EQ(g.generateActions(al), 9);
    EXPECT_EQ(al[0], Action::CHECK);
    EXPECT_EQ(al[1], Action::CALL);
    EXPECT_EQ(al[2], Action::BET50);
    EXPECT_EQ(al[3], Action::BET75);
    EXPECT_EQ(al[4], Action::BET100);
    EXPECT_EQ(al[5], Action::BET150);
    EXPECT_EQ(al[6], Action::BET200);
    EXPECT_EQ(al[7], Action::BET300);
    EXPECT_EQ(al[8], Action::ALLIN);

    g.makeMove(Action::CALL); // SB calls → BB faces toCall == 0
    int n = g.generateActions(al);
    for (int i = 0; i < n; i++)
        EXPECT_NE(al[i], Action::CALL);
}

// Hero (P0/SB) folds immediately → loses the small blind.
// Villain (BB) folds after a raise → hero wins the big blind.
TEST(CFRGame, Payout) {
    MAKE_GAME();
    g.makeMove(Action::CHECK); // SB folds
    EXPECT_FLOAT_EQ(g.payout(), -static_cast<float>(SMALL_BLIND));

    rng.seed(TEST_SEED);
    CFRGame g2;
    g2.makeMove(Action::BET50);
    g2.makeMove(Action::CHECK); // BB folds
    EXPECT_FLOAT_EQ(g2.payout(), static_cast<float>(BIG_BLIND));
}

// Normalised InfoSet scalars and street one-hot encoding.
TEST(CFRGame, GetInfo) {
    MAKE_GAME();
    InfoSet info = g.getInfo();
    float eff = static_cast<float>(STARTING_STACK);

    EXPECT_FLOAT_EQ(info.myStack,
                    static_cast<float>(STARTING_STACK - SMALL_BLIND) / eff);
    EXPECT_FLOAT_EQ(info.oppStack,
                    static_cast<float>(STARTING_STACK - BIG_BLIND) / eff);
    EXPECT_FLOAT_EQ(info.potSize,
                    static_cast<float>(SMALL_BLIND + BIG_BLIND) / eff);
    EXPECT_TRUE(info.streetEmbed[static_cast<int>(Round::PREFLOP)]);
    EXPECT_FALSE(info.streetEmbed[static_cast<int>(Round::FLOP)]);
    EXPECT_TRUE(info.isButton); // stm == 0

    g.makeMove(Action::CALL);
    g.makeMove(Action::CHECK); // → flop, stm = 1
    info = g.getInfo();
    EXPECT_TRUE(info.streetEmbed[static_cast<int>(Round::FLOP)]);
    EXPECT_FALSE(info.isButton);
}
