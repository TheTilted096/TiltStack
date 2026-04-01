#include <gtest/gtest.h>
#include "DeepCFR.h"

static constexpr uint64_t TEST_SEED = 42ULL;

static ActionList makeActions(std::initializer_list<Action> acts) {
    ActionList al{};
    int i = 0;
    for (Action a : acts) al[i++] = a;
    return al;
}

// ---------------------------------------------------------------------------

// All-negative regrets → uniform probability over the legal actions only.
// Actions outside the legal list must remain at zero.
TEST(DeepCFR, GetInstantStratAllNegative) {
    Regrets r     = {-1.0f, -2.0f, -3.0f, -4.0f, -5.0f};
    ActionList al = makeActions({Action::CHECK, Action::CALL, Action::BET50});
    Strategy s    = DeepCFR::getInstantStrat(r, al, 3);

    EXPECT_FLOAT_EQ(s[static_cast<int>(Action::CHECK)],  1.0f / 3);
    EXPECT_FLOAT_EQ(s[static_cast<int>(Action::CALL)],   1.0f / 3);
    EXPECT_FLOAT_EQ(s[static_cast<int>(Action::BET50)],  1.0f / 3);
    EXPECT_FLOAT_EQ(s[static_cast<int>(Action::BET100)], 0.0f);
    EXPECT_FLOAT_EQ(s[static_cast<int>(Action::ALLIN)],  0.0f);
}

// Positive regrets are proportionally normalized; non-positive floor to zero.
TEST(DeepCFR, GetInstantStratNormalized) {
    Regrets r     = {2.0f, 6.0f, 0.0f, -1.0f, -1.0f};
    ActionList al = makeActions({Action::CHECK, Action::CALL, Action::BET50});
    Strategy s    = DeepCFR::getInstantStrat(r, al, 3);

    EXPECT_FLOAT_EQ(s[static_cast<int>(Action::CHECK)], 0.25f);
    EXPECT_FLOAT_EQ(s[static_cast<int>(Action::CALL)],  0.75f);
    EXPECT_FLOAT_EQ(s[static_cast<int>(Action::BET50)], 0.0f);
}

// Actions not in the legal list must have zero probability even when their
// regret is positive.
TEST(DeepCFR, GetInstantStratIllegalActionsZero) {
    Regrets r     = {5.0f, 5.0f, 5.0f, 5.0f, 5.0f};
    ActionList al = makeActions({Action::CHECK, Action::CALL});
    Strategy s    = DeepCFR::getInstantStrat(r, al, 2);

    EXPECT_FLOAT_EQ(s[static_cast<int>(Action::BET50)],  0.0f);
    EXPECT_FLOAT_EQ(s[static_cast<int>(Action::BET100)], 0.0f);
    EXPECT_FLOAT_EQ(s[static_cast<int>(Action::ALLIN)],  0.0f);

    float total = s[static_cast<int>(Action::CHECK)] + s[static_cast<int>(Action::CALL)];
    EXPECT_FLOAT_EQ(total, 1.0f);
}

// A strategy with probability 1 on a single action must always sample that
// action regardless of the RNG state.
TEST(DeepCFR, SampleActionSureStrategy) {
    Strategy strat{};
    strat[static_cast<int>(Action::CALL)] = 1.0f;
    ActionList al = makeActions({Action::CHECK, Action::CALL});

    rng.seed(TEST_SEED);
    for (int i = 0; i < 1000; i++)
        EXPECT_EQ(DeepCFR::sampleAction(strat, al, 2), Action::CALL);
}

// An all-zero strategy means no cumulative threshold is ever crossed.
// The function must fall back to returning the last legal action.
TEST(DeepCFR, SampleActionFallback) {
    Strategy strat{};   // all zero
    ActionList al = makeActions({Action::CHECK, Action::CALL, Action::BET50});

    rng.seed(TEST_SEED);
    EXPECT_EQ(DeepCFR::sampleAction(strat, al, 3), Action::BET50);
}
