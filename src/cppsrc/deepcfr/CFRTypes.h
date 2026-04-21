#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <vector>

using Card = uint8_t;

constexpr int NUM_PLAYERS = 2;
constexpr int STARTING_STACK = 40000; // 20 BB in milli-chips
constexpr int SMALL_BLIND = 1000;     // milli-chips
constexpr int BIG_BLIND = 2000;       // milli-chips

const int NUM_ACTIONS = 10; // number of distinct actions
const int MAX_ACTIONS = 6;  // maximum number of actions per betting round

const int NUM_ROUNDS = 4;

enum class Action : uint8_t {
    CHECK,  // 0 — also FOLD when to_call > 0
    CALL,   // 1
    BET33,  // 2 — 0.33x pot
    BET50,  // 3 — 0.50x pot
    BET75,  // 4 — 0.75x pot
    BET100, // 5 — 1.00x pot
    BET150, // 6 — 1.50x pot
    BET200, // 7 — 2.00x pot
    BET300, // 8 — 3.00x pot
    ALLIN   // 9
};

enum class Round : uint8_t { PREFLOP, FLOP, TURN, RIVER };

using Strategy = std::array<float, NUM_ACTIONS>; // array of probabilities
using Regrets = std::array<float, NUM_ACTIONS>;  // array of regrets
using CardArr = std::array<float, 52>; // array of 0.0 or 1.0 depending on card

using ActionList = std::array<Action, NUM_ACTIONS>;

struct InfoSet {
    // 8-byte: card presence bitfields (52 LSBs used, one bit per card index)
    uint64_t hole, flop, turn, river;

    // 4-byte: normalised scalars and betting history
    float myStack, oppStack, potSize, toCall;
    float currentEHS;
    float betHist[NUM_ROUNDS][MAX_ACTIONS];

    // 4-byte: bitmask marking used betHist slots.
    // Bit (r * MAX_ACTIONS + a) is set when betHist[r][a] has been written.
    // Only the bottom 24 LSBs are used (4 rounds × 6 slots).
    uint32_t betHistMask;

    // 2-byte: street abstraction buckets (flop/turn/river)
    std::array<uint16_t, NUM_ROUNDS - 1> streetBucket;

    // 1-byte: one-hot round encoding and button flag
    std::array<bool, NUM_ROUNDS> streetEmbed;
    bool isButton; // true if stm == 0
};

static_assert(offsetof(InfoSet, betHist) == 52, "InfoSet layout changed");
static_assert(offsetof(InfoSet, betHistMask) == 148, "InfoSet layout changed");
static_assert(offsetof(InfoSet, streetBucket) == 152, "InfoSet layout changed");
static_assert(offsetof(InfoSet, streetEmbed) == 158, "InfoSet layout changed");
static_assert(offsetof(InfoSet, isButton) == 162, "InfoSet layout changed");
static_assert(sizeof(InfoSet) == 168, "InfoSet layout changed");

struct BoardState {
    int pot;
    int toCall;
    bool stm;
    Action act;
};
