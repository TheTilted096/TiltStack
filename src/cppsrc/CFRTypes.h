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

const int NUM_ACTIONS = 5; // number of distinct actions
const int MAX_ACTIONS = 6; // maximum number of actions per betting round

const int NUM_ROUNDS = 4;

enum class Action : uint8_t { 
    CHECK, // also FOLD
    CALL, 
    BET50, 
    BET100, 
    //BET200, 
    ALLIN 
};

enum class Round : uint8_t { PREFLOP, FLOP, TURN, RIVER };

using Strategy = std::array<float, NUM_ACTIONS>; // array of probabilities
using Regrets = std::array<float, NUM_ACTIONS>; // array of regrets
using CardArr = std::array<float, 52>; // array of 0.0 or 1.0 depending on card

using ActionList = std::array<Action, NUM_ACTIONS>;

struct InfoSet{
    // 8-byte: card presence bitfields (52 LSBs used, one bit per card index)
    uint64_t hole, flop, turn, river;

    // 4-byte: normalised scalars and betting history
    float myStack, oppStack, potSize, toCall;
    float currentEHS;
    float betHist[NUM_ROUNDS][MAX_ACTIONS];

    // 2-byte: street abstraction buckets (flop/turn/river)
    std::array<uint16_t, NUM_ROUNDS - 1> streetBucket;

    // 1-byte: one-hot round encoding and button flag
    std::array<bool, NUM_ROUNDS> streetEmbed;
    bool isButton; // true if stm == 0
};

struct BoardState{
    int pot;
    int toCall;
    bool stm;
    Action act;
};
