#pragma once
#include "CFRUtils.h"
// Include OMPEval before hand-isomorphism: deck.h defines CARDS/RANKS/SUITS
// macros that collide with OMPEval identifiers.
#include "HandEvaluator.h"
extern "C" {
#define _Bool bool
#include "hand_index.h"
#undef _Bool
}

// Single shared indexer for all CFRGame instances.  Initialized once before
// any worker threads are spawned; hand_index_all / hand_unindex only read
// the tables after init, so concurrent access is safe.
extern hand_indexer_t g_indexer;

class CFRGame {
  public:
    std::array<BoardState, MAX_ACTIONS * NUM_ROUNDS + 1> history;

    int ply;

    std::array<int, 2> stacks;
    std::array<int, 2> initialStacks;

    bool hero;
    int isTerminal;

    std::array<uint8_t, NUM_ROUNDS + 1> actionCount;
    Round currentRound;

    float betHist[NUM_ROUNDS][MAX_ACTIONS];
    uint16_t streetBucket[NUM_ROUNDS][2];
    float streetEHS[NUM_ROUNDS][2];
    hand_index_t streetIDs[NUM_ROUNDS][2];
    Card rawDeck[9];      // {p0h0, p0h1, p1h0, p1h1, f0, f1, f2, turn, river}
    uint32_t betHistMask; // mirrors InfoSet::betHistMask; maintained by
                          // makeMove/makeBet/unmakeMove

    CFRGame();
    ~CFRGame() = default;

    void begin(int, int, bool);

    // Initialise with explicitly provided cards rather than a random shuffle.
    // cards[9] = {p0h0, p0h1, p1h0, p1h1, flop0, flop1, flop2, turn, river}.
    // Unreached-street slots may be any unused card index in [0,51]; their
    // EHS/bucket values are never read before that street is reached.
    void beginWithCards(int, int, bool, const Card cards[9]);

    int isTerminalState(const Action &);
    bool isFold(const Action &);
    bool endsStreet(const Action &);

    bool stm();
    void makeMove(const Action &);
    void unmakeMove();

    InfoSet getInfo();

    float payout();

    int generateActions(ActionList &);

    // Apply a bet of amount_milli milli-chips.  The financial state is updated
    // with the exact amount; the recorded action label is the pot-fraction
    // abstract action whose fraction is closest to amount_milli / (pot +
    // toCall):
    //   CHECK  — amount_milli == 0
    //   CALL   — amount_milli <= toCall
    //   BET50  — raise fraction < 0.75  (closer to 0.5× pot)
    //   BET100 — raise fraction >= 0.75 and < all-in  (closer to 1.0× pot)
    //   ALLIN  — amount_milli >= effective all-in amount (clamped)
    void makeBet(int amount_milli);

  private:
    // Shared by begin() and beginWithCards(): zero-initialise all game state
    // and set up stacks / blinds / the initial BoardState.
    void initState(int ss1, int ss2, bool h);

    // Index each player's 7-card hand and precompute EHS + cluster buckets for
    // all streets.  deck[9] = {p0h0, p0h1, p1h0, p1h1, f0, f1, f2, turn,
    // river}.
    void indexCards(const Card deck[9]);
};