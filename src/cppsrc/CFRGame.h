#include "CFRUtils.h"
#include <cstring>
// Include OMPEval before hand-isomorphism: deck.h defines CARDS/RANKS/SUITS
// macros that collide with OMPEval identifiers.
#include "../third_party/OMPEval/HandEvaluator.h"
extern "C" {
#include "../third_party/hand-isomorphism/hand_index.h"
}

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

    hand_indexer_t
        indexer; // 4-round {2,3,1,1} indexer, initialized in CFRGame()

    CFRGame();
    ~CFRGame();

    // hand_indexer_t owns heap-allocated tables via malloc; disable copy,
    // enable move.  The moved-from object's destructor is made a no-op by
    // zeroing rounds, since hand_indexer_free() loops on that field.
    CFRGame(const CFRGame &) = delete;
    CFRGame &operator=(const CFRGame &) = delete;

    CFRGame(CFRGame &&other) noexcept {
        history = other.history;
        ply = other.ply;
        stacks = other.stacks;
        initialStacks = other.initialStacks;
        hero = other.hero;
        isTerminal = other.isTerminal;
        actionCount = other.actionCount;
        currentRound = other.currentRound;
        std::memcpy(betHist, other.betHist, sizeof(betHist));
        std::memcpy(streetBucket, other.streetBucket, sizeof(streetBucket));
        std::memcpy(streetEHS, other.streetEHS, sizeof(streetEHS));
        std::memcpy(streetIDs, other.streetIDs, sizeof(streetIDs));
        indexer = other.indexer;
        other.indexer.rounds = 0; // disarm moved-from destructor
    }
    CFRGame &operator=(CFRGame &&other) noexcept {
        if (this != &other) {
            hand_indexer_free(&indexer);
            history = other.history;
            ply = other.ply;
            stacks = other.stacks;
            initialStacks = other.initialStacks;
            hero = other.hero;
            isTerminal = other.isTerminal;
            actionCount = other.actionCount;
            currentRound = other.currentRound;
            std::memcpy(betHist, other.betHist, sizeof(betHist));
            std::memcpy(streetBucket, other.streetBucket, sizeof(streetBucket));
            std::memcpy(streetEHS, other.streetEHS, sizeof(streetEHS));
            std::memcpy(streetIDs, other.streetIDs, sizeof(streetIDs));
            indexer = other.indexer;
            other.indexer.rounds = 0;
        }
        return *this;
    }

    void begin(int, int, bool);

    int isTerminalState(const Action &);
    bool isFold(const Action &);
    bool endsStreet(const Action &);

    bool stm();
    void makeMove(const Action &);
    void unmakeMove();

    InfoSet getInfo();

    float payout();

    int generateActions(ActionList &);
};