#include "CFRUtils.h"
// Include OMPEval before hand-isomorphism: deck.h defines CARDS/RANKS/SUITS
// macros that collide with OMPEval identifiers.
#include "../third_party/OMPEval/HandEvaluator.h"
extern "C" {
#include "../third_party/hand-isomorphism/hand_index.h"
}

class CFRGame{
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

        hand_indexer_t indexer; // 4-round {2,3,1,1} indexer, initialized in CFRGame()

        CFRGame();
        ~CFRGame();

        void begin(int, int, bool);

        int isTerminalState(const Action&);
        bool isFold(const Action&);
        bool endsStreet(const Action&);

        bool stm();
        Action lastAction();

        void makeMove(const Action&);
        void unmakeMove();

        InfoSet getInfo();

        float payout();

        int generateActions(ActionList&);
};