#include "CFRTypes.h"
// Include OMPEval before deck.h: deck.h defines CARDS/RANKS/SUITS macros that
// collide with OMPEval identifiers.
#include "../third_party/OMPEval/HandEvaluator.h"
#include "../third_party/hand-isomorphism/deck.h"

// Shared xorshift64* PRNG — one instance across all CFRGame objects that use it.
// Not all CFRGame objects need to share this; callers may seed or bypass it as needed.
inline uint64_t rngState = 0xdeadbeefcafe1234ULL;

inline uint64_t fastRand(){
    rngState ^= rngState >> 12;
    rngState ^= rngState << 25;
    rngState ^= rngState >> 27;
    return rngState * 0x2545F4914F6CDD1DULL;
}

class CFRGame{
    public:
        std::array<BoardState, MAX_ACTIONS * NUM_ROUNDS + 1> history;
        CardArr hole, flop, turn, river;

        int ply;

        std::array<int, 2> stacks;
        std::array<int, 2> initialStacks;

        bool hero;
        int isTerminal;

        std::array<Card, 5> board; // pre-deal all 5 cards
        Card holeCards[2][2]; // deal hole cards

        std::array<uint8_t, NUM_ROUNDS + 1> actionCount;
        Round currentRound;

        float betHist[NUM_ROUNDS][MAX_ACTIONS]; 
        std::array<int16_t, NUM_ROUNDS - 1> streetBucket; 

        CFRGame();

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