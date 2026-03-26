#include "CFRGame.h"

CFRGame::CFRGame(){
    begin(STARTING_STACK, STARTING_STACK, 0);
}

void CFRGame::begin(int ss1, int ss2, bool h){
    // hero, board/hole cards, stacks, set externally
    ply = 0;
    hero = h;
    isTerminal = false;
    actionCount.fill(0);
    currentRound = Round::PREFLOP;
    streetBucket.fill(0);

    std::fill(&betHist[0][0], &betHist[0][0] + sizeof(betHist) / sizeof(float), -1.0f);

    initialStacks[0] = ss1;
    initialStacks[1] = ss2;
    stacks[0] = ss1 - SMALL_BLIND;
    stacks[1] = ss2 - BIG_BLIND;

    history[0].pot = SMALL_BLIND + BIG_BLIND;
    history[0].toCall = BIG_BLIND - SMALL_BLIND;
    history[0].stm = 0; // little blind

    history[0].act = Action::BET50; // big blind does a half pot bet to start

    history[0].ehs = 0.0; // TODO: LOOKUP

    // Partial Fisher-Yates: pick 9 unique cards from the 52-card deck.
    Card deck[CARDS];
    for (int i = 0; i < CARDS; i++){
        deck[i] = static_cast<Card>(i);
    }
    for (int i = 0; i < 9; i++){
        int j = i + static_cast<int>(fastRand() % static_cast<uint64_t>(CARDS - i));
        std::swap(deck[i], deck[j]);
    }

    std::memcpy(holeCards, deck, 4);
    std::memcpy(board.data(), deck + 4, 5);
}

bool CFRGame::isFold(const Action& a){
    return (a == Action::CHECK) and (history[ply].toCall > 0);
}

bool CFRGame::endsStreet(const Action& a){
    int roundNum = static_cast<int>(currentRound);
    int numActs = actionCount[roundNum];

    if (a == Action::CALL){
        return (currentRound != Round::PREFLOP) or (numActs > 0);
    }

    if ((a == Action::CHECK) and (history[ply].toCall == 0)){
        return (numActs > 0);
    }

    return false;
}

int CFRGame::isTerminalState(const Action& a){ // 2 showdown, 1 fold, 0 continue
    if (isFold(a)){
        return 1;
    }

    if (endsStreet(a)){
        if (currentRound == Round::RIVER){
            return 2;
        }

        if ((a == Action::CALL) and (lastAction() == Action::ALLIN)){
            return 2;
        }
    }

    return 0;
}

bool CFRGame::stm(){ return history[ply].stm; }

Action CFRGame::lastAction(){ return history[ply].act; }

void CFRGame::makeMove(const Action& a){
    bool streetEnded = endsStreet(a);
    isTerminal = isTerminalState(a);

    int bet = 0;

    const BoardState& last = history[ply];

    if (a == Action::CALL) {
        bet = last.toCall;
    } else if (a == Action::BET50) {
        // Match the call, then add 50% of the hypothetical called pot
        bet = last.toCall + (last.pot + last.toCall) / 2;
    } else if (a == Action::BET100) {
        // Match the call, then add 100% of the hypothetical called pot
        bet = last.toCall + (last.pot + last.toCall);
    } else if (a == Action::ALLIN) {
        bet = std::min(stacks[last.stm], last.toCall + stacks[!last.stm]);
    }

    ply++;

    BoardState& now = history[ply];

    now.act = a;
    now.ehs; // TODO: LOOKUP

    now.pot = last.pot + bet;
    now.toCall = bet - last.toCall;
    stacks[last.stm] -= bet;

    int roundNum = static_cast<int>(currentRound);
    int numActs = actionCount[roundNum];
    actionCount[roundNum]++;

    betHist[roundNum][numActs] = static_cast<float>(now.toCall) / (last.pot + last.toCall);

    currentRound = static_cast<Round>(roundNum + streetEnded);
    streetBucket[roundNum + streetEnded]; // TODO: LOOKUP BUCKET 

    now.stm = streetEnded or !last.stm;
}

void CFRGame::unmakeMove(){
    int bet = history[ply].pot - history[ply - 1].pot;
    stacks[history[ply - 1].stm] += bet;
    isTerminal = 0;

    ply--;

    int roundNum = static_cast<int>(currentRound);

    if (actionCount[roundNum] == 0){
        currentRound = static_cast<Round>(roundNum - 1);
        roundNum--;
    }

    actionCount[roundNum]--;
    betHist[roundNum][actionCount[roundNum]] = -1.0f;
}

float CFRGame::payout(){
    int my_contrib  = initialStacks[hero]  - stacks[hero];
    int opp_contrib = initialStacks[!hero] - stacks[!hero];

    bool heroWins;

    if (isTerminal == 1){
        // Fold: stm() is the winner (non-folder after the endsStreet fix).
        heroWins = (stm() == hero);
    } else {
        // Showdown: evaluate both 7-card hands (2 hole + 5 board).
        static omp::HandEvaluator evaluator;

        omp::Hand h0 = omp::Hand::empty();
        h0 += omp::Hand(holeCards[0][0]);
        h0 += omp::Hand(holeCards[0][1]);
        omp::Hand h1 = omp::Hand::empty();
        h1 += omp::Hand(holeCards[1][0]);
        h1 += omp::Hand(holeCards[1][1]);
        for (int i = 0; i < 5; i++){
            h0 += omp::Hand(board[i]);
            h1 += omp::Hand(board[i]);
        }

        uint16_t rank0 = evaluator.evaluate(h0);
        uint16_t rank1 = evaluator.evaluate(h1);

        if (rank0 == rank1){
            return 0.0f; // chop
        }

        bool winner = (rank1 > rank0);
        heroWins = (winner == hero);
    }

    return static_cast<float>(heroWins ? opp_contrib : -my_contrib);
}

int CFRGame::generateActions(ActionList& alist){
    const BoardState& cur = history[ply];
    int playerStack = stacks[cur.stm];
    int oppStack    = stacks[!cur.stm];

    int callAmt   = cur.toCall;
    int bet50Amt  = cur.toCall + (cur.pot + cur.toCall) / 2;
    int bet100Amt = cur.toCall + (cur.pot + cur.toCall);
    // Opponent may not have enough to cover a full raise; cap at what can be matched.
    int allinAmt  = std::min(playerStack, cur.toCall + oppStack);

    // Minimum legal raise: must be at least the previous raise increment (or BB when opening).
    // cur.toCall is the last raise increment (see makeMove: now.toCall = bet - last.toCall).
    int minRaise = cur.toCall + std::max(cur.toCall, BIG_BLIND);

    int n = 0;

    // CHECK / FOLD: always legal (fold when toCall > 0, check otherwise).
    alist[n++] = Action::CHECK;

    // CALL: legal whenever there is a bet to call.
    // Handles call-all-in transparently (we still emit CALL, not ALLIN).
    if (callAmt > 0) {
        alist[n++] = Action::CALL;
    }

    // BET50: valid if the computed amount meets the min raise and the player can afford it.
    if (bet50Amt >= minRaise && bet50Amt <= playerStack) {
        alist[n++] = Action::BET50;
    }

    // BET100: same validity check; skip if it somehow equals BET50 (rounding).
    if (bet100Amt >= minRaise && bet100Amt <= playerStack && bet100Amt != bet50Amt) {
        alist[n++] = Action::BET100;
    }

    // ALLIN: include whenever it is a true raise (more than a call).
    // Skip if the amount equals BET50 or BET100 — prefer the more passive label.
    if (allinAmt > callAmt && allinAmt != bet50Amt && allinAmt != bet100Amt) {
        alist[n++] = Action::ALLIN;
    }

    return n;
}