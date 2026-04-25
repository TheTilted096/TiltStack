#include "CFRGame.h"

hand_indexer_t g_indexer;

// ---------------------------------------------------------------------------
// classifyRaise — map a continuous raise into the nearest abstract action.
//
// raiseComp : amount committed above the call (amount_milli - toCall)
// effPot    : pot + toCall (the hypothetical pot if the caller called)
//
// Thresholds are arithmetic midpoints between consecutive fractions,
// compared via integer cross-multiplication (no floating point):
//   5/12, 5/8, 7/8, 5/4, 7/4, 5/2
// ---------------------------------------------------------------------------
static Action classifyRaise(int raiseComp, int effPot) {
    const auto &F = CFRGame::BetFractions;
    for (int i = 0; i + 1 < (int)F.size(); i++) {
        // Midpoint between F[i] and F[i+1] via cross-multiplication (no
        // floats). raiseComp/effPot < (F[i]+F[i+1])/2 iff:
        //   raiseComp * 2*F[i].den*F[i+1].den < effPot * (F[i].num*F[i+1].den +
        //   F[i].den*F[i+1].num)
        if (raiseComp * 2 * F[i].den * F[i + 1].den <
            effPot * (F[i].num * F[i + 1].den + F[i].den * F[i + 1].num))
            return static_cast<Action>(2 + i);
    }
    return Action::BET300;
}

CFRGame::CFRGame() { begin(STARTING_STACK, STARTING_STACK, 0); }

void CFRGame::initState(int ss1, int ss2, bool h) {
    ply = 0;
    hero = h;
    isTerminal = false;
    actionCount.fill(0);
    currentRound = Round::PREFLOP;

    std::fill(&betHist[0][0], &betHist[0][0] + sizeof(betHist) / sizeof(float),
              0.0f);
    betHistMask = 0;

    initialStacks[0] = ss1;
    initialStacks[1] = ss2;
    stacks[0] = ss1 - SMALL_BLIND;
    stacks[1] = ss2 - BIG_BLIND;

    history[0].pot = SMALL_BLIND + BIG_BLIND;
    history[0].toCall = BIG_BLIND - SMALL_BLIND;
    history[0].stm = 0;             // little blind
    history[0].act = Action::BET50; // big blind does a half pot bet to start
}

void CFRGame::indexCards(const Card deck[9]) {
    // deck[0..1] = p0 hole, deck[2..3] = p1 hole, deck[4..8] = board
    std::memcpy(rawDeck, deck, 9 * sizeof(Card));
    for (int p = 0; p < 2; p++) {
        uint8_t cards[7] = {deck[p * 2], deck[p * 2 + 1], deck[4], deck[5],
                            deck[6],     deck[7],         deck[8]};
        hand_index_t indices[NUM_ROUNDS];
        hand_index_all(&g_indexer, cards, indices);
        for (int r = 0; r < NUM_ROUNDS; r++)
            streetIDs[r][p] = indices[r];
    }

    // Precompute EHS and cluster bucket for every street and player.
    for (int r = 0; r < NUM_ROUNDS; r++) {
        for (int p = 0; p < 2; p++) {
            streetEHS[r][p] = gEHS[r][streetIDs[r][p]] / 65535.0f;
            streetBucket[r][p] = gLabels[r][streetIDs[r][p]];
        }
    }
}

void CFRGame::begin(int ss1, int ss2, bool h) {
    initState(ss1, ss2, h);

    // Partial Fisher-Yates: pick 9 unique cards from the 52-card deck.
    Card deck[CARDS];
    for (int i = 0; i < CARDS; i++) {
        deck[i] = static_cast<Card>(i);
    }
    for (int i = 0; i < 9; i++) {
        int j =
            i + static_cast<int>(rng.next() % static_cast<uint64_t>(CARDS - i));
        std::swap(deck[i], deck[j]);
    }

    indexCards(deck);
}

void CFRGame::beginWithCards(int ss1, int ss2, bool h, const Card cards[9]) {
    initState(ss1, ss2, h);
    indexCards(cards);
}

bool CFRGame::isFold(const Action &a) {
    return (a == Action::CHECK) and (history[ply].toCall > 0);
}

bool CFRGame::endsStreet(const Action &a) {
    int roundNum = static_cast<int>(currentRound);
    int numActs = actionCount[roundNum];

    if (a == Action::CALL) {
        return (currentRound != Round::PREFLOP) or (numActs > 0);
    }

    if ((a == Action::CHECK) and (history[ply].toCall == 0)) {
        return (numActs > 0);
    }

    return false;
}

int CFRGame::isTerminalState(
    const Action &a) { // 2 showdown, 1 fold, 0 continue
    if (isFold(a)) {
        return 1;
    }

    if (endsStreet(a)) {
        if (currentRound == Round::RIVER) {
            return 2;
        }

        if ((a == Action::CALL) and
            (stacks[!stm()] == 0 || stacks[stm()] == history[ply].toCall)) {
            return 2;
        }
    }

    return 0;
}

bool CFRGame::stm() { return history[ply].stm; }

void CFRGame::makeMove(const Action &a) {
    bool streetEnded = endsStreet(a);
    isTerminal = isTerminalState(a);

    const BoardState &last = history[ply];
    int effPot = last.pot + last.toCall;
    int bet = 0;

    if (a == Action::CALL) {
        bet = last.toCall;
    } else if (a == Action::ALLIN) {
        bet = std::min(stacks[last.stm], last.toCall + stacks[!last.stm]);
    } else if (a >= Action::BET33) {
        const Fraction &f = BetFractions[static_cast<int>(a) - 2];
        bet = last.toCall + effPot * f.num / f.den;
    }
    // CHECK: bet = 0

    ply++;

    BoardState &now = history[ply];

    now.act = a;
    now.pot = last.pot + bet;
    now.toCall = bet - last.toCall;
    stacks[last.stm] -= bet;

    int roundNum = static_cast<int>(currentRound);
    int numActs = actionCount[roundNum];
    actionCount[roundNum]++;

    betHist[roundNum][numActs] =
        static_cast<float>(now.toCall) / (last.pot + last.toCall);
    betHistMask ^= (1u << (roundNum * MAX_ACTIONS + numActs));

    currentRound = static_cast<Round>(roundNum + streetEnded);

    now.stm = streetEnded or !last.stm;
}

void CFRGame::makeBet(int amount_milli) {
    const BoardState &last = history[ply];
    int allinAmt = std::min(stacks[last.stm], last.toCall + stacks[!last.stm]);

    // Classify the bet as the nearest pot-fraction abstract action label.
    Action a;
    if (amount_milli == 0) {
        a = Action::CHECK;
    } else if (amount_milli <= last.toCall) {
        a = Action::CALL;
    } else if (amount_milli >= allinAmt) {
        a = Action::ALLIN;
        amount_milli = allinAmt; // clamp to avoid over-committing
    } else {
        int effPot = last.pot + last.toCall;
        a = classifyRaise(amount_milli - last.toCall, effPot);
    }

    bool streetEnded = endsStreet(a);
    isTerminal = isTerminalState(a);

    ply++;
    BoardState &now = history[ply];
    now.act = a;
    now.pot = last.pot + amount_milli;
    now.toCall = amount_milli - last.toCall;
    stacks[last.stm] -= amount_milli;

    int roundNum = static_cast<int>(currentRound);
    int numActs = actionCount[roundNum];
    actionCount[roundNum]++;

    betHist[roundNum][numActs] =
        static_cast<float>(now.toCall) / (last.pot + last.toCall);
    betHistMask |= (1u << (roundNum * MAX_ACTIONS + numActs));

    currentRound = static_cast<Round>(roundNum + streetEnded);
    now.stm = streetEnded or !last.stm;
}

void CFRGame::unmakeMove() {
    int bet = history[ply].pot - history[ply - 1].pot;
    stacks[history[ply - 1].stm] += bet;
    isTerminal = 0;

    ply--;

    int roundNum = static_cast<int>(currentRound);

    if (actionCount[roundNum] == 0) {
        currentRound = static_cast<Round>(roundNum - 1);
        roundNum--;
    }

    actionCount[roundNum]--;
    betHist[roundNum][actionCount[roundNum]] = 0.0f;
    betHistMask ^= (1u << (roundNum * MAX_ACTIONS + actionCount[roundNum]));
}

float CFRGame::payout() {
    int my_contrib = initialStacks[hero] - stacks[hero];
    int opp_contrib = initialStacks[!hero] - stacks[!hero];

    bool heroWins;

    if (isTerminal == 1) {
        // Fold: stm() is the winner (non-folder after the endsStreet fix).
        heroWins = (stm() == hero);
    } else {
        // Showdown: evaluate both 7-card hands (2 hole + 5 board).
        static omp::HandEvaluator evaluator;

        uint8_t cards0[7], cards1[7];
        hand_unindex(&g_indexer, 3, streetIDs[3][0], cards0);
        hand_unindex(&g_indexer, 3, streetIDs[3][1], cards1);

        omp::Hand h0 = omp::Hand::empty();
        omp::Hand h1 = omp::Hand::empty();
        for (int i = 0; i < 7; i++) {
            h0 += omp::Hand(cards0[i]);
            h1 += omp::Hand(cards1[i]);
        }

        uint16_t rank0 = evaluator.evaluate(h0);
        uint16_t rank1 = evaluator.evaluate(h1);

        if (rank0 == rank1) {
            return 0.0f; // chop
        }

        bool winner = (rank1 > rank0);
        heroWins = (winner == hero);
    }

    return static_cast<float>(heroWins ? opp_contrib : -my_contrib);
}

// BetFractions[i] is the pot fraction for the i-th raise action (BET33…BET300).
const std::array<Fraction, NUM_ACTIONS - 3> CFRGame::BetFractions = {
    {{1, 3}, {1, 2}, {3, 4}, {1, 1}, {3, 2}, {2, 1}, {3, 1}}};

const ActionList CFRGame::FullMenu = {
    Action::BET33,  Action::BET50,  Action::BET75,  Action::BET100,
    Action::BET150, Action::BET200, Action::BET300,
};

// SPR buckets: 0 = Shove (<0.5), 1 = Committed (0.5–1.5),
//              2 = Medium (1.5–4.0), 3 = Deep (≥4.0).
// Each row lists the fraction-based raise actions to offer; ALLIN is always
// appended by generateActions unless a fraction already hits the exact amount.
const ActionList CFRGame::RaiseMenu[NUM_ROUNDS][4] = {
    // PREFLOP
    {
        /* Shove     */ {},
        /* Committed */ {},
        /* Medium    */ {},
        /* Deep      */
        {Action::BET50, Action::BET75, Action::BET100, Action::BET150,
         Action::BET200, Action::BET300},
    },
    // FLOP
    {
        /* Shove     */ {},
        /* Committed */ {Action::BET75, Action::BET100},
        /* Medium    */
        {Action::BET33, Action::BET50, Action::BET75, Action::BET100,
         Action::BET150},
        /* Deep      */
        {Action::BET33, Action::BET50, Action::BET75, Action::BET100,
         Action::BET150, Action::BET200, Action::BET300},
    },
    // TURN
    {
        /* Shove     */ {},
        /* Committed */ {Action::BET75, Action::BET100},
        /* Medium    */
        {Action::BET50, Action::BET75, Action::BET100, Action::BET150},
        /* Deep      */
        {Action::BET50, Action::BET75, Action::BET100, Action::BET150,
         Action::BET200},
    },
    // RIVER
    {
        /* Shove     */ {},
        /* Committed */ {Action::BET75},
        /* Medium    */ {Action::BET75, Action::BET150},
        /* Deep      */ {Action::BET33, Action::BET75, Action::BET150},
    },
};

const int CFRGame::RaiseCount[NUM_ROUNDS][4] = {
    {0, 0, 0, 6}, // PREFLOP
    {0, 2, 5, 7}, // FLOP
    {0, 2, 4, 5}, // TURN
    {0, 1, 2, 3}, // RIVER
};

int CFRGame::generateActions(ActionList &alist, bool prune) {
    const BoardState &cur = history[ply];
    int playerStack = stacks[cur.stm];
    int oppStack = stacks[!cur.stm];

    int callAmt = cur.toCall;
    int effPot = cur.pot + cur.toCall;
    int allinAmt = std::min(playerStack, callAmt + oppStack);

    // Minimum legal raise: at least the previous raise increment (or BB when
    // opening). cur.toCall is the last raise increment (makeMove: now.toCall =
    // bet - last.toCall).
    int minRaise = callAmt + std::max(callAmt, BIG_BLIND);

    int roundNum = static_cast<int>(currentRound);
    bool penultimate = (actionCount[roundNum] == MAX_ACTIONS - 2);

    int n = 0;
    alist[n++] = Action::CHECK;
    if (callAmt > 0)
        alist[n++] = Action::CALL;
    if (allinAmt <= callAmt)
        return n;

    const ActionList *menu;
    int menuSize;
    if (prune) {
        float spr = static_cast<float>(playerStack) / cur.pot;
        int bucket = (spr >= 0.5f) + (spr >= 1.5f) + (spr >= 4.0f);
        menu = &RaiseMenu[roundNum][bucket];
        menuSize = RaiseCount[roundNum][bucket];
    } else {
        menu = &FullMenu;
        menuSize = static_cast<int>(BetFractions.size());
    }

    bool allinCovered = false;
    for (int i = 0; i < menuSize; i++) {
        Action act = (*menu)[i];
        const Fraction &f = BetFractions[static_cast<int>(act) - 2];
        int betAmt = callAmt + effPot * f.num / f.den;

        if (betAmt > allinAmt)
            break; // all subsequent raises (ascending order) also out of range
        if (betAmt < minRaise || (penultimate && betAmt != allinAmt))
            continue;
        if (prune && playerStack - betAmt < initialStacks[cur.stm] / 5)
            break; // bet leaves < 20% stack — absorb remainder into All-In

        alist[n++] = act;
        if (betAmt == allinAmt) {
            allinCovered = true;
            break;
        }
    }

    // ALLIN: always legal as a raise unless a fraction already hit that amount.
    if (!allinCovered)
        alist[n++] = Action::ALLIN;

    return n;
}

InfoSet CFRGame::getInfo() {
    InfoSet info;
    const BoardState &cur = history[ply];
    int roundNum = static_cast<int>(currentRound);
    bool stm = cur.stm;

    float effStack =
        static_cast<float>(std::min(initialStacks[0], initialStacks[1]));

    info.myStack = static_cast<float>(stacks[stm]) / effStack;
    info.oppStack = static_cast<float>(stacks[!stm]) / effStack;
    info.potSize = static_cast<float>(cur.pot) / effStack;
    info.toCall = static_cast<float>(cur.toCall) / effStack;
    info.explicitSPR = std::min(info.myStack / info.potSize, 20.0f) / 20.0f;
    info.currentEHS = streetEHS[roundNum][stm];

    std::memcpy(info.betHist, betHist, sizeof(betHist));
    info.betHistMask = betHistMask;

    // stm==0 → p0 hole cards at rawDeck[0,1]; stm==1 → p1 at rawDeck[2,3]
    int holeBase = stm * 2;
    info.hole = (1ULL << rawDeck[holeBase]) | (1ULL << rawDeck[holeBase + 1]);
    info.flop =
        ((1ULL << rawDeck[4]) | (1ULL << rawDeck[5]) | (1ULL << rawDeck[6])) *
        (currentRound >= Round::FLOP);
    info.turn = (1ULL << rawDeck[7]) * (currentRound >= Round::TURN);
    info.river = (1ULL << rawDeck[8]) * (currentRound >= Round::RIVER);

    // One-hot encode current round
    info.streetEmbed.fill(false);
    info.streetEmbed[roundNum] = true;
    info.isButton = (stm == 0);

    // streetBucket slots 0,1,2 correspond to flop, turn, river (rounds 1,2,3).
    // begin() stores raw 0-indexed cluster labels; +1 is applied here so that
    // 0 is unambiguously "unused" and valid labels occupy 1–8192.
    // Streets not yet reached in the current hand are written as 0.
    for (int i = 0; i < NUM_ROUNDS - 1; i++) {
        int round = i + 1; // 1=flop, 2=turn, 3=river
        info.streetBucket[i] =
            (currentRound >= static_cast<Round>(round))
                ? static_cast<uint16_t>(streetBucket[round][stm] + 1)
                : 0;
    }

    return info;
}

void loadTables(const std::string &clusters_dir) {
    const std::string d = clusters_dir + "/";

    readU16File(d + "preflop_ehs_fine.bin", gEHS[0]);
    readU16File(d + "flop_ehs_fine.bin", gEHS[1]);
    readU16File(d + "turn_ehs_fine.bin", gEHS[2]);
    readU16File(d + "river_ehs_fine.bin", gEHS[3]);

    gLabels[0].resize(gEHS[0].size());
    std::iota(gLabels[0].begin(), gLabels[0].end(), uint16_t{0});

    readU16File(d + "flop_labels.bin", gLabels[1]);
    readU16File(d + "turn_labels.bin", gLabels[2]);
    readU16File(d + "river_labels.bin", gLabels[3]);

    uint8_t rounds[] = {2, 3, 1, 1};
    hand_indexer_init(4, rounds, &g_indexer);
}