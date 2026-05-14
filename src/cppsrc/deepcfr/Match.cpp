#include "Match.h"
#include "CFRUtils.h"

namespace Match {

Task<float> singleGame(std::array<Card, 9> cards, bool passB, uint8_t flags,
                       Scheduler &sched) {
    CFRGame cfr;
    cfr.beginWithCards(STARTING_STACK, STARTING_STACK, /*hero=*/false,
                       cards.data());

    while (!cfr.isTerminal) {
        int netIdx = static_cast<int>(cfr.stm()) ^ static_cast<int>(passB);
        bool argmax = (flags >> netIdx) & 1u;
        bool prune = (flags >> (2 + netIdx)) & 1u;

        ActionList moves;
        int numMoves = cfr.generateActions(moves, prune);

        Regrets values =
            co_await InferenceAwaitable{cfr.getInfo(), sched, netIdx};

        Action action;
        if (argmax) {
            action = argmaxLegal(values, moves, numMoves);
        } else {
            Strategy s = normalizeLegal(values, moves, numMoves);
            action = sampleAction(s, moves, numMoves);
        }

        cfr.makeMove(action);
    }

    co_return cfr.payout(); // hero=false → P0's milli-chip gain/loss
}

Task<float> gamePair(uint8_t flags, Scheduler &sched) {
    std::array<Card, 9> cards;
    dealCards(cards);

    // Pass A: P0-net as P0 (SB).  payout() returns P0's payoff directly.
    float payoffA = co_await singleGame(cards, /*passB=*/false, flags, sched);

    // Pass B: seats swapped — P0-net as P1 (BB).
    // payout() still returns P0's payoff, so P0-net's BB payoff = -payoffB.
    float payoffB = co_await singleGame(cards, /*passB=*/true, flags, sched);

    sched.sbPayoffs.push_back(payoffA);
    sched.bbPayoffs.push_back(-payoffB);

    co_return payoffA - payoffB;
}

} // namespace Match
