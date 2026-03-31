#include "CFRGame.h"

// ---------------------------------------------------------------------------
// Replay buffers — populated during deepcfr traversals.
// Cleared and consumed by the Python training loop between CFR iterations.
// ---------------------------------------------------------------------------

std::vector<InfoSet> advantageInput;
std::vector<Regrets>  advantageOutput;

std::vector<InfoSet>  policyInput;
std::vector<Strategy> policyOutput;

// ---------------------------------------------------------------------------
// Regret matching over the legal action subset only.
// Illegal actions stay at 0 probability so indexing by Action cast is safe.
// ---------------------------------------------------------------------------

Action sampleAction(const Strategy& strat, const ActionList& moves, int numMoves){
    float sample = (fastRand() >> 40) * 0x1.0p-24f;

    float cumulative = 0.0f;
    for (int i = 0; i < numMoves; i++){
        cumulative += strat[static_cast<int>(moves[i])];
        if (sample < cumulative)
            return moves[i];
    }
    return moves[numMoves - 1];
}

Strategy getInstantStrat(const Regrets& r, const ActionList& moves, int numMoves){
    Strategy s{};
    float sum = 0.0f;

    for (int i = 0; i < numMoves; i++){
        int a = static_cast<int>(moves[i]);
        s[a] = std::max(0.0f, r[a]);
        sum += s[a];
    }

    if (sum > 0.0f){
        for (int i = 0; i < numMoves; i++)
            s[static_cast<int>(moves[i])] /= sum;
    } else {
        float uniform = 1.0f / numMoves;
        for (int i = 0; i < numMoves; i++)
            s[static_cast<int>(moves[i])] = uniform;
    }

    return s;
}

float deepcfr(CFRGame& game, bool hero){
    if (game.isTerminal){
        return game.payout();
    }

    ActionList moves;
    int numMoves = game.generateActions(moves);

    // inference the regret network here
    Regrets predictedRegrets{};
    // TODO: call regret network with game.getInfo() to fill predictedRegrets

    Strategy instantStrategy = getInstantStrat(predictedRegrets, moves, numMoves);

    float nodeEV = 0.0;

    if (game.stm() == hero){
        Regrets trueRegret{};
        std::array<float, NUM_ACTIONS> actionUtils{};

        for (int i = 0; i < numMoves; i++){
            game.makeMove(moves[i]);

            int actionInt = static_cast<int>(moves[i]);

            actionUtils[actionInt] = deepcfr(game, hero);

            game.unmakeMove();

            nodeEV += instantStrategy[actionInt] * actionUtils[actionInt];
        }

        float pot = static_cast<float>(game.history[game.ply].pot);
        for (int j = 0; j < numMoves; j++){
            int actionInt = static_cast<int>(moves[j]);
            trueRegret[actionInt] = (actionUtils[actionInt] - nodeEV) / pot;
        }

        advantageInput.push_back(game.getInfo());
        advantageOutput.push_back(trueRegret);

        return nodeEV;
    }

    policyInput.push_back(game.getInfo());
    policyOutput.push_back(instantStrategy);

    Action villainMove = sampleAction(instantStrategy, moves, numMoves);

    game.makeMove(villainMove);
    nodeEV = deepcfr(game, hero);
    game.unmakeMove();

    return nodeEV;
}
