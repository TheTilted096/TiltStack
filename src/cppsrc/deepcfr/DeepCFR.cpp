#include "DeepCFR.h"
#include <limits>

namespace DeepCFR {

Action sampleAction(const Strategy &strat, const ActionList &moves,
                    int numMoves) {
    float sample = rng.nextFloat();
    float cumulative = 0.0f;
    for (int i = 0; i < numMoves; i++) {
        cumulative += strat[static_cast<int>(moves[i])];
        if (sample < cumulative)
            return moves[i];
    }
    return moves[numMoves - 1];
}

Strategy getInstantStrat(const Regrets &r, const ActionList &moves,
                         int numMoves) {
    Strategy s{};
    float sum = 0.0f;

    for (int i = 0; i < numMoves; i++) {
        int a = static_cast<int>(moves[i]);
        s[a] = std::max(0.0f, r[a]);
        sum += s[a];
    }

    if (sum > 0.0f) {
        for (int i = 0; i < numMoves; i++)
            s[static_cast<int>(moves[i])] /= sum;
    } else {
        float uniform = 1.0f / numMoves;
        for (int i = 0; i < numMoves; i++)
            s[static_cast<int>(moves[i])] = uniform;
    }

    return s;
}

// ---------------------------------------------------------------------------

Task<float> rollout(CFRGame game, bool hero, int t, Scheduler &sched) {
    // game is owned by this frame for the lifetime of the traversal.
    co_return co_await traverse(game, hero, t, sched);
}

Task<float> traverse(CFRGame &game, bool hero, int t, Scheduler &sched) {
    if (game.isTerminal)
        co_return game.payout();

    ActionList moves;
    int numMoves = game.generateActions(moves, /*prune=*/true);

    Regrets predictedRegrets =
        co_await InferenceAwaitable{game.getInfo(), sched};
    Strategy instantStrategy =
        getInstantStrat(predictedRegrets, moves, numMoves);

    float nodeEV = 0.0f;

    if (game.stm() == hero) {
        Regrets trueRegret;
        trueRegret.fill(std::numeric_limits<float>::quiet_NaN());
        std::array<float, NUM_ACTIONS> actionUtils{};

        for (int i = 0; i < numMoves; i++) {
            int actionInt = static_cast<int>(moves[i]);
            game.makeMove(moves[i]);
            actionUtils[actionInt] = co_await traverse(game, hero, t, sched);
            game.unmakeMove();
            nodeEV += instantStrategy[actionInt] * actionUtils[actionInt];
        }

        for (int j = 0; j < numMoves; j++) {
            int actionInt = static_cast<int>(moves[j]);
            trueRegret[actionInt] = (actionUtils[actionInt] - nodeEV) /
                                    static_cast<float>(STARTING_STACK);
        }

        sched.advantageInputs.push_back(game.getInfo());
        sched.advantageOutputs.push_back(trueRegret);

        co_return nodeEV;
    }

    if (t > 50) {
        sched.policyInputs.push_back(game.getInfo());
        sched.policyOutputs.push_back(instantStrategy);
        sched.policyWeights.push_back(t);
    }

    Action villainMove = sampleAction(instantStrategy, moves, numMoves);
    game.makeMove(villainMove);
    nodeEV = co_await traverse(game, hero, t, sched);
    game.unmakeMove();

    co_return nodeEV;
}

} // namespace DeepCFR
