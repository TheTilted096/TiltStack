#include "TPO.h"

Task<float> TPO::rollout(CFRGame game, bool hero, Scheduler &sched) {
    // game is owned by this frame for the lifetime of the traversal.
    co_return co_await traverse(game, hero, 1.0f, sched);
}

Task<float> TPO::traverse(CFRGame &game, bool hero, float heroReach,
                          Scheduler &sched) {
    if (game.isTerminal)
        co_return game.payout();

    ActionList moves;
    int numMoves = game.generateActions(moves, game.stm() xor hero);

    Strategy instantStrategy =
        co_await InferenceAwaitable{game.getInfo(), sched};

    // Renormalize over legal actions; NaN-fill illegal slots for Python loss
    // masking.
    instantStrategy = normalizeLegal(instantStrategy, moves, numMoves);
    nanMaskIllegal(instantStrategy, moves, numMoves);

    float nodeEV = 0.0f;

    if (game.stm() == hero) {
        Regrets trueRegret;
        trueRegret.fill(std::numeric_limits<float>::quiet_NaN());
        std::array<float, NUM_ACTIONS> actionUtils{};

        for (int i = 0; i < numMoves; i++) {
            int actionInt = static_cast<int>(moves[i]);
            game.makeMove(moves[i]);
            float nextReach =
                heroReach * std::max(instantStrategy[actionInt], 0.05f);
            actionUtils[actionInt] =
                co_await traverse(game, hero, nextReach, sched);
            game.unmakeMove();
            nodeEV += instantStrategy[actionInt] * actionUtils[actionInt];
        }

        for (int j = 0; j < numMoves; j++) {
            int actionInt = static_cast<int>(moves[j]);
            trueRegret[actionInt] = (actionUtils[actionInt] - nodeEV);
        }

        sched.advantageInputs.push_back(game.getInfo());
        sched.advantageOutputs.push_back(trueRegret);
        sched.policyInputs.push_back(game.getInfo());
        sched.policyOutputs.push_back(instantStrategy);
        sched.policyWeights.push_back(heroReach);

        co_return nodeEV;
    }

    Action villainMove = sampleAction(instantStrategy, moves, numMoves);
    game.makeMove(villainMove);
    nodeEV = co_await traverse(game, hero, heroReach, sched);
    game.unmakeMove();

    co_return nodeEV;
}
