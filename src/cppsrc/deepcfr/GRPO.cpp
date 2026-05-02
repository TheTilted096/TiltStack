#include "GRPO.h"

Task<float> GRPO::rollout(CFRGame game, bool hero, Scheduler &sched) {
    // game is owned by this frame for the lifetime of the traversal.
    co_return co_await traverse(game, hero, 1.0f, sched);
}

Task<float> GRPO::traverse(CFRGame &game, bool hero, float heroReach, Scheduler &sched) {
    if (game.isTerminal)
        co_return game.payout();

    ActionList moves;
    int numMoves = game.generateActions(moves, game.stm() xor hero);

    Strategy instantStrategy = co_await InferenceAwaitable{game.getInfo(), sched};

    // Renormalize over legal actions and NaN-fill illegal slots so that
    // (a) the distribution is exact over legal actions and (b) Python can
    // detect illegal slots via isnan() for loss masking.
    float stratSum = 0.0f;
    for (int i = 0; i < numMoves; i++)
        stratSum += instantStrategy[static_cast<int>(moves[i])];
    Strategy normalized;
    normalized.fill(std::numeric_limits<float>::quiet_NaN());
    if (stratSum > 1e-8f) {
        for (int i = 0; i < numMoves; i++) {
            int a = static_cast<int>(moves[i]);
            normalized[a] = instantStrategy[a] / stratSum;
        }
    } else {
        float uniform = 1.0f / numMoves;
        for (int i = 0; i < numMoves; i++)
            normalized[static_cast<int>(moves[i])] = uniform;
    }
    instantStrategy = normalized;

    float nodeEV = 0.0f;

    if (game.stm() == hero) {
        Regrets trueRegret;
        trueRegret.fill(std::numeric_limits<float>::quiet_NaN());
        std::array<float, NUM_ACTIONS> actionUtils{};

        for (int i = 0; i < numMoves; i++) {
            int actionInt = static_cast<int>(moves[i]);
            game.makeMove(moves[i]);
            actionUtils[actionInt] = co_await traverse(game, hero,
                heroReach * instantStrategy[actionInt], sched);
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

    Action villainMove = DeepCFR::sampleAction(instantStrategy, moves, numMoves);
    game.makeMove(villainMove);
    nodeEV = co_await traverse(game, hero, heroReach, sched);
    game.unmakeMove();

    co_return nodeEV;
}