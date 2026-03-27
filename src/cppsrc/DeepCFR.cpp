#include "CFRGame.h"

Strategy getInstantStrat(const Regrets& r){
    // get regrets from strategy
}

float deepcfr(CFRGame& game, int id, bool hero){
    if (game.isTerminal){
        return game.payout();
    }

    ActionList moves;
    int numMoves = game.generateActions(moves);

    // inference the regret network here
    Regrets predictedRegrets;
    Strategy instantStrategy = getInstantStrat(predictedRegrets);

    float nodeEV = 0.0;

    if (game.stm() == hero){
        Regrets trueRegret{};
        std::array<float, NUM_ACTIONS> actionUtils{};

        for (int i = 0; i < numMoves; i++){
            game.makeMove(moves[i]);

            int actionInt = static_cast<int>(moves[i]);

            actionUtils[actionInt] = deepcfr(game, id, hero);

            game.unmakeMove();

            nodeEV += instantStrategy[actionInt] * actionUtils[actionInt];
        }

        for (int j = 0; j < numMoves; j++){
            int actionInt = static_cast<int>(moves[j]);
            trueRegret[actionInt] = actionUtils[actionInt] - nodeEV;
        }

        // stuff to train the advantage network on
        // advantageInput.push_back(game.info);
        // advantageOutput.push_back(trueRegret);

        return nodeEV;
    }

    // thread-safe:
    // policyInput.push_back(game.info);
    // policyOutput.push_back(instantStrategy);

    Action villainMove; // sample from distribution (computed at top)
    game.makeMove(villainMove);
    nodeEV = deepcfr(game, id, hero);
    game.unmakeMove();

    return nodeEV;    
}