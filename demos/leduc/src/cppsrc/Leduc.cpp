#include "Leduc.h"

LeducSolver::LeducSolver() : nodes(528) {}

float LeducSolver::cfr(const std::array<Rank, 3>& cards, Hash hash, std::array<double, 2> prob, int iteration, int update_player, bool accumulate_strategy){
    NodeInfo info(hash);
    int stm = info.stm();

    ActionList moves = info.moves();

    Strategy node_strat = nodes[hash].get_current_strategy(prob[stm], moves, iteration, accumulate_strategy);
    float node_util = 0.0f;
    float action_utils[3] = {};

    for(int i = 0; i < moves.len; i++){
        Action a = moves.al[i];
        std::array<double, 2> next_prob = prob;
        next_prob[stm] *= node_strat[a];

        if(info.ends_hand(a)){
            action_utils[i] = info.payout(a, cards[1 - stm]);
        } else {
            int ns = info.next_stm(a);
            action_utils[i] = (2 * (stm == ns) - 1) * cfr(cards, info.next_hash(a, cards[2], cards[ns]), next_prob, iteration, update_player, accumulate_strategy);
        }

        node_util += node_strat[a] * action_utils[i];
    }

    // CFR+: Accumulate regret deltas only for the updating player
    if(stm == update_player){
        for(int j = 0; j < moves.len; j++){
            float regret = action_utils[j] - node_util;
            nodes[hash].regret_deltas[moves.al[j]] += regret * static_cast<float>(prob[1 - stm]);
        }
    }

    return node_util;
}

void LeducSolver::flush_regrets(){
    // Apply accumulated deltas and floor all nodes
    for(int h = 0; h < 528; h++){
        nodes[h].flush_regrets();
    }
}

std::vector<Strategy> LeducSolver::get_all_strategies(){
    std::vector<Strategy> strategies(528);
    for(int h = 0; h < 528; h++){
        NodeInfo info(h);
        strategies[h] = nodes[h].get_stored_strategy(info.moves());
    }
    return strategies;
}
