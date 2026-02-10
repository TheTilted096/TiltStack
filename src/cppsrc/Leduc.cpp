#include "Leduc.h"

LeducSolver::LeducSolver() : nodes(528) {}

float LeducSolver::cfr(const std::array<Rank, 3>& cards, Hash hash, std::array<float, 2> prob){
    NodeInfo info(hash);
    int stm = info.stm();

    ActionList moves = info.moves();

    Strategy node_strat = nodes[hash].get_current_strategy(prob[stm], moves);
    float node_util = 0.0f;
    float action_utils[3] = {};

    for(int i = 0; i < moves.len; i++){
        Action a = moves.al[i];
        std::array<float, 2> next_prob = prob;
        next_prob[stm] *= node_strat[a];

        if(info.ends_hand(a)){
            action_utils[i] = info.payout(a, cards[1 - stm]);
        } else {
            int ns = info.next_stm(a);
            action_utils[i] = (2 * (stm == ns) - 1) * cfr(cards, info.next_hash(a, cards[2], cards[ns]), next_prob);
        }

        node_util += node_strat[a] * action_utils[i];
    }

    for(int j = 0; j < moves.len; j++){
        float regret = action_utils[j] - node_util;
        nodes[hash].regrets[moves.al[j]] += regret * prob[1 - stm];
    }

    return node_util;
}
