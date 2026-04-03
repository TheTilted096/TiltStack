#include "BestResponse.h"

BestResponse::BestResponse()
    : fixed_strategy(528, {0.0f, 0.0f, 0.0f}),
      br_strategy(528, {0.0f, 0.0f, 0.0f}),
      cached_ev(0.0f),
      action_values(528, {0.0, 0.0, 0.0}) {}

void BestResponse::load_strategy(const std::vector<Strategy>& strat){
    fixed_strategy = strat;
}

float BestResponse::compute(int br_player) {
    // Initialize BR strategy to uniform over legal actions
    br_strategy.assign(528, {0.0f, 0.0f, 0.0f});
    for (int h = 0; h < 528; h++) {
        NodeInfo info(h);

        if (info.stm() != br_player){
            continue;
        }

        ActionList moves = info.moves();
        float uniform = 1.0f / moves.len;
        for (int i = 0; i < moves.len; i++)
            br_strategy[h][moves.al[i]] = uniform;
    }

    // Iterate until BR strategy converges
    float ev = 0.0f;
    for (int iter = 0; iter < 20; iter++) {
        action_values.assign(528, {0.0, 0.0, 0.0});
        double total_ev = 0.0;
        double total_weight = 0.0;

        for (int p0 = 0; p0 < 3; p0++) {
            for (int p1 = 0; p1 < 3; p1++) {
                for (int c = 0; c < 3; c++) {
                    if (p0 == p1 && p1 == c) continue;

                    double w = (p0 == p1 || p0 == c || p1 == c) ? 4.0 : 8.0;
                    std::array<Rank, 3> cards = {static_cast<Rank>(p0), static_cast<Rank>(p1), static_cast<Rank>(c)};

                    float deal_ev = br_traverse(cards, p0 * 8, br_player, w);
                    total_ev += w * deal_ev;
                    total_weight += w;
                }
            }
        }

        // Update BR strategy from accumulated action values
        bool changed = false;
        for (int h = 0; h < 528; h++) {
            NodeInfo info(h);
            if (info.stm() != br_player){
                continue;
            }

            ActionList moves = info.moves();

            // Find the maximum action value
            double best_val = action_values[h][moves.al[0]];
            for (int i = 1; i < moves.len; i++) {
                if (action_values[h][moves.al[i]] > best_val) {
                    best_val = action_values[h][moves.al[i]];
                }
            }

            // Count actions with maximum value and distribute probability evenly
            Strategy new_strat = {0.0f, 0.0f, 0.0f};
            int num_best = 0;
            for (int i = 0; i < moves.len; i++) {
                if (action_values[h][moves.al[i]] == best_val) {
                    num_best++;
                }
            }
            float prob = 1.0f / num_best;
            for (int i = 0; i < moves.len; i++) {
                if (action_values[h][moves.al[i]] == best_val) {
                    new_strat[moves.al[i]] = prob;
                }
            }


            if (new_strat != br_strategy[h]) {
                changed = true;
                br_strategy[h] = new_strat;
            }
        }

        ev = static_cast<float>(total_ev / total_weight);
        if (!changed){
            break;
        }
    }

    cached_ev = ev;
    return cached_ev;
}

float BestResponse::get_ev() const {
    return cached_ev;
}

Strategy BestResponse::get_br_strategy_at(int hash) const {
    return br_strategy[hash];
}

std::vector<Strategy> BestResponse::get_full_br_strategy() const {
    return br_strategy;
}

float BestResponse::br_traverse(const std::array<Rank, 3>& cards, Hash hash, int br_player, double weight) {
    NodeInfo info(hash);
    int stm = info.stm();
    ActionList moves = info.moves();

    // sign flips payouts to br_player's perspective
    float sign = (stm == br_player) ? 1.0f : -1.0f;

    if (stm == br_player) {
        // BR player's node: compute all action values, use current BR strategy for return value
        float action_vals[3] = {};

        for (int i = 0; i < moves.len; i++) {
            Action a = moves.al[i];

            if (info.ends_hand(a)) {
                action_vals[i] = sign * info.payout(a, cards[1 - stm]);
            } else {
                int ns = info.next_stm(a);
                action_vals[i] = br_traverse(cards, info.next_hash(a, cards[2], cards[ns]), br_player, weight);
            }
        }

        // Accumulate weighted action values for strategy derivation
        for (int i = 0; i < moves.len; i++)
            action_values[hash][moves.al[i]] += weight * action_vals[i];

        // Return value using current BR strategy (not clairvoyant max)
        float val = 0.0f;
        for (int i = 0; i < moves.len; i++)
            val += br_strategy[hash][moves.al[i]] * action_vals[i];
        return val;

    } else {
        // Opponent's node: weight by fixed strategy, propagate opponent reach
        float val = 0.0f;

        for (int i = 0; i < moves.len; i++) {
            Action a = moves.al[i];
            float strat_prob = fixed_strategy[hash][a];

            float child_val;
            if (info.ends_hand(a)) {
                child_val = sign * info.payout(a, cards[1 - stm]);
            } else {
                int ns = info.next_stm(a);
                child_val = br_traverse(cards, info.next_hash(a, cards[2], cards[ns]), br_player, weight * strat_prob);
            }

            val += strat_prob * child_val;
        }

        return val;
    }
}
