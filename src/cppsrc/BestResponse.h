#pragma once

#include <vector>
#include <array>

#include "Node.h"

class BestResponse {
    public:
        BestResponse();

        void load_strategy(const std::vector<Strategy>& strat);

        float compute(int br_player);
        float get_ev() const;

        Strategy get_br_strategy_at(int hash) const;
        std::vector<Strategy> get_full_br_strategy() const;

    private:
        std::vector<Strategy> fixed_strategy;
        std::vector<Strategy> br_strategy;
        float cached_ev;

        std::vector<std::array<double, 3>> action_values;

        float br_traverse(const std::array<Rank, 3>& cards, Hash hash, int br_player, double weight);
};
