/**
 * Leduc.h - CFR+ Solver for Leduc Hold'em
 *
 * Implements alternating updates CFR+ with batched regret flooring.
 * Uses double precision for probability propagation to maintain
 * accuracy through deep recursion.
 */
#pragma once

#include <vector>

#include "Node.h"

class LeducSolver{
    public:
        std::vector<Node> nodes;

        LeducSolver();
        float cfr(const std::array<Rank, 3>&, Hash, std::array<double, 2>, int, int, bool);
        void flush_regrets();
        std::vector<Strategy> get_all_strategies();
};