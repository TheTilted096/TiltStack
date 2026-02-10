#pragma once

#include <vector>

#include "Node.h"

class LeducSolver{
    public:
        std::vector<Node> nodes; 

        LeducSolver();
        float cfr(const std::array<Rank, 3>&, Hash, std::array<float, 2>);
};