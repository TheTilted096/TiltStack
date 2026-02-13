/**
 * Node.h - Leduc Hold'em CFR+ Node Implementation
 *
 * Defines game state information (NodeInfo) and CFR nodes (Node) for
 * Leduc Hold'em poker. Nodes store accumulated strategy (double precision)
 * and regrets (float) with batched flooring for CFR+.
 */
#pragma once

#include <array>

enum Action{ Check = 0, Bet = 1, Raise = 2 };

enum Rank{ Jack = 0, Queen = 1, King = 2 };

// enum Suit{ Spades = 0, Hearts = 1 };

enum Outcome{ Loss = 0, Push = 1, Wins = 2 };

using Hash = int;

struct ActionList{
    std::array<Action, 3> al;
    int len;
};

Outcome compare_hands(Rank a, Rank b, Rank c);

class NodeInfo{
    public:
        Hash hash;
        Rank private_card;
        Rank shared_card;
        int bet_round;
        int raises;

        NodeInfo(const Hash&);

        int stm();
        int payout(const Action& a, Rank opp_card = Jack);
        ActionList moves();

        bool ends_hand(const Action&);
        int next_stm(const Action&);

        Action roundAction();
        Hash next_hash(const Action& a, const Rank& c, const Rank& next_card);
};

using Strategy = std::array<float, 3>;


class Node{
    public:
        std::array<double, 3> strategy;  // Use double for accumulated strategy (precision!)
        std::array<float, 3> regrets;
        std::array<float, 3> regret_deltas;  // Accumulated regret changes for batched flooring

        Node();

        Strategy get_current_strategy(const double&, const ActionList&, int, bool);
        Strategy get_stored_strategy(const ActionList&);
        void flush_regrets();  // Apply deltas and floor
};
