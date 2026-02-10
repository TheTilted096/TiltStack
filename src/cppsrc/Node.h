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
        Strategy strategy;
        std::array<float, 3> regrets;

        Node();

        Strategy get_current_strategy(const float&, const ActionList&);
        Strategy get_stored_strategy(const ActionList&);
};
