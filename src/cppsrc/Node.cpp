#include "Node.h"

// Determines if hand A or B wins given shared card C
Outcome compare_hands(Rank a, Rank b, Rank c){
    int sa = a + c + 4 * (a == c);
    int sb = b + c + 4 * (b == c);
    return static_cast<Outcome>((sa >= sb) + (sa > sb));
}

// --- NodeInfo ---

NodeInfo::NodeInfo(const Hash& h) : hash(h) {
    if(h < 24){
        private_card = static_cast<Rank>(h / 8);
        shared_card = Jack; // unused in round 1
        bet_round = 0;
        raises = h % 4;
        return;
    }

    int adj = h - 24;
    private_card = static_cast<Rank>((adj % 21) / 7);
    shared_card = static_cast<Rank>(adj / 168);
    bet_round = 1;
    raises = (adj / 21) % 4;
}

int NodeInfo::stm(){
    int seq;
    if(bet_round == 0)
        seq = hash % 8;
    else
        seq = ((hash - 24) / 21) % 8;

    return ((seq % 2) + (seq / 4)) % 2;
}

int NodeInfo::payout(const Action& a, Rank opp_card){
    // --- Corrected Payout Logic ---
    // Determine Round 1 raises from history
    int r1_raises;
    if (bet_round == 0) r1_raises = raises;
    else r1_raises = (((hash - 24) % 21) % 7 + 1) % 4;

    // CASE 1: FOLD
    if (a == Check && raises > 0) { // Fold
        if (bet_round == 0) {
            // R1 Fold: Lose ante + (N-1) bets of 2.
            return -(1 + 2 * (raises - 1));
        } else {
            // R2 Fold: Lose R1 investment + (N-1) bets of 4
            int r1_investment = 1 + 2 * r1_raises;
            int r2_investment = 4 * (raises - 1);
            return -(r1_investment + r2_investment);
        }
    }

    // CASE 2: SHOWDOWN (Call or Check-back)
    int total_investment = 1 + 2 * r1_raises + (bet_round == 1 ? 4 * raises : 0);
    Outcome result = compare_hands(private_card, opp_card, shared_card);
    return (result - 1) * total_investment;

    /*
    // --- Original Payout Logic ---
    if(bet_round == 0) { // must be fold
        return 1 - 2 * raises;
    }

    // Round 2: extract round 1 raises from hash
    int r1_seq = ((hash - 24) % 21) % 7 + 1;
    int r1_raises = r1_seq % 4;

    // Fold in round 2
    if(a == Check && raises > 0) {
        return 3 - 2 * r1_raises - 4 * raises;
    }

    // Showdown (call or check-check)
    int pot_per_player = 1 + 2 * r1_raises + 4 * raises;
    Outcome result = compare_hands(private_card, opp_card, shared_card);
    return (result - 1) * pot_per_player;
    */
    
}

ActionList NodeInfo::moves(){
    ActionList ml;

    if(raises == 0){
        ml.al = {Check, Raise};
        ml.len = 2;
        return ml;
    }

    if(raises == 3){
        ml.al = {Check, Bet};
        ml.len = 2;
        return ml;
    }

    ml.al = {Check, Bet, Raise};
    ml.len = 3;
    return ml;
}

bool NodeInfo::ends_hand(const Action& a){
    // Fold ends the hand in either round
    if(a == Check && raises > 0)
        return true;

    // In round 2, roundAction (call or check-check) at non-initial state leads to showdown
    if(bet_round == 1){
        int seq = ((hash - 24) / 21) % 8;
        if(a == roundAction() && seq != 0)
            return true;
    }

    return false;
}

int NodeInfo::next_stm(const Action& a){
    // Transition to round 2: P0 acts first (stm = 0)
    // Otherwise: control transfers to opponent (stm flips)
    int is_transition = (bet_round == 0) * (a == roundAction()) * (hash % 8 != 0);
    return (1 - is_transition) * (1 - stm());
}

Action NodeInfo::roundAction(){
    if(raises > 0)
        return Bet;
    return Check;
}

Hash NodeInfo::next_hash(const Action& a, const Rank& c, const Rank& next_card){
    if(bet_round == 0){
        if(a == roundAction() && (hash % 8 != 0)){
            // Transition to round 2: r2_seq starts at 0
            int r1_seq = hash % 8;
            return 24 + c * 168 + next_card * 7 + (r1_seq - 1);
        }

        // Stay in round 1: update seq and use next player's card
        int old_seq = hash % 8;
        int new_seq = old_seq + (a == Check ? 4 : 1);
        return next_card * 8 + new_seq;
    }

    // Round 2: extract components and rebuild with next player's card
    int adj = hash - 24;
    int sc = adj / 168;
    int r2_seq = (adj / 21) % 8;
    int r1_info = adj % 7; // r1_seq - 1

    int new_r2_seq = r2_seq + (a == Check ? 4 : 1);
    return 24 + sc * 168 + new_r2_seq * 21 + next_card * 7 + r1_info;
}

// --- Node ---

Node::Node() : strategy({0.0f, 0.0f, 0.0f}), regrets({0.0f, 0.0f, 0.0f}) {}

Strategy Node::get_current_strategy(const float& p, const ActionList& legal){
    Strategy strat = {0.0f, 0.0f, 0.0f};

    // Sum positive regrets for legal actions
    float normalizing_sum = 0.0f;
    for(int i = 0; i < legal.len; i++){
        float r = regrets[legal.al[i]];
        if(r > 0) normalizing_sum += r;
    }

    if(normalizing_sum == 0.0f){
        float uniform = 1.0f / legal.len;
        for(int i = 0; i < legal.len; i++)
            strat[legal.al[i]] = uniform;
    } else {
        for(int i = 0; i < legal.len; i++){
            Action a = legal.al[i];
            strat[a] = (regrets[a] > 0) * regrets[a] / normalizing_sum;
        }
    }

    // Accumulate weighted strategy
    for(int i = 0; i < 3; i++)
        strategy[i] += strat[i] * p;

    return strat;
}

Strategy Node::get_stored_strategy(const ActionList& legal){
    float normalizing_sum = 0.0f;
    for(int i = 0; i < legal.len; i++)
        normalizing_sum += strategy[legal.al[i]];

    Strategy stored = {0.0f, 0.0f, 0.0f};

    if(normalizing_sum == 0.0f){
        float uniform = 1.0f / legal.len;
        for(int i = 0; i < legal.len; i++)
            stored[legal.al[i]] = uniform;
    } else {
        for(int i = 0; i < legal.len; i++)
            stored[legal.al[i]] = strategy[legal.al[i]] / normalizing_sum;
    }

    return stored;
}
