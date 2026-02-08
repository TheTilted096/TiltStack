from enum import Enum

class Action(Enum):
    CHECK = 0 # also is fold
    BET = 1 # aka CALL
    RAISE = 2 # bet and raise

# Actions = [Action.CHECK, Action.BET, Action.RAISE]

class Rank(Enum):
    JACK = 0
    QUEEN = 1
    KING = 2

# Ranks = [Rank.JACK, Rank.QUEEN, Rank.KING]

class Suit(Enum):
    SPADES = 0
    HEARTS = 1

# Suits = [Suit.SPADES, Suit.HEARTS]

class Outcome(Enum):
    LOSS = 0
    PUSH = 1
    WINS = 2

# Outcomes = [Outcome.LOSS, Outcome.PUSH, Outcome.WINS]

def compare_hands(a: Rank, b: Rank, c: Rank) -> Outcome: # determines if hand A or B wins.
    sa = a.value + c.value + 4 * (a.value == c.value)
    sb = b.value + c.value + 4 * (b.value == c.value)

    return Outcome((sa >= sb) + (sa > sb))

class NodeInfo: # a struct that unpacks the node encoding
    def __init__(self, hash: int):    
        self.hash = hash

        if hash < 24: # first round hand
            self.private_card = Rank(hash // 8)
            self.shared_card = None
            self.bet_round = 0
            self.raises = hash % 4 # i define a 'raise' as raising the stakes, which includes the initial bet

            return

        # second round hand

        hash = hash - 24

        self.private_card = Rank((hash % 21) // 7)
        self.shared_card = hash // 168
        self.bet_round = 1
        self.raises = (hash // 21) % 4

    def stm(self) -> int: # gets the side to move
        seq = 0

        if self.bet_round == 0:
            seq = self.hash % 8
        else:
            seq = ((self.hash - 24) // 21) % 8

        return ((seq % 2) + (seq // 4)) % 2

    def payout(self, a: Action, opp_card: Rank = None) -> int: # payout after the hand ends from action a
        if self.bet_round == 0: # must be fold
            return 1 - 2 * self.raises # you should verify for yourself this works.

        # Round 2: extract round 1 raises from hash
        r1_seq = ((self.hash - 24) % 21) % 7 + 1
        r1_raises = r1_seq % 4

        # Fold in round 2
        if a == Action.CHECK and self.raises > 0:
            return 3 - 2 * r1_raises - 4 * self.raises # you should verify for yourself this works.

        # Showdown (call or check-check)
        pot_per_player = 1 + 2 * r1_raises + 4 * self.raises

        result = compare_hands(self.private_card, opp_card, Rank(self.shared_card))

        return (result.value - 1) * pot_per_player

    '''
    def payout(self, a: Action, opp_card: Rank = None) -> int:
        # Helper to get Round 1 Raises
        if self.bet_round == 0:
            r1_raises = self.raises
        else:
            r1_seq = ((self.hash - 24) % 21) % 7 + 1
            r1_raises = r1_seq % 4

        # --- CASE 1: FOLD (Round 1) ---
        if self.bet_round == 0 and a == Action.CHECK and self.raises > 0:
            # You fold. You lose what you put in BEFORE this current raise.
            # If raises=1 (Bet 2), you put in Ante (1). Loss: 1.
            # If raises=2 (Raise 4), you put in Ante (1) + Bet (2). Loss: 3.
            return -(1 + 2 * (self.raises - 1))

        # --- CASE 2: FOLD (Round 2) ---
        if self.bet_round == 1 and a == Action.CHECK and self.raises > 0:
            # Calculate what we put in during Round 1
            pot_r1 = 1 + 2 * r1_raises
            
            # You fold. You lose R1 investment + R2 investment BEFORE this raise.
            # In R2, bets are 4 chips.
            return -(pot_r1 + 4 * (self.raises - 1))

        # --- CASE 3: SHOWDOWN (Call or Check-Check) ---
        # If we get here, the player Matched the bet.
        # Calculate full committed amount including the current raise.
        
        # Round 1 contribution
        committed = 1 + 2 * r1_raises
        
        # Round 2 contribution (if applicable)
        if self.bet_round == 1:
            committed += 4 * self.raises

        result = compare_hands(self.private_card, opp_card, Rank(self.shared_card))
        
        if result == Outcome.WINS:
            return committed
        elif result == Outcome.LOSS:
            return -committed
            
        return 0 # PUSH
    '''

    def moves(self) -> list[Action]: # finds the list of possible moves in a node
        if self.raises == 0: # only check
            return [Action.CHECK, Action.RAISE]
        
        if self.raises == 3:
            return [Action.CHECK, Action.BET]
        
        return [Action.CHECK, Action.BET, Action.RAISE]

    def ends_hand(self, a: Action) -> bool: # does this action end the hand?
        # Fold ends the hand in either round
        if a == Action.CHECK and self.raises > 0:
            return True

        # In round 2, roundAction (call or check-check) at non-initial state leads to showdown
        if self.bet_round == 1:
            seq = ((self.hash - 24) // 21) % 8
            if a == self.roundAction() and seq != 0:
                return True

        return False

    def next_stm(self, a: Action) -> int: # side to move after taking action a
        # Transition to round 2: P0 acts first (stm = 0)
        # Otherwise: control transfers to opponent (stm flips)
        is_transition = (self.bet_round == 0) * (a == self.roundAction()) * (self.hash % 8 != 0)
        return (1 - is_transition) * (1 - self.stm())

    def roundAction(self) -> Action: # computes the (only) action that continues to the next round
        if self.raises > 0:
            return Action.BET
        
        return Action.CHECK

    def next_hash(self, a: Action, c: Rank, next_card: Rank) -> int: # hash of next decision node from action
        if self.bet_round == 0:
            if a == self.roundAction() and (self.hash % 8 != 0):
                # Transition to round 2: r2_seq starts at 0
                r1_seq = self.hash % 8
                return 24 + c.value * 168 + next_card.value * 7 + (r1_seq - 1)

            # Stay in round 1: update seq and use next player's card
            old_seq = self.hash % 8
            new_seq = old_seq + (4 if a == Action.CHECK else 1)
            return next_card.value * 8 + new_seq

        # Round 2: extract components and rebuild with next player's card
        h = self.hash - 24
        shared_card = h // 168
        r2_seq = (h // 21) % 8
        r1_info = h % 7  # This is r1_seq - 1

        new_r2_seq = r2_seq + (4 if a == Action.CHECK else 1)
        return 24 + shared_card * 168 + new_r2_seq * 21 + next_card.value * 7 + r1_info

class Node:
    def __init__(self):
        self.strategy = [0.0, 0.0, 0.0] # note that some actions are not always legal
        self.regrets = [0.0, 0.0, 0.0]

    def get_current_strategy(self, p: float, legal_actions: list[Action]) -> list[float]:
        # Calculate current strategy from regrets (only for legal actions)
        strategy = [0.0, 0.0, 0.0]

        # Sum positive regrets for legal actions only
        normalizing_sum = sum(max(0, self.regrets[a.value]) for a in legal_actions)

        if normalizing_sum == 0:
            # If no positive regrets, use uniform strategy over legal actions
            uniform_prob = 1.0 / len(legal_actions)
            for a in legal_actions:
                strategy[a.value] = uniform_prob
        else:
            # Normalize positive regrets for legal actions only
            for a in legal_actions:
                strategy[a.value] = (self.regrets[a.value] > 0) * self.regrets[a.value] / normalizing_sum

        # Increment self.strategy by strategy scaled by p
        for i in range(len(self.strategy)):
            self.strategy[i] += strategy[i] * p

        return strategy

    def get_stored_strategy(self, legal_actions: list[Action]) -> list[float]:
        # Sum strategy values for legal actions only
        normalizing_sum = sum(self.strategy[a.value] for a in legal_actions)

        stored = [0.0, 0.0, 0.0]

        if normalizing_sum == 0:
            # If no accumulated strategy, use uniform over legal actions
            uniform_prob = 1.0 / len(legal_actions)
            for a in legal_actions:
                stored[a.value] = uniform_prob
        else:
            # Normalize only legal actions
            for a in legal_actions:
                stored[a.value] = self.strategy[a.value] / normalizing_sum

        return stored

