from enum import Enum

class Action(Enum): 
    CHECK = 'c' 
    BET = 'b'

actions = [Action.CHECK, Action.BET]

class Card(Enum):
    JACK = 'J'
    QUEEN = 'Q'
    KING = 'K'

cards = [Card.JACK, Card.QUEEN, Card.KING]

card_rank = {Card.JACK: 0, Card.QUEEN: 1, Card.KING: 2}

class Node:
    def __init__(self):
        self.strategy = [0.0, 0.0]
        self.regrets = [0.0, 0.0]

    def get_current_strategy(self, p : float) -> list[float]:
        # Calculate current strategy from regrets
        strategy = [0.0, 0.0]

        if all(x == 0 for x in self.regrets):
            strategy = [0.5, 0.5]
        else:
            # Sum positive regrets
            normalizing_sum = sum(max(0, r) for r in self.regrets)

            # Normalize positive regrets
            for i in range(len(self.regrets)):
                if self.regrets[i] > 0:
                    strategy[i] = self.regrets[i] / normalizing_sum
                else:
                    strategy[i] = 0.0

        # Increment self.strategy by strategy scaled by p
        for i in range(len(self.strategy)):
            self.strategy[i] += strategy[i] * p

        return strategy
    
    def get_stored_strategy(self) -> list[float]:
        normalizing_sum = sum(self.strategy)

        if normalizing_sum == 0:
            return [0.5, 0.5]

        return [s / normalizing_sum for s in self.strategy]
        
