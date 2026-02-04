from Node import *

class Kuhn:
    def __init__(self):
        self.nodes = {}
    
    def print_all(self) -> None:
        tails = ['', 'c', 'b', 'cc', 'cb', 'bb', 'cbb', 'cbc']
        keys = []
        for card in cards:
            for t in tails:
                keys.append(card.value + t)

        for k in keys:
            if k in self.nodes:
                strat = self.nodes[k].get_stored_strategy()
                rounded = [round(s, 3) for s in strat]
                print(f"{k}: {rounded}")

    def train(self) -> None:
        score = 0.0
        for i in range(3):
            for j in range(3):
                if (i == j):
                    continue

                score += self.cfr([cards[i], cards[j]], "", [1.0, 1.0])

    def cfr(self, cards: list[Card],
            history: str, prob: list[float]) -> float:
        game_len = len(history)
        stm = game_len % 2

        if game_len > 1:
            # there are three terminal states

            showdown_val = card_rank[cards[stm]] > card_rank[cards[1 - stm]]

            if history[-1] == Action.CHECK.value:
                if history[-2] == Action.CHECK.value:
                    return showdown_val * 2 - 1 # return the payout

                return 1

            if history[-2:] == Action.BET.value + Action.BET.value:
                return showdown_val * 4 - 2

        node_key = cards[stm].value + history
        if node_key not in self.nodes:
            self.nodes[node_key] = Node()
        node_strat = self.nodes[node_key].get_current_strategy(prob[stm])

        node_util = 0.0
        action_utils = [0.0, 0.0]
        for i in range(2):
            next_hist = history + actions[i].value
            next_prob = prob.copy()
            next_prob[stm] *= node_strat[i]


            action_utils[i] = -self.cfr(cards, next_hist, next_prob)

            node_util += node_strat[i] * action_utils[i]

        for j in range(2):
            regret = action_utils[j] - node_util
            self.nodes[node_key].regrets[j] += regret * prob[1 - stm]

        return node_util
            
if __name__ == "__main__":
    ku = Kuhn()

    for i in range(100001):
        if (i % 10000 == 0):
            ku.print_all()
            print()
            print(i, "iterations complete")
            print()
        
        ku.train()
        


