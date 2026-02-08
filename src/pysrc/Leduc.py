from Node import *

class Leduc:
    def __init__(self):
        self.nodes = [Node() for _ in range(528)]

    def cfr(self, cards: list[Rank], hash: int, prob: list[float]) -> float:
        info = NodeInfo(hash)
        stm = info.stm()

        moves = info.moves()

        node_strat = self.nodes[hash].get_current_strategy(prob[stm], moves)
        node_util = 0.0
        action_utils = [0.0] * len(moves)

        for i in range(len(moves)):
            a = moves[i]
            next_prob = prob.copy()
            next_prob[stm] *= node_strat[a.value]

            if info.ends_hand(a):
                action_utils[i] = info.payout(a, cards[1 - stm])

            else:
                next_stm = info.next_stm(a)
                action_utils[i] = self.cfr(cards, info.next_hash(a, cards[2], cards[next_stm]), next_prob)

                if stm != next_stm:
                    action_utils[i] *= -1

            node_util += node_strat[a.value] * action_utils[i]

        for j in range(len(moves)):
            regret = action_utils[j] - node_util
            self.nodes[hash].regrets[moves[j].value] += regret * prob[1 - stm]

        return node_util

    def train(self, n: int):
        for _ in range(n):
            for p0 in range(3):
                for p1 in range(3):
                    for c in range(3):
                        # Skip if all three are the same rank (only 2 suits per rank)
                        if p0 == p1 == c:
                            continue

                        # Weight by combinatorial frequency: 2*2*2=8 if all different, 2*1*2=4 if any pair
                        weight = 4.0 if (p0 == p1 or p0 == c or p1 == c) else 8.0

                        cards = [Rank(p0), Rank(p1), Rank(c)]
                        self.cfr(cards, p0 * 8, [weight, weight])

    @staticmethod
    def hash_to_string(hash: int) -> str:
        """Convert hash to readable format like 'J:cr' or 'Q|K:rbc'"""
        rank_names = ['J', 'Q', 'K']

        if hash < 24:
            # Round 1: hash = private_card * 8 + seq
            private_card = hash // 8
            seq = hash % 8
        else:
            # Round 2: more complex encoding
            h = hash - 24
            private_card = (h % 21) // 7
            shared_card = h // 168
            seq = (h // 21) % 8
            r1_seq = (h % 21) % 7 + 1

            # Build round 1 history
            r1_history = ""
            r1_s = r1_seq
            if r1_s >= 4:
                r1_history += 'c'
                r1_s -= 4
            r1_history += 'r' * (r1_s % 4)
            if r1_s > 0:
                r1_history += 'b'  # call that ended round 1

            # Build round 2 history
            r2_history = ""
            if seq >= 4:
                r2_history += 'c'
                seq -= 4
            for _ in range(seq):
                r2_history += 'r'

            return f"{rank_names[private_card]}|{rank_names[shared_card]}:{r1_history}/{r2_history}"

        # Round 1 history
        history = ""
        if seq >= 4:
            history += 'c'
            seq -= 4
        for _ in range(seq):
            history += 'r'

        return f"{rank_names[private_card]}:{history}"

    def write_results(self, filename: str):
        """Write all 528 strategies to a file"""
        with open(filename, 'w') as f:
            f.write("Leduc Poker CFR Results\n")
            f.write("=" * 60 + "\n\n")

            f.write("Round 1 Nodes (0-23):\n")
            f.write("-" * 40 + "\n")
            for h in range(24):
                info = NodeInfo(h)
                moves = info.moves()
                strat = self.nodes[h].get_stored_strategy(moves)

                readable = self.hash_to_string(h)
                strat_str = [f"{action_chars[a.value]}:{strat[a.value]:.4f}"
                            for a in moves]
                f.write(f"{readable:12} -> {', '.join(strat_str)}\n")

            f.write("\nRound 2 Nodes (24-527):\n")
            f.write("-" * 40 + "\n")
            for h in range(24, 528):
                info = NodeInfo(h)
                moves = info.moves()
                strat = self.nodes[h].get_stored_strategy(moves)

                readable = self.hash_to_string(h)
                strat_str = [f"{action_chars[a.value]}:{strat[a.value]:.4f}"
                            for a in moves]
                f.write(f"{readable:20} -> {', '.join(strat_str)}\n")


action_chars = ['c', 'b', 'r']

if __name__ == "__main__":
    leduc = Leduc()

    iterations = 10000
    print(f"Training Leduc Poker CFR for {iterations} iterations...")

    for i in range(iterations):
        if (i + 1) % 100 == 0:
            print(f"Iteration {i + 1}/{iterations} complete")
        leduc.train(1)

    print("\nTraining complete! Writing results to leduc_results.txt...")
    leduc.write_results("leduc_results.txt")
    print("Results written to leduc_results.txt")

        

                

