"""
Compute the Best Response against a trained Leduc Poker CFR strategy.

Usage:
    python BestResponse.py <strategy_file.csv>
    python BestResponse.py                      # defaults to leduc_strategy.csv
"""

import csv
import sys
import os
from leducsolver import BestResponse as BRSolver, NodeInfo
from Leduc import Leduc

ACTION_NAMES = ['check', 'bet', 'raise']


class BestResponse:
    def __init__(self):
        self.solver = BRSolver()
        self.br_strategy = []
        self.ev = 0.0
        self.player = 0

    def load_strategy_csv(self, filename: str):
        """Load opponent strategy from CSV file."""
        strategy = [[0.0, 0.0, 0.0] for _ in range(528)]

        with open(filename, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                h = int(row['hash'])
                strategy[h] = [float(row['check']), float(row['bet']), float(row['raise'])]

        self.solver.load_strategy(strategy)

    def compute(self, player: int):
        """Compute best response for the specified player."""
        self.player = player
        self.ev = self.solver.compute(player)
        self.br_strategy = self.solver.get_full_br_strategy()

    @staticmethod
    def write_unified_results(br_p0: 'BestResponse', br_p1: 'BestResponse', filename: str):
        """Write both players' BR strategies to a single human-readable file."""
        action_chars = ['c', 'b', 'r']

        if os.path.exists(filename):
            print(f"\033[93mWarning: '{filename}' already exists and will be overwritten.\033[0m")

        with open(filename, 'w') as f:
            f.write("Leduc Poker Best Response Results\n")
            f.write("=" * 60 + "\n")
            f.write(f"Player 0 Expected Value: {br_p0.ev:+.6f}\n")
            f.write(f"Player 1 Expected Value: {br_p1.ev:+.6f}\n")
            f.write(f"Exploitability: {(br_p0.ev + br_p1.ev) / 2.0:.6f}\n")
            f.write("=" * 60 + "\n\n")

            f.write("Round 1 Nodes (0-23):\n")
            f.write("-" * 40 + "\n")
            for h in range(24):
                info = NodeInfo(h)
                player = info.stm()
                moves = info.moves()
                strat = br_p0.br_strategy[h] if player == 0 else br_p1.br_strategy[h]

                readable = Leduc.hash_to_string(h)
                strat_str = [f"{action_chars[moves[i].value]}:{strat[moves[i].value]:.2f}"
                            for i in range(len(moves))]
                f.write(f"({player}) {readable:12} -> {', '.join(strat_str)}\n")

            f.write("\nRound 2 Nodes (24-527):\n")
            f.write("-" * 40 + "\n")
            for h in range(24, 528):
                info = NodeInfo(h)
                player = info.stm()
                moves = info.moves()
                strat = br_p0.br_strategy[h] if player == 0 else br_p1.br_strategy[h]

                readable = Leduc.hash_to_string(h)
                strat_str = [f"{action_chars[moves[i].value]}:{strat[moves[i].value]:.2f}"
                            for i in range(len(moves))]
                f.write(f"({player}) {readable:20} -> {', '.join(strat_str)}\n")


def main():
    strategy_file = sys.argv[1] if len(sys.argv) > 1 else "leduc_strategy.csv"

    if not os.path.exists(strategy_file):
        print(f"Error: Strategy file '{strategy_file}' not found.")
        print("Run Leduc.py first to generate the strategy CSV.")
        sys.exit(1)

    print(f"Loading strategy from '{strategy_file}'...")

    # Compute best response for Player 0
    br_p0 = BestResponse()
    br_p0.load_strategy_csv(strategy_file)
    br_p0.compute(0)

    print(f"\nBest Response for Player 0:")
    print(f"  Expected Value: {br_p0.ev:+.6f}")

    # Compute best response for Player 1
    br_p1 = BestResponse()
    br_p1.load_strategy_csv(strategy_file)
    br_p1.compute(1)

    print(f"\nBest Response for Player 1:")
    print(f"  Expected Value: {br_p1.ev:+.6f}")

    exploitability = (br_p0.ev + br_p1.ev) / 2.0
    print(f"\nExploitability: {exploitability:.6f}")

    # Write unified results to file
    print(f"\nWriting results...")
    BestResponse.write_unified_results(br_p0, br_p1, "br_results.txt")
    print(f"Best Response results written to br_results.txt")


if __name__ == "__main__":
    main()
