"""
Leduc Hold'em Poker CFR+ Solver

Implements Counterfactual Regret Minimization Plus (CFR+) for solving
Leduc Hold'em poker to Nash equilibrium.

Features:
- Alternating player updates (avoid simultaneous update bugs)
- Batched regret flooring (CFR+)
- Linear strategy weighting with 500-iteration delay
- Double precision accumulation (prevents drift at high iterations)

Converges to 0.00 mBB exploitability in ~14k iterations.
"""
import os
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from leducsolver import LeducSolver, BestResponse as BRSolver, NodeInfo, Rank, Action

class Leduc:
    def __init__(self):
        self.solver = LeducSolver()
        self.iteration = 0

    def train(self, n: int):
        for _ in range(n):
            self.iteration += 1

            # CFR+: Alternating updates - Player 0 first (accumulate strategy)
            for p0 in range(3):
                for p1 in range(3):
                    for c in range(3):
                        if p0 == p1 == c:
                            continue
                        weight = 4.0 if (p0 == p1 or p0 == c or p1 == c) else 8.0
                        cards = [Rank(p0), Rank(p1), Rank(c)]
                        self.solver.cfr(cards, p0 * 8, [weight, weight], self.iteration, 0, True)

            # Flush Player 0 regrets (apply deltas and floor)
            self.solver.flush_regrets()

            # CFR+: Alternating updates - Player 1 against P0's new strategy (don't accumulate strategy)
            for p0 in range(3):
                for p1 in range(3):
                    for c in range(3):
                        if p0 == p1 == c:
                            continue
                        weight = 4.0 if (p0 == p1 or p0 == c or p1 == c) else 8.0
                        cards = [Rank(p0), Rank(p1), Rank(c)]
                        self.solver.cfr(cards, p0 * 8, [weight, weight], self.iteration, 1, False)

            # Flush Player 1 regrets (apply deltas and floor)
            self.solver.flush_regrets()

    def compute_best_response(self):
        """Compute best response for both players. Returns (ev0, ev1, br_strategies_p0, br_strategies_p1)."""
        strategies = self.solver.get_all_strategies()
        br = BRSolver()
        br.load_strategy(strategies)
        ev0 = br.compute(0)
        br_p0 = br.get_full_br_strategy()
        ev1 = br.compute(1)
        br_p1 = br.get_full_br_strategy()
        return ev0, ev1, br_p0, br_p1

    def compute_exploitability(self) -> float:
        ev0, ev1, _, _ = self.compute_best_response()
        return (ev0 + ev1) / 2.0

    def write_br_results(self, filename: str):
        """Write best response results to a human-readable file."""
        action_chars = ['c', 'b', 'r']
        ev0, ev1, br_p0, br_p1 = self.compute_best_response()

        if os.path.exists(filename):
            print(f"\033[93mWarning: '{filename}' already exists and will be overwritten.\033[0m")

        with open(filename, 'w') as f:
            f.write("Leduc Poker Best Response Results\n")
            f.write("=" * 60 + "\n")
            f.write(f"Player 0 Expected Value: {ev0:+.6f}\n")
            f.write(f"Player 1 Expected Value: {ev1:+.6f}\n")
            f.write(f"Exploitability: {(ev0 + ev1) / 2.0:.6f}\n")
            f.write("=" * 60 + "\n\n")

            f.write("Round 1 Nodes (0-23):\n")
            f.write("-" * 40 + "\n")
            for h in range(24):
                info = NodeInfo(h)
                player = info.stm()
                moves = info.moves()
                strat = br_p0[h] if player == 0 else br_p1[h]

                readable = self.hash_to_string(h)
                strat_str = [f"{action_chars[moves[i].value]}:{strat[moves[i].value]:.2f}"
                            for i in range(len(moves))]
                f.write(f"({player}) {readable:12} -> {', '.join(strat_str)}\n")

            f.write("\nRound 2 Nodes (24-527):\n")
            f.write("-" * 40 + "\n")
            for h in range(24, 528):
                info = NodeInfo(h)
                player = info.stm()
                moves = info.moves()
                strat = br_p0[h] if player == 0 else br_p1[h]

                readable = self.hash_to_string(h)
                strat_str = [f"{action_chars[moves[i].value]}:{strat[moves[i].value]:.2f}"
                            for i in range(len(moves))]
                f.write(f"({player}) {readable:20} -> {', '.join(strat_str)}\n")

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
            else:
                r1_history += 'c'  # second check that ended round 1

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
        """Write all 528 strategies to a human-readable file"""
        action_chars = ['c', 'b', 'r']

        # Warn if overwriting
        if os.path.exists(filename):
            print(f"\033[93mWarning: '{filename}' already exists and will be overwritten.\033[0m")

        with open(filename, 'w') as f:
            f.write("Leduc Poker CFR Results\n")
            f.write("=" * 60 + "\n\n")

            f.write("Round 1 Nodes (0-23):\n")
            f.write("-" * 40 + "\n")
            for h in range(24):
                info = NodeInfo(h)
                player = info.stm()
                moves = info.moves()
                strat = self.solver[h].get_stored_strategy(moves)

                readable = self.hash_to_string(h)
                strat_str = [f"{action_chars[a.value]}:{strat[a.value]:.2f}"
                            for a in moves]
                f.write(f"({player}) {readable:12} -> {', '.join(strat_str)}\n")

            f.write("\nRound 2 Nodes (24-527):\n")
            f.write("-" * 40 + "\n")
            for h in range(24, 528):
                info = NodeInfo(h)
                player = info.stm()
                moves = info.moves()
                strat = self.solver[h].get_stored_strategy(moves)

                readable = self.hash_to_string(h)
                strat_str = [f"{action_chars[a.value]}:{strat[a.value]:.2f}"
                            for a in moves]
                f.write(f"({player}) {readable:20} -> {', '.join(strat_str)}\n")

    def write_strategy_csv(self, filename: str):
        """Write all 528 strategies to a machine-readable CSV file"""
        if os.path.exists(filename):
            print(f"\033[93mWarning: '{filename}' already exists and will be overwritten.\033[0m")

        with open(filename, 'w') as f:
            f.write("hash,check,bet,raise\n")
            for h in range(528):
                info = NodeInfo(h)
                moves = info.moves()
                strat = self.solver[h].get_stored_strategy(moves)
                f.write(f"{h},{strat[0]:.6f},{strat[1]:.6f},{strat[2]:.6f}\n")


def fmt_iters(n: int) -> str:
    if n >= 1000 and n % 1000 == 0:
        return f"{n // 1000}k"
    return str(n)


if __name__ == "__main__":
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    leduc = Leduc()

    iterations = 100000
    sample_interval = 1000
    print(f"Training Leduc Poker CFR for {fmt_iters(iterations)} iterations...")

    # Set up live plot
    plt.ion()
    fig, ax = plt.subplots()
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Exploitability (mBB/hand)")
    ax.set_title("Leduc CFR Convergence")
    ax.set_yscale("log")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: fmt_iters(int(x))))
    line, = ax.plot([], [], 'b-')
    fig.show()

    iter_history = []
    expl_history = []

    for i in range(iterations):
        leduc.train(1)

        if (i + 1) % sample_interval == 0:
            expl = leduc.compute_exploitability()
            expl_mbb = expl * 1000
            iter_history.append(i + 1)
            expl_history.append(expl_mbb)

            line.set_xdata(iter_history)
            line.set_ydata(expl_history)
            ax.relim()
            ax.autoscale_view()
            fig.canvas.draw_idle()
            fig.canvas.flush_events()

            print(f"{fmt_iters(i + 1)}/{fmt_iters(iterations)} | E: {expl_mbb:.2f} mBB/hand")

    plt.ioff()
    with open(os.path.join(output_dir, "exploitability.csv"), 'w') as f:
        f.write("iteration,exploitability_mbb\n")
        for it, ex in zip(iter_history, expl_history):
            f.write(f"{it},{ex:.4f}\n")
    fig.savefig(os.path.join(output_dir, "exploitability.png"), dpi=150)

    print(f"\nWriting results to {output_dir}/...")
    leduc.write_results(os.path.join(output_dir, "leduc_results.txt"))
    leduc.write_strategy_csv(os.path.join(output_dir, "leduc_strategy.csv"))
    leduc.write_br_results(os.path.join(output_dir, "br_results.txt"))
    print("Done.")

    plt.show()
