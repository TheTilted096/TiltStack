from Node import *
import matplotlib.pyplot as plt
import numpy as np

class Kuhn:
    def __init__(self):
        self.nodes = {}
        self.strategy_history = {}  # Stores strategy evolution for each node
    
    def save_strategy_snapshot(self) -> None:
        """Save current average strategy for all nodes"""
        for node_key, node in self.nodes.items():
            if node_key not in self.strategy_history:
                self.strategy_history[node_key] = []

            strat = node.get_stored_strategy()
            self.strategy_history[node_key].append(strat.copy())

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

    def visualize_convergence(self, node_key: str) -> None:
        """Visualize how a node's strategy converged over iterations"""
        if node_key not in self.strategy_history:
            print(f"No history found for node '{node_key}'")
            print(f"Available nodes: {list(self.strategy_history.keys())}")
            return

        history = self.strategy_history[node_key]
        if len(history) == 0:
            print(f"No strategy history recorded for node '{node_key}'")
            return

        # Extract CHECK and BET probabilities
        check_probs = [strat[0] for strat in history]
        bet_probs = [strat[1] for strat in history]

        # Create figure
        plt.figure(figsize=(5, 4))

        # Draw the probability simplex constraint line (CHECK + BET = 1)
        plt.plot([0, 1], [1, 0], 'k--', alpha=0.2, linewidth=2, label='Probability simplex (p₁ + p₂ = 1)')

        # Mark the uniform strategy starting point (0.5, 0.5)
        plt.plot(0.5, 0.5, 'D', color='gold', markersize=18,
                label='Uniform Strategy (0.5, 0.5)', markeredgecolor='black',
                markeredgewidth=2.5, zorder=2, alpha=0.9)

        # Plot trajectory line
        plt.plot(check_probs, bet_probs, 'b-', alpha=0.3, linewidth=1, label='Strategy trajectory')

        # Create color gradient for points (early iterations = yellow, late = dark blue)
        colors = plt.cm.viridis(np.linspace(0, 1, len(history)))

        # Plot each snapshot point with color gradient
        for i in range(len(history)):
            plt.scatter(check_probs[i], bet_probs[i], c=[colors[i]], s=30,
                       edgecolors='black', linewidths=0.5, zorder=3)

        # Mark start and end points more prominently
        plt.plot(check_probs[0], bet_probs[0], 'o', color='lime', markersize=15,
                label=f'Start (iter 0)', markeredgecolor='black', markeredgewidth=2, zorder=4)
        plt.plot(check_probs[-1], bet_probs[-1], 'o', color='red', markersize=15,
                label=f'End (iter {len(history)-1})', markeredgecolor='black', markeredgewidth=2, zorder=4)

        # Add iteration labels at key points
        label_interval = max(1, len(history) // 5)
        for i in range(0, len(history), label_interval):
            if i > 0 and i < len(history) - 1:  # Skip start and end
                plt.annotate(f'iter {i}', (check_probs[i], bet_probs[i]),
                           xytext=(5, 5), textcoords='offset points',
                           fontsize=8, alpha=0.7)

        # Calculate dynamic bounds with padding
        all_x = check_probs + [0.5]  # Include uniform strategy point
        all_y = bet_probs + [0.5]

        x_min, x_max = min(all_x), max(all_x)
        y_min, y_max = min(all_y), max(all_y)

        # Add 15% padding on each side
        x_range = max(x_max - x_min, 0.1)  # Minimum range of 0.1
        y_range = max(y_max - y_min, 0.1)

        x_padding = x_range * 0.15
        y_padding = y_range * 0.15

        x_lim_min = max(0, x_min - x_padding)
        x_lim_max = min(1, x_max + x_padding)
        y_lim_min = max(0, y_min - y_padding)
        y_lim_max = min(1, y_max + y_padding)

        plt.xlabel('P(CHECK)', fontsize=12, fontweight='bold')
        plt.ylabel('P(BET)', fontsize=12, fontweight='bold')
        plt.title(f'Strategy Convergence for Node: {node_key}\n({len(history)} snapshots)', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.xlim(x_lim_min, x_lim_max)
        plt.ylim(y_lim_min, y_lim_max)

        # Add colorbar to show time progression
        sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=plt.Normalize(vmin=0, vmax=len(history)-1))
        sm.set_array([])
        plt.colorbar(sm, ax=plt.gca(), label='Snapshot Index')

        plt.tight_layout()
        plt.show()

    def train(self) -> None: # iterate over every deal in kuhn poker
        score = 0.0
        for i in range(3):
            for j in range(3):
                if (i == j):
                    continue

                score += self.cfr([cards[i], cards[j]], "", [1.0, 1.0])

    def cfr(self, cards: list[Card],
            history: str, prob: list[float]) -> float:
        game_len = len(history)
        stm = game_len % 2 # side to move

        if game_len > 1:
            # there are three terminal states

            showdown_val = card_rank[cards[stm]] > card_rank[cards[1 - stm]] # who wins the showdown

            if history[-1] == Action.CHECK.value: # game ends in a check or fold
                if history[-2] == Action.CHECK.value: # there was a check before that, double check
                    return showdown_val * 2 - 1 # return the payout

                return 1 # otherwise, the opponent folded, so this side gets a point

            if history[-2:] == Action.BET.value + Action.BET.value: # game ends in two bets
                return showdown_val * 4 - 2 # return showdown winner but with higher stakes

        node_key = cards[stm].value + history # get the key for the next node
        if node_key not in self.nodes: # if it's not in the dictionary, we must initialize it
            self.nodes[node_key] = Node()
        node_strat = self.nodes[node_key].get_current_strategy(prob[stm]) # compute a temporary strategy internally

        node_util = 0.0 # EV of this node
        action_utils = [0.0, 0.0] # EV of each child node
        for i in range(2):
            next_hist = history + actions[i].value # next history
            next_prob = prob.copy() # copy the 'allowance' probabilities
            next_prob[stm] *= node_strat[i] # adjust our 'allowance' probability

            action_utils[i] = -self.cfr(cards, next_hist, next_prob) # compute utility of child node

            node_util += node_strat[i] * action_utils[i] # weight and add to EV of this node

        for j in range(2): # regret updating, the core mechanism
            regret = action_utils[j] - node_util
            self.nodes[node_key].regrets[j] += regret * prob[1 - stm]

        return node_util
            
if __name__ == "__main__":
    ku = Kuhn()

    snapshot_interval = 10  # Save strategy every 100 iterations
    total_iterations = 1000

    print("Training Kuhn Poker with CFR...")
    print(f"Saving strategy snapshots every {snapshot_interval} iterations\n")

    for i in range(total_iterations):
        if (i % snapshot_interval == 0):
            #ku.print_all()
            #print()
            print(i, "iterations complete")
            ku.save_strategy_snapshot()
            #print()

        ku.train()

    print()
    ku.print_all()       

    print("\nTraining complete!")
    print(f"Total snapshots saved: {len(ku.strategy_history[list(ku.strategy_history.keys())[0]]) if ku.strategy_history else 0}")
    print("\n" + "="*50)
    print("STRATEGY CONVERGENCE VISUALIZATION")
    print("="*50)

    # Interactive loop for node visualization
    while True:
        print("\nEnter a node key to visualize (e.g., 'K', 'Qcb', 'Jb')")
        print("Available nodes:", sorted(ku.strategy_history.keys()))
        print("Type 'quit' or 'exit' to stop")

        user_input = input("\nNode key: ").strip()

        if user_input.lower() in ['quit', 'exit']:
            print("Exiting visualization mode.")
            break

        if user_input in ku.strategy_history:
            ku.visualize_convergence(user_input)
        else:
            print(f"\nNode '{user_input}' not found.")
            print(f"Available nodes: {sorted(ku.strategy_history.keys())}")



