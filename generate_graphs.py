#!/usr/bin/env python3
"""
Generate graphs for presentation
"""
import matplotlib.pyplot as plt
import rlcard
from src.gto_agent import GTOAgent
from src.exploitative_agent import ExploitativeAgent
from src.opponents import TightBot, LoosePassiveBot

def run_comparison():
    gto = GTOAgent()
    
    # Test both agents over many hands
    num_hands = 100
    
    # vs TightBot
    tight_opp = TightBot(4)
    env = rlcard.make('leduc-holdem')
    
    gto_profits = []
    exp_profits = []
    
    print("Running simulations...")
    
    # GTO vs Tight
    env.set_agents([gto, tight_opp])
    cumulative = 0
    for i in range(num_hands):
        _, payoffs = env.run(is_training=False)
        cumulative += payoffs[0]
        gto_profits.append(cumulative)
    
    # Exploitative vs Tight
    exp = ExploitativeAgent(gto, "TightBot")
    env.set_agents([exp, TightBot(4)])
    cumulative = 0
    for i in range(num_hands):
        _, payoffs = env.run(is_training=False)
        cumulative += payoffs[0]
        exp_profits.append(cumulative)
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, num_hands+1), gto_profits, label='GTO', linewidth=2)
    plt.plot(range(1, num_hands+1), exp_profits, label='Exploitative', linewidth=2)
    plt.xlabel('Hands Played', fontsize=12)
    plt.ylabel('Cumulative Profit (chips)', fontsize=12)
    plt.title('GTO vs Exploitative Performance Against Tight Opponent', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('results_comparison.png', dpi=300)
    print("✓ Saved: results_comparison.png")
    
    # Bar chart
    fig, ax = plt.subplots(figsize=(8, 6))
    agents = ['GTO', 'Exploitative']
    payoffs = [gto_profits[-1]/num_hands, exp_profits[-1]/num_hands]
    
    bars = ax.bar(agents, payoffs, color=['#3498db', '#e74c3c'])
    ax.set_ylabel('Average Profit (chips/hand)', fontsize=12)
    ax.set_title('Performance Comparison vs Tight Opponent', fontsize=14)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}',
                ha='center', va='bottom', fontsize=12)
    
    plt.tight_layout()
    plt.savefig('results_bar.png', dpi=300)
    print("✓ Saved: results_bar.png")

if __name__ == '__main__':
    run_comparison()
