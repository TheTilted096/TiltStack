#!/usr/bin/env python3
"""
Evaluate GTO vs Exploitative against different opponents
"""
import rlcard
from src.gto_agent import GTOAgent
from src.opponents import TightBot, LoosePassiveBot, AggressiveBot
from rlcard.agents import RandomAgent

def evaluate(agent1, agent2, num_games=1000):
    """Run tournament and return win rate for agent1"""
    env = rlcard.make('leduc-holdem')
    env.set_agents([agent1, agent2])
    
    wins = 0
    total_payoff = 0
    
    for _ in range(num_games):
        _, payoffs = env.run(is_training=False)
        if payoffs[0] > 0:
            wins += 1
        total_payoff += payoffs[0]
    
    return {
        'win_rate': wins / num_games,
        'avg_payoff': total_payoff / num_games
    }

if __name__ == '__main__':
    print("Loading GTO agent...")
    gto = GTOAgent()
    
    print("\n=== GTO vs Different Opponents ===\n")
    
    # Test vs Random
    print("GTO vs Random...")
    result = evaluate(gto, RandomAgent(num_actions=4))
    print(f"  Win rate: {result['win_rate']:.1%}")
    print(f"  Avg payoff: {result['avg_payoff']:.2f} chips/hand\n")
    
    # Test vs Tight
    print("GTO vs TightBot...")
    result = evaluate(gto, TightBot(num_actions=4))
    print(f"  Win rate: {result['win_rate']:.1%}")
    print(f"  Avg payoff: {result['avg_payoff']:.2f} chips/hand\n")
    
    # Test vs Loose-Passive
    print("GTO vs LoosePassiveBot...")
    result = evaluate(gto, LoosePassiveBot(num_actions=4))
    print(f"  Win rate: {result['win_rate']:.1%}")
    print(f"  Avg payoff: {result['avg_payoff']:.2f} chips/hand\n")
    
    # Test vs Aggressive
    print("GTO vs AggressiveBot...")
    result = evaluate(gto, AggressiveBot(num_actions=4))
    print(f"  Win rate: {result['win_rate']:.1%}")
    print(f"  Avg payoff: {result['avg_payoff']:.2f} chips/hand\n")
