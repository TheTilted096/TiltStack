#!/usr/bin/env python3
"""
Benchmark GTO and Exploitative agents against different opponents.
"""
import rlcard
from src.showcase.gto_agent import GTOAgent
from src.showcase.exploitative_agent import ExploitativeAgent
from src.showcase.opponents import TightBot, LoosePassiveBot, AggressiveBot
from rlcard.agents import RandomAgent


def evaluate(agent1, agent2, num_games=1000):
    """Run a tournament and return win rate and average payoff for agent1."""
    env = rlcard.make('leduc-holdem')
    env.set_agents([agent1, agent2])

    wins = 0
    total_payoff = 0
    for _ in range(num_games):
        _, payoffs = env.run(is_training=False)
        if payoffs[0] > 0:
            wins += 1
        total_payoff += payoffs[0]

    return {'win_rate': wins / num_games, 'avg_payoff': total_payoff / num_games}


if __name__ == '__main__':
    print("Loading GTO agent...")
    gto = GTOAgent()

    print("\n=== GTO baseline ===\n")
    for name, opp in [
        ("Random",          RandomAgent(num_actions=4)),
        ("TightBot",        TightBot(4)),
        ("LoosePassiveBot", LoosePassiveBot(4)),
        ("AggressiveBot",   AggressiveBot(4)),
    ]:
        r = evaluate(gto, opp)
        print(f"  vs {name:<18}  win rate {r['win_rate']:.1%}  avg payoff {r['avg_payoff']:+.3f}")

    print("\n=== GTO vs Exploitative ===\n")
    for opponent_type, opp_cls in [
        ("TightBot",        TightBot),
        ("LoosePassiveBot", LoosePassiveBot),
        ("AggressiveBot",   AggressiveBot),
    ]:
        exp = ExploitativeAgent(gto, opponent_type=opponent_type)
        gto_r = evaluate(gto, opp_cls(4))
        exp_r = evaluate(exp, opp_cls(4))
        mult = (exp_r['avg_payoff'] / gto_r['avg_payoff']
                if abs(gto_r['avg_payoff']) > 1e-6 else float('inf'))
        print(f"  vs {opponent_type:<18}  GTO {gto_r['avg_payoff']:+.3f}"
              f"  Exploit {exp_r['avg_payoff']:+.3f}  ({mult:.2f}x)")
