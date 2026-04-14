#!/usr/bin/env python3
import rlcard
from rlcard.agents import RandomAgent
import pickle

print("Loading 100k trained CFR agent...")
with open('./models/leduc_cfr_100k.pkl', 'rb') as f:
    cfr_agent = pickle.load(f)

print("Creating Leduc environment...")
env = rlcard.make('leduc-holdem')

print("\nTesting CFR vs Random over 1000 games...")
env.set_agents([cfr_agent, RandomAgent(env.num_actions)])

wins = 0
total_payoff = 0
for i in range(1000):
    trajectories, payoffs = env.run(is_training=False)
    if payoffs[0] > 0:
        wins += 1
    total_payoff += payoffs[0]

win_rate = wins / 1000
avg_payoff = total_payoff / 1000

print(f"\nResults:")
print(f"Win Rate: {win_rate:.1%}")
print(f"Average Payoff: {avg_payoff:.2f} chips/hand")
print(f"Expected: >85% win rate")

if win_rate > 0.85:
    print("✓ CFR agent working correctly!")
else:
    print("⚠ CFR agent might need more training or has issues")
