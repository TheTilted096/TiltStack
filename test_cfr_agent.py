#!/usr/bin/env python3
import rlcard
from rlcard.agents import RandomAgent
import pickle

print("Loading trained CFR agent...")
with open('./models/leduc_cfr_10k.pkl', 'rb') as f:
    cfr_agent = pickle.load(f)

print("Creating Leduc environment...")
env = rlcard.make('leduc-holdem')

print("\nTesting CFR vs Random over 1000 games...")
env.set_agents([cfr_agent, RandomAgent(env.num_actions)])

wins = 0
for i in range(1000):
    trajectories, payoffs = env.run(is_training=False)
    if payoffs[0] > 0:
        wins += 1

win_rate = wins / 1000
print(f"\nCFR Win Rate: {win_rate:.1%}")
print(f"Expected: >85% (CFR should dominate random)")

if win_rate > 0.85:
    print("✓ CFR agent working correctly!")
else:
    print("⚠ CFR agent might need more training")
