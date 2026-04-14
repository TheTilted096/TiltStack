#!/usr/bin/env python3
import rlcard
from rlcard.agents import CFRAgent, RandomAgent
import pickle

# Load your trained agent
with open('./models/leduc_cfr_100k.pkl', 'rb') as f:
    trained_cfr = pickle.load(f)

# Create a fresh untrained agent for comparison
env = rlcard.make('leduc-holdem', config={'allow_step_back': True})
untrained_cfr = CFRAgent(env)

# Test both
print("Testing UNTRAINED CFR vs Random (1000 games)...")
env_test = rlcard.make('leduc-holdem')
env_test.set_agents([untrained_cfr, RandomAgent(env_test.num_actions)])

untrained_wins = 0
for i in range(1000):
    _, payoffs = env_test.run(is_training=False)
    if payoffs[0] > 0:
        untrained_wins += 1

print(f"Untrained CFR: {untrained_wins} wins ({untrained_wins/1000:.1%})")

print("\nTesting TRAINED CFR vs Random (1000 games)...")
env_test.set_agents([trained_cfr, RandomAgent(env_test.num_actions)])

trained_wins = 0
for i in range(1000):
    _, payoffs = env_test.run(is_training=False)
    if payoffs[0] > 0:
        trained_wins += 1

print(f"Trained CFR: {trained_wins} wins ({trained_wins/1000:.1%})")

improvement = trained_wins - untrained_wins
print(f"\nImprovement: +{improvement} wins")

if improvement > 50:
    print("✓ CFR is learning!")
else:
    print("⚠ CFR not learning much")
