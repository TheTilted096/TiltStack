#!/usr/bin/env python3
import rlcard
from rlcard.agents import CFRAgent
from rlcard.utils import set_seed
import os
import pickle

set_seed(42)
os.makedirs('models', exist_ok=True)

# Use Leduc instead - MUCH faster
print("Creating Leduc Hold'em environment...")
env = rlcard.make('leduc-holdem', config={'allow_step_back': True})

print("Initializing CFR agent...")
cfr_agent = CFRAgent(env, model_path='./cfr_model')

print("Training CFR for 10,000 iterations...")
for episode in range(10000):
    cfr_agent.train()
    if (episode + 1) % 1000 == 0:
        print(f"  Iteration {episode + 1}/10,000")

print("\nSaving model...")
# Use pickle to save the agent
with open('./models/leduc_cfr_10k.pkl', 'wb') as f:
    pickle.dump(cfr_agent, f)

print("✓ Done! Model saved to ./models/leduc_cfr_10k.pkl")
