#!/usr/bin/env python3
import rlcard
from rlcard.agents import CFRAgent
from rlcard.utils import set_seed
import pickle

set_seed(42)

print("Creating Leduc Hold'em environment...")
env = rlcard.make('leduc-holdem', config={'allow_step_back': True})

print("Initializing CFR agent...")
cfr_agent = CFRAgent(env, model_path='./cfr_model')

print("Training CFR for 100,000 iterations (will take ~15-20 min)...")
for episode in range(100000):
    cfr_agent.train()
    if (episode + 1) % 10000 == 0:
        print(f"  Iteration {episode + 1}/100,000")

print("\nSaving model...")
with open('./models/leduc_cfr_100k.pkl', 'wb') as f:
    pickle.dump(cfr_agent, f)

print("✓ Done! Model saved to ./models/leduc_cfr_100k.pkl")
