#!/usr/bin/env python3
"""
Train a CFR agent on Leduc Hold'em and save it to disk.

Usage:
    python src/tests/train_cfr_leduc.py              # 100k iterations (default)
    python src/tests/train_cfr_leduc.py --iters 10000
"""
import argparse
import os
import pickle
import rlcard
from rlcard.agents import CFRAgent
from rlcard.utils import set_seed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--iters', type=int, default=100_000,
                        help='Number of CFR training iterations (default: 100000)')
    args = parser.parse_args()

    set_seed(42)
    os.makedirs('models', exist_ok=True)

    iters = args.iters
    k = iters // 1000
    model_path = f'./models/leduc_cfr_{k}k.pkl'

    print(f"Creating Leduc Hold'em environment...")
    env = rlcard.make('leduc-holdem', config={'allow_step_back': True})

    print(f"Initializing CFR agent...")
    cfr_agent = CFRAgent(env, model_path='./cfr_model')

    print(f"Training CFR for {iters:,} iterations...")
    log_every = max(1000, iters // 10)
    for episode in range(iters):
        cfr_agent.train()
        if (episode + 1) % log_every == 0:
            print(f"  Iteration {episode + 1:,}/{iters:,}")

    print(f"\nSaving model to {model_path}...")
    with open(model_path, 'wb') as f:
        pickle.dump(cfr_agent, f)
    print(f"✓ Done!")


if __name__ == '__main__':
    main()
