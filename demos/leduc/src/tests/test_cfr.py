#!/usr/bin/env python3
"""
Test a trained CFR agent against a random player and compare to an untrained baseline.

Usage:
    python src/tests/test_cfr.py                                   # default 100k model
    python src/tests/test_cfr.py --model models/leduc_cfr_10k.pkl
"""
import argparse
import pickle
import rlcard
from rlcard.agents import CFRAgent, RandomAgent


def win_rate_vs_random(agent, num_games=1000):
    env = rlcard.make('leduc-holdem')
    env.set_agents([agent, RandomAgent(env.num_actions)])
    wins = 0
    total_payoff = 0
    for _ in range(num_games):
        _, payoffs = env.run(is_training=False)
        if payoffs[0] > 0:
            wins += 1
        total_payoff += payoffs[0]
    return wins / num_games, total_payoff / num_games


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='./models/leduc_cfr_100k.pkl',
                        help='Path to trained model pickle (default: leduc_cfr_100k.pkl)')
    parser.add_argument('--games', type=int, default=1000,
                        help='Number of test games (default: 1000)')
    args = parser.parse_args()

    print(f"Loading trained agent from {args.model}...")
    with open(args.model, 'rb') as f:
        trained = pickle.load(f)

    print("Creating untrained CFR baseline...")
    env_train = rlcard.make('leduc-holdem', config={'allow_step_back': True})
    untrained = CFRAgent(env_train)

    print(f"\nTesting both agents vs Random over {args.games:,} games...")
    wr_u, pay_u = win_rate_vs_random(untrained, args.games)
    print(f"  Untrained CFR  {wr_u:.1%} win rate  {pay_u:+.3f} chips/hand")

    wr_t, pay_t = win_rate_vs_random(trained, args.games)
    print(f"  Trained CFR    {wr_t:.1%} win rate  {pay_t:+.3f} chips/hand")
    print(f"  Improvement    +{(wr_t - wr_u) * 100:.1f}pp win rate")

    if wr_t > 0.85:
        print("\n✓ Trained agent performing correctly (>85% vs random)")
    else:
        print("\n⚠ Trained agent below expected threshold — may need more training")


if __name__ == '__main__':
    main()
