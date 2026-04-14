#!/usr/bin/env python3
"""Smoke tests for the rlcard installation and environment setup."""
import rlcard
from rlcard.agents import RandomAgent


def test_leduc():
    env = rlcard.make('leduc-holdem')
    env.set_agents([RandomAgent(env.num_actions), RandomAgent(env.num_actions)])
    _, payoffs = env.run()
    print(f"  Leduc Hold'em        players={env.num_players}  actions={env.num_actions}  payoffs={payoffs}")


def test_no_limit():
    env = rlcard.make('no-limit-holdem')
    env.set_agents([RandomAgent(env.num_actions), RandomAgent(env.num_actions)])
    _, payoffs = env.run()
    print(f"  No-Limit Hold'em     players={env.num_players}  actions={env.num_actions}  payoffs={payoffs}")


def test_no_limit_custom_config():
    env = rlcard.make('no-limit-holdem', config={'game_num_players': 2})
    env.set_agents([RandomAgent(env.num_actions), RandomAgent(env.num_actions)])
    payoff_list = []
    for _ in range(3):
        _, payoffs = env.run()
        payoff_list.append(payoffs)
    print(f"  No-Limit (custom)    3-game payoffs: {payoff_list}")


if __name__ == '__main__':
    print(f"rlcard {rlcard.__version__}\n")
    test_leduc()
    test_no_limit()
    test_no_limit_custom_config()
    print("\n✓ All smoke tests passed!")
