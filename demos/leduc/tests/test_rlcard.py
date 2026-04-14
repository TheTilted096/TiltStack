#!/usr/bin/env python3
import rlcard
from rlcard.agents import RandomAgent

print(f"RLCard version: {rlcard.__version__}")

# Create no-limit hold'em environment
print("\nCreating No-Limit Hold'em environment...")
env = rlcard.make('no-limit-holdem')

print(f"Number of players: {env.num_players}")
print(f"Number of actions: {env.num_actions}")

# Set up two random agents
env.set_agents([
    RandomAgent(env.num_actions), 
    RandomAgent(env.num_actions)
])

# Run one game
print("\nRunning one game...")
trajectories, payoffs = env.run()
print(f"Payoffs: {payoffs}")
print("\n✓ Test successful! RLCard is working.")
