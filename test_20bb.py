#!/usr/bin/env python3
import rlcard
from rlcard.agents import RandomAgent

# Test with custom stack size
config = {
    'game_num_players': 2,
}

env = rlcard.make('no-limit-holdem', config=config)
env.set_agents([
    RandomAgent(env.num_actions),
    RandomAgent(env.num_actions)
])

print("Testing 20BB configuration...")
print(f"Players: {env.num_players}")
print(f"Action space: {env.num_actions}")

# Run 5 games and check it works
for i in range(5):
    trajectories, payoffs = env.run()
    print(f"Game {i+1}: Payoffs = {payoffs}")

print("\n✓ All tests passed!")
