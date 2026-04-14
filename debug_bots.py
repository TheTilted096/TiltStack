#!/usr/bin/env python3
import rlcard
from src.opponents import TightBot

env = rlcard.make('leduc-holdem')
env.reset()
state = env.get_state(0)

print("State legal actions:", state['legal_actions'])
print()

tight = TightBot(4)
for i in range(5):
    action, info = tight.eval_step(state)
    print(f"Attempt {i+1}: action = {action}, type = {type(action)}")
