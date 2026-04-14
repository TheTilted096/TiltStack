#!/usr/bin/env python3
"""
Clean wrapper for trained GTO agent
"""
import pickle

class GTOAgent:
    def __init__(self, model_path='models/leduc_cfr_100k.pkl'):
        with open(model_path, 'rb') as f:
            self.agent = pickle.load(f)
        self.use_raw = False

    def eval_step(self, state):
        return self.agent.eval_step(state)

    def step(self, state):
        return self.agent.step(state)
