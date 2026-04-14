#!/usr/bin/env python3
"""
Synthetic opponent bots - using ACTION IDs not strings
In Leduc: 0=call, 1=raise, 2=fold, 3=check (approximately)
"""
import numpy as np

class TightBot:
    """Folds almost everything"""
    def __init__(self, num_actions):
        self.num_actions = num_actions
        self.use_raw = False

    def eval_step(self, state):
        legal_actions = list(state['legal_actions'].keys())

        # If fold is available (usually action 2), take it 90% of time
        if 2 in legal_actions:
            if np.random.random() < 0.9:
                return 2, {}  # Fold

        # Otherwise check (3) or call (0)
        if 3 in legal_actions:
            return 3, {}  # Check
        if 0 in legal_actions:
            return 0, {}  # Call

        # Random fallback
        action = np.random.choice(legal_actions)
        return action, {}

class LoosePassiveBot:
    """NEVER folds, NEVER raises - always call/check"""
    def __init__(self, num_actions):
        self.num_actions = num_actions
        self.use_raw = False

    def eval_step(self, state):
        legal_actions = list(state['legal_actions'].keys())

        # Prefer check (3) over call (0), NEVER fold (2) or raise (1)
        if 3 in legal_actions:
            return 3, {}  # Check
        if 0 in legal_actions:
            return 0, {}  # Call

        # If only raise/fold available, call anyway
        if 1 in legal_actions:
            return 1, {}

        action = np.random.choice(legal_actions)
        return action, {}

class AggressiveBot:
    """Always raises when possible, never folds"""
    def __init__(self, num_actions):
        self.num_actions = num_actions
        self.use_raw = False

    def eval_step(self, state):
        legal_actions = list(state['legal_actions'].keys())

        # ALWAYS raise (1) if possible
        if 1 in legal_actions:
            return 1, {}  # Raise

        # NEVER fold - prefer call (0) or check (3)
        if 0 in legal_actions:
            return 0, {}  # Call
        if 3 in legal_actions:
            return 3, {}  # Check

        action = np.random.choice(legal_actions)
        return action, {}
