#!/usr/bin/env python3
"""
Track opponent statistics in real-time
"""

class OpponentTracker:
    def __init__(self):
        self.hands_seen = 0
        self.hands_played = 0  # Voluntarily put chips in pot
        self.hands_raised = 0
        self.times_called = 0
        self.times_raised = 0
        self.times_folded = 0
        
    def update(self, action, is_voluntary=True):
        """Update stats after each action"""
        self.hands_seen += 1
        
        if action in ['call', 'raise'] and is_voluntary:
            self.hands_played += 1
            
        if action == 'raise':
            self.hands_raised += 1
            self.times_raised += 1
        elif action == 'call':
            self.times_called += 1
        elif action == 'fold':
            self.times_folded += 1
    
    def get_vpip(self):
        """Voluntarily Put In Pot %"""
        if self.hands_seen == 0:
            return 0.0
        return self.hands_played / self.hands_seen
    
    def get_pfr(self):
        """Pre-Flop Raise %"""
        if self.hands_seen == 0:
            return 0.0
        return self.hands_raised / self.hands_seen
    
    def get_aggression(self):
        """Aggression Factor = raises / calls"""
        if self.times_called == 0:
            return 99.0 if self.times_raised > 0 else 0.0
        return self.times_raised / self.times_called
    
    def classify(self):
        """Simple rule-based classification"""
        if self.hands_seen < 20:
            return "Unknown"
        
        vpip = self.get_vpip()
        aggression = self.get_aggression()
        
        if vpip < 0.3:
            return "Tight"
        elif vpip > 0.6 and aggression < 1.0:
            return "Loose-Passive"
        elif aggression > 2.0:
            return "Aggressive"
        else:
            return "Unknown"
    
    def get_stats(self):
        return {
            'hands': self.hands_seen,
            'vpip': self.get_vpip(),
            'pfr': self.get_pfr(),
            'aggression': self.get_aggression(),
            'type': self.classify()
        }
