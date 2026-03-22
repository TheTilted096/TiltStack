# TiltStack - Adaptive Poker AI

Fast adaptation in adversarial poker environments.

## Quick Start
```bash
# Install dependencies
pip install rlcard torch matplotlib

# Run live demo
python3 demo_live.py

# Run evaluation
python3 test_exploitative.py
```

## Results

- **1.43x improvement** vs tight opponents
- **1.26x improvement** vs loose-passive opponents

## Architecture

1. **GTO Baseline**: CFR-trained Nash equilibrium strategy
2. **Opponent Modeling**: Rule-based classification (Tight/Loose-Passive/Aggressive)
3. **Exploitative Strategies**: Hardcoded counter-strategies per opponent type

## Team

- Corey Zhang (PM, Algorithms)
- [Other team members]

## Midway Showcase: April 4, 2026
