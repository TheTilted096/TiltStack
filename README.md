# TiltStack

An exploitative poker AI using GTO followed by opponent modelling.

## Overview

TiltStack is a poker AI project that combines Game Theory Optimal (GTO) strategy computation with opponent modeling to create an exploitative poker agent. The project uses Counterfactual Regret Minimization (CFR) to compute Nash equilibrium strategies, then adapts play based on observed opponent tendencies.

## Repository Structure

```
TiltStack/
├── src/
│   ├── pysrc/          # High-level Python implementation
│   │   ├── Node.py     # Game tree node definitions and utilities
│   │   ├── Leduc.py    # Leduc Hold'em CFR solver
│   │   └── ...
│   └── cppsrc/         # Performance-critical C++ code (PyBind11)
│       └── ...
├── demos/
│   └── kuhn/           # Kuhn Poker reference implementation
│       ├── Node.py
│       └── Kuhn.py
└── README.md
```

### src/pysrc

Contains the high-level Python implementation of the poker AI. This includes:

- **Node.py**: Core game tree abstractions including `Action`, `Rank`, `NodeInfo`, and `Node` classes. Handles game state representation, legal move generation, payout calculation, and regret tracking for CFR.

- **Leduc.py**: CFR solver for Leduc Hold'em poker. Implements the full CFR algorithm with proper information set handling and strategy accumulation.

### src/cppsrc

Will contain performance-critical C++ code exposed to Python via PyBind11. This allows the project to maintain a clean Python API while achieving the performance necessary for solving larger poker variants.

### demos/kuhn

Reference implementation of CFR for Kuhn Poker, a simplified 3-card poker game. Useful for understanding the core CFR algorithm before tackling more complex variants.

## Getting Started

```bash
make install  # Install dependencies (pybind11)
make          # Build C++ extensions
make test     # Run tests
```

See [QUICK_START.md](QUICK_START.md) for more details. The Makefile works on both Linux and Windows.

## Current Status

- Leduc Hold'em CFR solver implemented and verified
- Kuhn Poker demo available for reference
- C++ backend with PyBind11 integration for performance optimization

## Authors

**PMs:** Nathaniel Potter && Corey Zhang

- Forrest Chai
- Sophie Fong
- Michelle Wang
- Chris Yoon
