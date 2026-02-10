# TiltStack

An exploitative poker AI using GTO followed by opponent modelling.

## Overview

TiltStack is a poker AI project that combines Game Theory Optimal (GTO) strategy computation with opponent modeling to create an exploitative poker agent. The project uses Counterfactual Regret Minimization (CFR) to compute Nash equilibrium strategies, then adapts play based on observed opponent tendencies.

The CFR solver is implemented in C++ for performance and exposed to Python via PyBind11 as the `leducsolver` module. Training orchestration, output, and analysis remain in Python.

## Repository Structure

```
TiltStack/
├── Makefile              # Build system (install, build, test, clean)
├── setup.py              # PyBind11 compilation config
├── src/
│   ├── pysrc/
│   │   ├── Leduc.py      # Training loop, output, and analysis (uses C++ solver)
│   │   └── Node.py       # Pure-Python reference implementation
│   └── cppsrc/
│       ├── Node.h         # Game tree types: Action, Rank, NodeInfo, Node
│       ├── Node.cpp        # Node/NodeInfo implementation
│       ├── Leduc.h         # LeducSolver class declaration
│       ├── Leduc.cpp       # CFR solver implementation
│       └── bindings.cpp    # PyBind11 module (leducsolver)
├── demos/
│   └── kuhn/              # Kuhn Poker reference implementation
│       ├── Node.py
│       └── Kuhn.py
└── README.md
```

### src/cppsrc — C++ Solver

The performance-critical CFR solver, compiled into the `leducsolver` Python module:

- **Node.h / Node.cpp**: Core game tree types — `Action`, `Rank`, `Outcome` enums, `NodeInfo` (game state decoding, legal moves, payouts, hash transitions), and `Node` (regret-matched strategy computation).
- **Leduc.h / Leduc.cpp**: `LeducSolver` class containing the node table and recursive `cfr()` method.
- **bindings.cpp**: PyBind11 bindings exposing `LeducSolver`, `Node`, `NodeInfo`, `Action`, `Rank`, and `ActionList` to Python.

### src/pysrc — Python Layer

- **Leduc.py**: `Leduc` class that wraps a C++ `LeducSolver`. Handles the training loop (card dealing and weighting), strategy output, and hash-to-string conversion.
- **Node.py**: Pure-Python reference implementation of the same logic (useful for prototyping and verification).

### demos/kuhn

Reference implementation of CFR for Kuhn Poker, a simplified 3-card poker game. Useful for understanding the core CFR algorithm before tackling more complex variants.

## Getting Started

```bash
make install  # Install dependencies (pybind11)
make          # Build C++ extensions
make test     # Build and run the Leduc solver
make clean    # Remove build artifacts
```

See [QUICK_START.md](QUICK_START.md) for detailed setup instructions and [PYBIND11_README.md](PYBIND11_README.md) for the C++/Python binding architecture.

## Current Status

- Leduc Hold'em CFR solver implemented in C++ and exposed via PyBind11
- Python training loop and strategy output working against the C++ backend
- Kuhn Poker demo available for reference

## Authors

**PMs:** Nathaniel Potter && Corey Zhang

- Forrest Chai
- Sophie Fong
- Michelle Wang
- Chris Yoon
