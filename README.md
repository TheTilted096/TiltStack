# TiltStack

An exploitative poker AI using GTO followed by opponent modelling.

## Overview

TiltStack is a poker AI project that combines Game Theory Optimal (GTO) strategy computation with opponent modeling to create an exploitative poker agent. The project uses Counterfactual Regret Minimization (CFR) to compute Nash equilibrium strategies, then adapts play based on observed opponent tendencies.

## Repository Structure

```
TiltStack/
├── src/
│   ├── cppsrc/              # (Hold'em — coming soon)
│   └── pysrc/               # (Hold'em — coming soon)
├── demos/
│   ├── kuhn/                # Kuhn Poker reference implementation
│   │   ├── Node.py
│   │   └── Kuhn.py
│   └── leduc/               # Leduc Hold'em CFR solver
│       ├── Makefile
│       ├── setup.py
│       ├── requirements.txt
│       └── src/
│           ├── cppsrc/      # C++ solver (PyBind11)
│           └── pysrc/       # Python training layer
├── README.md
├── QUICK_START.md
├── PYBIND11_README.md
└── LICENSE
```

### demos/leduc — Leduc Hold'em Solver

A self-contained Leduc Hold'em CFR solver with C++ performance-critical code exposed to Python via PyBind11.

**C++ Solver** (`demos/leduc/src/cppsrc/`):
- **Node.h / Node.cpp**: Core game tree types — `Action`, `Rank`, `Outcome` enums, `NodeInfo` (game state decoding, legal moves, payouts, hash transitions), and `Node` (regret-matched strategy computation).
- **Leduc.h / Leduc.cpp**: `LeducSolver` class containing the node table and recursive `cfr()` method.
- **BestResponse.h / BestResponse.cpp**: `BestResponse` class that computes the optimal counter-strategy against a fixed opponent strategy. Used to measure exploitability of CFR strategies.
- **bindings.cpp**: PyBind11 bindings exposing `LeducSolver`, `BestResponse`, `Node`, `NodeInfo`, `Action`, `Rank`, and `ActionList` to Python.

**Python Layer** (`demos/leduc/src/pysrc/`):
- **Leduc.py**: `Leduc` class that wraps a C++ `LeducSolver`. Handles the training loop (card dealing and weighting), strategy output, and hash-to-string conversion.
- **BestResponse.py**: `BestResponse` class that wraps the C++ `BestResponse` solver. Computes optimal counter-strategies and measures exploitability.

### demos/kuhn — Kuhn Poker

Reference implementation of CFR for Kuhn Poker, a simplified 3-card poker game. Useful for understanding the core CFR algorithm before tackling more complex variants.

## Getting Started

```bash
cd demos/leduc
make install  # Install dependencies (pybind11)
make          # Build C++ extensions
make test     # Build and run the Leduc CFR solver
make best     # Build and compute Best Response exploitability
make clean    # Remove build artifacts and output files
```

See [QUICK_START.md](QUICK_START.md) for detailed setup instructions and [PYBIND11_README.md](PYBIND11_README.md) for the C++/Python binding architecture.

## Output Files

After running the solvers (from `demos/leduc/`), the following files are generated:

- **leduc_results.txt**: Human-readable CFR strategy for all 528 nodes with player markers
- **leduc_strategy.csv**: Machine-readable CFR strategy in CSV format
- **br_results.txt**: Best Response strategies for both players with exploitability metrics

## Current Status

- Leduc Hold'em CFR solver implemented in C++ and exposed via PyBind11
- Best Response computation for exploitability measurement
- Python training loop and strategy output working against the C++ backend
- Kuhn Poker demo available for reference

## Authors

**PMs:** Nathaniel Potter && Corey Zhang

- Forrest Chai
- Sophie Fong
- Michelle Wang
- Chris Yoon
