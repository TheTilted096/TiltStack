# TiltStack

An exploitative poker AI using GTO followed by opponent modelling.

## Overview

TiltStack is a poker AI project that combines Game Theory Optimal (GTO) strategy computation with opponent modeling to create an exploitative poker agent. The project uses Counterfactual Regret Minimization (CFR) to compute Nash equilibrium strategies, then adapts play based on observed opponent tendencies.

## Repository Structure

```
TiltStack/
├── src/                        # Texas Hold'em infrastructure
│   ├── Makefile                #   Build system & clustering pipeline
│   ├── setup.py                #   PyBind11 config for hand_indexer
│   ├── cppsrc/
│   │   ├── river_expander.cpp  #   C++ river equity computation engine
│   │   └── bindings.cpp        #   PyBind11 bindings for hand_indexer
│   ├── pysrc/
│   │   ├── river_clusterer.py  #   K-means clustering (2.4B river states)
│   │   ├── generate_sample_indices.py  # Random sampling for K-means training
│   │   ├── pipe_to_npy.py      #   Binary-to-NumPy converter
│   │   └── visualize_labels.py #   Cluster diagnostic visualizations
│   └── third_party/
│       ├── OMPEval/            #   Fast poker hand evaluator (C++)
│       └── hand-isomorphism/   #   Isomorphic hand indexing (C)
├── demos/
│   ├── kuhn/                   # Kuhn Poker — vanilla CFR reference
│   │   ├── Node.py             #   Basic CFR node
│   │   └── Kuhn.py             #   CFR solver with convergence viz
│   └── leduc/                  # Leduc Hold'em — CFR+ solver
│       ├── Makefile
│       ├── setup.py
│       ├── requirements.txt
│       ├── docs/               #   Implementation notes
│       └── src/
│           ├── cppsrc/         #   C++ solver core (PyBind11)
│           └── pysrc/          #   Python training & output layer
├── requirements.txt            # Root dependencies (numpy, faiss-cpu)
├── README.md
├── QUICK_START.md
└── LICENSE
```

## src/ — Texas Hold'em Infrastructure

The `src/` directory contains the Hold'em river equity clustering pipeline, which abstracts 2.4 billion river card states into ~30,000 strategic buckets.

### C++ Engine (`src/cppsrc/`)

- **river_expander.cpp**: Enumerates all river card combinations, computes equity vectors (169-dimensional, one entry per opponent hand class) using OMPEval for hand evaluation and hand-isomorphism for canonical indexing. Supports full enumeration (~2.4B states) and sampled subsets. Multi-threaded via OpenMP.
- **bindings.cpp**: PyBind11 wrapper exposing the hand-isomorphism indexer to Python as the `hand_indexer` module.

### Python Scripts (`src/pysrc/`)

- **river_clusterer.py**: Two-phase K-means clustering using FAISS. Phase 1: train centroids on a ~20M sample. Phase 2: stream all 2.4B vectors and assign each to the nearest centroid. Supports GPU acceleration.
- **generate_sample_indices.py**: Generates N sorted unique random indices from [0, 2.4B) for sampling the training set. Outputs a binary uint64 file.
- **pipe_to_npy.py**: Reads raw uint8 equity vectors from stdin and saves them as a float32 `.npy` file. Used in the `river_expander | pipe_to_npy` pipeline.
- **visualize_labels.py**: Produces diagnostic plots (histograms, PCA projections, cosine similarity heatmaps, example hands) to evaluate cluster quality.

### Third-Party Libraries (`src/third_party/`)

- **OMPEval/**: Fast 5–7 card poker hand evaluator by [zekyll](https://github.com/zekyll/OMPEval). Used by `river_expander.cpp` for showdown evaluation.
- **hand-isomorphism/**: Isomorphic hand indexing by [kdub0](https://github.com/kdub0/hand-isern). Maps strategically equivalent hands (suit permutations) to canonical indices, reducing the state space.

### Build & Pipeline (`src/Makefile`)

```bash
cd src
make              # Build river_expander executable
make pybind       # Build hand_indexer Python module
make pipeline     # Run full 3-step clustering pipeline
make clean        # Remove build artifacts
```

Pipeline parameters (override via command line):
- `K=30000` — number of clusters
- `SAMPLE_SIZE=20000000` — training sample size
- `THREADS=16` — CPU thread limit
- `GPU=auto` — GPU acceleration (`yes`/`no`/`auto`)

## demos/leduc — Leduc Hold'em Solver

A self-contained Leduc Hold'em CFR+ solver with C++ performance-critical code exposed to Python via PyBind11.

**C++ Solver** (`demos/leduc/src/cppsrc/`):
- **Node.h / Node.cpp**: Core game tree types — `Action`, `Rank`, `Outcome` enums, `NodeInfo` (game state decoding, legal moves, payouts, hash transitions), and `Node` (regret-matched strategy computation with double-precision accumulation).
- **Leduc.h / Leduc.cpp**: `LeducSolver` class containing the 528-node table and recursive `cfr()` method with alternating updates.
- **BestResponse.h / BestResponse.cpp**: `BestResponse` class that computes the optimal counter-strategy against a fixed opponent strategy. Used to measure exploitability of CFR strategies.
- **bindings.cpp**: PyBind11 bindings exposing `LeducSolver`, `BestResponse`, `Node`, `NodeInfo`, `Action`, `Rank`, and `ActionList` to Python.

**Python Layer** (`demos/leduc/src/pysrc/`):
- **Leduc.py**: `Leduc` class that wraps a C++ `LeducSolver`. Handles the training loop (card dealing and weighting), strategy output, best response computation, and live convergence plotting.
- **BestResponse.py**: `BestResponse` class that wraps the C++ `BestResponse` solver. Computes optimal counter-strategies and measures exploitability.

See the [Leduc README](demos/leduc/README.md) for performance benchmarks and usage examples.

## demos/kuhn — Kuhn Poker

Reference implementation of vanilla CFR for Kuhn Poker, a simplified 3-card poker game (Jack, Queen, King). Includes interactive convergence visualization on 2D strategy simplices. Useful for understanding the core CFR algorithm before tackling more complex variants.

## Dependencies

### Root (`requirements.txt`)
```
numpy
faiss-cpu
```

These are required for the Hold'em river clustering pipeline in `src/`. Install with:
```bash
pip install -r requirements.txt
```

For GPU-accelerated clustering, install `faiss-gpu` instead of `faiss-cpu`.

### Leduc Demo (`demos/leduc/requirements.txt`)
```
pybind11>=2.10.0
setuptools>=42.0.0
matplotlib>=3.5.0
```

### System Requirements

- **Python**: 3.8+
- **C++ compiler**: C++20 support required for Leduc solver, C++11 for river_expander
  - Windows: MSVC (Visual Studio Build Tools)
  - Linux/Mac: g++ or clang++
- **OpenMP**: Required for multi-threaded river equity computation (included with most compilers)

No conda environment file is provided — use pip with the requirements files above.

## Getting Started

### Leduc Demo (recommended starting point)
```bash
cd demos/leduc
make install  # Install pybind11
make build    # Build C++ extension
make test     # Train CFR+ solver (100k iterations)
make best     # Compute Best Response exploitability
```

### Hold'em Pipeline
```bash
cd src
pip install -r ../requirements.txt
make              # Build river_expander
make pipeline     # Run full clustering pipeline (~2.4B states)
```

See [QUICK_START.md](QUICK_START.md) for detailed setup instructions.

## Current Status

- Leduc Hold'em CFR+ solver — converges to 0.00 mBB exploitability in 14k iterations
- Best Response computation for exploitability measurement
- Hold'em river equity clustering pipeline — 2.4B states into 30k buckets
- Hand evaluation and isomorphic indexing via third-party C/C++ libraries
- Kuhn Poker demo available for reference

## Authors

**PMs:** Nathaniel Potter && Corey Zhang

- Forrest Chai
- Sophie Fong
- Michelle Wang
- Chris Yoon
