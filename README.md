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
│   │   ├── turn_expander.cpp   #   C++ turn histogram computation engine
│   │   └── bindings.cpp        #   PyBind11 bindings for hand_indexer
│   ├── pysrc/
│   │   ├── river_clusterer.py  #   K-means clustering (2.4B river states)
│   │   ├── turn_clusterer.py   #   K-means clustering (~55M turn states)
│   │   ├── river_visualize_labels.py # Cluster diagnostic visualizations
│   │   └── turn_visualize_labels.py  # Cluster diagnostic visualizations
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

The `src/` directory contains the abstraction pipeline for Texas Hold'em, which consists of two main stages:
1.  **River Clustering**: Abstracts ~2.4 billion river card states into 8,192 strategic buckets based on hand equity.
2.  **Turn Clustering**: Abstracts ~55 million turn card states into 8,192 buckets based on the probability distribution of hitting each river bucket.

This two-stage process reduces the complexity of the game tree while preserving key strategic information.

### C++ Engine (`src/cppsrc/`)

- **river_expander.cpp**: Enumerates all river card combinations, computes equity vectors (169-dimensional, one entry per opponent hand class) using OMPEval for hand evaluation and hand-isomorphism for canonical indexing. Supports full enumeration (~2.4B states) and sampled subsets. Multi-threaded via OpenMP.
- **turn_expander.cpp**: For each canonical turn state, this enumerates all 46 possible river cards. For each river card, it finds the corresponding river cluster label (from the previous pipeline stage) and builds a 256-dimensional histogram of "wide" river buckets hit. This histogram serves as the feature vector for turn clustering.
- **bindings.cpp**: PyBind11 wrapper exposing the `hand-isomorphism` indexer to Python as the `hand_indexer` module. This module includes `RiverExpander` and `TurnExpander` classes.

### Python Scripts (`src/pysrc/`)

- **river_cluster_pipeline.py / river_clusterer.py**: A two-phase K-means clustering pipeline using FAISS. Phase 1 trains centroids on a sample of river states. Phase 2 streams all ~2.4B vectors, assigning each to the nearest centroid to generate the final labels.
- **turn_cluster_pipeline.py / turn_clusterer.py**: A similar K-means pipeline for turn states. It uses the histograms from `TurnExpander` as feature vectors and clusters them using L1 distance. This depends on the outputs (`river_labels.bin` and `river_centroids.npy`) of the river pipeline.
- **river_visualize_labels.py / turn_visualize_labels.py**: Scripts that produce diagnostic plots (histograms, PCA projections, example hands) to evaluate the quality of the river and turn clusters, respectively.

### Third-Party Libraries (`src/third_party/`)

- **OMPEval/**: Fast 5–7 card poker hand evaluator by [zekyll](https://github.com/zekyll/OMPEval). Used by `river_expander.cpp` for showdown evaluation.
- **hand-isomorphism/**: Isomorphic hand indexing by [kdub0](https://github.com/kdub0/hand-isern). Maps strategically equivalent hands (suit permutations) to canonical indices, reducing the state space.

### Build & Pipeline (`src/Makefile`)

The `Makefile` is used for building the `hand_indexer` PyBind11 module, which is a prerequisite for running the clustering pipelines.

```bash
cd src
make pybind       # Build hand_indexer Python module
make clean        # Remove build artifacts
```

The clustering pipelines are executed directly via their Python scripts:
```bash
# Run the clustering pipelines
python pysrc/river_cluster_pipeline.py
python pysrc/turn_cluster_pipeline.py
```
Pipeline parameters can be passed as command line arguments to these scripts.

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
pybind11
matplotlib
```

These are required for the Hold'em clustering pipelines in `src/`. Install with:
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
- **OpenMP**: Required for multi-threaded equity computation (included with most compilers)

Requires a conda environment and python venv (TODO: Specify dependency setup)

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
make pybind       # Build hand_indexer Python module

# Run the clustering pipelines
python pysrc/river_cluster_pipeline.py
python pysrc/turn_cluster_pipeline.py
```

See [QUICK_START.md](QUICK_START.md) for detailed setup instructions.

## Current Status

- Leduc Hold'em CFR+ solver — converges to 0.00 mBB exploitability in 14k iterations
- Best Response computation for exploitability measurement
- Hold'em river equity clustering pipeline — 2.4B states into 8,192 buckets
- Hold'em turn histogram clustering pipeline — ~55M states into 8,192 buckets
- Hand evaluation and isomorphic indexing via third-party C/C++ libraries
- Kuhn Poker demo available for reference

## Authors

**PMs:** Nathaniel Potter && Corey Zhang

- Forrest Chai
- Sophie Fong
- Michelle Wang
- Chris Yoon
