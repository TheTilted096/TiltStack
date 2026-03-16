# TiltStack

An exploitative poker AI using GTO strategy followed by opponent modelling.

## Overview

TiltStack is a poker AI project that combines Game Theory Optimal (GTO) strategy computation with opponent modeling to create an exploitative agent. The project uses Counterfactual Regret Minimization (CFR) to compute Nash equilibrium strategies, then adapts play based on observed opponent tendencies.

The `src/` directory contains the Texas Hold'em abstraction pipeline — a three-stage K-means clustering system that compresses the game tree into a tractable state space. The `demos/` directory contains self-contained solver implementations used for development and reference.

## Repository Structure

```
TiltStack/
├── src/                        # Texas Hold'em infrastructure
│   ├── Makefile                #   Build system
│   ├── setup.py                #   PyBind11 config for hand_indexer
│   ├── docs/
│   │   └── CLUSTERING.md       #   Full clustering pipeline documentation
│   ├── cppsrc/
│   │   ├── river_expander.cpp  #   C++ river equity computation engine
│   │   ├── turn_expander.cpp   #   C++ turn histogram computation engine
│   │   ├── flop_expander.cpp   #   C++ flop histogram computation engine
│   │   └── bindings.cpp        #   PyBind11 bindings for hand_indexer
│   ├── pysrc/
│   │   ├── river_cluster_pipeline.py   #   River clustering orchestrator
│   │   ├── river_clusterer.py          #   River K-means library
│   │   ├── river_visualize_labels.py   #   River cluster diagnostics
│   │   ├── turn_cluster_pipeline.py    #   Turn clustering orchestrator
│   │   ├── turn_clusterer.py           #   Turn K-means library
│   │   ├── turn_visualize_labels.py    #   Turn cluster diagnostics
│   │   ├── flop_cluster_pipeline.py    #   Flop clustering orchestrator
│   │   ├── flop_clusterer.py           #   Flop K-means library
│   │   └── flop_visualize_labels.py    #   Flop cluster diagnostics
│   └── third_party/
│       ├── OMPEval/            #   Fast poker hand evaluator (C++)
│       └── hand-isomorphism/   #   Isomorphic hand indexing (C)
├── demos/
│   ├── kuhn/                   # Kuhn Poker — vanilla CFR reference implementation
│   └── leduc/                  # Leduc Hold'em — CFR+ solver
│       ├── docs/IMPLEMENTATION.md
│       └── src/
│           ├── cppsrc/         #   C++ CFR+ solver (PyBind11)
│           └── pysrc/          #   Python training & output layer
├── requirements.txt            # Root dependencies (numpy, faiss-gpu, matplotlib)
├── README.md
└── LICENSE
```

## Quick Start

### Kuhn Poker (no build step)

```bash
cd demos/kuhn
python Kuhn.py
```

### Leduc Hold'em CFR+ Solver

```bash
cd demos/leduc
make install   # Install pybind11
make build     # Compile C++ extension
make test      # Train CFR+ (100k iterations)
make best      # Compute Best Response exploitability
```

See the [Leduc README](demos/leduc/README.md) for performance benchmarks and usage.

### Hold'em Abstraction Pipeline

```bash
cd src
pip install -e . --no-build-isolation   # Build hand_indexer pybind module

# Run in order — each stage depends on the previous
python pysrc/river_cluster_pipeline.py
python pysrc/river_visualize_labels.py

python pysrc/turn_cluster_pipeline.py
python pysrc/turn_visualize_labels.py

python pysrc/flop_cluster_pipeline.py
python pysrc/flop_visualize_labels.py
```

GPU (FAISS) is required for all clustering pipelines. See [src/docs/CLUSTERING.md](src/docs/CLUSTERING.md) for parameters, output files, and technical details.

## Components

### Hold'em Abstraction Pipeline (`src/`)

Three-stage K-means clustering pipeline that abstracts Texas Hold'em into buckets for a GTO solver:

| Stage | States | Features | Distance | Clusters |
|-------|-------:|----------|----------|-------:|
| River | 2.4B | 169-dim equity vectors | L2 | 8,192 |
| Turn | ~55M | 256-dim CDF vectors (wide river buckets) | L1 / EMD | 8,192 |
| Flop | 1.29M | 256-dim CDF vectors (wide turn buckets) | L1 / EMD | 2,048 |

Each stage reads the labels and centroids from the previous stage. All pipelines use FAISS GPU K-means and write `uint16` label files plus `float32` centroid arrays.

See [src/docs/CLUSTERING.md](src/docs/CLUSTERING.md) for the full technical documentation.

### Leduc Hold'em Solver (`demos/leduc/`)

Self-contained CFR+ solver for Leduc Hold'em with a C++ core exposed to Python via PyBind11. Converges to 0.00 mBB exploitability in 14k iterations at ~3,810 iterations/second.

See the [Leduc README](demos/leduc/README.md) and [Implementation Notes](demos/leduc/docs/IMPLEMENTATION.md).

### Kuhn Poker (`demos/kuhn/`)

Reference implementation of vanilla CFR for 3-card Kuhn Poker with interactive convergence visualization on 2D strategy simplices.

## Dependencies

### Root (`requirements.txt`)

```
numpy
faiss-gpu
pybind11
matplotlib
```

Install with `pip install -r requirements.txt`. Use `faiss-cpu` if no GPU is available (clustering pipelines will not run, but visualizations will).

### System Requirements

- **Python**: 3.8+
- **C++ compiler**: C++20 (Leduc solver), C++17 (Hold'em expanders)
  - Windows: MSVC via Visual Studio Build Tools
  - Linux/Mac: g++ or clang++
- **OpenMP**: Required for multi-threaded equity computation
- **GPU**: Required for clustering pipelines (FAISS GPU)

## Current Status

- Hold'em river equity clustering — 2.4B states into 8,192 buckets
- Hold'em turn histogram clustering — ~55M states into 8,192 buckets
- Hold'em flop histogram clustering — 1.29M states into 2,048 buckets
- Leduc Hold'em CFR+ solver — converges to 0.00 mBB exploitability in 14k iterations
- Best Response computation for exploitability measurement
- Kuhn Poker vanilla CFR demo

## Authors

**PMs:** Nathaniel Potter && Corey Zhang

- Forrest Chai
- Sophie Fong
- Michelle Wang
- Chris Yoon
