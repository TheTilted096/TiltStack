# TiltStack

Neural-network-accelerated Counterfactual Regret Minimization (DeepCFR) for no-limit Texas Hold'em.

## Overview

TiltStack is a poker AI system that uses deep learning to accelerate convergence to Nash equilibrium in heads-up no-limit poker. The core innovation is **DeepCFR**: neural networks (advantage networks and strategy networks) predict regrets and strategies during game-tree traversal, enabling efficient GPU-accelerated training via C++20 coroutines.

**Current performance (500k hands):**
- vs baseline opponent: **-207.60 ± 10.20 mBB/hand overall**
- SB position: -162.76 mBB/hand
- BB position: -252.43 mBB/hand

The system comprises:

1. **Texas Hold'em abstraction pipeline** (`src/`): Four-stage K-means clustering that compresses 2.4B river states → 8,192 clusters, enabling network-based game solving
2. **DeepCFR solver** (C++ core + Python interface): Coroutine-based parallel rollouts with GPU batching
3. **Neural networks**: Shared architecture for advantage prediction (regrets) and strategy learning
4. **Match evaluation** (`src/pysrc/evaluation/`): OpenSpiel integration for duplicate-pair match evaluation

## Repository Structure

```
TiltStack/
├── src/                        # Texas Hold'em infrastructure
│   ├── Makefile                #   Build system
│   ├── setup.py                #   PyBind11 config for hand_indexer
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
│       └── src/
│           ├── cppsrc/         #   C++ CFR+ solver (PyBind11)
│           └── pysrc/          #   Python training & output layer
├── docs/                       # Technical documentation
│   ├── SETUP.md                #   Environment setup and pipeline instructions
│   ├── DEEPCFR_TRAVERSAL.md    #   DeepCFR algorithm and game traversal
│   ├── NETWORK_ARCHITECTURE.md #   Neural network design and training
│   ├── COROUTINE_ROLLOUTS.md   #   C++20 coroutines and GPU batching
│   ├── EVALUATION_METHODOLOGY.md #  Match evaluation and convergence metrics
│   ├── CLUSTERING.md           #   Hold'em abstraction pipeline technical reference
│   ├── LEDUC.md                #   Leduc Hold'em CFR+ solver overview
│   └── LEDUC_IMPLEMENTATION.md #   Leduc algorithm details and convergence
├── environment.yml             # Conda environment (Python 3.11, faiss-gpu, numpy, matplotlib)
├── requirements.txt            # pip dependencies (numpy, faiss-cpu, pybind11, matplotlib)
├── README.md
└── LICENSE
```

## Quick Start

See [docs/SETUP.md](docs/SETUP.md) for full environment setup instructions, including Conda environment creation and build troubleshooting.

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

See the [Leduc README](docs/leduc/README.md) for performance benchmarks and usage.

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

GPU (FAISS) is required for all clustering pipelines. See [docs/CLUSTERING.md](docs/CLUSTERING.md) for parameters, output files, and technical details.

## Documentation

Key technical documents:

| Document | Purpose |
|----------|---------|
| [DEEPCFR_TRAVERSAL.md](docs/DEEPCFR_TRAVERSAL.md) | Core DeepCFR algorithm, game representation, regret computation |
| [NETWORK_ARCHITECTURE.md](docs/NETWORK_ARCHITECTURE.md) | Neural network design (embeddings, layers, loss functions) |
| [COROUTINE_ROLLOUTS.md](docs/COROUTINE_ROLLOUTS.md) | C++20 coroutines, frame pooling, GPU batching machinery |
| [EVALUATION_METHODOLOGY.md](docs/EVALUATION_METHODOLOGY.md) | Match evaluation, duplicate pairs, exploitability |
| [CLUSTERING.md](docs/CLUSTERING.md) | Hold'em abstraction pipeline (four-stage K-means) |
| [SETUP.md](docs/SETUP.md) | Build instructions, environment setup, troubleshooting |

## Components

### DeepCFR Solver (`src/cppsrc/deepcfr/`, `src/pysrc/deepcfr/`)

Neural-network-accelerated CFR with:
- **C++ core**: Game traversal, coroutines, C++ types (CFRGame, Scheduler, Orchestrator)
- **Python interface**: Network training, checkpoint management, match evaluation
- **GPU acceleration**: Batched neural inference via FAISS and PyTorch
- **Parallel rollouts**: Multi-threaded games with lock-free scheduling

See [docs/DEEPCFR_TRAVERSAL.md](docs/DEEPCFR_TRAVERSAL.md) for algorithm details.

### Hold'em Abstraction Pipeline (`src/cppsrc/clustering/`, `src/pysrc/clustering/`)

Four-stage K-means clustering pipeline:

| Stage | States | Features | Distance | Clusters | Purpose |
|-------|-------:|----------|----------|-------:|---------|
| River | 2.4B | 169-dim equity | L2 | 8,192 | Network input features |
| Turn | ~55M | 256-dim histogram CDF | L1 / EMD | 8,192 | Summarize river distribution |
| Flop | 1.3M | 256-dim histogram CDF | L1 / EMD | 2,048 | Summarize turn distribution |
| Preflop | 169 | — | — | 169 | Canonical hand classes |

Each stage reads labels/centroids from the previous stage. FAISS GPU K-means. Output: `uint16` label files + `float32` centroid arrays.

See [docs/CLUSTERING.md](docs/CLUSTERING.md).

### Match Evaluation (`src/pysrc/evaluation/`)

OpenSpiel-based match runner for duplicate-pair evaluation:
- **TiltStack agent**: Neural network inference during OpenSpiel game simulation
- **Baseline agents**: Best-response, random
- **Duplicate protocol**: Swapped seats cancel deal variance
- **Metrics**: Per-seat edges (SB/BB), overall edge, confidence intervals

See [docs/EVALUATION_METHODOLOGY.md](docs/EVALUATION_METHODOLOGY.md).

### Reference Solvers (`demos/`)

**Leduc Hold'em CFR+ Solver** (`demos/leduc/`): Self-contained C++/Python solver for a smaller game. Converges to 0.00 mBB in 14k iterations (~3.8k iter/sec).

**Kuhn Poker** (`demos/kuhn/`): Vanilla CFR on 3-card Kuhn Poker. No build step; interactive convergence plots.

## Dependencies

### Python & PyPI (`requirements.txt`)

```
numpy              # Array operations
faiss-gpu          # GPU K-means clustering (required; CPU impractical)
pybind11           # C++/Python bindings
matplotlib         # Visualization
open_spiel         # Game simulation (evaluation only)
```

Install: `pip install -r requirements.txt` (GPU required).

### Conda (`environment.yml` — Recommended)

```yaml
channels:
  - pytorch
  - nvidia
  - conda-forge
dependencies:
  - python=3.11
  - numpy
  - matplotlib
  - pybind11
  - faiss-gpu       # GPU K-means (CUDA 12.x)
  - gtest           # C++ testing
  - pytorch=2.5.1=py3.11_cuda12.4_cudnn9.1.0_0  # GPU training
  - pytorch-cuda=12.4
  - pip
  - pip:
      - setuptools>=42.0.0
      - open_spiel
```

Install: `conda env create -f environment.yml && conda activate tiltstack`

### System Requirements

- **Python**: 3.8+ (3.11 recommended for conda)
- **C++ compiler**: 
  - C++20 (DeepCFR solver, Leduc demo)
  - C++17 (Hold'em clustering pipeline)
  - Linux/Mac: g++ 10+ or clang++ 12+
  - Windows: MSVC 2019+
- **CUDA**: 11.x or 12.x (GPU clustering and training)
- **OpenMP**: Multi-threaded equity computation (libomp-dev on Ubuntu, libgomp-devel on RHEL)
- **GPU VRAM**: ≥4 GB (2 GB for clustering, 2 GB for inference during training)

## Current Status

### Completed
- ✓ Hold'em abstraction pipeline: river (2.4B → 8k), turn (55M → 8k), flop (1.3M → 2k) clustering
- ✓ DeepCFR solver: Neural networks + coroutine rollouts + GPU batching
- ✓ Leduc Hold'em CFR+ reference solver (0.00 mBB in 14k iterations)
- ✓ Match evaluation framework (OpenSpiel + duplicate pairs)
- ✓ Kuhn Poker vanilla CFR demo

### In Progress
- DeepCFR training convergence (current: -207.60 mBB/hand @ 500k hands)
- Advantage network refinement
- Strategy network regularization

## Authors

**PMs:** Nathaniel Potter && Corey Zhang

- Forrest Chai
- Sophie Fong
- Michelle Wang
- Chris Yoon
