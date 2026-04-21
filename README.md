# TiltStack

Neural-network-accelerated Counterfactual Regret Minimization (DeepCFR) for
no-limit Texas Hold'em, with an adaptive exploitative layer that outperforms
GTO against predictable opponents.

## Overview

TiltStack is a poker AI system built around two ideas:

1. **DeepCFR** — neural networks (advantage + strategy nets) accelerate
   convergence to Nash equilibrium in heads-up no-limit Hold'em via C++20
   coroutines and GPU-batched inference.

2. **Adaptive exploitation** — a BestResponse solver computes the
   mathematically optimal counter-strategy for a classified opponent profile,
   yielding 3–8× more chips/hand than the GTO baseline against weak opponents.

**Current training performance (500k hands):**
- vs baseline: **−207.60 ± 10.20 mBB/hand** (training still converging)

---

## Quick Start

### Leduc Hold'em demo (no GPU required)

The fastest way to see TiltStack in action — runs on any laptop:

```bash
cd demos/leduc
make install   # streamlit, plotly, numpy, matplotlib
make           # compile C++ extension via CMake
make demo      # opens Streamlit UI at localhost:8501
```

See [demos/leduc/README.md](demos/leduc/README.md) for full demo documentation.

### Kuhn Poker (no build step)

```bash
cd demos/kuhn
uv run python Kuhn.py
```

### Hold'em training pipeline (GPU required)

```bash
# Install all Python dependencies
uv sync

# Build C++ extensions (hand_indexer + deepcfr)
cd src && make

# Run clustering pipeline (once; order matters)
uv run python pysrc/clustering/river_cluster_pipeline.py
uv run python pysrc/clustering/turn_cluster_pipeline.py
uv run python pysrc/clustering/flop_cluster_pipeline.py
uv run python pysrc/clustering/preflop_ehs_pipeline.py

# Train DeepCFR
uv run python pysrc/deepcfr/NLHE_Trainer.py
```

See [docs/SETUP.md](docs/SETUP.md) for full instructions and troubleshooting.

---

## Repository Structure

```
TiltStack/
├── pyproject.toml              # Python dependencies (uv)
├── uv.lock
├── src/                        # Texas Hold'em training infrastructure
│   ├── CMakeLists.txt          #   CMake build (hand_indexer + deepcfr)
│   ├── Makefile                #   Thin wrapper around cmake targets
│   ├── cppsrc/
│   │   ├── clustering/         #   River/turn/flop equity engines (C++)
│   │   ├── deepcfr/            #   CFR traversal, reservoir, scheduler (C++20)
│   │   └── test/               #   GoogleTest unit tests
│   └── pysrc/
│       ├── clustering/         #   K-means pipeline orchestrators
│       ├── deepcfr/            #   Network training, checkpoint management
│       └── evaluation/         #   OpenSpiel match runner
├── demos/
│   ├── kuhn/                   # Kuhn Poker — vanilla CFR reference
│   └── leduc/                  # Leduc Hold'em — CFR+ solver + adaptive demo
│       ├── CMakeLists.txt
│       ├── Makefile
│       └── src/
│           ├── cppsrc/         #   C++ CFR+ solver (pybind11)
│           ├── pysrc/          #   Python training wrapper
│           ├── showcase/       #   Streamlit UI + matplotlib demo
│           └── tests/
└── docs/                       # Technical documentation
    ├── SETUP.md                #   Build instructions and troubleshooting
    ├── DEEPCFR.md              #   DeepCFR algorithm and traversal
    ├── NETWORK_ARCH.md         #   Neural network design
    ├── COROUTINES.md           #   C++20 coroutines and GPU batching
    ├── EVALUATION.md           #   Match evaluation methodology
    ├── CLUSTERING.md           #   Hold'em abstraction pipeline
    ├── LEDUC.md                #   Leduc CFR+ solver overview
    └── LEDUC_IMPLEMENTATION.md #   Leduc algorithm details
```

---

## Dependencies

### Python — managed by [uv](https://docs.astral.sh/uv/)

```bash
uv sync               # core deps (torch, faiss-gpu-cu12, numpy, open-spiel)
uv sync --extra demos # also installs streamlit, plotly, rlcard
```

> **macOS / no GPU**: `uv sync` will fail because `faiss-gpu-cu12` has no
> macOS wheel. Use the Leduc demo instead — `make install` inside
> `demos/leduc/` installs only the demo packages via pip.

### System

| Requirement | Notes |
|-------------|-------|
| Python 3.11+ | Managed by uv for the training pipeline |
| CMake 3.20+ | Fetches all C++ deps automatically |
| g++ 10+ / clang++ 12+ | C++20 required |
| CUDA 12.4 | GPU training and FAISS clustering (Linux) |
| OpenMP | `libomp-dev` (Ubuntu) or `libgomp-devel` (RHEL) |

---

## Components

### DeepCFR Solver (`src/`)

- **C++ core**: Game traversal (`CFRGame`), reservoir sampling, coroutine
  scheduler (`Orchestrator`, `Scheduler`)
- **Python interface**: Network training, tensorboard logging, checkpoint
  management (`NLHE_Trainer.py`, `network_training.py`)
- **GPU acceleration**: Batched inference via PyTorch (CUDA 12.4)

See [docs/DEEPCFR.md](docs/DEEPCFR.md) and [docs/COROUTINES.md](docs/COROUTINES.md).

### Hold'em Abstraction Pipeline (`src/pysrc/clustering/`)

Four-stage FAISS GPU K-means compressing the game tree:

| Stage | States | Clusters |
|-------|-------:|--------:|
| River | 2.4B | 8,192 |
| Turn | ~55M | 8,192 |
| Flop | 1.3M | 2,048 |
| Preflop | 169 | 169 |

See [docs/CLUSTERING.md](docs/CLUSTERING.md).

### Match Evaluation (`src/pysrc/evaluation/`)

OpenSpiel duplicate-pair evaluation cancels deal variance. Reports per-seat
and overall edges in mBB/hand with confidence intervals.

See [docs/EVALUATION.md](docs/EVALUATION.md).

### Leduc Hold'em Demo (`demos/leduc/`)

Self-contained reference implementation. No GPU needed.

- CFR+ converges to **0.00 mBB exploitability** in ~14k iterations
- Interactive Streamlit UI: train GTO → compute BR → watch live simulation
- Exploitative agent earns **3–8× more chips/hand** than GTO vs weak opponents

See [demos/leduc/README.md](demos/leduc/README.md).

---

## Current Status

| Component | Status |
|-----------|--------|
| Hold'em abstraction pipeline | ✓ Complete |
| DeepCFR solver (C++ + Python) | ✓ Complete |
| Leduc CFR+ reference solver | ✓ Complete |
| Leduc adaptive demo (Streamlit) | ✓ Complete |
| Match evaluation framework | ✓ Complete |
| DeepCFR training convergence | ⏳ In progress (−207 mBB @ 500k hands) |
| Opponent classifier (LSTM) | 🔲 Planned |
| Full Hold'em adaptive agent | 🔲 Planned |

---

## Authors

**PMs:** Nathaniel Potter, Corey Zhang

Forrest Chai · Sophie Fong · Michelle Wang · Chris Yoon
