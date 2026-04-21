# Setup Guide

## Prerequisites

### System Dependencies

- **C++ compiler**: g++ 10+ or clang++ 12+ with C++20 support
- **CMake**: 3.20+
- **OpenMP**: Required for multi-threaded equity computation
- **CUDA**: Required for FAISS GPU clustering (tested with CUDA 12.4)
- **uv**: [Install uv](https://docs.astral.sh/uv/getting-started/installation/)

On RHEL/Fedora:
```bash
dnf install gcc-c++ libgomp-devel cmake
```

On Ubuntu/Debian:
```bash
apt-get install build-essential libomp-dev cmake
```

---

## Environment Setup

All Python dependencies are managed by [uv](https://docs.astral.sh/uv/).

```bash
# Install core dependencies (torch, faiss-gpu-cu12, numpy, etc.)
uv sync

# Also install demo dependencies (rlcard, streamlit, plotly)
uv sync --extra demos
```

`uv sync` creates `.venv/` at the repo root and resolves everything from
`pyproject.toml`. PyTorch is fetched from the PyTorch CUDA 12.4 index
automatically — no separate install step needed.

---

## Hold'em Abstraction Pipeline

### 1. Build the C++ Extensions

CMake fetches all C++ dependencies (pybind11, hand-isomorphism, OMPEval,
GoogleTest) automatically on first configure. An internet connection is
required the first time; subsequent builds use the cached `build/_deps/`.

```bash
cd src
make          # configure + build hand_indexer and deepcfr
make clean    # remove build/ artifacts and .so files
make clean-hard  # also remove clusters/
```

This compiles two pybind11 modules:

- **`hand_indexer`** — wraps `RiverExpander`, `TurnExpander`, `FlopExpander`,
  and the isomorphic hand indexers (hand-isomorphism + OMPEval).
- **`deepcfr`** — wraps the C++ DeepCFR traversal engine, reservoir sampling,
  and the inference scheduler.

### 2. Run the Pipelines

Pipelines must run in order. Each stage depends on output files from the
previous stage.

```bash
cd src

# Stage 1: River (2.4B states → 8,192 clusters)
uv run python pysrc/clustering/river_cluster_pipeline.py
uv run python pysrc/clustering/river_visualize_labels.py

# Stage 2: Turn (~55M states → 8,192 clusters)
uv run python pysrc/clustering/turn_cluster_pipeline.py
uv run python pysrc/clustering/turn_visualize_labels.py

# Stage 3: Flop (1.29M states → 2,048 clusters)
uv run python pysrc/clustering/flop_cluster_pipeline.py
uv run python pysrc/clustering/flop_visualize_labels.py

# Stage 4: Preflop (169 canonical classes → EHS values)
uv run python pysrc/clustering/preflop_ehs_pipeline.py
uv run python pysrc/clustering/preflop_ehs_visualize.py
```

### Pipeline Parameters

Each pipeline accepts flags to override defaults:

```bash
uv run python pysrc/clustering/river_cluster_pipeline.py -k 8192 --sample-size 20000000 -i 25 -t 16
uv run python pysrc/clustering/turn_cluster_pipeline.py  -k 8192 --sample-size 10000000 -i 25 -t 16
uv run python pysrc/clustering/flop_cluster_pipeline.py  -k 2048 -i 25 -t 16
```

| Flag | River | Turn | Flop | Description |
|------|:-----:|:----:|:----:|-------------|
| `-k` | 8192 | 8192 | 2048 | Number of clusters |
| `-s` | 20M | 10M | — | Training sample size (flop fits in RAM) |
| `-i` | 25 | 25 | 25 | K-means iterations |
| `-t` | 16 | 16 | 16 | OpenMP thread count |
| `-q` | — | — | — | Quiet mode (suppress progress output) |

See [CLUSTERING.md](CLUSTERING.md) for full technical details on each stage.

### Resource Requirements

| Stage | GPU VRAM | RAM |
|-------|:--------:|:---:|
| River | ~2 GB | ~6 GB |
| Turn | ~2 GB | ~12 GB (river_labels + river_ehs_fine ≈ 9.8 GB) |
| Flop | ~1 GB | ~3 GB |
| Preflop | — | < 1 GB |

### Output

All output is written to `src/clusters/`:

```
src/clusters/
├── river_centroids.npy     # 8192 × 169 float32
├── river_labels.bin        # 2.4B uint16 cluster assignments
├── river_ehs_fine.bin      # 2.4B uint16 per-state EHS (/ 65535.0 to decode)
├── river_ehs.bin           # 8192 float32 per-cluster EHS (sorted ascending)
├── turn_centroids.npy      # 8192 × 256 float32
├── turn_labels.bin         # ~55M uint16 cluster assignments
├── turn_ehs_fine.bin       # ~55M uint16 per-state EHS
├── turn_ehs.bin            # 8192 float32 per-cluster EHS (sorted ascending)
├── flop_centroids.npy      # 2048 × 256 float32
├── flop_labels.bin         # 1,286,792 uint16 cluster assignments
├── flop_ehs_fine.bin       # 1,286,792 uint16 per-state EHS
├── flop_ehs.bin            # 2048 float32 per-cluster EHS (sorted ascending)
├── preflop_ehs_fine.bin    # 169 float32 per-class EHS
└── *_viz.png               # Diagnostic plots (3 per stage)
```

---

## Evaluation

OpenSpiel is required only to run the match evaluation pipeline
(`src/pysrc/evaluation/`). It is included in the core dependencies and
installed by `uv sync`.

### Configuring the Game

The evaluation uses OpenSpiel's `universal_poker` game engine with parameters
matched to the training setup:

| Parameter | Value | Notes |
|-----------|:-----:|-------|
| Starting stacks | 2000 chips (= 20 BB) | Must match `STARTING_STACK = 40,000` milli-chips in `CFRTypes.h` |
| Blinds | 50 / 100 | Small / Big |
| Betting | No-limit | `betting=no_limit` |
| Players | 2 | Heads-up |

Recommended game string:
```python
game = pyspiel.load_game(
    "universal_poker(betting=no_limit,numPlayers=2,numSuits=4,numRanks=13,"
    "numHoleCards=2,numRounds=4,blind=50 100,raiseSize=0 0,maxRaises=0 0,"
    "numBoardCards=0 3 1 1,stack=2000 2000)"
)
```

### Running the Match Evaluation

```bash
cd src
uv run python pysrc/evaluation/match_runner.py \
    --strat-net  ../checkpoints/policy0050.pt \
    --br-net0    ../br_checkpoints/br_adv0_0040.pt \
    --br-net1    ../br_checkpoints/br_adv1_0040.pt \
    --clusters   ../clusters \
    --num-games  10000
```

The script reports P0 win rate in chips/hand. Run duplicate matches (swap
seats) to cancel out deal variance.

---

## Demos

### Kuhn Poker (no build step)

```bash
cd demos/kuhn
uv run python Kuhn.py
```

Runs 1,000 iterations of vanilla CFR on 3-card Kuhn Poker and displays
interactive convergence plots on 2D strategy simplices. Requires only
`numpy` and `matplotlib`.

### Leduc Hold'em CFR+ Solver & Demo

No GPU required — works on any laptop (macOS or Linux).

```bash
cd demos/leduc

# Install demo Python packages (streamlit, plotly, rlcard, numpy, matplotlib)
make install

# Compile C++ extension via CMake (downloads pybind11 automatically)
make

# Launch the interactive Streamlit demo at localhost:8501
make demo

# Or run the CFR+ solver standalone for 100k iterations (~27s)
make test
```

See [LEDUC.md](LEDUC.md) and [LEDUC_IMPLEMENTATION.md](LEDUC_IMPLEMENTATION.md)
for algorithm details, and [`demos/leduc/README.md`](../demos/leduc/README.md)
for the full demo guide including how to add new opponent profiles.

---

## Code Style

Both C++ and Python formatting are driven by a single Make target from `src/`:

```bash
make tidy
```

This runs `clang-format` over all `.h`/`.cpp` files and `ruff format` over
`pysrc/`.

---

## Troubleshooting

**`ModuleNotFoundError: No module named 'hand_indexer'`**
Run `make` from `src/` and make sure you are running Python via
`uv run python` (not the system or conda Python).

**`ModuleNotFoundError: No module named 'faiss'`**
You are running the system Python instead of the uv venv. Use
`uv run python <script>` or activate the venv with
`source .venv/bin/activate`.

**`faiss.GpuIndexFlatL1` or GPU errors**
Verify CUDA is available: `uv run python -c "import faiss; print(faiss.get_num_gpus())"`.
If it returns 0, check your CUDA driver version (requires CUDA 12.4+).

**CMake cannot find Python**
CMake is picking up a system or conda Python. The `src/Makefile` passes
`-DPython_EXECUTABLE` automatically via `uv run python`. If invoking CMake
directly, pass it manually:
```bash
cmake -S src -B src/build -DPython_EXECUTABLE=$(uv run python -c "import sys; print(sys.executable)")
```

**OpenMP not found during CMake configure**
Install `libgomp-devel` (RHEL) or `libomp-dev` (Ubuntu).

**C++ compiler errors (Leduc solver)**
The Leduc solver requires C++20. Verify with `g++ --version` (needs g++ 10+).
