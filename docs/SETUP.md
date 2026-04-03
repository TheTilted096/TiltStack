# Setup Guide

## Prerequisites

### System Dependencies

- **C++ compiler**: g++ 10+ or clang++ 12+ with C++17 support (C++20 for the Leduc demo)
- **OpenMP**: Required for multi-threaded equity computation in the Hold'em pipeline
- **CUDA**: Required for FAISS GPU clustering (tested with CUDA 11.x / 12.x)
- **Conda**: [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or Anaconda

On Ubuntu/Debian:
```bash
sudo apt-get install build-essential libomp-dev
```

On RHEL/Fedora:
```bash
sudo dnf install gcc-c++ libgomp-devel
```

---

## Environment Setup

### Option A: Conda (Recommended)

```bash
conda env create -f environment.yml
conda activate tiltstack
```

This installs Python 3.11, NumPy, Matplotlib, PyBind11, and FAISS with GPU support.

A GPU is required — the clustering pipelines are not practical to run on CPU given the dataset sizes (2.4B river states, ~55M turn states).

### Option B: pip

```bash
pip install -r requirements.txt
```

The root `requirements.txt` lists `faiss-cpu`. Replace it with `faiss-gpu` — a GPU is required to run the clustering pipelines.

---

## Hold'em Abstraction Pipeline

### 1. Build the C++ Extension

From the `src/` directory:

```bash
cd src
pip install -e . --no-build-isolation
```

Or via Make:

```bash
cd src
make          # equivalent to pip install -e . --no-build-isolation
make clean    # remove build artifacts
```

This compiles the `hand_indexer` PyBind11 module, which wraps:
- `RiverExpander` — equity vector + per-state EHS + multiplicity computation via OMPEval
- `TurnExpander` — wide-bucket histogram + per-state EHS + multiplicity computation
- `FlopExpander` — wide-bucket histogram + per-state EHS + multiplicity computation
- `PreflopIndexer`, `RiverIndexer`, `TurnIndexer`, `FlopIndexer` — isomorphic hand indexing via hand-isomorphism

### 2. Run the Pipelines

Pipelines must run in order. Each stage depends on output files from the previous stage.

```bash
cd src

# Stage 1: River (2.4B states → 8,192 clusters)
python pysrc/river_cluster_pipeline.py
python pysrc/river_visualize_labels.py

# Stage 2: Turn (~55M states → 8,192 clusters)
python pysrc/turn_cluster_pipeline.py
python pysrc/turn_visualize_labels.py

# Stage 3: Flop (1.29M states → 2,048 clusters)
python pysrc/flop_cluster_pipeline.py
python pysrc/flop_visualize_labels.py

# Stage 4: Preflop (169 canonical classes → EHS values)
python pysrc/preflop_ehs_pipeline.py
python pysrc/preflop_ehs_visualize.py
```

### Pipeline Parameters

Each pipeline accepts flags to override defaults:

```bash
python pysrc/river_cluster_pipeline.py -k 8192 --sample-size 20000000 -i 25 -t 16
python pysrc/turn_cluster_pipeline.py  -k 8192 --sample-size 10000000 -i 25 -t 16
python pysrc/flop_cluster_pipeline.py  -k 2048 -i 25 -t 16
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

### Installing OpenSpiel

OpenSpiel is required only to run the match evaluation pipeline (`src/pysrc/evaluation/`). It is installed automatically via conda:

```bash
conda env create -f environment.yml   # open_spiel is in the pip section
conda activate tiltstack
```

Or manually if the environment already exists:

```bash
pip install open_spiel
```

OpenSpiel wheels are pre-built for Python 3.11 on Linux x86_64. If you encounter a build-from-source fallback, verify your Python version and OS.

### Configuring the Game

The evaluation uses OpenSpiel's `universal_poker` game engine with parameters matched to the training setup:

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
python pysrc/evaluation/match_runner.py \
    --strat-net  ../checkpoints/policy0050.pt \
    --br-net0    ../br_checkpoints/br_adv0_0040.pt \
    --br-net1    ../br_checkpoints/br_adv1_0040.pt \
    --clusters   ../clusters \
    --num-games  10000
```

The script reports P0 win rate in chips/hand. Run duplicate matches (swap seats) to cancel out deal variance.

---

## Demos

### Kuhn Poker (no build step)

```bash
cd demos/kuhn
python Kuhn.py
```

Runs 1,000 iterations of vanilla CFR on 3-card Kuhn Poker and displays interactive convergence plots on 2D strategy simplices. Requires only `numpy` and `matplotlib`.

### Leduc Hold'em CFR+ Solver

```bash
cd demos/leduc
make install   # install pybind11 + setuptools
make build     # compile C++ extension (requires C++20)
make test      # train for 100k iterations
make best      # compute best response / exploitability
```

See [LEDUC.md](LEDUC.md) and [LEDUC_IMPLEMENTATION.md](LEDUC_IMPLEMENTATION.md) for details.

---

## Troubleshooting

**`ModuleNotFoundError: No module named 'hand_indexer'`**
Run `pip install -e . --no-build-isolation` from the `src/` directory and make sure your conda environment is active.

**`faiss.GpuIndexFlatL1` or GPU errors**
Verify CUDA is available: `python -c "import faiss; print(faiss.get_num_gpus())"`. If it returns 0, check your CUDA installation or switch to `faiss-cpu`.

**OpenMP not found during build**
Install `libomp-dev` (Ubuntu) or `libgomp-devel` (RHEL). On macOS, install via `brew install libomp`.

**C++ compiler errors (Leduc solver)**
The Leduc solver requires C++20. Verify with `g++ --version` (needs g++ 10+) or `clang++ --version` (needs clang 12+).
