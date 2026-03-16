# Quick Start

## Prerequisites

- **Python 3.8+**
- **C++ compiler** with C++20 support
  - Windows: MSVC via [Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
  - Linux/Mac: g++ or clang++

---

## Kuhn Poker Demo

No build step needed — pure Python:

```bash
cd demos/kuhn
python Kuhn.py    # Trains vanilla CFR (1000 iterations) with convergence visualization
```

---

## Leduc Hold'em Demo

### Install & Build

```bash
cd demos/leduc
make install   # Install pybind11 via pip
make build     # Compile C++ extension → leducsolver.pyd/.so
```

### Run

```bash
make test      # Train CFR+ solver (100k iterations)
make best      # Compute Best Response exploitability
make clean     # Remove build artifacts and output files
```

Or directly from `src/pysrc/`:

```bash
python Leduc.py         # Train CFR+ (100k iterations)
python BestResponse.py  # Compute exploitability of trained strategy
```

### Output

- `output/leduc_results.txt` — human-readable strategy
- `output/leduc_strategy.csv` — machine-readable strategy
- `output/br_results.txt` — Best Response strategies and exploitability

Strategy files use player markers `(0)` or `(1)` before each node:
```
(0) J:           -> c:0.91, r:0.09
(1) J:r          -> c:0.79, b:0.16, r:0.05
```

### Project Structure

```
demos/leduc/
├── Makefile
├── setup.py                # C++20 + pybind11 config
├── requirements.txt
├── docs/
│   └── IMPLEMENTATION.md
└── src/
    ├── cppsrc/             # Node, Leduc (CFR+), BestResponse, bindings
    └── pysrc/              # Leduc.py, BestResponse.py
```

---

## Hold'em Clustering Pipeline

Clusters 2.4B river equity states and ~55M turn states into 8192 buckets each using K-means (FAISS, GPU required).

### Build

```bash
cd src
pip install -e . --no-build-isolation   # Compile hand_indexer pybind module
```

Or via make:

```bash
make        # Equivalent to pip install -e . --no-build-isolation
make clean       # Remove build artifacts
make clean-hard  # Also remove output/
```

### Run Pipelines

From `src/pysrc/`:

```bash
python pysrc/river_cluster_pipeline.py
python pysrc/river_visualize_labels.py
python pysrc/turn_cluster_pipeline.py
python pysrc/turn_visualize_labels.py
```

The turn pipeline depends on `river_labels.bin` and `river_centroids.npy` — run river first.

### Pipeline Parameters

```bash
python pysrc/river_cluster_pipeline.py -k 8192 --sample-size 20000000 -t 16
python pysrc/turn_cluster_pipeline.py  -k 8192 --sample-size 10000000  -t 16
```

| Parameter | River default | Turn default | Description |
|-----------|--------------|-------------|-------------|
| `-k` | 8192 | 8192 | Number of clusters |
| `-s` | 20,000,000 | 10,000,000 | Training sample size |
| `-i` | 25 | 25 | K-means iterations |
| `-t` | 16 | 16 | OMP thread count |

### Pipeline Steps (each pipeline)

1. **Generate indices** — random sample of state indices, sorted and saved
2. **Compute sample** — equity/CDF vectors for sampled indices via C++ pybind
3. **Train centroids** — FAISS K-means on GPU; centroids sorted by EHS
4. **Assign labels** — stream all states, write uint16 cluster assignments

Intermediate sample files (`*_sample_indices.bin`, `*_sample.npy`) are deleted automatically after the pipeline completes.

### Output Files

```
src/output/
├── river_centroids.npy              # 8192 × 169 float32 centroid matrix
├── river_labels.bin                 # 2.4B uint16 cluster assignments
├── river_labels_viz.png             # Cluster size distribution + PCA overview
├── river_labels_viz_representatives.png
├── river_labels_viz_hands.png
├── turn_centroids.npy               # 8192 × 256 float32 CDF centroid matrix
├── turn_labels.bin                  # ~55M uint16 cluster assignments
├── turn_labels_viz.png
├── turn_labels_viz_representatives.png
└── turn_labels_viz_hands.png
```

### Technical Notes

- **River**: 2,428,287,420 states; 169-dim equity vectors (one per preflop bucket); L2 K-means
- **Turn**: ~55M states; 256-dim CDF vectors over wide buckets (32 river clusters each); L1 K-means (Earth Mover's Distance)
- Both pipelines require FAISS with GPU support — CPU-only runs are not supported
