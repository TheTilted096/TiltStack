# Hold'em Abstraction Pipeline

Three-stage K-means clustering pipeline that abstracts Texas Hold'em into a tractable state space for a GTO solver. Each stage depends on the outputs of the previous one.

```
River  →  Turn  →  Flop
```

---

## Build

```bash
cd src
pip install -e . --no-build-isolation   # Compile hand_indexer pybind module
```

Or via make:

```bash
make             # Equivalent to pip install -e . --no-build-isolation
make clean       # Remove build artifacts
make clean-hard  # Also remove output/
```

---

## Running the Pipelines

Run in order from `src/`:

```bash
python pysrc/river_cluster_pipeline.py
python pysrc/river_visualize_labels.py

python pysrc/turn_cluster_pipeline.py
python pysrc/turn_visualize_labels.py

python pysrc/flop_cluster_pipeline.py
python pysrc/flop_visualize_labels.py
```

Each pipeline skips already-completed steps (centroids and labels files are checked before recomputing).

---

## Pipeline Parameters

```bash
python pysrc/river_cluster_pipeline.py -k 8192 --sample-size 20000000 -t 16
python pysrc/turn_cluster_pipeline.py  -k 8192 --sample-size 10000000 -t 16
python pysrc/flop_cluster_pipeline.py  -k 2048
```

| Parameter | River default | Turn default | Flop default | Description |
|-----------|:---:|:---:|:---:|-------------|
| `-k` | 8192 | 8192 | 2048 | Number of clusters |
| `-s` | 20,000,000 | 10,000,000 | — | Training sample size (flop fits fully in RAM) |
| `-i` | 25 | 25 | 25 | K-means iterations |
| `-t` | 16 | 16 | 16 | OMP thread count |

GPU (FAISS) is required for all three pipelines.

---

## Stage Details

### River

**Input:** ~2.4 billion canonical river states
**Feature:** 169-dim float32 equity vector — one entry per preflop isomorphic hand class (pairs, suited, offsuit), values in [0, 255]
**Distance:** L2
**Output:** 8,192 clusters

**Steps:**

1. **Sample indices** — draw a random subset of state indices, sort, write to `river_sample_indices.bin`
2. **Compute sample** — call `RiverExpander.compute_sample()` to evaluate equity vectors for sampled indices
3. **Train centroids** — FAISS K-means on GPU; sort by mean EHS (ascending)
4. **Assign labels** — stream all 2.4B states in batches via `RiverExpander.expand_all()`, write `river_labels.bin`

Intermediate files (`river_sample_indices.bin`, `river_sample.npy`) are deleted automatically on completion.

---

### Turn

**Input:** ~55 million canonical turn states
**Feature:** 256-dim float32 CDF vector — cumulative sum of a uint8[256] count histogram over 256 wide river buckets (each wide bucket = 32 consecutive river clusters). Counts sum to 46 (one per possible river card); CDF values ∈ [0, 46].
**Distance:** L1 (equivalent to Wasserstein-1 / Earth Mover's Distance on the underlying distributions)
**Output:** 8,192 clusters
**Depends on:** `river_labels.bin`, `river_centroids.npy`

**Steps:**

1. **Sample indices** — random subset of turn state indices
2. **Compute sample** — `TurnExpander.compute_sample()` returns uint8 count histograms; pipeline converts to CDF via cumsum
3. **Train centroids** — FAISS K-means (L1) on GPU; sort by expected EHS computed from river centroids
4. **Assign labels** — stream all turn states in batches, converting each batch to CDF before searching

Intermediate files (`turn_sample_indices.bin`, `turn_sample.npy`) are deleted automatically on completion.

---

### Flop

**Input:** 1,286,792 canonical flop states (fits entirely in RAM)
**Feature:** 256-dim float32 CDF vector — cumulative sum of a uint8[256] count histogram over 256 wide turn buckets (each wide bucket = 32 consecutive turn clusters). Counts sum to 47 (one per possible turn card); CDF values ∈ [0, 47].
**Distance:** L1 (Wasserstein-1)
**Output:** 2,048 clusters
**Depends on:** `turn_labels.bin`, `turn_centroids.npy`, `river_centroids.npy`

Because all flop states fit in RAM, no sampling or intermediate disk files are needed.

**Steps:**

1. **Compute CDFs** — load all 1.286M flop states into RAM via `FlopExpander.compute_sample()`, convert uint8 histograms to CDFs
2. **Train centroids** — FAISS K-means (L1) on GPU; sort by expected EHS via two-level averaging (river centroids → wide river EHS → turn centroid EHS → wide turn EHS → flop centroid EHS)
3. **Assign labels** — GPU search over the in-memory CDF matrix (no streaming needed)

---

## Output Files

```
src/output/
├── river_centroids.npy                  # 8192 × 169  float32
├── river_labels.bin                     # 2,428,287,420 × uint16
├── river_labels_viz.png                 # Cluster size distribution + PCA + equity profiles
├── river_labels_viz_representatives.png # PCA + rank-size with EHS percentile clusters marked
├── river_labels_viz_hands.png           # Example hands per representative cluster
│
├── turn_centroids.npy                   # 8192 × 256  float32 (CDF vectors)
├── turn_labels.bin                      # ~55M × uint16
├── turn_labels_viz.png
├── turn_labels_viz_representatives.png
├── turn_labels_viz_hands.png
│
├── flop_centroids.npy                   # 2048 × 256  float32 (CDF vectors)
├── flop_labels.bin                      # 1,286,792 × uint16
├── flop_labels_viz.png
├── flop_labels_viz_representatives.png
└── flop_labels_viz_hands.png
```

---

## C++ Engine (`src/cppsrc/`)

| File | Role |
|------|------|
| `river_expander.cpp/.h` | Enumerates river states; computes 169-dim equity vectors via OMPEval |
| `turn_expander.cpp/.h` | For each turn state, enumerates 46 river cards and builds wide-bucket histogram using river labels |
| `flop_expander.cpp/.h` | For each flop state, enumerates 47 turn cards and builds wide-bucket histogram using turn labels |
| `bindings.cpp` | PyBind11 bindings exposing `RiverExpander`, `TurnExpander`, `FlopExpander` (and indexers) as the `hand_indexer` module |

All expanders are multi-threaded via OpenMP.

---

## Third-Party Libraries

- **OMPEval** (`src/third_party/OMPEval/`): Fast 5–7 card poker hand evaluator by [zekyll](https://github.com/zekyll/OMPEval). Used by `river_expander.cpp` for showdown equity computation.
- **hand-isomorphism** (`src/third_party/hand-isomorphism/`): Isomorphic hand indexing by [kdub0](https://github.com/kdub0/hand-isomorphism). Maps suit-equivalent hands to canonical indices, reducing the state space.
