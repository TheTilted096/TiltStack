# Hold'em Abstraction Pipeline

Four-stage K-means clustering pipeline that abstracts Texas Hold'em into a tractable state space for a GTO solver. Each stage depends on the outputs of the previous one.

```
River  →  Turn  →  Flop  →  Preflop
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
make clean-hard  # Also remove clusters/
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

python pysrc/preflop_ehs_pipeline.py
python pysrc/preflop_ehs_visualize.py
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

GPU (FAISS) is required for all three clustering pipelines.

---

## Stage Details

### River

**Input:** ~2.4 billion canonical river states
**Feature:** 169-dim uint8 equity vector — one entry per preflop isomorphic hand class (pairs, suited, offsuit); decode: `byte / 255.0 = equity`
**Distance:** L2
**Output:** 8,192 clusters

**Steps:**

1. **Sample indices** — draw a random subset of state indices, sort, write to `river_sample_indices.bin`
2. **Compute sample** — `RiverExpander.compute_sample()` evaluates equity vectors for sampled indices; pipeline converts to float32 for training
3. **Train centroids** — FAISS K-means on GPU; centroids are initially unsorted
4. **Assign labels + EHS** — stream all 2.4B states in batches via `RiverExpander.expand_all_with_ehs_mult()`, which produces equity vectors, per-state EHS, and suit-isomorphism multiplicities in a single parallel pass per batch; writes `river_labels.bin` and `river_ehs_fine.bin`
5. **Sort & remap** — sort centroids by multiplicity-weighted per-cluster EHS (ascending); remap labels accordingly; write `river_ehs.bin`

Per-state EHS is `totalEqSum / totalCount` over all concrete opponent hole-card pairs — correctly probability-weighted, not an unweighted average over the 169 canonical buckets.

Intermediate files (`river_sample_indices.bin`, `river_sample.npy`) are deleted automatically on completion.

---

### Turn

**Input:** ~55 million canonical turn states
**Feature:** 256-dim float32 CDF vector — cumulative sum of a uint8[256] count histogram over 256 wide river buckets (each wide bucket = 32 consecutive river clusters). Counts sum to 46 (one per possible river card); CDF values ∈ [0, 46].
**Distance:** L1 (equivalent to Wasserstein-1 / Earth Mover's Distance on the underlying distributions)
**Output:** 8,192 clusters
**Depends on:** `river_labels.bin`, `river_ehs_fine.bin`

**Steps:**

1. **Sample indices** — random subset of turn state indices
2. **Compute sample** — `TurnExpander.compute_sample()` returns uint8 count histograms; pipeline converts to CDF via cumsum
3. **Train centroids** — FAISS K-means (L1) on GPU; centroids are initially unsorted
4. **Assign labels + EHS** — stream all turn states in batches via `TurnExpander.expand_all_with_ehs_mult()`, producing histograms, per-state EHS (averaged from `river_ehs_fine` lookups), and multiplicities in a single parallel pass; writes `turn_labels.bin` and `turn_ehs_fine.bin`
5. **Sort & remap** — sort centroids by multiplicity-weighted per-cluster EHS; remap labels; write `turn_ehs.bin`

Intermediate files (`turn_sample_indices.bin`, `turn_sample.npy`) are deleted automatically on completion.

---

### Flop

**Input:** 1,286,792 canonical flop states (fits entirely in RAM)
**Feature:** 256-dim float32 CDF vector — cumulative sum of a uint8[256] count histogram over 256 wide turn buckets (each wide bucket = 32 consecutive turn clusters). Counts sum to 47 (one per possible turn card); CDF values ∈ [0, 47].
**Distance:** L1 (Wasserstein-1)
**Output:** 2,048 clusters
**Depends on:** `turn_labels.bin`, `turn_ehs_fine.bin`

Because all flop states fit in RAM, no sampling or intermediate disk files are needed.

**Steps:**

1. **Compute data** — `FlopExpander.compute_sample_ehs_mult()` returns histograms, per-state EHS (averaged from `turn_ehs_fine` lookups), and multiplicities for all states in a single parallel pass; pipeline converts histograms to CDFs via cumsum; writes `flop_ehs_fine.bin`
2. **Train centroids** — FAISS K-means (L1) on GPU; centroids are initially unsorted
3. **Assign labels, sort & remap** — GPU search over the in-memory CDF matrix (no streaming needed); sort centroids by multiplicity-weighted per-cluster EHS; remap labels; write `flop_ehs.bin`

---

### Preflop

**Input:** 169 canonical preflop hole-hand classes
**Output:** 169 EHS values (no clustering — state space is already small enough)
**Depends on:** `flop_ehs_fine.bin`

For each of the C(52,2) = 1,326 concrete hole-card pairs, all C(50,3) = 19,600 possible flop boards are enumerated. Each 5-card combination is mapped to a canonical flop index via `FlopIndexer.index()` and looked up in `flop_ehs_fine.bin`. Values are averaged per canonical preflop class (no multiplicity weighting needed — concrete deals are enumerated uniformly).

Output: `preflop_ehs_fine.bin` — 169 × float32 (~676 bytes).

---

## Output Files

```
src/clusters/
├── river_centroids.npy                  # 8192 × 169  float32
├── river_labels.bin                     # 2,428,287,420 × uint16
├── river_ehs_fine.bin                   # 2,428,287,420 × uint16  (decode: value / 65535.0)
├── river_ehs.bin                        # 8192 × float32  (per-cluster EHS, sorted ascending)
├── river_labels_viz.png
├── river_labels_viz_representatives.png
├── river_labels_viz_hands.png
│
├── turn_centroids.npy                   # 8192 × 256  float32 (CDF vectors)
├── turn_labels.bin                      # ~55M × uint16
├── turn_ehs_fine.bin                    # ~55M × uint16  (decode: value / 65535.0)
├── turn_ehs.bin                         # 8192 × float32  (per-cluster EHS, sorted ascending)
├── turn_labels_viz.png
├── turn_labels_viz_representatives.png
├── turn_labels_viz_hands.png
│
├── flop_centroids.npy                   # 2048 × 256  float32 (CDF vectors)
├── flop_labels.bin                      # 1,286,792 × uint16
├── flop_ehs_fine.bin                    # 1,286,792 × uint16  (decode: value / 65535.0)
├── flop_ehs.bin                         # 2048 × float32  (per-cluster EHS, sorted ascending)
├── flop_labels_viz.png
├── flop_labels_viz_representatives.png
├── flop_labels_viz_hands.png
│
└── preflop_ehs_fine.bin                 # 169 × float32  (per-class EHS)
```

---

## C++ Engine (`src/cppsrc/`)

| File | Role |
|------|------|
| `river_expander.cpp/.h` | Enumerates river states; computes 169-dim equity vectors and per-state EHS via OMPEval; exposes `compute_rows` (sampling) and `compute_range_ehs_mult` (streaming) |
| `turn_expander.cpp/.h` | For each turn state, enumerates 46 river cards and builds wide-bucket histogram + per-state EHS from `river_ehs_fine` in a single loop; exposes `compute_rows` and `compute_range_ehs_mult` |
| `flop_expander.cpp/.h` | For each flop state, enumerates 47 turn cards and builds wide-bucket histogram + per-state EHS from `turn_ehs_fine` in a single loop; exposes `compute_rows` and `compute_rows_ehs_mult` |
| `bindings.cpp` | PyBind11 bindings exposing `RiverExpander`, `TurnExpander`, `FlopExpander` (and indexers) as the `hand_indexer` module |

All expanders are multi-threaded via OpenMP. Each `computeRowEhsMult` private method runs one inner loop that fills the equity/histogram vector, the scalar EHS, and the multiplicity simultaneously — no redundant passes per state.

---

## Third-Party Libraries

- **OMPEval** (`src/third_party/OMPEval/`): Fast 5–7 card poker hand evaluator by [zekyll](https://github.com/zekyll/OMPEval). Used by `river_expander.cpp` for showdown equity computation.
- **hand-isomorphism** (`src/third_party/hand-isomorphism/`): Isomorphic hand indexing by [kdub0](https://github.com/kdub0/hand-isomorphism). Maps suit-equivalent hands to canonical indices, reducing the state space.
