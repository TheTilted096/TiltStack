# Quick Start

## Prerequisites

- **Python 3.8+**
- **C++ compiler** with C++20 support (C++11 for `src/`)
  - Windows: MSVC via [Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
  - Linux/Mac: g++ or clang++ (usually pre-installed)

## Leduc Demo (Recommended Starting Point)

### Install & Run

```bash
cd demos/leduc
make install   # Install pybind11 via pip
make build     # Compile C++ extension → leducsolver.pyd/.so
make test      # Train CFR+ solver (100k iterations)
make best      # Compute Best Response exploitability
make clean     # Remove build artifacts and output files
```

After building, run Python scripts directly:
```bash
cd demos/leduc/src/pysrc
python Leduc.py         # Train CFR solver (100k iterations)
python BestResponse.py  # Compute exploitability of trained strategy
```

**Note**: On Linux/Mac, use `python3` instead of `python`.

### Project Structure

```
demos/leduc/
├── Makefile                # Build commands
├── setup.py                # C++ compilation config (C++20, pybind11)
├── requirements.txt        # Python dependencies
├── docs/
│   └── IMPLEMENTATION.md   # Algorithm details and design decisions
└── src/
    ├── cppsrc/             # C++ source
    │   ├── Node.h/cpp           # Game tree types, regret matching
    │   ├── Leduc.h/cpp          # CFR+ solver (alternating updates)
    │   ├── BestResponse.h/cpp   # Exploitability computation
    │   └── bindings.cpp         # PyBind11 wrappers
    └── pysrc/              # Python source
        ├── leducsolver.pyd      # Compiled module (copied after build)
        ├── Leduc.py             # Training loop, strategy output, plotting
        └── BestResponse.py      # Best Response computation wrapper
```

### Workflow

**Training CFR Strategy:**
1. Edit C++ solver code in `demos/leduc/src/cppsrc/` if needed
2. Run `make build` to compile
3. Run `make test` or `python src/pysrc/Leduc.py` to train
4. Check `output/leduc_results.txt` for human-readable strategy
5. Check `output/leduc_strategy.csv` for machine-readable strategy

**Measuring Exploitability:**
1. After training CFR, run `make best` or `python src/pysrc/BestResponse.py`
2. Check `output/br_results.txt` for Best Response strategies and exploitability

### Output Format

Strategy files use player markers `(0)` or `(1)` before each node:
```
(0) J:           -> c:0.91, r:0.09
(1) J:r          -> c:0.79, b:0.16, r:0.05
```

## Hold'em River Clustering Pipeline

### Install Dependencies

```bash
pip install -r requirements.txt   # numpy, faiss-cpu
```

For GPU acceleration, install `faiss-gpu` instead of `faiss-cpu`.

### Build & Run

```bash
cd src
make              # Build river_expander executable
make pybind       # Build hand_indexer Python module
make pipeline     # Run full 3-step clustering pipeline
make clean        # Remove build artifacts
make clean-hard   # Remove build/ and output/ entirely
```

### Pipeline Parameters

Override on the command line:
```bash
make pipeline K=5000 SAMPLE_SIZE=10000000 THREADS=8 GPU=no
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `K` | 30000 | Number of clusters |
| `SAMPLE_SIZE` | 20000000 | Training sample size |
| `NITER` | 25 | K-means iterations |
| `THREADS` | 16 | CPU thread limit |
| `GPU` | auto | GPU acceleration (`yes`/`no`/`auto`) |

### Pipeline Steps

1. **Sample**: Generate random indices → compute equity vectors for sample
2. **Train**: K-means centroids on the sample (FAISS)
3. **Assign**: Stream all 2.4B states, assign each to nearest centroid

### Output Files

- `output/river_centroids.npy` — K×169 float32 centroid matrix
- `output/river_labels.bin` — 2.4B uint16 cluster assignments
- `output/river_labels_viz*.png` — Diagnostic visualizations

## Kuhn Poker Demo

No build step needed — pure Python:
```bash
cd demos/kuhn
python Kuhn.py    # Train vanilla CFR (1000 iterations) with convergence visualization
```

## Adding C++ Bindings (PyBind11)

Example from Leduc (`bindings.cpp`):
```cpp
#include <pybind11/pybind11.h>
#include "Leduc.cpp"
#include "Node.cpp"

namespace py = pybind11;

PYBIND11_MODULE(leducsolver, m) {
    py::class_<LeducSolver>(m, "LeducSolver")
        .def(py::init<>())
        .def("cfr", &LeducSolver::cfr);
}
```

Python usage:
```python
from leducsolver import LeducSolver, Rank
solver = LeducSolver()
solver.cfr([Rank.JACK, Rank.QUEEN, Rank.KING], 0, [1.0, 1.0])
```
