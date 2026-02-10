# Quick Start

## Initial Setup

Install required dependencies:
```bash
make install
```

This installs `pybind11` via pip. You also need a C++ compiler:
- **Windows**: MSVC (install via Visual Studio Build Tools)
- **Linux/Mac**: g++ or clang++ (usually pre-installed)

## Build and Run

The Makefile works on both Linux and Windows:

```bash
make          # Build the leducsolver module, copy to src/pysrc/
make test     # Build and run the Leduc CFR solver
make clean    # Remove build artifacts
```

After building, run Python scripts directly:
```bash
cd src/pysrc
python Leduc.py   # On Linux: python3
```

## Project Structure

```
TiltStack/
├── Makefile              # Build commands
├── setup.py              # C++ compilation config
├── build/                # Build artifacts (auto-created)
└── src/
    ├── cppsrc/           # C++ source
    │   ├── Node.h        # Game tree types and declarations
    │   ├── Node.cpp      # Node/NodeInfo implementation
    │   ├── Leduc.h       # LeducSolver declaration
    │   ├── Leduc.cpp     # CFR solver implementation
    │   └── bindings.cpp  # PyBind11 wrappers (leducsolver module)
    └── pysrc/            # Python source
        ├── leducsolver.pyd  # Compiled module (copied here after build)
        └── Leduc.py         # Training loop and output
```

## Workflow

1. Edit C++ solver code in `src/cppsrc/`
2. Run `make`
3. Run `python src/pysrc/Leduc.py`

## Adding C++ Bindings

**src/cppsrc/bindings.cpp:**
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

**Python usage:**
```python
from leducsolver import LeducSolver, Rank
solver = LeducSolver()
solver.cfr([Rank.JACK, Rank.QUEEN, Rank.KING], 0, [1.0, 1.0])
```
