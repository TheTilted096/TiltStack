# PyBind11 Setup

C++ to Python bindings using PyBind11 with a simple Makefile.

## Quick Start

```bash
make install  # Install pybind11
make          # Build
make test     # Build and run solver
make clean    # Clean
```

## How It Works

1. **`make install`** installs `pybind11` via pip
2. **`make build`** compiles C++ into `build/leducsolver.pyd` (Windows) or `build/leducsolver.so` (Linux)
3. The compiled module is copied to `src/pysrc/`
4. Python scripts in `src/pysrc/` can `import leducsolver` directly

## Project Structure

```
TiltStack/
├── Makefile              # Build system
├── setup.py              # Compiler configuration
├── build/                # Build artifacts
└── src/
    ├── cppsrc/
    │   ├── Node.h         # Types: Action, Rank, NodeInfo, Node
    │   ├── Node.cpp        # Game tree implementation
    │   ├── Leduc.h         # LeducSolver declaration
    │   ├── Leduc.cpp       # CFR solver implementation
    │   └── bindings.cpp    # PyBind11 bindings (leducsolver module)
    └── pysrc/
        ├── *.pyd/*.so      # Compiled module (copied here after build)
        └── Leduc.py        # Python training loop and output
```

## Architecture

The `leducsolver` module exposes the C++ solver to Python:

| C++ Type | Python Access | Purpose |
|---|---|---|
| `LeducSolver` | `LeducSolver()` | Node table + `cfr()` method |
| `Node` | `solver[hash]` | Strategy/regret storage per info set |
| `NodeInfo` | `NodeInfo(hash)` | Decode hash into game state |
| `ActionList` | Iterable | Legal actions at a node |
| `Action` | `Action.CHECK`, `.BET`, `.RAISE` | Move enum |
| `Rank` | `Rank.JACK`, `.QUEEN`, `.KING` | Card rank enum |

### Binding Pattern

`bindings.cpp` includes the `.cpp` files directly (single-translation-unit build) and wraps them with pybind11:

```cpp
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "Leduc.cpp"
#include "Node.cpp"

PYBIND11_MODULE(leducsolver, m) {
    py::class_<LeducSolver>(m, "LeducSolver")
        .def(py::init<>())
        .def("cfr", &LeducSolver::cfr)
        .def("__getitem__", ...);  // solver[hash] -> Node&
}
```

### Python Usage

```python
from leducsolver import LeducSolver, NodeInfo, Rank

solver = LeducSolver()

# Run one CFR iteration
solver.cfr([Rank.JACK, Rank.QUEEN, Rank.KING], 0, [8.0, 8.0])

# Read back a strategy
info = NodeInfo(0)
moves = info.moves()
strategy = solver[0].get_stored_strategy(moves)
```

## Compilation

### setup.py

Compiler flags live in `extra_compile_args` in `setup.py`:

```python
extra_compile_args=['/std:c++20', '/O2'] if sys.platform == 'win32' else ['-std=c++20', '-O3'],
```

To add your own flags, append to the appropriate list. Common additions:

| Flag (MSVC) | Flag (GCC/Clang) | Purpose |
|---|---|---|
| `/O2` | `-O3` | Max optimization (already enabled) |
| `/arch:AVX2` | `-mavx2` | AVX2 vectorization |
| `/GL` | `-flto` | Link-time optimization |
| `/fp:fast` | `-ffast-math` | Relaxed floating point |
| `/W4` | `-Wall -Wextra` | Extra warnings |

### Build output

`make` will print a verbose wall of MSVC/GCC flags — this is normal setuptools behavior. The output shows the full compiler invocation including all auto-detected include paths and default flags. There is no built-in way to suppress it.

### Platform notes

- **Windows**: Uses MSVC (auto-detected via Visual Studio Build Tools), compiles to `.pyd`. The Makefile sets `SHELL := cmd.exe` so that `>nul` redirects work correctly (GNU Make on Windows defaults to `sh.exe`, which creates a literal `nul` file instead).
- **Linux/Mac**: Uses g++/clang++ (auto-detected), compiles to `.so`.
- The Makefile auto-detects the platform and uses the correct Python command (`py` on Windows, `python3` elsewhere).

### Precision note

The C++ solver uses `float` (32-bit) for strategy and regret arrays, while the pure-Python reference uses `float` (64-bit double). Over many iterations, small rounding differences accumulate, so results will be close but not bit-identical. Strongly-converged nodes (e.g. mandatory calls) will match exactly.
