# PyBind11 Quick Start

## Initial Setup

Install required dependencies:
```bash
make install
```

## Build and Run

The Makefile works on both Linux and Windows:

```bash
make          # Build module, copy to src/pysrc/
make test     # Build and run tests
make clean    # Remove build artifacts
make help     # Show available commands
```

After building, run Python scripts directly:
```bash
cd src/pysrc
python test_example.py   # On Linux: python3
```

## Project Structure

```
TiltStack/
├── Makefile              # Build commands
├── setup.py              # C++ compilation config
├── build/                # Build artifacts (auto-created)
└── src/
    ├── cppsrc/           # C++ source
    │   ├── example.cpp   # Implementation
    │   └── bindings.cpp  # PyBind11 wrappers
    └── pysrc/            # Python source
        ├── *.pyd         # Compiled module (copied here)
        └── test_example.py
```

## Workflow

1. Edit C++ in `src/cppsrc/`
2. Run `make`
3. Run `python src/pysrc/your_script.py`

## Adding C++ Functions

**src/cppsrc/example.cpp:**
```cpp
int subtract(int a, int b) { return a - b; }
```

**src/cppsrc/bindings.cpp:**
```cpp
m.def("subtract", &subtract);
```

**Build and use:**
```bash
make
python -c "from src.pysrc import example_module as em; print(em.subtract(10, 3))"
```
