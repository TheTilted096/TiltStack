# PyBind11 Setup

C++ to Python bindings using PyBind11 with a simple Makefile.

## Quick Start

```bash
make          # Build
make test     # Test
make clean    # Clean

# Or run directly after building:
cd src/pysrc
python test_example.py
```

## Project Structure

```
TiltStack/
├── Makefile              # Build system
├── setup.py              # Compiler configuration
├── build/                # Build artifacts
└── src/
    ├── cppsrc/
    │   ├── example.cpp   # C++ implementation
    │   └── bindings.cpp  # PyBind11 bindings
    └── pysrc/
        ├── *.pyd/*.so    # Compiled module (copied here after build)
        └── *.py          # Python scripts
```

## How It Works

1. **`make build`** compiles C++ → `build/example_module.pyd`
2. Module is copied to `src/pysrc/`
3. Python scripts in `src/pysrc/` can `import example_module` directly

This matches the workflow from cmake-based projects but without cmake/conan/poetry.

## Files

### setup.py
Minimal setuptools config for compiling the C++ extension:
```python
from setuptools import setup, Extension
import pybind11

setup(
    name='example_module',
    ext_modules=[Extension(
        'example_module',
        sources=['src/cppsrc/bindings.cpp'],
        include_dirs=[pybind11.get_include(), 'src/cppsrc'],
        language='c++',
        extra_compile_args=['/std:c++17'],  # or '-std=c++17' on Linux
    )],
)
```

### Makefile
```makefile
PYTHON := $(shell python -c "import pybind11; import sys; print(sys.executable)")

build:
	"$(PYTHON)" setup.py build_ext --build-lib=build
	@cp -f build/*.pyd src/pysrc/ 2>/dev/null || cp -f build/*.so src/pysrc/

test: build
	@cd src/pysrc && "$(PYTHON)" test_example.py

clean:
	@rm -rf build
	@rm -f src/pysrc/*.pyd src/pysrc/*.so
```

## Adding New C++ Code

1. Write functions/classes in `src/cppsrc/example.cpp`
2. Add bindings in `src/cppsrc/bindings.cpp`
3. Run `make`
4. Import from Python

Example:
```cpp
// example.cpp
int subtract(int a, int b) { return a - b; }

// bindings.cpp
m.def("subtract", &subtract, "Subtract two integers");
```

```python
import example_module as em
print(em.subtract(10, 3))  # 7
```

## Platform Notes

- **Windows**: Uses MSVC (auto-detected)
- **Linux/Mac**: Uses g++/clang++ (auto-detected)
- The Makefile auto-detects the correct Python with pybind11 installed
