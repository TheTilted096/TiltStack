# ---------------------------------------------------------------------------
#  demos/leduc/Makefile
#
#  Targets
#  -------
#    make            Build the leducsolver C++ extension (default)
#    make install    Install Python demo dependencies
#    make test       Train CFR+ for 100k iterations
#    make demo       Launch the Streamlit interactive demo
#    make clean      Remove build/ artifacts and compiled .so files
#
#  Quick-start
#  -----------
#    cd demos/leduc/
#    make install   # install streamlit, plotly, rlcard, numpy, matplotlib
#    make           # compile leducsolver C++ extension
#    make demo      # launch Streamlit UI at localhost:8501
# ---------------------------------------------------------------------------

BUILD_DIR = build

# Detect the Python interpreter: prefer uv-managed venv, fall back to
# python3.11 or python3 for environments without a full uv sync (e.g. macOS
# without CUDA, where faiss-gpu-cu12 cannot install).
PYTHON_EXE := $(shell uv run python -c "import sys; print(sys.executable)" 2>/dev/null \
              || python3.11 -c "import sys; print(sys.executable)" 2>/dev/null \
              || python3 -c "import sys; print(sys.executable)")

.PHONY: build install test demo clean

build: $(BUILD_DIR)/CMakeCache.txt
	cmake --build $(BUILD_DIR) --target leducsolver

$(BUILD_DIR)/CMakeCache.txt:
	cmake -S . -B $(BUILD_DIR) -DCMAKE_BUILD_TYPE=Release \
	    -DPython_EXECUTABLE=$(PYTHON_EXE) \
	    -DPython3_EXECUTABLE=$(PYTHON_EXE)

# Install only the packages required for the demo.  Uses `uv pip install`
# instead of `uv sync` so it works on macOS (no CUDA / faiss-gpu-cu12).
install:
	$(PYTHON_EXE) -m pip install "streamlit>=1.30.0" "plotly>=5.18.0" \
	    "rlcard>=1.0.5" numpy matplotlib

test: build
	$(PYTHON_EXE) src/pysrc/Leduc.py

demo: build
	$(PYTHON_EXE) -m streamlit run src/showcase/app.py

clean:
	rm -rf $(BUILD_DIR) src/pysrc/*.so src/pysrc/__pycache__ \
	    src/showcase/__pycache__
