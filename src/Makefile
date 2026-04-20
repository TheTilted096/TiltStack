# ---------------------------------------------------------------------------
#  src/Makefile
#
#  Targets
#  -------
#    make hand_indexer   Build the hand_indexer pybind11 module
#    make deepcfr        Build the deepcfr pybind11 module
#    make all            Build both modules (default)
#    make test           Compile and run C++ unit tests via CTest
#    make clean          Remove build/ artifacts and pysrc/ .so files
#    make clean-hard     Also remove clusters/
#
#  Quick-start
#  -----------
#    cd src/
#    make                              # configures + builds hand_indexer + deepcfr
#    uv run python pysrc/NLHE_Trainer.py
#
#  Smoke-test
#    uv run python pysrc/NLHE_Trainer.py \
#        --iterations 1 --threads 1 --total-samples 10000 \
#        --clusters-dir clusters
#
#  Clustering pipeline (run once; outputs go to clusters/)
#    uv run python pysrc/preflop_ehs_pipeline.py
#    uv run python pysrc/flop_cluster_pipeline.py
#    uv run python pysrc/turn_cluster_pipeline.py
#    uv run python pysrc/river_cluster_pipeline.py
# ---------------------------------------------------------------------------

BUILD_DIR  = build
PYTHON_EXE := $(shell uv run python -c "import sys; print(sys.executable)")

.PHONY: all hand_indexer deepcfr test clean clean-hard tidy

all: hand_indexer deepcfr

$(BUILD_DIR)/CMakeCache.txt:
	cmake -S . -B $(BUILD_DIR) -DCMAKE_BUILD_TYPE=Release \
	    -DPython_EXECUTABLE=$(PYTHON_EXE) \
	    -DPython3_EXECUTABLE=$(PYTHON_EXE)

hand_indexer: $(BUILD_DIR)/CMakeCache.txt
	cmake --build $(BUILD_DIR) --target hand_indexer

deepcfr: $(BUILD_DIR)/CMakeCache.txt
	cmake --build $(BUILD_DIR) --target deepcfr

test: $(BUILD_DIR)/CMakeCache.txt
	cmake --build $(BUILD_DIR) --target cfr_tests
	ctest --test-dir $(BUILD_DIR) --output-on-failure

tidy:
	find . -name "*.h"   -exec clang-format -i -style="{BasedOnStyle: llvm, IndentWidth: 4}" {} +
	find . -name "*.cpp" -exec clang-format -i -style="{BasedOnStyle: llvm, IndentWidth: 4}" {} +
	uv run ruff format pysrc/

# ---------------------------------------------------------------------------
clean:
	rm -rf $(BUILD_DIR) pysrc/clustering/hand_indexer.* pysrc/deepcfr/deepcfr.*

clean-hard:
	rm -rf $(BUILD_DIR) runs .ruff_cache clusters checkpoints br_checkpoints pysrc/clustering/hand_indexer.* pysrc/deepcfr/deepcfr.*
