# ---------------------------------------------------------------------------
#  src/Makefile
#
#  Targets
#  -------
#    make hand_indexer   Build the hand_indexer pybind11 module and copy to pysrc/
#    make deepcfr        Build the deepcfr pybind11 module and copy to pysrc/
#    make all            Build both modules (default)
#    make test           Compile and run C++ unit tests
#    make clean          Remove build/ artifacts and pysrc/ .so files
#    make clean-hard     Also remove clusters/
#
#  Quick-start
#  -----------
#    cd src/
#    make                              # builds hand_indexer + deepcfr
#    python pysrc/NLHE_Trainer.py      # pilot run (default: 20 iterations)
#
#  Smoke-test (fast sanity check before a full run)
#    python pysrc/NLHE_Trainer.py \
#        --iterations 1 --threads 1 --total-samples 10000 \
#        --clusters-dir clusters
#
#  Clustering pipeline (run once; outputs go to clusters/)
#    python pysrc/preflop_ehs_pipeline.py
#    python pysrc/flop_cluster_pipeline.py
#    python pysrc/turn_cluster_pipeline.py
#    python pysrc/river_cluster_pipeline.py
# ---------------------------------------------------------------------------

PYTHON      = python3
MKDIR_BUILD = mkdir -p build

# GTest headers and libs (provided by the active conda environment).
GTEST_INC = $(CONDA_PREFIX)/include
GTEST_LIB = $(CONDA_PREFIX)/lib

.PHONY: all hand_indexer deepcfr test clean clean-hard

all: hand_indexer deepcfr

hand_indexer: | build
	$(PYTHON) setup.py build_ext --build-lib=build --build-temp=build/temp
	cp -f build/hand_indexer*.so pysrc/clustering/

deepcfr: | build
	$(PYTHON) setup_deepcfr.py build_ext --build-lib=build --build-temp=build/temp
	cp -f build/deepcfr*.so pysrc/deepcfr/

test: | build
	gcc -O2 \
	    -I third_party/hand-isomorphism \
	    -c third_party/hand-isomorphism/hand_index.c -o build/hand_index.o
	gcc -O2 \
	    -I third_party/hand-isomorphism \
	    -c third_party/hand-isomorphism/deck.c -o build/deck.o
	g++ -std=c++20 -O2 \
	    $(wildcard cppsrc/test/*.cpp) \
	    cppsrc/deepcfr/CFRGame.cpp \
	    cppsrc/deepcfr/DeepCFR.cpp \
	    cppsrc/deepcfr/Scheduler.cpp \
	    third_party/OMPEval/HandEvaluator.cpp \
	    build/hand_index.o build/deck.o \
	    -I cppsrc/deepcfr \
	    -I third_party/OMPEval \
	    -I third_party/hand-isomorphism \
	    -I $(GTEST_INC) \
	    -L $(GTEST_LIB) -lgtest -lgtest_main -lpthread \
	    -Wl,-rpath,$(GTEST_LIB) \
	    -o build/cfr_tests
	./build/cfr_tests

build:
	$(MKDIR_BUILD)

tidy:
	find . -name "*.h"   -exec clang-format -i -style="{BasedOnStyle: llvm, IndentWidth: 4}" {} +
	find . -name "*.cpp" -exec clang-format -i -style="{BasedOnStyle: llvm, IndentWidth: 4}" {} +

# ---------------------------------------------------------------------------
clean:
	rm -rf build *.egg-info pysrc/clustering/hand_indexer.* pysrc/deepcfr/deepcfr.*

clean-hard:
	rm -rf build clusters checkpoints br_checkpoints *.egg-info pysrc/clustering/hand_indexer.* pysrc/deepcfr/deepcfr.*
