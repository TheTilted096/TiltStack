# ---------------------------------------------------------------------------
#  src/Makefile
#
#  Builds the hand_indexer pybind11 module.
#
#  Usage (from src/):
#    make                  # build hand_indexer pybind module
#    make clean            # remove build/ artifacts and pysrc/hand_indexer.*
#    make clean-hard       # also remove clusters/
#
#  Clustering pipeline:
#    python pysrc/river_cluster_pipeline.py [options]
#
#  See pysrc/river_cluster_pipeline.py for pipeline parameters.
# ---------------------------------------------------------------------------

# OS-specific commands
ifeq ($(OS),Windows_NT)
  SHELL        = cmd.exe
  PYTHON       = py
  MKDIR_BUILD  = if not exist build mkdir build
  COPY_PYBIND  = copy /Y build\hand_indexer*.pyd pysrc\ >nul 2>&1 || echo.
else
  PYTHON       = python3
  MKDIR_BUILD  = mkdir -p build
  COPY_PYBIND  = cp -f build/hand_indexer*.so pysrc/
endif

# ---- Build targets --------------------------------------------------------

# GTest headers and libs are provided by the active conda environment.
GTEST_INC = $(CONDA_PREFIX)/include
GTEST_LIB = $(CONDA_PREFIX)/lib

.PHONY: all pybind test clean clean-hard


all: pybind

pybind: | build
	$(PYTHON) setup.py build_ext --build-lib=build --build-temp=build/temp
	$(COPY_PYBIND)

test: | build
	gcc -O2 \
	    -I third_party/hand-isomorphism \
	    -c third_party/hand-isomorphism/hand_index.c -o build/hand_index.o
	gcc -O2 \
	    -I third_party/hand-isomorphism \
	    -c third_party/hand-isomorphism/deck.c -o build/deck.o
	g++ -std=c++20 -O2 \
	    $(wildcard cppsrc/test/*.cpp) \
	    cppsrc/CFRGame.cpp \
	    cppsrc/DeepCFR.cpp \
	    cppsrc/Scheduler.cpp \
	    third_party/OMPEval/HandEvaluator.cpp \
	    build/hand_index.o build/deck.o \
	    -I cppsrc \
	    -I third_party/OMPEval \
	    -I third_party/hand-isomorphism \
	    -I $(GTEST_INC) \
	    -L $(GTEST_LIB) -lgtest -lgtest_main -lpthread \
	    -Wl,-rpath,$(GTEST_LIB) \
	    -o build/cfr_tests
	./build/cfr_tests

build:
	$(MKDIR_BUILD)

# ---- Clean targets ---------------------------------------------------------

ifeq ($(OS),Windows_NT)
clean:
	-if exist build rmdir /S /Q build
	-if exist hand_indexer.egg-info rmdir /S /Q hand_indexer.egg-info
	-del /Q pysrc\hand_indexer.* 2>nul

clean-hard:
	-if exist build rmdir /S /Q build
	-if exist clusters rmdir /S /Q clusters
	-if exist hand_indexer.egg-info rmdir /S /Q hand_indexer.egg-info
	-del /Q pysrc\hand_indexer.* 2>nul
else
clean:
	rm -rf build *.egg-info pysrc/hand_indexer.*

clean-hard:
	rm -rf build clusters *.egg-info pysrc/hand_indexer.*
endif
