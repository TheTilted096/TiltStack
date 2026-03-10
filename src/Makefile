# ---------------------------------------------------------------------------
#  src/Makefile
#
#  Builds poker equity expanders (river_expander, etc.) and the hand_indexer
#  pybind11 module for Python-side hand lookup.
#
#  Usage (from src/):
#    make                  # build river_expander (default)
#    make pybind           # build hand_indexer Python module
#    make clean            # remove build/ and intermediate build files
#    make clean-hard       # remove build/ and output/ entirely
#
#  Clustering pipeline:
#    python pysrc/cluster_pipeline.py [options]
#
#  See pysrc/cluster_pipeline.py for pipeline parameters.
# ---------------------------------------------------------------------------

CXX      = g++
CC       = gcc
CXXFLAGS = -O3 -std=c++11 -fopenmp -Wall
CFLAGS   = -O3 -std=c99 -Wall

# Directories
CPPSRC      = cppsrc
PYSRC       = pysrc
THIRD_PARTY = third_party
BUILDDIR    = build
OUTDIR      = output

# OS-specific commands
ifeq ($(OS),Windows_NT)
  SHELL        = cmd.exe
  PYTHON       = py
  MKDIR_BUILD  = if not exist $(BUILDDIR) mkdir $(BUILDDIR)
  MKDIR_OUT    = if not exist $(OUTDIR) mkdir $(OUTDIR)
  COPY_PYBIND  = copy /Y $(BUILDDIR)\hand_indexer*.pyd $(PYSRC)\ >nul 2>&1 || echo.
else
  PYTHON       = python3
  MKDIR_BUILD  = mkdir -p $(BUILDDIR)
  MKDIR_OUT    = mkdir -p $(OUTDIR)
  COPY_PYBIND  = cp -f $(BUILDDIR)/hand_indexer*.so $(PYSRC)/
endif

# Build outputs
EXPANDER = $(BUILDDIR)/river_expander

# Object files
OBJS = $(BUILDDIR)/river_expander.o \
       $(BUILDDIR)/HandEvaluator.o \
       $(BUILDDIR)/hand_index.o \
       $(BUILDDIR)/deck.o

# ---- Build targets --------------------------------------------------------

.PHONY: all pybind clean clean-hard

all: $(EXPANDER)

$(BUILDDIR):
	$(MKDIR_BUILD)

$(EXPANDER): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^

$(BUILDDIR)/river_expander.o: $(CPPSRC)/river_expander.cpp | $(BUILDDIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILDDIR)/HandEvaluator.o: $(THIRD_PARTY)/OMPEval/omp/HandEvaluator.cpp | $(BUILDDIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILDDIR)/hand_index.o: $(THIRD_PARTY)/hand-isomorphism/hand_index.c | $(BUILDDIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILDDIR)/deck.o: $(THIRD_PARTY)/hand-isomorphism/deck.c | $(BUILDDIR)
	$(CC) $(CFLAGS) -c $< -o $@

# ---- Pybind11 module (hand_indexer) ----------------------------------------

pybind:
	$(PYTHON) setup.py build_ext --build-lib=$(BUILDDIR) --build-temp=$(BUILDDIR)/temp
	$(COPY_PYBIND)

# ---- Clean targets ---------------------------------------------------------

ifeq ($(OS),Windows_NT)
clean:
	-if exist $(BUILDDIR) rmdir /S /Q $(BUILDDIR)
	-if exist hand_indexer.egg-info rmdir /S /Q hand_indexer.egg-info
	-del /Q $(PYSRC)\hand_indexer.* 2>nul

clean-hard:
	-if exist $(BUILDDIR) rmdir /S /Q $(BUILDDIR)
	-if exist $(OUTDIR) rmdir /S /Q $(OUTDIR)
else
clean:
	rm -rf $(BUILDDIR) *.egg-info $(PYSRC)/hand_indexer.*

clean-hard:
	rm -rf $(BUILDDIR) $(OUTDIR)
endif
