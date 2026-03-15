# ---------------------------------------------------------------------------
#  src/Makefile
#
#  Builds the hand_indexer pybind11 module.
#
#  Usage (from src/):
#    make                  # build hand_indexer pybind module
#    make clean            # remove build/ artifacts and pysrc/hand_indexer.*
#    make clean-hard       # also remove output/
#
#  Clustering pipeline:
#    python pysrc/cluster_pipeline.py [options]
#
#  See pysrc/cluster_pipeline.py for pipeline parameters.
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

.PHONY: all pybind clean clean-hard

all: pybind

pybind: | build
	$(PYTHON) setup.py build_ext --build-lib=build --build-temp=build/temp
	$(COPY_PYBIND)

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
	-if exist output rmdir /S /Q output
	-if exist hand_indexer.egg-info rmdir /S /Q hand_indexer.egg-info
	-del /Q pysrc\hand_indexer.* 2>nul
else
clean:
	rm -rf build *.egg-info pysrc/hand_indexer.*

clean-hard:
	rm -rf build output *.egg-info pysrc/hand_indexer.*
endif
