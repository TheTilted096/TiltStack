# ---------------------------------------------------------------------------
#  src/Makefile
#
#  Builds river_expander and runs the 3-step river equity clustering pipeline.
#
#  Usage (from src/):
#    make                  # build river_expander
#    make pipeline         # build + run full pipeline
#    make clean            # remove build artifacts
#    make clean-intermediate  # remove pipeline intermediates (keeps centroids + labels)
#    make clean-output     # remove all pipeline output files
#
#  Tunable parameters (override on command line):
#    THREADS=16            # CPU thread limit (OpenMP + BLAS)
#    GPU=yes               # force GPU on;  GPU=no to force off;  GPU=auto (default)
#    SAMPLE_SIZE=10000000  # training sample size
#    K=30000               # number of clusters
#    NITER=15              # K-means iterations
# ---------------------------------------------------------------------------

CXX      = g++
CC       = gcc
CXXFLAGS = -O3 -std=c++11 -fopenmp -Wall
CFLAGS   = -O3 -std=c99 -Wall

# Directories
CPPSRC      = cppsrc
THIRD_PARTY = third_party
BUILDDIR    = build
OUTDIR      = output

# Executables / scripts
EXPANDER       = $(BUILDDIR)/river_expander
INDICES_SCRIPT = pysrc/generate_sample_indices.py
CLUSTER_SCRIPT = pysrc/river_clusterer.py
PIPE_SCRIPT    = pysrc/pipe_to_npy.py

# Pipeline parameters (override on command line, e.g. make pipeline K=5000)
K           = 30000
SAMPLE_SIZE = 5000000
NITER       = 15
SEED        = 42
GPU         = auto
THREADS     = 8

# Pipeline output paths
INDICES_PATH   = $(OUTDIR)/sample_indices.bin
SAMPLE_PATH    = $(OUTDIR)/river_sample.npy
CENTROIDS_PATH = $(OUTDIR)/river_centroids.npy
LABELS_PATH    = $(OUTDIR)/river_labels.bin

# Object files
OBJS = $(BUILDDIR)/river_expander.o \
       $(BUILDDIR)/HandEvaluator.o \
       $(BUILDDIR)/hand_index.o \
       $(BUILDDIR)/deck.o

# ---- Build targets --------------------------------------------------------

.PHONY: all clean clean-intermediate clean-output pipeline

all: $(EXPANDER)

$(BUILDDIR):
	mkdir -p $(BUILDDIR)

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

clean:
	rm -rf $(BUILDDIR)

clean-intermediate:
	rm -f $(INDICES_PATH) $(SAMPLE_PATH)

clean-output:
	rm -rf $(OUTDIR)

# Thread limit — controls OpenMP (river_expander) and FAISS/BLAS (clusterer)
THREAD_ENV = OMP_NUM_THREADS=$(THREADS) MKL_NUM_THREADS=$(THREADS) OPENBLAS_NUM_THREADS=$(THREADS)

# Resolve GPU flag: "auto" probes at runtime, "yes"/"no" forces
ifeq ($(GPU),auto)
  GPU_FLAG = $(shell python -c "import faiss; print('--gpu' if hasattr(faiss,'StandardGpuResources') and faiss.get_num_gpus()>0 else '')" 2>/dev/null)
else ifeq ($(GPU),yes)
  GPU_FLAG = --gpu
else
  GPU_FLAG =
endif

# ---- Pipeline targets -----------------------------------------------------

pipeline: $(LABELS_PATH)
	@echo "==> Pipeline complete."
	@echo "    Centroids: $(CENTROIDS_PATH)"
	@echo "    Labels:    $(LABELS_PATH)"
	@echo "    (You can delete $(SAMPLE_PATH) and $(INDICES_PATH) to reclaim ~3.4 GB)"

$(OUTDIR):
	mkdir -p $(OUTDIR)

# Step 1a: Generate random sample indices
$(INDICES_PATH): $(INDICES_SCRIPT) | $(OUTDIR)
	@echo "==> Step 1/3: Generating $(SAMPLE_SIZE) random indices..."
	python $< -n $(SAMPLE_SIZE) --seed $(SEED) -o $@

# Step 1b: Compute equity vectors for sampled indices
$(SAMPLE_PATH): $(EXPANDER) $(INDICES_PATH) $(PIPE_SCRIPT) | $(OUTDIR)
	@echo "==> Step 1/3: Computing equity vectors for $(SAMPLE_SIZE) sampled hands..."
	$(THREAD_ENV) $(EXPANDER) --sample $(INDICES_PATH) - | python $(PIPE_SCRIPT) -o $@

# Step 2: Train K-means centroids on the sample
$(CENTROIDS_PATH): $(SAMPLE_PATH) $(CLUSTER_SCRIPT) | $(OUTDIR)
	@echo "==> Step 2/3: Training K=$(K) centroids ($(NITER) iterations, $(THREADS) threads)..."
	$(THREAD_ENV) python $(CLUSTER_SCRIPT) --train-only $(SAMPLE_PATH) -k $(K) -i $(NITER) --seed $(SEED) $(GPU_FLAG) -o $(OUTDIR)/river

# Step 3: Stream all 2.4B states, assign cluster labels
$(LABELS_PATH): $(EXPANDER) $(CENTROIDS_PATH) $(CLUSTER_SCRIPT) | $(OUTDIR)
	@echo "==> Step 3/3: Assigning labels (streaming all states, $(THREADS) threads)..."
	$(THREAD_ENV) $(EXPANDER) - | $(THREAD_ENV) python $(CLUSTER_SCRIPT) --assign-only $(CENTROIDS_PATH) --stdin $(GPU_FLAG) -o $(OUTDIR)/river
