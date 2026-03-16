#!/usr/bin/env python3
"""
Clustering pipeline for turn equity states.

Orchestrates the full K-means clustering workflow:
  1. Generate random sample indices
  2. Compute wide-bucket CDF vectors for sampled indices
  3. Train K-means centroids (L1 / EMD distance) on the sample
  4. Stream all states and assign cluster labels

Each turn state is represented as a 256-dim float32 CDF vector:
  - The C++ TurnExpander produces a uint8[256] count histogram over 256 wide
    buckets (each wide bucket spans 32 consecutive river fine buckets).
  - Counts sum to 46 (one per possible river card).
  - The pipeline computes the cumulative sum of counts (NOT divided by 46),
    yielding a CDF vector with values in [0, 46].  L1 distance between CDFs is
    equivalent to the Earth Mover's Distance (Wasserstein-1) on the underlying
    probability distributions, which is more sensitive to distributional ordering
    than L1 on raw PDFs.
  - Centroids are sorted by expected EHS, computed by reverting each CDF centroid
    to a PDF via finite differences and dot-producting with wide bucket EHS values
    derived from river_centroids.npy (which are scaled by 255.0).

GPU (FAISS) is required — CPU clustering is not supported due to infeasible
runtime on the ~55M turn states dataset.

Requires the hand_indexer pybind module to be built first:
    cd src/ && pip install -e . --no-build-isolation

Requires river_labels.bin and river_centroids.npy from the river clustering pipeline.

Then run:
    python turn_cluster_pipeline.py
    python turn_cluster_pipeline.py -k 8192 --sample-size 10000000 -t 16

Nathaniel Potter, 03-15-2026
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

import hand_indexer
from turn_clusterer import (assign_turn_labels_streaming, compute_wide_ehs,
                             gpu_available, sort_turn_centroids, train_turn_centroids)

# ---------------------------------------------------------------------------
# Fixed paths
# ---------------------------------------------------------------------------

OUTPUT_DIR            = Path("output")
RIVER_LABELS_PATH     = OUTPUT_DIR / "river_labels.bin"
RIVER_CENTROIDS_PATH  = OUTPUT_DIR / "river_centroids.npy"
INDICES_PATH          = OUTPUT_DIR / "turn_sample_indices.bin"
SAMPLE_PATH           = OUTPUT_DIR / "turn_sample.npy"
CENTROIDS_PATH        = OUTPUT_DIR / "turn_centroids.npy"
LABELS_PATH           = OUTPUT_DIR / "turn_labels.bin"


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class TurnClusterPipeline:
    """Orchestrates the turn clustering pipeline."""

    def __init__(self, k: int, sample_size: int, niter: int,
                 seed: int, threads: int, verbose: bool = True):
        self.k           = k
        self.sample_size = sample_size
        self.niter       = niter
        self.seed        = seed
        self.threads     = threads
        self.verbose     = verbose
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        for path, name in [(RIVER_LABELS_PATH,    "river_labels.bin"),
                           (RIVER_CENTROIDS_PATH, "river_centroids.npy")]:
            if not path.exists():
                raise FileNotFoundError(
                    f"{name} not found at {path}. "
                    "Run the river clustering pipeline first.")

    def log(self, msg: str):
        if self.verbose:
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", file=sys.stderr)

    def _set_thread_env(self):
        for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
            os.environ[var] = str(self.threads)

    def _make_expander(self):
        self.log("  Loading river_labels.bin into RAM (~4.9 GB)...")
        return hand_indexer.TurnExpander(str(RIVER_LABELS_PATH))

    # ------------------------------------------------------------------
    # Pipeline steps
    # ------------------------------------------------------------------

    def step_generate_indices(self):
        """Generate sorted random sample indices and save to disk."""
        if INDICES_PATH.exists():
            self.log("Indices already exist, skipping generation.")
            return

        self.log(f"==> Step 1/4: Sampling {self.sample_size:,} indices...")
        t0  = time.time()
        expander = self._make_expander()
        num_states = expander.num_states()
        self.log(f"  Turn states: {num_states:,}")

        rng     = np.random.default_rng(self.seed)
        indices = np.sort(
            rng.choice(num_states, size=self.sample_size, replace=False)
        ).astype(np.uint64)
        indices.tofile(INDICES_PATH)
        self.log(f"  {self.sample_size:,} indices written "
                 f"({INDICES_PATH.stat().st_size / 1e6:.1f} MB, "
                 f"{time.time() - t0:.1f}s)")

    def step_compute_sample(self):
        """Compute wide-bucket CDF vectors for the sampled indices."""
        if SAMPLE_PATH.exists():
            self.log("Sample already exists, skipping computation.")
            return

        self.log(f"==> Step 2/4: Computing CDF vectors for "
                 f"{self.sample_size:,} samples...")
        self._set_thread_env()
        expander = self._make_expander()
        indices  = np.fromfile(INDICES_PATH, dtype=np.uint64)
        t0       = time.time()

        # expander returns uint8 counts; compute CDF (cumsum) without normalising
        sample = np.cumsum(expander.compute_sample(indices).astype(np.float32), axis=1)
        np.save(SAMPLE_PATH, sample)
        self.log(f"  Sample: {sample.shape[0]:,} x {sample.shape[1]}  "
                 f"({sample.nbytes / 1e9:.2f} GB, {time.time() - t0:.1f}s)")

    def step_train_centroids(self):
        """Train K-means centroids (L1) on the saved sample."""
        if CENTROIDS_PATH.exists():
            self.log("Centroids already exist, skipping training.")
            return

        self.log(f"==> Step 3/4: Training K={self.k:,} centroids "
                 f"({self.niter} iterations, L1)...")
        sample          = np.load(SAMPLE_PATH)
        river_centroids = np.load(RIVER_CENTROIDS_PATH)
        wide_ehs        = compute_wide_ehs(river_centroids)
        self.log(f"  Wide EHS range: [{wide_ehs.min():.4f}, {wide_ehs.max():.4f}]")

        centroids = train_turn_centroids(sample, self.k, self.niter, self.seed)
        centroids = sort_turn_centroids(centroids, wide_ehs)
        np.save(CENTROIDS_PATH, centroids)
        self.log(f"  Centroids saved: {CENTROIDS_PATH}")

    def step_assign_labels(self):
        """Stream all turn states and assign cluster labels."""
        if LABELS_PATH.exists():
            self.log("Labels already exist, skipping assignment.")
            return

        self.log("==> Step 4/4: Assigning labels to all turn states...")
        self._set_thread_env()
        expander  = self._make_expander()
        centroids = np.load(CENTROIDS_PATH)
        assign_turn_labels_streaming(expander, centroids, str(LABELS_PATH),
                                     batch_size=500_000)
        self.log(f"  Labels saved: {LABELS_PATH}")

    def run(self):
        """Execute the full pipeline."""
        if not gpu_available():
            sys.exit("Error: No FAISS GPU support detected. A GPU is required to run this pipeline.")
        start = time.time()
        self.log("Starting turn clustering pipeline")
        self.log(f"Parameters: K={self.k:,}, sample_size={self.sample_size:,}, "
                 f"niter={self.niter}, threads={self.threads}")

        self.step_generate_indices()
        self.step_compute_sample()
        self.step_train_centroids()
        self.step_assign_labels()

        self.log("  Cleaning up intermediate sample files...")
        for path in (SAMPLE_PATH, INDICES_PATH):
            if path.exists():
                path.unlink()
                self.log(f"  Deleted: {path}")

        elapsed = time.time() - start
        self.log(f"==> Pipeline complete in {elapsed / 60:.1f} minutes")
        self.log(f"Centroids: {CENTROIDS_PATH}")
        self.log(f"Labels:    {LABELS_PATH}")


def main():
    parser = argparse.ArgumentParser(
        description="K-means clustering pipeline for turn states (L1/EMD, CDF vectors, GPU required).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python turn_cluster_pipeline.py
  python turn_cluster_pipeline.py -k 8192 --sample-size 10000000 -t 16
        """,
    )
    parser.add_argument("-k", "--clusters", type=int, default=8_192,
                        help="Number of clusters (default: 8192)")
    parser.add_argument("-s", "--sample-size", type=int, default=10_000_000,
                        help="Training sample size (default: 10000000)")
    parser.add_argument("-i", "--niter", type=int, default=25,
                        help="K-means iterations (default: 25)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("-t", "--threads", type=int, default=16,
                        help="OMP thread count (default: 16)")
    parser.add_argument("-q", "--quiet", action="store_true",
                        help="Suppress status messages")

    args = parser.parse_args()

    TurnClusterPipeline(
        k=args.clusters,
        sample_size=args.sample_size,
        niter=args.niter,
        seed=args.seed,
        threads=args.threads,
        verbose=not args.quiet,
    ).run()


if __name__ == "__main__":
    main()
