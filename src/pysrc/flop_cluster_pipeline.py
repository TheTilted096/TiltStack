#!/usr/bin/env python3
"""
Clustering pipeline for flop equity states.

Orchestrates the full K-means clustering workflow:
  1. Compute wide-bucket CDF vectors for all 1,286,792 flop states into RAM
  2. Train K-means centroids (L1 / EMD distance) on the full dataset
  3. Assign cluster labels to all states directly from the in-memory CDF matrix

Each flop state is represented as a 256-dim float32 CDF vector:
  - The C++ FlopExpander produces a uint8[256] count histogram over 256 wide
    buckets (each wide bucket spans 32 consecutive turn fine buckets).
  - Counts sum to 47 (one per possible turn card).
  - The pipeline computes the cumulative sum of counts (NOT divided by 47),
    yielding a CDF vector with values in [0, 47].  L1 distance between CDFs is
    equivalent to the Earth Mover's Distance (Wasserstein-1).
  - Centroids are sorted by expected EHS via a two-level wide-bucket average:
    river_centroids -> wide_river_ehs -> E[EHS] per turn centroid ->
    wide_turn_ehs -> E[EHS] per flop centroid.

Because all 1,286,792 states fit comfortably in RAM (~1.31 GB as float32 CDFs),
no sampling or intermediate disk file is needed.  The pipeline runs in 3 steps
(vs. 4 for the turn pipeline).

GPU (FAISS) is required.

Requires the hand_indexer pybind module to be built first:
    cd src/ && pip install -e . --no-build-isolation

Requires turn_labels.bin, turn_centroids.npy, and river_centroids.npy from the
upstream clustering pipelines.

Then run:
    python flop_cluster_pipeline.py
    python flop_cluster_pipeline.py -k 2048 --niter 25 -t 16

Nathaniel Potter, 03-15-2026
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

import hand_indexer
from flop_clusterer import (assign_flop_labels, compute_wide_turn_ehs,
                             gpu_available, sort_flop_centroids,
                             train_flop_centroids)

# ---------------------------------------------------------------------------
# Fixed paths
# ---------------------------------------------------------------------------

OUTPUT_DIR            = Path("output")
TURN_LABELS_PATH      = OUTPUT_DIR / "turn_labels.bin"
TURN_CENTROIDS_PATH   = OUTPUT_DIR / "turn_centroids.npy"
RIVER_CENTROIDS_PATH  = OUTPUT_DIR / "river_centroids.npy"
CENTROIDS_PATH        = OUTPUT_DIR / "flop_centroids.npy"
LABELS_PATH           = OUTPUT_DIR / "flop_labels.bin"


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class FlopClusterPipeline:
    """Orchestrates the flop clustering pipeline."""

    def __init__(self, k: int, niter: int, seed: int, threads: int,
                 verbose: bool = True):
        self.k       = k
        self.niter   = niter
        self.seed    = seed
        self.threads = threads
        self.verbose = verbose
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        for path, name in [(TURN_LABELS_PATH,     "turn_labels.bin"),
                           (TURN_CENTROIDS_PATH,  "turn_centroids.npy"),
                           (RIVER_CENTROIDS_PATH, "river_centroids.npy")]:
            if not path.exists():
                raise FileNotFoundError(
                    f"{name} not found at {path}. "
                    "Run the upstream clustering pipeline first.")

    def log(self, msg: str):
        if self.verbose:
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", file=sys.stderr)

    def _set_thread_env(self):
        for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
            os.environ[var] = str(self.threads)

    def _make_expander(self):
        self.log("  Loading turn_labels.bin into RAM (~110 MB)...")
        return hand_indexer.FlopExpander(str(TURN_LABELS_PATH))

    # ------------------------------------------------------------------
    # Pipeline steps
    # ------------------------------------------------------------------

    def step_compute_cdfs(self):
        """Compute wide-bucket CDF vectors for all flop states into RAM."""
        self.log("==> Step 1/3: Computing CDF vectors for all flop states...")
        self._set_thread_env()
        expander   = self._make_expander()
        num_states = expander.num_states()
        self.log(f"  Flop states: {num_states:,}")

        t0      = time.time()
        indices = np.arange(num_states, dtype=np.uint64)

        # expander returns uint8 counts; compute CDF (cumsum) without normalising
        all_cdfs = np.cumsum(
            expander.compute_sample(indices).astype(np.float32), axis=1
        )
        self.log(f"  CDF matrix: {all_cdfs.shape[0]:,} x {all_cdfs.shape[1]}  "
                 f"({all_cdfs.nbytes / 1e9:.2f} GB, {time.time() - t0:.1f}s)")
        return all_cdfs

    def step_train_centroids(self, all_cdfs: np.ndarray) -> np.ndarray:
        """Train K-means centroids (L1) on the full CDF matrix."""
        if CENTROIDS_PATH.exists():
            self.log("Centroids already exist, skipping training.")
            return np.load(CENTROIDS_PATH)

        self.log(f"==> Step 2/3: Training K={self.k:,} centroids "
                 f"({self.niter} iterations, L1)...")
        turn_centroids  = np.load(TURN_CENTROIDS_PATH)
        river_centroids = np.load(RIVER_CENTROIDS_PATH)
        wide_turn_ehs   = compute_wide_turn_ehs(turn_centroids, river_centroids)
        self.log(f"  Wide turn EHS range: [{wide_turn_ehs.min():.4f}, "
                 f"{wide_turn_ehs.max():.4f}]")

        centroids = train_flop_centroids(all_cdfs, self.k, self.niter, self.seed)
        centroids = sort_flop_centroids(centroids, wide_turn_ehs)
        np.save(CENTROIDS_PATH, centroids)
        self.log(f"  Centroids saved: {CENTROIDS_PATH}")
        return centroids

    def step_assign_labels(self, all_cdfs: np.ndarray,
                           centroids: np.ndarray) -> None:
        """Assign cluster labels to all flop states from the in-memory CDFs."""
        if LABELS_PATH.exists():
            self.log("Labels already exist, skipping assignment.")
            return

        self.log("==> Step 3/3: Assigning labels to all flop states...")
        assign_flop_labels(all_cdfs, centroids, str(LABELS_PATH))
        self.log(f"  Labels saved: {LABELS_PATH}")

    def run(self):
        """Execute the full pipeline."""
        if not gpu_available():
            sys.exit("Error: No FAISS GPU support detected. A GPU is required to run this pipeline.")
        start = time.time()
        self.log("Starting flop clustering pipeline")
        self.log(f"Parameters: K={self.k:,}, niter={self.niter}, threads={self.threads}")

        all_cdfs  = self.step_compute_cdfs()
        centroids = self.step_train_centroids(all_cdfs)
        self.step_assign_labels(all_cdfs, centroids)

        elapsed = time.time() - start
        self.log(f"==> Pipeline complete in {elapsed / 60:.1f} minutes")
        self.log(f"Centroids: {CENTROIDS_PATH}")
        self.log(f"Labels:    {LABELS_PATH}")


def main():
    parser = argparse.ArgumentParser(
        description="K-means clustering pipeline for flop states (L1/EMD, CDF vectors, GPU required).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python flop_cluster_pipeline.py
  python flop_cluster_pipeline.py -k 2048 --niter 25 -t 16
        """,
    )
    parser.add_argument("-k", "--clusters", type=int, default=2_048,
                        help="Number of clusters (default: 2048)")
    parser.add_argument("-i", "--niter", type=int, default=25,
                        help="K-means iterations (default: 25)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("-t", "--threads", type=int, default=16,
                        help="OMP thread count (default: 16)")
    parser.add_argument("-q", "--quiet", action="store_true",
                        help="Suppress status messages")

    args = parser.parse_args()

    FlopClusterPipeline(
        k=args.clusters,
        niter=args.niter,
        seed=args.seed,
        threads=args.threads,
        verbose=not args.quiet,
    ).run()


if __name__ == "__main__":
    main()
