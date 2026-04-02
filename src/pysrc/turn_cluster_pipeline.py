#!/usr/bin/env python3
"""
Clustering pipeline for turn equity states.

Orchestrates the full K-means clustering workflow:
  1. Generate random sample indices
  2. Compute wide-bucket CDF vectors for sampled indices
  3. Train K-means centroids (L1 / EMD distance) on the sample (unsorted)
  4. Stream all states in a single pass: assign cluster labels and compute
     turn_ehs_fine.bin simultaneously
  5. Sort centroids by true multiplicity-weighted per-cluster EHS; remap labels

Each turn state is represented as a 256-dim float32 CDF vector:
  - The C++ TurnExpander produces a uint8[256] count histogram over 256 wide
    buckets (each wide bucket spans 32 consecutive river fine buckets).
  - Counts sum to 46 (one per possible river card).
  - The pipeline computes the cumulative sum of counts (NOT divided by 46),
    yielding a CDF vector with values in [0, 46].  L1 distance between CDFs is
    equivalent to the Earth Mover's Distance (Wasserstein-1).
  - Centroids are sorted by the multiplicity-weighted average EHS of their
    members, computed during step 4.

The TurnExpander loads both river_labels.bin (~4.9 GB) and river_ehs_fine.bin
(~4.65 GB) simultaneously (~9.55 GB total), enabling a single streaming pass
that produces cluster labels and per-state EHS together.

GPU (FAISS) is required.

Requires the hand_indexer pybind module to be built first:
    cd src/ && pip install -e . --no-build-isolation

Requires river_labels.bin and river_ehs_fine.bin from the river clustering pipeline.

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
from turn_clusterer import (assign_turn_labels_and_ehs_fine_streaming,
                             gpu_available, remap_labels_inplace,
                             train_turn_centroids)

# ---------------------------------------------------------------------------
# Fixed paths
# ---------------------------------------------------------------------------

OUTPUT_DIR            = Path("clusters")
RIVER_LABELS_PATH     = OUTPUT_DIR / "river_labels.bin"
RIVER_EHS_FINE_PATH   = OUTPUT_DIR / "river_ehs_fine.bin"
INDICES_PATH          = OUTPUT_DIR / "turn_sample_indices.bin"
SAMPLE_PATH           = OUTPUT_DIR / "turn_sample.npy"
CENTROIDS_PATH        = OUTPUT_DIR / "turn_centroids.npy"
EHS_PATH              = OUTPUT_DIR / "turn_ehs.bin"
EHS_FINE_PATH         = OUTPUT_DIR / "turn_ehs_fine.bin"
LABELS_PATH           = OUTPUT_DIR / "turn_labels.bin"
CLUSTER_EHS_PATH      = OUTPUT_DIR / "turn_cluster_ehs_accum.npy"  # temp


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

        for path, name in [(RIVER_LABELS_PATH,   "river_labels.bin"),
                           (RIVER_EHS_FINE_PATH, "river_ehs_fine.bin")]:
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
        self.log("  Loading river_labels.bin (~4.9 GB) and "
                 "river_ehs_fine.bin (~4.65 GB) into RAM...")
        return hand_indexer.TurnExpander(str(RIVER_LABELS_PATH),
                                         str(RIVER_EHS_FINE_PATH))

    # ------------------------------------------------------------------
    # Pipeline steps
    # ------------------------------------------------------------------

    def step_generate_indices(self):
        """Generate sorted random sample indices and save to disk."""
        if INDICES_PATH.exists():
            self.log("Indices already exist, skipping generation.")
            return

        self.log(f"==> Step 1/5: Sampling {self.sample_size:,} indices...")
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

        self.log(f"==> Step 2/5: Computing CDF vectors for "
                 f"{self.sample_size:,} samples...")
        self._set_thread_env()
        expander = self._make_expander()
        indices  = np.fromfile(INDICES_PATH, dtype=np.uint64)
        t0       = time.time()

        sample = np.cumsum(expander.compute_sample(indices).astype(np.float32), axis=1)
        np.save(SAMPLE_PATH, sample)
        self.log(f"  Sample: {sample.shape[0]:,} x {sample.shape[1]}  "
                 f"({sample.nbytes / 1e9:.2f} GB, {time.time() - t0:.1f}s)")

    def step_train_centroids(self):
        """Train K-means centroids (L1) on the saved sample (unsorted)."""
        if CENTROIDS_PATH.exists():
            self.log("Centroids already exist, skipping training.")
            return

        self.log(f"==> Step 3/5: Training K={self.k:,} centroids "
                 f"({self.niter} iterations, L1)...")
        sample    = np.load(SAMPLE_PATH)
        centroids = train_turn_centroids(sample, self.k, self.niter, self.seed)
        np.save(CENTROIDS_PATH, centroids)
        self.log(f"  Centroids saved: {CENTROIDS_PATH}")

    def step_assign_labels_and_ehs_fine(self):
        """Stream all turn states in a single pass: assign labels and compute EHS fine."""
        if LABELS_PATH.exists() and EHS_FINE_PATH.exists() and CLUSTER_EHS_PATH.exists():
            self.log("Labels and EHS fine already exist, skipping.")
            return

        self.log("==> Step 4/5: Assigning labels and computing EHS fine (single pass)...")
        self._set_thread_env()
        expander  = self._make_expander()
        centroids = np.load(CENTROIDS_PATH)
        per_cluster_ehs = assign_turn_labels_and_ehs_fine_streaming(
            expander, centroids, str(LABELS_PATH), str(EHS_FINE_PATH),
            batch_size=1_000_000)
        np.save(CLUSTER_EHS_PATH, per_cluster_ehs)
        self.log(f"  Labels saved: {LABELS_PATH}")
        self.log(f"  EHS fine saved: {EHS_FINE_PATH}")
        self.log(f"  Per-cluster EHS range: "
                 f"[{per_cluster_ehs.min():.4f}, {per_cluster_ehs.max():.4f}]")

    def step_sort_by_true_ehs(self):
        """Sort centroids by weighted per-cluster EHS; remap labels; write turn_ehs.bin."""
        if EHS_PATH.exists():
            self.log("EHS file already exists, skipping sort step.")
            return

        self.log("==> Step 5/5: Sorting by true EHS and remapping labels...")
        centroids       = np.load(CENTROIDS_PATH)
        per_cluster_ehs = np.load(CLUSTER_EHS_PATH)

        sort_order = np.argsort(per_cluster_ehs)
        centroids  = centroids[sort_order]
        per_cluster_ehs[sort_order].astype(np.float32).tofile(EHS_PATH)
        np.save(CENTROIDS_PATH, centroids)

        self.log("  Remapping labels...")
        remap_labels_inplace(str(LABELS_PATH), sort_order)

        CLUSTER_EHS_PATH.unlink()
        self.log(f"  Centroids saved: {CENTROIDS_PATH}")
        self.log(f"  EHS saved:       {EHS_PATH}")
        self.log(f"  Labels remapped: {LABELS_PATH}")

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
        self.step_assign_labels_and_ehs_fine()
        self.step_sort_by_true_ehs()

        self.log("  Cleaning up intermediate sample files...")
        for path in (SAMPLE_PATH, INDICES_PATH):
            if path.exists():
                path.unlink()
                self.log(f"  Deleted: {path}")

        elapsed = time.time() - start
        self.log(f"==> Pipeline complete in {elapsed / 60:.1f} minutes")
        self.log(f"Centroids: {CENTROIDS_PATH}")
        self.log(f"Labels:    {LABELS_PATH}")
        self.log(f"EHS fine:  {EHS_FINE_PATH}")


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
