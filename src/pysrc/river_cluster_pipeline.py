#!/usr/bin/env python3
"""
Clustering pipeline for river equity states.

Orchestrates the full K-means clustering workflow:
  1. Generate random sample indices
  2. Compute equity vectors for sampled indices
  3. Train K-means centroids on the sample
  4. Stream all states and assign cluster labels

GPU (FAISS) is required — CPU clustering is not supported.

Requires the hand_indexer pybind module to be built first:
    cd src/ && make

Then run:
    python river_cluster_pipeline.py
    python river_cluster_pipeline.py -k 8192 --sample-size 20000000 -t 16

Nathaniel Potter, 03-10-2026
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

import hand_indexer
from river_clusterer import (assign_labels_streaming, gpu_available,
                              sort_centroids_by_ehs, train_centroids)

# ---------------------------------------------------------------------------
# Fixed output paths
# ---------------------------------------------------------------------------

OUTPUT_DIR     = Path("clusters")
INDICES_PATH   = OUTPUT_DIR / "river_sample_indices.bin"
SAMPLE_PATH    = OUTPUT_DIR / "river_sample.npy"
CENTROIDS_PATH = OUTPUT_DIR / "river_centroids.npy"
EHS_PATH       = OUTPUT_DIR / "river_ehs.bin"
LABELS_PATH    = OUTPUT_DIR / "river_labels.bin"

NUM_RIVER_STATES = 2_428_287_420


class ClusterPipeline:
    """Orchestrates the river clustering pipeline."""

    def __init__(self, k: int, sample_size: int, niter: int,
                 seed: int, threads: int, verbose: bool = True):
        self.k           = k
        self.sample_size = sample_size
        self.niter       = niter
        self.seed        = seed
        self.threads     = threads
        self.verbose     = verbose
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    def log(self, msg: str):
        if self.verbose:
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", file=sys.stderr)

    def _set_thread_env(self):
        for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
            os.environ[var] = str(self.threads)

    # ------------------------------------------------------------------
    # Pipeline steps
    # ------------------------------------------------------------------

    def step_generate_indices(self):
        """Generate sorted random sample indices and save to disk."""
        if INDICES_PATH.exists():
            self.log("Indices already exist, skipping generation.")
            return

        self.log(f"==> Step 1/4: Sampling {self.sample_size:,} indices...")
        t0 = time.time()
        rng = np.random.default_rng(self.seed)
        indices = np.sort(
            rng.choice(NUM_RIVER_STATES, size=self.sample_size, replace=False)
        ).astype(np.uint64)
        indices.tofile(INDICES_PATH)
        self.log(f"  {self.sample_size:,} indices written "
                 f"({INDICES_PATH.stat().st_size / 1e6:.1f} MB, "
                 f"{time.time() - t0:.1f}s)")

    def step_compute_sample(self):
        """Compute equity vectors for the sampled indices via pybind."""
        if SAMPLE_PATH.exists():
            self.log("Sample already exists, skipping computation.")
            return

        self.log(f"==> Step 2/4: Computing equity vectors for "
                 f"{self.sample_size:,} samples...")
        self._set_thread_env()
        expander = hand_indexer.RiverExpander()
        indices = np.fromfile(INDICES_PATH, dtype=np.uint64)
        t0 = time.time()
        sample = expander.compute_sample(indices).astype(np.float32)
        np.save(SAMPLE_PATH, sample)
        self.log(f"  Sample: {sample.shape[0]:,} x {sample.shape[1]}  "
                 f"({sample.nbytes / 1e9:.2f} GB, {time.time() - t0:.1f}s)")

    def step_train_centroids(self):
        """Train K-means centroids on the saved sample."""
        if CENTROIDS_PATH.exists():
            self.log("Centroids already exist, skipping training.")
            return

        self.log(f"==> Step 3/4: Training K={self.k:,} centroids "
                 f"({self.niter} iterations)...")
        sample = np.load(SAMPLE_PATH)
        centroids = train_centroids(sample, self.k, self.niter, self.seed)
        centroids = sort_centroids_by_ehs(centroids)
        ehs = (centroids.mean(axis=1) / 255.0).astype(np.float32)
        ehs.tofile(EHS_PATH)
        np.save(CENTROIDS_PATH, centroids)
        self.log(f"  Centroids saved: {CENTROIDS_PATH}")
        self.log(f"  EHS saved:       {EHS_PATH}")

    def step_assign_labels(self):
        """Stream all states and assign cluster labels."""
        if LABELS_PATH.exists():
            self.log("Labels already exist, skipping assignment.")
            return

        self.log("==> Step 4/4: Assigning labels to all states...")
        self._set_thread_env()
        expander  = hand_indexer.RiverExpander()
        centroids = np.load(CENTROIDS_PATH)
        assign_labels_streaming(expander, centroids, str(LABELS_PATH),
                                batch_size=1_000_000)
        self.log(f"  Labels saved: {LABELS_PATH}")

    def run(self):
        """Execute the full pipeline."""
        if not gpu_available():
            sys.exit("Error: No FAISS GPU support detected. A GPU is required to run this pipeline.")
        start = time.time()
        self.log(f"Starting river clustering pipeline")
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
        description="K-means clustering pipeline for river equity states (GPU required).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python river_cluster_pipeline.py
  python river_cluster_pipeline.py -k 8192 --sample-size 20000000 -t 16
        """,
    )
    parser.add_argument("-k", "--clusters", type=int, default=8_192,
                        help="Number of clusters (default: 8192)")
    parser.add_argument("-s", "--sample-size", type=int, default=20_000_000,
                        help="Training sample size (default: 20000000)")
    parser.add_argument("-i", "--niter", type=int, default=25,
                        help="K-means iterations (default: 25)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("-t", "--threads", type=int, default=16,
                        help="OMP thread count (default: 16)")
    parser.add_argument("-q", "--quiet", action="store_true",
                        help="Suppress status messages")

    args = parser.parse_args()

    ClusterPipeline(
        k=args.clusters,
        sample_size=args.sample_size,
        niter=args.niter,
        seed=args.seed,
        threads=args.threads,
        verbose=not args.quiet,
    ).run()


if __name__ == "__main__":
    main()
