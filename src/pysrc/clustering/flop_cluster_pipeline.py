#!/usr/bin/env python3
"""
Clustering pipeline for flop equity states.

Orchestrates the full K-means clustering workflow:
  1. Compute wide-bucket CDF vectors, per-state EHS, and multiplicities
     for all 1,286,792 flop states into RAM; write flop_ehs_fine.bin
  2. Train K-means centroids (L1 / EMD distance) on the full CDF dataset
  3. Assign cluster labels; sort centroids by true multiplicity-weighted
     per-cluster EHS; remap labels; write final output files

Each flop state is represented as a 256-dim float32 CDF vector:
  - The C++ FlopExpander produces a uint8[256] count histogram over 256 wide
    buckets (each wide bucket spans 32 consecutive turn fine buckets).
  - Counts sum to 47 (one per possible turn card).
  - The pipeline computes the cumulative sum of counts (NOT divided by 47),
    yielding a CDF vector with values in [0, 47].  L1 distance between CDFs is
    equivalent to the Earth Mover's Distance (Wasserstein-1).
  - Centroids are sorted by the multiplicity-weighted average EHS of their
    members, computed during step 3.

Because all 1,286,792 states fit comfortably in RAM (~1.31 GB as float32 CDFs),
no sampling or intermediate disk file is needed.  The pipeline runs in 3 steps.

The FlopExpander loads both turn_labels.bin (~110 MB) and turn_ehs_fine.bin
(~110 MB) simultaneously — total footprint ~220 MB.

GPU (FAISS) is required.

Requires the hand_indexer pybind module to be built first:
    cd src/ && pip install -e . --no-build-isolation

Requires turn_labels.bin and turn_ehs_fine.bin from the upstream turn pipeline.

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
from clusterer_lib import (
    METRIC_L1,
    assign_labels,
    copy_outputs,
    encode_ehs,
    gpu_available,
    remap_labels_inplace,
    train_centroids,
    weighted_cluster_ehs,
)

# ---------------------------------------------------------------------------
# Fixed paths
# ---------------------------------------------------------------------------

OUTPUT_DIR = Path(__file__).parent.parent.parent / "clusters"
TMP_OUTPUT_DIR = Path("/tmp") / "tiltstack-clusters"
TURN_LABELS_PATH = OUTPUT_DIR / "turn_labels.bin"
TURN_EHS_FINE_PATH = OUTPUT_DIR / "turn_ehs_fine.bin"
CENTROIDS_PATH = TMP_OUTPUT_DIR / "flop_centroids.npy"
EHS_PATH = TMP_OUTPUT_DIR / "flop_ehs.bin"
EHS_FINE_PATH = TMP_OUTPUT_DIR / "flop_ehs_fine.bin"
LABELS_PATH = TMP_OUTPUT_DIR / "flop_labels.bin"


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def set_tmp_output_dir(tmpdir: Path):
    """Set temp output paths from the user-provided temp root."""
    global TMP_OUTPUT_DIR
    global CENTROIDS_PATH, EHS_PATH, EHS_FINE_PATH, LABELS_PATH

    TMP_OUTPUT_DIR = Path(tmpdir) / "tiltstack-clusters"
    CENTROIDS_PATH = TMP_OUTPUT_DIR / "flop_centroids.npy"
    EHS_PATH = TMP_OUTPUT_DIR / "flop_ehs.bin"
    EHS_FINE_PATH = TMP_OUTPUT_DIR / "flop_ehs_fine.bin"
    LABELS_PATH = TMP_OUTPUT_DIR / "flop_labels.bin"


class FlopClusterPipeline:
    """Orchestrates the flop clustering pipeline."""

    def __init__(
        self, k: int, niter: int, seed: int, threads: int, verbose: bool = True
    ):
        self.k = k
        self.niter = niter
        self.seed = seed
        self.threads = threads
        self.verbose = verbose
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        TMP_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        for path, name in [
            (TURN_LABELS_PATH, "turn_labels.bin"),
            (TURN_EHS_FINE_PATH, "turn_ehs_fine.bin"),
        ]:
            if not path.exists():
                raise FileNotFoundError(
                    f"{name} not found at {path}. "
                    "Run the upstream turn clustering pipeline first."
                )

    def log(self, msg: str):
        if self.verbose:
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", file=sys.stderr)

    def _set_thread_env(self):
        for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
            os.environ[var] = str(self.threads)

    def _make_expander(self):
        self.log(
            "  Loading turn_labels.bin (~110 MB) and "
            "turn_ehs_fine.bin (~110 MB) into RAM..."
        )
        return hand_indexer.FlopExpander(str(TURN_LABELS_PATH), str(TURN_EHS_FINE_PATH))

    # ------------------------------------------------------------------
    # Pipeline steps
    # ------------------------------------------------------------------

    def step_compute_data(self):
        """Compute CDF vectors, EHS, and multiplicities for all flop states."""
        self.log("==> Step 1/3: Computing data for all flop states...")
        self._set_thread_env()
        expander = self._make_expander()
        num_states = expander.num_states()
        self.log(f"  Flop states: {num_states:,}")

        t0 = time.time()
        indices = np.arange(num_states, dtype=np.uint64)

        hist, all_ehs, all_mult = expander.compute_sample_ehs_mult(indices)
        all_cdfs = np.cumsum(hist.astype(np.float32), axis=1)
        self.log(
            f"  CDF matrix: {all_cdfs.shape[0]:,} x {all_cdfs.shape[1]}  "
            f"({all_cdfs.nbytes / 1e9:.2f} GB, {time.time() - t0:.1f}s)"
        )

        encode_ehs(all_ehs).tofile(EHS_FINE_PATH)
        self.log(f"  EHS fine saved: {EHS_FINE_PATH}")

        return all_cdfs, all_ehs, all_mult

    def step_train_centroids(self, all_cdfs: np.ndarray) -> np.ndarray:
        """Train K-means centroids (L1) on the full CDF matrix (unsorted)."""
        self.log(
            f"==> Step 2/3: Training K={self.k:,} centroids "
            f"({self.niter} iterations, L1)..."
        )
        centroids = train_centroids(
            all_cdfs,
            self.k,
            self.niter,
            self.seed,
            metric=METRIC_L1,
            label="flop centroids",
        )
        np.save(CENTROIDS_PATH, centroids)
        self.log(f"  Centroids saved: {CENTROIDS_PATH}")
        return centroids

    def step_assign_sort_remap(
        self,
        all_cdfs: np.ndarray,
        all_ehs: np.ndarray,
        all_mult: np.ndarray,
        centroids: np.ndarray,
    ) -> None:
        """Assign labels, sort by true per-cluster EHS, remap, write output files."""
        self.log("==> Step 3/3: Assigning labels, sorting, and remapping...")
        k = len(centroids)

        labels = assign_labels(
            all_cdfs, centroids, str(LABELS_PATH), metric=METRIC_L1, label="flop"
        )

        ehs_sum, wt_sum = weighted_cluster_ehs(labels, all_ehs, all_mult, k)
        per_cluster_ehs = (ehs_sum / np.maximum(wt_sum, 1.0)).astype(np.float32)
        self.log(
            f"  Per-cluster EHS range: "
            f"[{per_cluster_ehs.min():.4f}, {per_cluster_ehs.max():.4f}]"
        )

        sort_order = np.argsort(per_cluster_ehs)
        centroids = centroids[sort_order]
        per_cluster_ehs[sort_order].tofile(EHS_PATH)
        np.save(CENTROIDS_PATH, centroids)

        self.log("  Remapping labels...")
        remap_labels_inplace(str(LABELS_PATH), sort_order)

        self.log(f"  Centroids saved: {CENTROIDS_PATH}")
        self.log(f"  EHS saved:       {EHS_PATH}")
        self.log(f"  Labels saved:    {LABELS_PATH}")

    def run(self):
        """Execute the full pipeline."""
        if not gpu_available():
            sys.exit(
                "Error: No FAISS GPU support detected. A GPU is required to run this pipeline."
            )
        start = time.time()
        self.log("Starting flop clustering pipeline")
        self.log(
            f"Parameters: K={self.k:,}, niter={self.niter}, threads={self.threads}"
        )

        all_cdfs, all_ehs, all_mult = self.step_compute_data()
        centroids = self.step_train_centroids(all_cdfs)
        self.step_assign_sort_remap(all_cdfs, all_ehs, all_mult, centroids)

        final_paths = copy_outputs(
            (CENTROIDS_PATH, EHS_PATH, EHS_FINE_PATH, LABELS_PATH), OUTPUT_DIR
        )

        elapsed = time.time() - start
        self.log(
            f"==> Pipeline complete in {elapsed / 60:.1f} minutes; "
            f"wrote {len(final_paths)} files to {OUTPUT_DIR}"
        )


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
    parser.add_argument(
        "-k",
        "--clusters",
        type=int,
        default=2_048,
        help="Number of clusters (default: 2048)",
    )
    parser.add_argument(
        "-i", "--niter", type=int, default=25, help="K-means iterations (default: 25)"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed (default: 42)"
    )
    parser.add_argument(
        "-t", "--threads", type=int, default=16, help="OMP thread count (default: 16)"
    )
    parser.add_argument(
        "-q", "--quiet", action="store_true", help="Suppress status messages"
    )
    parser.add_argument(
        "--tmpdir",
        type=Path,
        default=Path("/tmp"),
        help="Temporary directory root (default: /tmp)",
    )

    args = parser.parse_args()
    set_tmp_output_dir(args.tmpdir)

    FlopClusterPipeline(
        k=args.clusters,
        niter=args.niter,
        seed=args.seed,
        threads=args.threads,
        verbose=not args.quiet,
    ).run()


if __name__ == "__main__":
    main()
