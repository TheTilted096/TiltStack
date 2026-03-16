"""
K-means clustering for turn wide-bucket histogram vectors.

Library functions used by turn_cluster_pipeline.py:
  train_turn_centroids(sample, k, niter, seed)  -> np.ndarray  (K, 256) float32
  compute_wide_ehs(river_centroids)              -> np.ndarray  (256,) float32
  sort_turn_centroids(centroids, wide_ehs)       -> np.ndarray  sorted copy
  assign_turn_labels_streaming(expander, centroids, out_path, batch_size)

Each turn state is represented as a 256-dim float32 CDF vector:
  - The C++ TurnExpander produces a uint8[256] count histogram over 256 wide
    buckets (each wide bucket spans 32 consecutive river fine buckets).
  - Counts sum to 46 (one per possible river card).
  - The CDF is the cumulative sum of these counts (NOT divided by 46), yielding
    values in [0, 46].
  - L1 distance between CDFs equals the Earth Mover's Distance (Wasserstein-1)
    on the underlying probability distributions, making clustering more sensitive
    to distributional ordering than L1 on raw PDFs.
  - Centroids are sorted by expected EHS computed via PDF (finite differences of
    the CDF) dot-producted with wide bucket EHS values from river_centroids.npy.

GPU (FAISS) is required — CPU clustering is not supported due to infeasible
runtime on the ~55M turn states dataset.

Nathaniel Potter, 03-15-2026
"""

import sys
import time

import faiss
import numpy as np

# Number of possible river cards per turn state (counts sum to this value).
RIVER_CARDS_PER_TURN = 46


# ---------------------------------------------------------------------------
# Public library API
# ---------------------------------------------------------------------------

def gpu_available() -> bool:
    """Check whether FAISS GPU support is installed and a GPU is visible."""
    try:
        return hasattr(faiss, 'StandardGpuResources') and faiss.get_num_gpus() > 0
    except Exception:
        return False


def train_turn_centroids(sample: np.ndarray, k: int, niter: int,
                         seed: int) -> np.ndarray:
    """Train K-means centroids on float32 CDF vectors with L1 distance on GPU.

    L1 distance between CDFs is equivalent to the Earth Mover's Distance
    (Wasserstein-1) on the underlying probability distributions.

    Args:
        sample:  (N, 256) float32 CDF vectors (cumsum of counts, NOT divided by
                 46.0, so values range from 0 to 46).
        k:       Number of clusters.
        niter:   K-means iterations.
        seed:    Random seed.

    Returns:
        (k, 256) float32 centroid matrix (CDF vectors, values in [0, 46]).
    """
    d = sample.shape[1]
    print(f"Training K={k:,} turn centroids, {niter} iterations on GPU "
          f"({sample.shape[0]:,} x {d} vectors, L1)...", file=sys.stderr)

    clus = faiss.Clustering(d, k)
    clus.niter   = niter
    clus.verbose = True
    clus.seed    = seed
    clus.max_points_per_centroid = sample.shape[0] // k + 1

    cpu_index = faiss.IndexFlat(d, faiss.METRIC_L1)
    res       = faiss.StandardGpuResources()
    index     = faiss.index_cpu_to_gpu(res, 0, cpu_index)

    t0 = time.time()
    clus.train(sample, index)
    print(f"K-means training done in {time.time() - t0:.1f}s", file=sys.stderr)
    return faiss.vector_to_array(clus.centroids).reshape(k, d).copy()


def compute_wide_ehs(river_centroids: np.ndarray) -> np.ndarray:
    """Compute the average EHS for each of the 256 wide buckets.

    River centroids are (8192, 169) float32 equity vectors sorted weakest-first.
    The mean across the 169 opponent buckets gives each cluster's EHS.
    Grouping consecutive 32 clusters into one wide bucket and averaging yields
    a (256,) array that maps wide-bucket index -> expected EHS.

    Args:
        river_centroids: (8192, 169) sorted river centroid matrix.

    Returns:
        (256,) float32 array of average EHS per wide bucket.
    """
    fine_ehs = river_centroids.mean(axis=1)          # (8192,) EHS per river cluster
    return fine_ehs.reshape(256, 32).mean(axis=1)    # (256,)  EHS per wide bucket


def sort_turn_centroids(centroids: np.ndarray, wide_ehs: np.ndarray) -> np.ndarray:
    """Return centroids sorted by expected EHS (ascending = weakest first).

    Centroids are CDF vectors; we revert to PDF via finite differences before
    computing E[EHS] so the dot product yields a true probability-weighted sum.

    E[EHS] = sum_i( pdf[i] * wide_ehs[i] )   where pdf = diff(cdf)

    Args:
        centroids: (K, 256) float32 turn centroid matrix (CDF vectors, values
                   in [0, 46], NOT divided by 46.0).
        wide_ehs:  (256,) float32 average EHS per wide bucket from compute_wide_ehs().

    Returns:
        Sorted copy of centroids with label 0 = lowest expected EHS (weakest).
    """
    # Convert CDF → PDF (differences; first bucket equals first CDF value)
    pdf = np.hstack([centroids[:, :1], np.diff(centroids, axis=1)])  # (K, 256)
    expected_ehs = pdf @ wide_ehs    # (K,) – proportional to E[EHS], sufficient for ordering
    return centroids[np.argsort(expected_ehs)]


def assign_turn_labels_streaming(expander, centroids: np.ndarray, out_path: str,
                                 batch_size: int = 500_000) -> None:
    """Assign turn cluster labels to all states via the TurnExpander on GPU.

    Converts each uint8 count batch to a CDF (cumsum, NOT divided by 46),
    then searches with L1 distance (equivalent to Earth Mover's Distance).
    Writes raw uint16 labels to out_path.
    """
    k, d = centroids.shape
    assert k <= 65535, "K must fit in uint16"

    cpu_index = faiss.IndexFlat(d, faiss.METRIC_L1)
    cpu_index.add(centroids)
    res   = faiss.StandardGpuResources()
    index = faiss.index_cpu_to_gpu(res, 0, cpu_index)

    total_states = expander.num_states()
    print(f"Assigning {total_states:,} turn vectors on GPU "
          f"(batch_size={batch_size:,})...", file=sys.stderr)
    t0 = time.time()
    total_assigned = 0

    with open(out_path, 'wb') as f:
        def process_batch(batch_uint8):
            nonlocal total_assigned
            batch = np.cumsum(batch_uint8.astype(np.float32), axis=1)
            _, labels = index.search(batch, 1)
            f.write(labels.ravel().astype(np.uint16).tobytes())
            total_assigned += len(batch)

            elapsed   = time.time() - t0
            done_frac = total_assigned / total_states
            eta       = elapsed / done_frac * (1 - done_frac) if done_frac > 0 else 0
            rate      = total_assigned / elapsed
            print(f"\r  {100 * done_frac:.2f}%  ({total_assigned:,} / {total_states:,})  "
                  f"{rate / 1e6:.1f}M vec/s  ETA {eta / 60:.0f}m",
                  end='', file=sys.stderr)

        expander.expand_all(process_batch, batch_size)

    elapsed = time.time() - t0
    rate = total_assigned / elapsed if elapsed > 0 else 0
    print(f"\nAssignment done: {total_assigned:,} vectors in {elapsed:.1f}s "
          f"({rate / 1e6:.1f}M vec/s)", file=sys.stderr)
    if total_assigned != total_states:
        print(f"WARNING: assigned {total_assigned:,}, expected {total_states:,}",
              file=sys.stderr)
