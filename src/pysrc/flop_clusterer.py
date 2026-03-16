"""
K-means clustering for flop wide-bucket histogram vectors.

Library functions used by flop_cluster_pipeline.py:
  train_flop_centroids(sample, k, niter, seed)          -> np.ndarray  (K, 256) float32
  compute_wide_turn_ehs(turn_centroids, river_centroids) -> np.ndarray  (256,) float32
  sort_flop_centroids(centroids, wide_turn_ehs)          -> np.ndarray  sorted copy
  assign_flop_labels(all_cdfs, centroids, out_path)

Each flop state is represented as a 256-dim float32 CDF vector:
  - The C++ FlopExpander produces a uint8[256] count histogram over 256 wide
    buckets (each wide bucket spans 32 consecutive turn fine buckets).
  - Counts sum to 47 (one per possible turn card).
  - The CDF is the cumulative sum of these counts (NOT divided by 47), yielding
    values in [0, 47].
  - L1 distance between CDFs equals the Earth Mover's Distance (Wasserstein-1)
    on the underlying probability distributions.
  - Centroids are sorted by expected EHS computed via two-level wide-bucket
    averaging: river_centroids -> wide_river_ehs (256,) -> E[EHS] per turn
    centroid -> wide_turn_ehs (256,) -> E[EHS] per flop centroid.

All 1,286,792 flop states fit in RAM (~1.31 GB as float32 CDFs), so no sampling
or intermediate disk file is needed.

GPU (FAISS) is required.

Nathaniel Potter, 03-15-2026
"""

import sys
import time

import faiss
import numpy as np

# Number of possible turn cards per flop state (counts sum to this value).
TURN_CARDS_PER_FLOP = 47


# ---------------------------------------------------------------------------
# Public library API
# ---------------------------------------------------------------------------

def gpu_available() -> bool:
    """Check whether FAISS GPU support is installed and a GPU is visible."""
    try:
        return hasattr(faiss, 'StandardGpuResources') and faiss.get_num_gpus() > 0
    except Exception:
        return False


def train_flop_centroids(sample: np.ndarray, k: int, niter: int,
                         seed: int) -> np.ndarray:
    """Train K-means centroids on float32 CDF vectors with L1 distance on GPU.

    L1 distance between CDFs is equivalent to the Earth Mover's Distance
    (Wasserstein-1) on the underlying probability distributions.

    Args:
        sample:  (N, 256) float32 CDF vectors (cumsum of counts, NOT divided by
                 47.0, so values range from 0 to 47).
        k:       Number of clusters.
        niter:   K-means iterations.
        seed:    Random seed.

    Returns:
        (k, 256) float32 centroid matrix (CDF vectors, values in [0, 47]).
    """
    d = sample.shape[1]
    print(f"Training K={k:,} flop centroids, {niter} iterations on GPU "
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


def compute_wide_turn_ehs(turn_centroids: np.ndarray,
                          river_centroids: np.ndarray) -> np.ndarray:
    """Compute the average EHS for each of the 256 wide turn buckets.

    Two-level computation:
      1. wide_river_ehs (256,): mean equity per wide river bucket from river
         centroids (values in [0, 255] scale since river centroid values are
         equity * 255).
      2. fine_turn_ehs (8192,): E[EHS] per turn centroid (in [0, 46*255] scale)
         computed as pdf_turn @ wide_river_ehs.
      3. wide_turn_ehs (256,): mean of 32 consecutive turn centroid EHS values
         per wide turn bucket.

    Args:
        turn_centroids:  (8192, 256) sorted turn centroid matrix (CDF vectors,
                         values in [0, 46], NOT divided by 46.0).
        river_centroids: (8192, 169) sorted river centroid matrix (equity
                         vectors, values in [0, 255] scale).

    Returns:
        (256,) float32 array of average EHS per wide turn bucket.
        Values are in [0, 46*255] scale (proportional to true E[EHS]).
    """
    # Level 1: wide river EHS from river centroids
    fine_river_ehs = river_centroids.mean(axis=1)           # (8192,) in [0, 255]
    wide_river_ehs = fine_river_ehs.reshape(256, 32).mean(axis=1)  # (256,)

    # Level 2: E[EHS] per turn centroid via PDF dot product
    pdf_turn = np.hstack([turn_centroids[:, :1],
                          np.diff(turn_centroids, axis=1)])  # (8192, 256)
    fine_turn_ehs = pdf_turn @ wide_river_ehs                # (8192,) in [0, 46*255]

    # Level 3: group into 256 wide turn buckets
    return fine_turn_ehs.reshape(256, 32).mean(axis=1)       # (256,)


def sort_flop_centroids(centroids: np.ndarray,
                        wide_turn_ehs: np.ndarray) -> np.ndarray:
    """Return centroids sorted by expected EHS (ascending = weakest first).

    Centroids are CDF vectors; we revert to PDF via finite differences before
    computing E[EHS] so the dot product yields a true probability-weighted sum.

    E[EHS] = sum_i( pdf[i] * wide_turn_ehs[i] )   where pdf = diff(cdf)

    Args:
        centroids:     (K, 256) float32 flop centroid matrix (CDF vectors,
                       values in [0, 47], NOT divided by 47.0).
        wide_turn_ehs: (256,) float32 average EHS per wide turn bucket from
                       compute_wide_turn_ehs().

    Returns:
        Sorted copy of centroids with label 0 = lowest expected EHS (weakest).
    """
    pdf = np.hstack([centroids[:, :1], np.diff(centroids, axis=1)])  # (K, 256)
    expected_ehs = pdf @ wide_turn_ehs  # (K,) – proportional to E[EHS], sufficient for ordering
    return centroids[np.argsort(expected_ehs)]


def assign_flop_labels(all_cdfs: np.ndarray, centroids: np.ndarray,
                       out_path: str) -> None:
    """Assign flop cluster labels to all states via GPU search.

    All 1,286,792 CDF vectors are already in RAM, so no streaming is needed.
    Searches with L1 distance (equivalent to Earth Mover's Distance).
    Writes raw uint16 labels to out_path.

    Args:
        all_cdfs:   (N, 256) float32 CDF matrix for all flop states.
        centroids:  (K, 256) float32 sorted centroid matrix.
        out_path:   Path to write uint16 label file.
    """
    k, d = centroids.shape
    assert k <= 65535, "K must fit in uint16"

    cpu_index = faiss.IndexFlat(d, faiss.METRIC_L1)
    cpu_index.add(centroids)
    res   = faiss.StandardGpuResources()
    index = faiss.index_cpu_to_gpu(res, 0, cpu_index)

    total = all_cdfs.shape[0]
    print(f"Assigning {total:,} flop vectors on GPU...", file=sys.stderr)
    t0 = time.time()

    _, labels = index.search(all_cdfs, 1)
    labels = labels.ravel().astype(np.uint16)

    with open(out_path, 'wb') as f:
        f.write(labels.tobytes())

    elapsed = time.time() - t0
    print(f"Assignment done: {total:,} vectors in {elapsed:.1f}s "
          f"({total / elapsed / 1e6:.1f}M vec/s)", file=sys.stderr)
