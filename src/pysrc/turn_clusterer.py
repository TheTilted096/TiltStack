"""
K-means clustering for turn wide-bucket histogram vectors.

Library functions used by turn_cluster_pipeline.py:
  train_turn_centroids(sample, k, niter, seed, gpu)  -> np.ndarray  (K, 256) float32
  compute_wide_ehs(river_centroids)                  -> np.ndarray  (256,) float32
  sort_turn_centroids(centroids, wide_ehs)            -> np.ndarray  sorted copy
  assign_turn_labels_streaming(expander, centroids, out_path, batch_size, gpu)

Nathaniel Potter, 03-15-2026
"""

import sys
import time

import faiss
import numpy as np

# Normalisation denominator: one histogram count per possible river card.
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
                         seed: int, gpu: bool) -> np.ndarray:
    """Train K-means centroids on float32 probability vectors with L1 distance.

    Args:
        sample:  (N, 256) float32 probability vectors (counts / 46.0).
        k:       Number of clusters.
        niter:   K-means iterations.
        seed:    Random seed.
        gpu:     Use GPU if available.

    Returns:
        (k, 256) float32 centroid matrix.
    """
    d = sample.shape[1]
    use_gpu = gpu and gpu_available()
    if gpu and not use_gpu:
        print("WARNING: --gpu requested but no FAISS GPU support found; "
              "falling back to CPU.", file=sys.stderr)
    backend = "GPU" if use_gpu else "CPU"
    print(f"Training K={k:,} turn centroids, {niter} iterations on {backend} "
          f"({sample.shape[0]:,} x {d} vectors, L1)...", file=sys.stderr)

    clus = faiss.Clustering(d, k)
    clus.niter   = niter
    clus.verbose = True
    clus.seed    = seed
    clus.max_points_per_centroid = sample.shape[0] // k + 1

    index = faiss.IndexFlat(d, faiss.METRIC_L1)
    if use_gpu:
        res   = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, 0, index)

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

    E[EHS] = sum_i( wide_ehs[i] * prob[i] ) = centroids @ wide_ehs

    Args:
        centroids: (K, 256) float32 turn centroid matrix (probability vectors).
        wide_ehs:  (256,) float32 average EHS per wide bucket from compute_wide_ehs().

    Returns:
        Sorted copy of centroids with label 0 = lowest expected EHS (weakest).
    """
    expected_ehs = centroids @ wide_ehs    # (K,) scalar EHS per centroid
    return centroids[np.argsort(expected_ehs)]


def assign_turn_labels_streaming(expander, centroids: np.ndarray, out_path: str,
                                 batch_size: int = 500_000, gpu: bool = False) -> None:
    """Assign turn cluster labels to all states via the TurnExpander.

    Normalises each uint8 batch by 46.0, then searches with L1 distance.
    Writes raw uint16 labels to out_path.
    """
    k, d = centroids.shape
    assert k <= 65535, "K must fit in uint16"

    index = faiss.IndexFlat(d, faiss.METRIC_L1)
    index.add(centroids)
    if gpu and gpu_available():
        res   = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, 0, index)

    total_states = expander.num_states()
    backend = "GPU" if gpu and gpu_available() else "CPU"
    print(f"Assigning {total_states:,} turn vectors on {backend} "
          f"(batch_size={batch_size:,})...", file=sys.stderr)
    t0 = time.time()
    total_assigned = 0

    with open(out_path, 'wb') as f:
        def process_batch(batch_uint8):
            nonlocal total_assigned
            batch = batch_uint8.astype(np.float32) / RIVER_CARDS_PER_TURN
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
