"""
K-means clustering for flop wide-bucket histogram vectors.

Library functions used by flop_cluster_pipeline.py:
  gpu_available()
  train_flop_centroids(sample, k, niter, seed)          -> (K, 256) float32
  assign_flop_labels(all_cdfs, centroids, out_path)     -> (N,) uint16
  remap_labels_inplace(labels_path, sort_order)

Each flop state is represented as a 256-dim float32 CDF vector:
  - The C++ FlopExpander produces a uint8[256] count histogram over 256 wide
    buckets (each wide bucket spans 32 consecutive turn fine buckets).
  - Counts sum to 47 (one per possible turn card).
  - The CDF is the cumulative sum of these counts (NOT divided by 47), yielding
    values in [0, 47].
  - L1 distance between CDFs equals the Earth Mover's Distance (Wasserstein-1)
    on the underlying probability distributions.

All 1,286,792 flop states fit comfortably in RAM, so no sampling or streaming
is needed.  The FlopExpander loads both turn_labels.bin (~110 MB) and
turn_ehs_fine.bin (~110 MB) simultaneously.

Centroids are sorted by the multiplicity-weighted average EHS of their members,
computed in the pipeline after label assignment.

GPU (FAISS) is required.

Nathaniel Potter, 03-15-2026
"""

import sys
import time

import faiss
import numpy as np

# ---------------------------------------------------------------------------
# Public library API
# ---------------------------------------------------------------------------


def gpu_available() -> bool:
    """Check whether FAISS GPU support is installed and a GPU is visible."""
    try:
        return hasattr(faiss, "StandardGpuResources") and faiss.get_num_gpus() > 0
    except Exception:
        return False


def train_flop_centroids(
    sample: np.ndarray, k: int, niter: int, seed: int
) -> np.ndarray:
    """Train K-means centroids on float32 CDF vectors with L1 distance on GPU.

    Args:
        sample:  (N, 256) float32 CDF vectors (cumsum of counts, NOT divided by
                 47.0, so values range from 0 to 47).
        k:       Number of clusters.
        niter:   K-means iterations.
        seed:    Random seed.

    Returns:
        (k, 256) float32 centroid matrix (CDF vectors, values in [0, 47], unsorted).
    """
    d = sample.shape[1]
    print(
        f"Training K={k:,} flop centroids, {niter} iterations on GPU "
        f"({sample.shape[0]:,} x {d} vectors, L1)...",
        file=sys.stderr,
    )

    clus = faiss.Clustering(d, k)
    clus.niter = niter
    clus.verbose = True
    clus.seed = seed
    clus.max_points_per_centroid = sample.shape[0] // k + 1

    cpu_index = faiss.IndexFlat(d, faiss.METRIC_L1)
    res = faiss.StandardGpuResources()
    index = faiss.index_cpu_to_gpu(res, 0, cpu_index)

    t0 = time.time()
    clus.train(sample, index)
    print(f"K-means training done in {time.time() - t0:.1f}s", file=sys.stderr)
    return faiss.vector_to_array(clus.centroids).reshape(k, d).copy()


def assign_flop_labels(
    all_cdfs: np.ndarray, centroids: np.ndarray, out_path: str
) -> np.ndarray:
    """Assign flop cluster labels to all states via GPU search.

    All 1,286,792 CDF vectors are already in RAM, so no streaming is needed.
    Searches with L1 distance (equivalent to Earth Mover's Distance).
    Writes raw uint16 labels to out_path and returns the label array.

    Args:
        all_cdfs:   (N, 256) float32 CDF matrix for all flop states.
        centroids:  (K, 256) float32 centroid matrix (unsorted).
        out_path:   Path to write uint16 label file.

    Returns:
        (N,) uint16 label array.
    """
    k, d = centroids.shape
    assert k <= 65535, "K must fit in uint16"

    cpu_index = faiss.IndexFlat(d, faiss.METRIC_L1)
    cpu_index.add(centroids)
    res = faiss.StandardGpuResources()
    index = faiss.index_cpu_to_gpu(res, 0, cpu_index)

    total = all_cdfs.shape[0]
    print(f"Assigning {total:,} flop vectors on GPU...", file=sys.stderr)
    t0 = time.time()

    _, labels = index.search(all_cdfs, 1)
    labels = labels.ravel().astype(np.uint16)

    with open(out_path, "wb") as f:
        f.write(labels.tobytes())

    elapsed = time.time() - t0
    print(
        f"Assignment done: {total:,} vectors in {elapsed:.1f}s "
        f"({total / elapsed / 1e6:.1f}M vec/s)",
        file=sys.stderr,
    )
    return labels


def remap_labels_inplace(labels_path: str, sort_order: np.ndarray) -> None:
    """Remap uint16 labels file in-place according to a sort permutation.

    sort_order is the output of np.argsort(per_cluster_ehs), so
    sort_order[new_label] = old_label and label 0 is the weakest cluster.
    """
    labels = np.fromfile(labels_path, dtype=np.uint16)
    rank = np.empty(len(sort_order), dtype=np.uint16)
    rank[sort_order] = np.arange(len(sort_order), dtype=np.uint16)
    rank[labels].tofile(labels_path)
