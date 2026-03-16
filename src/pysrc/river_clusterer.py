"""
K-means clustering for river equity vectors.

Library functions used by river_cluster_pipeline.py:
  train_centroids(sample, k, niter, seed)  -> np.ndarray  (K, 169) float32
  sort_centroids_by_ehs(centroids)         -> np.ndarray  sorted copy
  assign_labels_streaming(expander, centroids, out_path, batch_size)

GPU (FAISS) is required — CPU clustering is not supported due to infeasible
runtime on the ~2.4B river states dataset.

Nathaniel Potter, 03-08-2026
"""

import sys
import time

import faiss
import numpy as np

NUM_DIMS = 169


# ---------------------------------------------------------------------------
# Public library API
# ---------------------------------------------------------------------------

def gpu_available() -> bool:
    """Check whether FAISS GPU support is installed and a GPU is visible."""
    try:
        return hasattr(faiss, 'StandardGpuResources') and faiss.get_num_gpus() > 0
    except Exception:
        return False


def train_centroids(sample: np.ndarray, k: int, niter: int,
                    seed: int) -> np.ndarray:
    """Train K-means centroids using FAISS on GPU.

    Args:
        sample:  (N, D) float32 training vectors.
        k:       Number of clusters.
        niter:   K-means iterations.
        seed:    Random seed.

    Returns:
        (k, D) float32 centroid matrix.
    """
    d = sample.shape[1]
    print(f"Training K={k:,} centroids, {niter} iterations on GPU "
          f"({sample.shape[0]:,} x {d} vectors)...", file=sys.stderr)
    kmeans = faiss.Kmeans(
        d, k,
        niter=niter,
        verbose=True,
        seed=seed,
        gpu=True,
        max_points_per_centroid=sample.shape[0] // k + 1,
    )
    t0 = time.time()
    kmeans.train(sample)
    print(f"K-means training done in {time.time() - t0:.1f}s", file=sys.stderr)
    return kmeans.centroids.copy()


def sort_centroids_by_ehs(centroids: np.ndarray) -> np.ndarray:
    """Return a copy of centroids sorted by mean equity (ascending)."""
    ehs = centroids.mean(axis=1)
    return centroids[np.argsort(ehs)]


def assign_labels_streaming(expander, centroids: np.ndarray, out_path: str,
                             batch_size: int = 1_000_000) -> None:
    """Assign cluster labels to all states via a pybind expander.

    Streams equity vectors in batches via expander.expand_all() to avoid
    materialising all states in memory at once. Writes raw uint16 labels
    to out_path.
    """
    k, d = centroids.shape
    assert k <= 65535, "K must fit in uint16"

    cpu_index = faiss.IndexFlatL2(d)
    cpu_index.add(centroids)
    res   = faiss.StandardGpuResources()
    index = faiss.index_cpu_to_gpu(res, 0, cpu_index)

    total_states = expander.num_states()
    print(f"Assigning {total_states:,} vectors on GPU "
          f"(batch_size={batch_size:,})...", file=sys.stderr)
    t0 = time.time()
    total_assigned = 0

    with open(out_path, 'wb') as f:
        def process_batch(batch_uint8):
            nonlocal total_assigned
            batch = batch_uint8.astype(np.float32)
            _, labels = index.search(batch, 1)
            f.write(labels.ravel().astype(np.uint16).tobytes())
            total_assigned += len(batch)

            elapsed = time.time() - t0
            done_frac = total_assigned / total_states
            eta = elapsed / done_frac * (1 - done_frac) if done_frac > 0 else 0
            rate = total_assigned / elapsed
            print(f"\r  {100 * done_frac:.2f}%  ({total_assigned:,} / {total_states:,})  "
                  f"{rate / 1e6:.1f}M vec/s  ETA {eta / 60:.0f}m",
                  end='', file=sys.stderr)

        expander.expand_all(process_batch, batch_size)

    elapsed = time.time() - t0
    rate = total_assigned / elapsed if elapsed > 0 else 0
    print(f"\nAssignment done: {total_assigned:,} vectors in {elapsed:.1f}s "
          f"({rate / 1e6:.1f}M vec/s)", file=sys.stderr)
    if total_assigned != total_states:
        print(f"WARNING: assigned {total_assigned:,} vectors, "
              f"expected {total_states:,}", file=sys.stderr)
