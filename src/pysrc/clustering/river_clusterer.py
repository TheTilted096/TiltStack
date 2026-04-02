"""
K-means clustering for river equity vectors.

Library functions used by river_cluster_pipeline.py:
  gpu_available()
  train_centroids(sample, k, niter, seed)                   -> (K, 169) float32
  assign_labels_and_ehs_fine_streaming(                     -> (K,) float32 per-cluster EHS
      expander, centroids, labels_path, ehs_fine_path,
      batch_size)
  remap_labels_inplace(labels_path, sort_order)

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
        sample:  (N, 169) float32 equity vectors.
        k:       Number of clusters.
        niter:   K-means iterations.
        seed:    Random seed.

    Returns:
        (k, 169) float32 centroid matrix (unsorted).
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


def assign_labels_and_ehs_fine_streaming(
        expander, centroids: np.ndarray,
        labels_path: str, ehs_fine_path: str,
        batch_size: int = 1_000_000) -> np.ndarray:
    """Assign cluster labels and write per-state EHS to disk in a single pass.

    Uses expander.expand_all_with_ehs_mult() to stream equity vectors, per-state
    EHS, and multiplicities simultaneously.  Per-state EHS is the
    multiplicity-weighted equity over all concrete opponent hands (computed in
    C++ as totalEqSum / totalCount).

    Writes:
      labels_path:   uint16 label for every river state (in canonical index order)
      ehs_fine_path: uint16 EHS for every river state (decode: value / 65535.0)

    Returns:
      (K,) float32 array of multiplicity-weighted per-cluster EHS.
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

    ehs_sum  = np.zeros(k, dtype=np.float64)
    mult_sum = np.zeros(k, dtype=np.float64)

    with open(labels_path, 'wb') as lf, open(ehs_fine_path, 'wb') as ef:
        def process_batch(equity_uint8, ehs_float32, mult_uint8):
            nonlocal total_assigned, ehs_sum, mult_sum
            equity = equity_uint8.astype(np.float32)
            _, label_arr = index.search(equity, 1)
            label_arr = label_arr.ravel().astype(np.uint16)
            lf.write(label_arr.tobytes())

            ehs_u16 = np.rint(ehs_float32 * 65535.0).astype(np.uint16)
            ef.write(ehs_u16.tobytes())

            ehs_f64 = ehs_float32.astype(np.float64)
            mult    = mult_uint8.astype(np.float64)
            ehs_sum  += np.bincount(label_arr, weights=ehs_f64 * mult,
                                    minlength=k)
            mult_sum += np.bincount(label_arr, weights=mult, minlength=k)
            total_assigned += len(label_arr)

            elapsed   = time.time() - t0
            done_frac = total_assigned / total_states
            eta       = elapsed / done_frac * (1 - done_frac) if done_frac > 0 else 0
            rate      = total_assigned / elapsed
            print(f"\r  {100 * done_frac:.2f}%  ({total_assigned:,} / {total_states:,})  "
                  f"{rate / 1e6:.1f}M vec/s  ETA {eta / 60:.0f}m",
                  end='', file=sys.stderr)

        expander.expand_all_with_ehs_mult(process_batch, batch_size)

    elapsed = time.time() - t0
    rate = total_assigned / elapsed if elapsed > 0 else 0
    print(f"\nAssignment done: {total_assigned:,} vectors in {elapsed:.1f}s "
          f"({rate / 1e6:.1f}M vec/s)", file=sys.stderr)
    if total_assigned != total_states:
        print(f"WARNING: assigned {total_assigned:,}, expected {total_states:,}",
              file=sys.stderr)

    per_cluster_ehs = ehs_sum / np.maximum(mult_sum, 1.0)
    return per_cluster_ehs.astype(np.float32)


def remap_labels_inplace(labels_path: str, sort_order: np.ndarray) -> None:
    """Remap uint16 labels file in-place according to a sort permutation.

    sort_order is the output of np.argsort(per_cluster_ehs), so
    sort_order[new_label] = old_label and label 0 is the weakest cluster.
    """
    labels = np.fromfile(labels_path, dtype=np.uint16)
    rank = np.empty(len(sort_order), dtype=np.uint16)
    rank[sort_order] = np.arange(len(sort_order), dtype=np.uint16)
    rank[labels].tofile(labels_path)
