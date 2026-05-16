"""Shared clustering utilities for the HUNL abstraction pipeline."""

import shutil
import sys
import time
from pathlib import Path
from typing import Callable

import faiss
import numpy as np

METRIC_L1 = faiss.METRIC_L1
METRIC_L2 = faiss.METRIC_L2


def gpu_available() -> bool:
    """Check whether FAISS GPU support is installed and a GPU is visible."""
    try:
        return hasattr(faiss, "StandardGpuResources") and faiss.get_num_gpus() > 0
    except Exception:
        return False


def copy_outputs(paths: tuple[Path, ...], output_dir: Path) -> tuple[Path, ...]:
    """Copy completed temp artifacts back to the published output directory."""
    output_dir.mkdir(parents=True, exist_ok=True)
    copied = []
    for path in paths:
        target = output_dir / path.name
        tmp_target = target.with_name(f"{target.name}.copying")
        shutil.copy2(path, tmp_target)
        tmp_target.replace(target)
        copied.append(target)
    return tuple(copied)


def train_centroids(
    sample: np.ndarray,
    k: int,
    niter: int,
    seed: int,
    *,
    metric: int = faiss.METRIC_L2,
    label: str = "centroids",
) -> np.ndarray:
    """Train K-means centroids on GPU."""
    d = sample.shape[1]
    metric_name = "L1" if metric == faiss.METRIC_L1 else "L2"
    print(
        f"Training K={k:,} {label}, {niter} iterations on GPU "
        f"({sample.shape[0]:,} x {d} vectors, {metric_name})...",
        file=sys.stderr,
    )
    t0 = time.time()

    if metric == faiss.METRIC_L2:
        kmeans = faiss.Kmeans(
            d,
            k,
            niter=niter,
            verbose=True,
            seed=seed,
            gpu=True,
            max_points_per_centroid=sample.shape[0] // k + 1,
        )
        kmeans.train(sample)
        centroids = kmeans.centroids.copy()
    else:
        clus = faiss.Clustering(d, k)
        clus.niter = niter
        clus.verbose = True
        clus.seed = seed
        clus.max_points_per_centroid = sample.shape[0] // k + 1

        cpu_index = faiss.IndexFlat(d, metric)
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, 0, cpu_index)

        clus.train(sample, index)
        centroids = faiss.vector_to_array(clus.centroids).reshape(k, d).copy()

    print(f"K-means training done in {time.time() - t0:.1f}s", file=sys.stderr)
    return centroids


def make_gpu_index(centroids: np.ndarray, metric: int):
    """Build a FAISS GPU flat index containing centroid vectors."""
    d = centroids.shape[1]
    if metric == faiss.METRIC_L2:
        cpu_index = faiss.IndexFlatL2(d)
    else:
        cpu_index = faiss.IndexFlat(d, metric)
    cpu_index.add(centroids)
    res = faiss.StandardGpuResources()
    return faiss.index_cpu_to_gpu(res, 0, cpu_index)


def encode_ehs(ehs: np.ndarray) -> np.ndarray:
    """Encode float EHS values into uint16 fixed-point values."""
    return np.rint(ehs * 65535.0).astype(np.uint16)


def weighted_cluster_ehs(
    labels: np.ndarray,
    ehs: np.ndarray,
    multiplicity: np.ndarray,
    k: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return weighted EHS and multiplicity sums for one labeled batch."""
    mult = multiplicity.astype(np.float64)
    ehs_sum = np.bincount(labels, weights=ehs.astype(np.float64) * mult, minlength=k)
    mult_sum = np.bincount(labels, weights=mult, minlength=k)
    return ehs_sum, mult_sum


def assign_labels(
    vectors: np.ndarray,
    centroids: np.ndarray,
    out_path: str,
    *,
    metric: int,
    label: str,
) -> np.ndarray:
    """Assign labels for an in-memory matrix and write raw uint16 labels."""
    k, _ = centroids.shape
    assert k <= 65535, "K must fit in uint16"

    index = make_gpu_index(centroids, metric)
    total = vectors.shape[0]
    print(f"Assigning {total:,} {label} vectors on GPU...", file=sys.stderr)
    t0 = time.time()

    _, labels = index.search(vectors, 1)
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


def assign_labels_and_ehs_fine_streaming(
    expander,
    centroids: np.ndarray,
    labels_path: str,
    ehs_fine_path: str,
    *,
    batch_size: int,
    metric: int,
    label: str,
    vector_transform: Callable[[np.ndarray], np.ndarray],
) -> np.ndarray:
    """Stream vectors, assign labels, write labels/EHS, and accumulate cluster EHS."""
    k, _ = centroids.shape
    assert k <= 65535, "K must fit in uint16"

    index = make_gpu_index(centroids, metric)
    total_states = expander.num_states()
    print(
        f"Assigning {total_states:,} {label} vectors on GPU "
        f"(batch_size={batch_size:,})...",
        file=sys.stderr,
    )
    t0 = time.time()
    total_assigned = 0

    ehs_sum = np.zeros(k, dtype=np.float64)
    mult_sum = np.zeros(k, dtype=np.float64)

    with open(labels_path, "wb") as lf, open(ehs_fine_path, "wb") as ef:

        def process_batch(raw_vectors, ehs_float32, mult_uint8):
            nonlocal total_assigned, ehs_sum, mult_sum
            vectors = vector_transform(raw_vectors)
            _, label_arr = index.search(vectors, 1)
            label_arr = label_arr.ravel().astype(np.uint16)
            lf.write(label_arr.tobytes())
            ef.write(encode_ehs(ehs_float32).tobytes())

            batch_ehs_sum, batch_mult_sum = weighted_cluster_ehs(
                label_arr, ehs_float32, mult_uint8, k
            )
            ehs_sum += batch_ehs_sum
            mult_sum += batch_mult_sum
            total_assigned += len(label_arr)

            elapsed = time.time() - t0
            done_frac = total_assigned / total_states
            eta = elapsed / done_frac * (1 - done_frac) if done_frac > 0 else 0
            rate = total_assigned / elapsed
            print(
                f"\r  {100 * done_frac:.2f}%  ({total_assigned:,} / {total_states:,})  "
                f"{rate / 1e6:.1f}M vec/s  ETA {eta / 60:.0f}m",
                end="",
                file=sys.stderr,
            )

        expander.expand_all_with_ehs_mult(process_batch, batch_size)

    elapsed = time.time() - t0
    rate = total_assigned / elapsed if elapsed > 0 else 0
    print(
        f"\nAssignment done: {total_assigned:,} vectors in {elapsed:.1f}s "
        f"({rate / 1e6:.1f}M vec/s)",
        file=sys.stderr,
    )
    if total_assigned != total_states:
        print(
            f"WARNING: assigned {total_assigned:,}, expected {total_states:,}",
            file=sys.stderr,
        )

    per_cluster_ehs = ehs_sum / np.maximum(mult_sum, 1.0)
    return per_cluster_ehs.astype(np.float32)


def remap_labels_inplace(labels_path: str, sort_order: np.ndarray) -> None:
    """Remap uint16 labels file in-place according to a sort permutation."""
    labels = np.fromfile(labels_path, dtype=np.uint16)
    rank = np.empty(len(sort_order), dtype=np.uint16)
    rank[sort_order] = np.arange(len(sort_order), dtype=np.uint16)
    rank[labels].tofile(labels_path)
