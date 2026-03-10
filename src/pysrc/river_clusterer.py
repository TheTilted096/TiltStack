"""
Cluster the river equity vectors produced by river_expander.cpp.

Supports two modes of operation:

1. File mode (original): reads from a 410 GB river_equities.bin on disk.
       python river_clusterer.py river_equities.bin -k 30000

2. Streaming pipeline (no large file on disk):
       Step 1 — Sample:   ./river_expander - | python reservoir_sample.py -n 20M -o sample.npy
       Step 2 — Train:    python river_clusterer.py --train-only sample.npy -k 30000 -o river
       Step 3 — Assign:   ./river_expander - | python river_clusterer.py --assign-only river_centroids.npy --stdin -o river

Method: Two-phase K-means (default K=30,000).
  Phase 1 — Train centroids on a sample (default 20M vectors) using FAISS.
  Phase 2 — Stream all vectors and assign each to its nearest centroid.

Output:
  <prefix>_centroids.npy  — (K, 169) float32 centroid matrix
  <prefix>_labels.bin     — 2,428,287,420 x uint16 cluster assignments (raw binary)

The labels array is indexed by the hand-isomorphism library's river indexer.
To look up the cluster for a 7-card river hand (2 hole + 5 board), compute
  hand_index_last(&river_indexer, cards)
and read the uint16 at that offset in the labels file.

Nathaniel Potter, 03-08-2026
"""

import argparse
import os
import sys
import time

import faiss
import numpy as np

NUM_VECTORS = 2_428_287_420
NUM_DIMS = 169
BYTES_PER_ROW = NUM_DIMS  # uint8


def create_memmap(path: str) -> np.ndarray:
    """Memory-map the binary file as a (NUM_VECTORS, 169) uint8 array."""
    expected_size = NUM_VECTORS * BYTES_PER_ROW
    actual_size = os.path.getsize(path)
    if actual_size != expected_size:
        print(f"WARNING: file size {actual_size} != expected {expected_size} "
              f"({actual_size // BYTES_PER_ROW} rows detected)", file=sys.stderr)
    nrows = actual_size // BYTES_PER_ROW
    return np.memmap(path, dtype=np.uint8, mode='r', shape=(nrows, NUM_DIMS))


def sample_vectors(mmap: np.ndarray, n_sample: int, seed: int) -> np.ndarray:
    """Draw a random sample from the memory-mapped file, returned as float32."""
    rng = np.random.default_rng(seed)
    nrows = mmap.shape[0]
    n_sample = min(n_sample, nrows)
    indices = rng.choice(nrows, size=n_sample, replace=False)
    indices.sort()  # sequential access is much faster on mmap

    print(f"Sampling {n_sample:,} vectors...", file=sys.stderr)
    t0 = time.time()

    # Read in chunks to keep memory bounded and allow progress reporting.
    CHUNK = 1_000_000
    parts = []
    for i in range(0, len(indices), CHUNK):
        chunk_idx = indices[i:i + CHUNK]
        parts.append(mmap[chunk_idx].astype(np.float32))
        if (i // CHUNK) % 5 == 0:
            print(f"  sampled {min(i + CHUNK, len(indices)):,} / {n_sample:,}",
                  file=sys.stderr)

    sample = np.concatenate(parts, axis=0)
    print(f"Sampling done in {time.time() - t0:.1f}s", file=sys.stderr)
    return sample


def gpu_available() -> bool:
    """Check whether FAISS GPU support is installed and a GPU is visible."""
    try:
        return hasattr(faiss, 'StandardGpuResources') and faiss.get_num_gpus() > 0
    except Exception:
        return False


def make_gpu_index(index: faiss.Index) -> faiss.Index:
    """Move a FAISS index to GPU if possible, otherwise return the CPU index."""
    if not gpu_available():
        return index
    res = faiss.StandardGpuResources()
    return faiss.index_cpu_to_gpu(res, 0, index)


def train_centroids(sample: np.ndarray, k: int, niter: int,
                    seed: int, gpu: bool) -> np.ndarray:
    """Train K-means centroids using FAISS."""
    d = sample.shape[1]
    use_gpu = gpu and gpu_available()
    if gpu and not use_gpu:
        print("WARNING: --gpu requested but no FAISS GPU support found; "
              "falling back to CPU.", file=sys.stderr)
    backend = "GPU" if use_gpu else "CPU"
    print(f"Training K={k:,} centroids, {niter} iterations on {backend} "
          f"({sample.shape[0]:,} x {d} vectors)...", file=sys.stderr)
    kmeans = faiss.Kmeans(
        d, k,
        niter=niter,
        verbose=True,
        seed=seed,
        gpu=use_gpu,
        max_points_per_centroid=sample.shape[0] // k + 1,
    )
    t0 = time.time()
    kmeans.train(sample)
    print(f"K-means training done in {time.time() - t0:.1f}s", file=sys.stderr)
    return kmeans.centroids.copy()  # (k, d) float32


def assign_labels(mmap: np.ndarray, centroids: np.ndarray,
                  out_path: str, batch_size: int, gpu: bool) -> None:
    """Stream the full file and assign each vector to its nearest centroid.

    Writes uint16 labels to out_path in raw binary.
    Uses FAISS IndexFlatL2 for fast batch nearest-neighbor search.
    """
    nrows = mmap.shape[0]
    k, d = centroids.shape
    assert k <= 65535, "K must fit in uint16"

    index = faiss.IndexFlatL2(d)
    index.add(centroids)
    if gpu:
        index = make_gpu_index(index)

    backend = "GPU" if gpu and gpu_available() else "CPU"
    print(f"Assigning {nrows:,} vectors on {backend} "
          f"(batch_size={batch_size:,})...", file=sys.stderr)
    t0 = time.time()

    with open(out_path, 'wb') as f:
        for start in range(0, nrows, batch_size):
            end = min(start + batch_size, nrows)
            batch = mmap[start:end].astype(np.float32)

            _, labels = index.search(batch, 1)
            labels = labels.ravel().astype(np.uint16)

            f.write(labels.tobytes())

            elapsed = time.time() - t0
            done_frac = end / nrows
            eta = elapsed / done_frac * (1 - done_frac) if done_frac > 0 else 0
            rate = end / elapsed
            print(f"\r  {100 * done_frac:.2f}%  ({end:,} / {nrows:,})  "
                  f"{rate / 1e6:.1f}M vec/s  ETA {eta / 60:.0f}m",
                  end='', file=sys.stderr)

    elapsed = time.time() - t0
    rate = nrows / elapsed
    print(f"\nAssignment done: {nrows:,} vectors in {elapsed:.1f}s "
          f"({rate / 1e6:.1f}M vec/s)", file=sys.stderr)


def assign_labels_stdin(centroids: np.ndarray, out_path: str,
                        batch_size: int, gpu: bool) -> None:
    """Read uint8 equity vectors from stdin and assign to nearest centroid.

    Writes uint16 labels to out_path in raw binary, one per input vector.
    """
    k, d = centroids.shape
    assert k <= 65535, "K must fit in uint16"

    index = faiss.IndexFlatL2(d)
    index.add(centroids)
    if gpu:
        index = make_gpu_index(index)

    stream = sys.stdin.buffer
    batch_bytes = batch_size * BYTES_PER_ROW

    backend = "GPU" if gpu and gpu_available() else "CPU"
    print(f"Assigning vectors from stdin on {backend} "
          f"(batch_size={batch_size:,})...", file=sys.stderr)
    t0 = time.time()
    total_assigned = 0

    with open(out_path, 'wb') as f:
        while True:
            raw = stream.read(batch_bytes)
            if not raw:
                break
            n_rows = len(raw) // BYTES_PER_ROW
            if n_rows == 0:
                break

            batch = np.frombuffer(raw[:n_rows * BYTES_PER_ROW],
                                  dtype=np.uint8).reshape(n_rows, NUM_DIMS)
            batch_f32 = batch.astype(np.float32)

            _, labels = index.search(batch_f32, 1)
            labels = labels.ravel().astype(np.uint16)

            f.write(labels.tobytes())
            total_assigned += n_rows

            elapsed = time.time() - t0
            done_frac = total_assigned / NUM_VECTORS
            eta = elapsed / done_frac * (1 - done_frac) if done_frac > 0 else 0
            rate = total_assigned / elapsed
            print(f"\r  {100 * done_frac:.2f}%  ({total_assigned:,} / {NUM_VECTORS:,})  "
                  f"{rate / 1e6:.1f}M vec/s  ETA {eta / 60:.0f}m",
                  end='', file=sys.stderr)

    elapsed = time.time() - t0
    rate = total_assigned / elapsed if elapsed > 0 else 0
    print(f"\nAssignment done: {total_assigned:,} vectors in {elapsed:.1f}s "
          f"({rate / 1e6:.1f}M vec/s)", file=sys.stderr)
    if total_assigned != NUM_VECTORS:
        print(f"WARNING: assigned {total_assigned:,} vectors, expected {NUM_VECTORS:,}",
              file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(
        description="Cluster river equity vectors via K-means.")
    parser.add_argument("input", nargs='?', default=None,
                        help="Path to river_equities.bin or sample.npy (omit with --stdin)")
    parser.add_argument("-k", "--clusters", type=int, default=30_000,
                        help="Number of clusters (default: 30000)")
    parser.add_argument("-s", "--sample-size", type=int, default=20_000_000,
                        help="Number of vectors to sample for training (default: 20M)")
    parser.add_argument("-i", "--niter", type=int, default=25,
                        help="K-means iterations (default: 25)")
    parser.add_argument("-b", "--batch-size", type=int, default=1_000_000,
                        help="Batch size for assignment pass (default: 1M)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--gpu", action="store_true",
                        help="Use GPU for FAISS K-means training")
    parser.add_argument("-o", "--output-prefix", default=None,
                        help="Output file prefix (default: derived from input)")
    parser.add_argument("--train-only", action="store_true",
                        help="Train centroids from a sample .npy file, then exit")
    parser.add_argument("--centroids-only", action="store_true",
                        help="Only train centroids, skip assignment pass")
    parser.add_argument("--assign-only", default=None, metavar="CENTROIDS.npy",
                        help="Skip training; load centroids from file and assign")
    parser.add_argument("--stdin", action="store_true",
                        help="Read equity vectors from stdin instead of a file (for piped mode)")
    args = parser.parse_args()

    # Determine output prefix
    if args.output_prefix:
        prefix = args.output_prefix
    elif args.input:
        prefix = os.path.splitext(args.input)[0]
    else:
        prefix = "river"
    centroids_path = f"{prefix}_centroids.npy"
    labels_path = f"{prefix}_labels.bin"

    use_gpu = args.gpu and gpu_available()
    if args.gpu:
        if use_gpu:
            print(f"FAISS GPU enabled ({faiss.get_num_gpus()} GPU(s) detected).",
                  file=sys.stderr)
        else:
            print("WARNING: --gpu requested but no FAISS GPU support found; "
                  "using CPU.", file=sys.stderr)

    # --- Phase 1: Train centroids ---
    if args.assign_only:
        print(f"Loading centroids from {args.assign_only}...", file=sys.stderr,
              end='', flush=True)
        t0 = time.time()
        centroids = np.load(args.assign_only)
        print(f" {centroids.shape[0]:,} x {centroids.shape[1]} float32 "
              f"({time.time() - t0:.1f}s)", file=sys.stderr)
    elif args.train_only:
        if not args.input:
            parser.error("--train-only requires an input sample .npy file")
        print(f"Loading sample from {args.input}...", file=sys.stderr,
              end='', flush=True)
        t0 = time.time()
        sample = np.load(args.input)
        sz_mb = sample.nbytes / 1e6
        print(f" {sample.shape[0]:,} x {sample.shape[1]} float32 "
              f"({sz_mb:.0f} MB, {time.time() - t0:.1f}s)", file=sys.stderr)
        centroids = train_centroids(
            sample, args.clusters, args.niter, args.seed, args.gpu)
        print(f"Saving centroids to {centroids_path}...", file=sys.stderr,
              end='', flush=True)
        t0 = time.time()
        np.save(centroids_path, centroids)
        print(f" done ({time.time() - t0:.1f}s)", file=sys.stderr)
        print("Done.", file=sys.stderr)
        return
    else:
        if not args.input:
            parser.error("input file required (or use --assign-only with --stdin)")
        mmap = create_memmap(args.input)
        print(f"Mapped {mmap.shape[0]:,} x {mmap.shape[1]} uint8 matrix", file=sys.stderr)

        sample = sample_vectors(mmap, args.sample_size, args.seed)
        centroids = train_centroids(
            sample, args.clusters, args.niter, args.seed, args.gpu)

        print(f"Saving centroids to {centroids_path}...", file=sys.stderr,
              end='', flush=True)
        t0 = time.time()
        np.save(centroids_path, centroids)
        print(f" done ({time.time() - t0:.1f}s)", file=sys.stderr)

        del sample  # free ~12 GB

    # --- Phase 2: Assign all vectors ---
    if not args.centroids_only:
        if args.stdin:
            assign_labels_stdin(centroids, labels_path, args.batch_size,
                                use_gpu)
        else:
            if not args.input:
                parser.error("input file required for assignment (or use --stdin)")
            if 'mmap' not in dir():
                mmap = create_memmap(args.input)
            assign_labels(mmap, centroids, labels_path, args.batch_size,
                          use_gpu)
        print(f"Labels saved to {labels_path}", file=sys.stderr)

    print("Done.", file=sys.stderr)


if __name__ == "__main__":
    main()
