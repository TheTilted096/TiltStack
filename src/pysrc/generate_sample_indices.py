"""
Pre-select random river state indices for K-means training.

Generates N sorted unique indices from [0, 2,428,287,420) and saves them
as a binary file of uint64 values that river_expander --sample can read.

Usage:
    python generate_sample_indices.py -n 20000000 -o sample_indices.bin

Nathaniel Potter, 03-09-2026
"""

import argparse
import os
import sys
import time

import numpy as np

NUM_STATES = 2_428_287_420


def main():
    parser = argparse.ArgumentParser(
        description="Generate sorted random river state indices.")
    parser.add_argument("-n", "--size", type=int, default=20_000_000,
                        help="Number of indices to sample (default: 20M)")
    parser.add_argument("-o", "--output", required=True,
                        help="Output path for the binary index file")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    print(f"Sampling {args.size:,} unique indices from {NUM_STATES:,}...",
          file=sys.stderr, end='', flush=True)
    t0 = time.time()
    indices = rng.choice(NUM_STATES, size=args.size, replace=False)
    print(f" done ({time.time() - t0:.1f}s)", file=sys.stderr)

    print(f"Sorting {args.size:,} indices...", file=sys.stderr,
          end='', flush=True)
    t0 = time.time()
    indices = np.sort(indices).astype(np.uint64)
    print(f" done ({time.time() - t0:.1f}s)", file=sys.stderr)

    print(f"Writing to {args.output}...", file=sys.stderr, end='', flush=True)
    t0 = time.time()
    indices.tofile(args.output)
    sz_mb = os.path.getsize(args.output) / 1e6
    print(f" {len(indices):,} uint64 indices ({sz_mb:.1f} MB, {time.time() - t0:.1f}s)",
          file=sys.stderr)


if __name__ == "__main__":
    main()
