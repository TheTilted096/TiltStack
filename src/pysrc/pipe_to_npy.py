"""
Read raw uint8 equity vectors from stdin and save as a float32 .npy file.

Used by the Makefile to convert piped river_expander output to NumPy format.

Usage:
    river_expander --sample indices.bin - | python pipe_to_npy.py -o output.npy
"""

import argparse
import sys
import time

import numpy as np

NUM_DIMS = 169


def main():
    parser = argparse.ArgumentParser(
        description="Convert piped uint8 equity vectors to float32 .npy")
    parser.add_argument("-o", "--output", required=True,
                        help="Output .npy path")
    args = parser.parse_args()

    print("Reading expander output from pipe...", file=sys.stderr, flush=True)
    t0 = time.time()
    raw = sys.stdin.buffer.read()
    n = len(raw) // NUM_DIMS
    print(f"Read {len(raw) / 1e9:.2f} GB ({n:,} vectors) in "
          f"{time.time() - t0:.1f}s", file=sys.stderr)

    print("Converting to float32 and saving...", file=sys.stderr, flush=True)
    t0 = time.time()
    arr = np.frombuffer(raw, dtype=np.uint8).reshape(n, NUM_DIMS).astype(np.float32)
    np.save(args.output, arr)
    print(f"Saved {n:,} x {NUM_DIMS} float32 -> {args.output} "
          f"({arr.nbytes / 1e9:.2f} GB, {time.time() - t0:.1f}s)",
          file=sys.stderr)


if __name__ == "__main__":
    main()
