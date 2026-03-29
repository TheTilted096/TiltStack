#!/usr/bin/env python3
"""
Compute and save per-state EHS for all 169 canonical preflop hole-hand classes.

Algorithm:
  For each of the 169 canonical preflop hole-hand classes, enumerate all
  C(50, 3) = 19,600 possible flop boards (3 cards from the 50 remaining cards).
  Map each (hole_cards, flop_board) combination to a canonical flop index via
  FlopIndexer.index(), look up flop_ehs_fine.bin, and average.

  No multiplicity weighting is required here because we enumerate concrete deals
  uniformly — each canonical preflop class is implicitly weighted by the number
  of concrete deals it represents (which is proportional to its multiplicity).

Output:
  clusters/preflop_ehs_fine.bin — 169 × float32 (~676 B)

Requires:
  clusters/flop_ehs_fine.bin from the flop clustering pipeline.

Then run:
    python preflop_ehs_pipeline.py

Nathaniel Potter, 03-15-2026
"""

import argparse
import sys
import time
from itertools import combinations
from pathlib import Path

import numpy as np

import hand_indexer

# ---------------------------------------------------------------------------
# Fixed paths
# ---------------------------------------------------------------------------

OUTPUT_DIR          = Path("clusters")
FLOP_EHS_FINE_PATH  = OUTPUT_DIR / "flop_ehs_fine.bin"
PREFLOP_EHS_PATH    = OUTPUT_DIR / "preflop_ehs_fine.bin"

NUM_CARDS    = 52
NUM_PREFLOP  = 169


def compute_preflop_ehs(verbose: bool = True) -> np.ndarray:
    """Enumerate all concrete (hole, flop) deals and average flop EHS.

    Returns:
        (169,) float32 array of per-class EHS, sorted by canonical index.
    """
    def log(msg):
        if verbose:
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", file=sys.stderr)

    log(f"Loading flop_ehs_fine.bin...")
    flop_ehs = np.fromfile(FLOP_EHS_FINE_PATH, dtype=np.uint16).astype(np.float32) / 65535.0
    log(f"  Loaded {len(flop_ehs):,} flop EHS values")

    preflop_indexer = hand_indexer.PreflopIndexer()
    flop_indexer    = hand_indexer.FlopIndexer()

    assert preflop_indexer.size() == NUM_PREFLOP

    ehs_sum   = np.zeros(NUM_PREFLOP, dtype=np.float64)
    board_cnt = np.zeros(NUM_PREFLOP, dtype=np.int64)

    hole_pairs = list(combinations(range(NUM_CARDS), 2))
    total_pairs = len(hole_pairs)

    log(f"Enumerating {total_pairs:,} hole-card pairs × 19,600 flop boards...")
    t0 = time.time()

    for pair_idx, (c0, c1) in enumerate(hole_pairs):
        hole = np.array([c0, c1], dtype=np.uint8)
        preflop_idx = preflop_indexer.index(hole)

        remaining = np.array([c for c in range(NUM_CARDS) if c != c0 and c != c1],
                             dtype=np.uint8)
        boards = np.array(list(combinations(remaining.tolist(), 3)), dtype=np.uint8)  # (19600, 3)

        cards5 = np.empty((len(boards), 5), dtype=np.uint8)
        cards5[:, 0] = c0
        cards5[:, 1] = c1
        cards5[:, 2:] = boards

        flop_indices = flop_indexer.batch_index(cards5)
        ehs_sum[preflop_idx]   += flop_ehs[flop_indices].sum()
        board_cnt[preflop_idx] += len(flop_indices)

        if verbose and (pair_idx + 1) % 100 == 0:
            elapsed   = time.time() - t0
            done_frac = (pair_idx + 1) / total_pairs
            eta       = elapsed / done_frac * (1 - done_frac) if done_frac > 0 else 0
            print(f"\r  {100 * done_frac:.1f}%  ETA {eta:.0f}s",
                  end='', file=sys.stderr)

    if verbose:
        print(f"\r  100.0%  Done in {time.time() - t0:.1f}s          ", file=sys.stderr)

    per_class_ehs = (ehs_sum / np.maximum(board_cnt, 1)).astype(np.float32)
    log(f"  EHS range: [{per_class_ehs.min():.4f}, {per_class_ehs.max():.4f}]")
    return per_class_ehs


def main():
    parser = argparse.ArgumentParser(
        description="Compute EHS for all 169 canonical preflop hole-hand classes.",
    )
    parser.add_argument("-q", "--quiet", action="store_true",
                        help="Suppress status messages")
    args = parser.parse_args()

    verbose = not args.quiet

    if not FLOP_EHS_FINE_PATH.exists():
        sys.exit(f"Error: {FLOP_EHS_FINE_PATH} not found. "
                 "Run the flop clustering pipeline first.")

    if PREFLOP_EHS_PATH.exists():
        if verbose:
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] "
                  f"{PREFLOP_EHS_PATH} already exists, skipping.",
                  file=sys.stderr)
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ehs = compute_preflop_ehs(verbose=verbose)
    ehs.tofile(PREFLOP_EHS_PATH)

    if verbose:
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] "
              f"Saved {len(ehs)} preflop EHS values to {PREFLOP_EHS_PATH}",
              file=sys.stderr)


if __name__ == "__main__":
    main()
