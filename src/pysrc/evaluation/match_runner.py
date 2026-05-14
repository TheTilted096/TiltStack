"""
match_runner.py — Batched head-to-head evaluation between two TiltStack strategy networks.

Each game pair is a self-contained Task<float> coroutine (Match::gamePair) that
plays both passes (same shuffled cards, swapped seats) and records per-pass payoffs
into per-thread Scheduler vectors.  The Orchestrator manages the coroutine pool;
Python only services inference requests.

At each inference suspension, the Scheduler's net_idx_data() array tells Python
which of the two networks should handle each request.  Python partitions the batch
by network, runs both forward passes (on separate CUDA streams when available),
writes outputs back into the shared output buffer, and calls submit_batch().

Usage:
    cd src
    python pysrc/evaluation/match_runner.py \\
        --p0     checkpoints/policy_v1.pt \\
        --p1     checkpoints/policy_v2.pt \\
        --pairs  100000 \\
        --device cpu
"""

import argparse
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

_EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_EVAL_DIR, "..", "deepcfr"))

import deepcfr
from tilt_agents import load_net_auto
from network_training import decode_batch, NUM_ACTIONS

# ---------------------------------------------------------------------------
# Game configuration — must match CFRTypes.h training parameters
# ---------------------------------------------------------------------------

BIG_BLIND_MILLI = 2000  # must match CFRTypes.h BIG_BLIND
CHIPS_TO_MBB = 1000 / BIG_BLIND_MILLI  # = 0.5  (milli-chips → mBB)

_CLUSTERS = os.path.join(_EVAL_DIR, "..", "..", "clusters")

# ---------------------------------------------------------------------------
# flags bit layout (see MatchGame.h)
#   bit 0 : net0 argmax    bit 2 : net0 prune
#   bit 1 : net1 argmax    bit 3 : net1 prune
# ---------------------------------------------------------------------------


def _make_flags(
    p0_argmax: bool, p1_argmax: bool, p0_prune: bool, p1_prune: bool
) -> int:
    return (
        int(p0_argmax)
        | (int(p1_argmax) << 1)
        | (int(p0_prune) << 2)
        | (int(p1_prune) << 3)
    )


# ---------------------------------------------------------------------------
# Inference service loop
# ---------------------------------------------------------------------------


def _service_inference(
    orch, net_p0, net_p1, device: torch.device, num_threads: int
) -> None:
    """
    Pop inference batches from the Orchestrator and service them until all
    workers have pushed their completion sentinels.

    Each batch may contain requests for net0, net1, or both.  net_idx_data()
    partitions the batch; outputs are written back into the shared buffer
    before submit_batch() resumes the waiting worker.
    """
    use_cuda = device.type == "cuda"
    if use_cuda:
        stream0 = torch.cuda.Stream(device)
        stream1 = torch.cuda.Stream(device)

    sentinels = 0
    while sentinels < num_threads:
        sched = orch.pop()
        if sched is None:
            sentinels += 1
            continue

        n = sched.batch_size()
        net_idx = sched.net_idx_data()  # (n,) int32, zero-copy
        outputs = sched.output_data()  # (n, NUM_ACTIONS) float32, writable

        idx0 = np.where(net_idx == 0)[0]
        idx1 = np.where(net_idx == 1)[0]

        with torch.no_grad():
            if use_cuda:
                if len(idx0):
                    with torch.cuda.stream(stream0):
                        x0, b0 = decode_batch(sched.input_data()[idx0])
                        logits0 = F.softmax(
                            net_p0(
                                x0.to(device, non_blocking=True),
                                b0.to(device, non_blocking=True),
                            ),
                            dim=1,
                        )
                if len(idx1):
                    with torch.cuda.stream(stream1):
                        x1, b1 = decode_batch(sched.input_data()[idx1])
                        logits1 = F.softmax(
                            net_p1(
                                x1.to(device, non_blocking=True),
                                b1.to(device, non_blocking=True),
                            ),
                            dim=1,
                        )
                torch.cuda.synchronize(device)
                if len(idx0):
                    outputs[idx0] = logits0.cpu().numpy()
                if len(idx1):
                    outputs[idx1] = logits1.cpu().numpy()
            else:
                if len(idx0):
                    x0, b0 = decode_batch(sched.input_data()[idx0])
                    outputs[idx0] = F.softmax(
                        net_p0(x0.to(device), b0.to(device)), dim=1
                    ).numpy()
                if len(idx1):
                    x1, b1 = decode_batch(sched.input_data()[idx1])
                    outputs[idx1] = F.softmax(
                        net_p1(x1.to(device), b1.to(device)), dim=1
                    ).numpy()

        sched.submit_batch()


# ---------------------------------------------------------------------------
# Main evaluation function
# ---------------------------------------------------------------------------


def run_match(
    net_p0,
    net_p1,
    device: torch.device,
    pairs: int,
    num_threads: int = 0,
    verbose: bool = True,
    p0_argmax: bool = False,
    p1_argmax: bool = False,
    p0_prune: bool = True,
    p1_prune: bool = True,
    p0_label: str = "P0",
    report_every: int = 0,
) -> tuple[float, float]:
    """
    Run a batched duplicate-pair evaluation of net_p0 vs net_p1.

    Parameters
    ----------
    net_p0, net_p1 : DeepCFRNet  — pre-loaded, eval()-mode strategy networks
    device         : torch.device
    pairs          : number of duplicate pairs to complete
    num_threads    : worker thread count (0 → hardware concurrency)
    verbose        : print overall result on completion
    report_every   : print rolling edge+CI every N pairs (0 = off).
                     The match is chunked at this boundary; reporting
                     happens between wait_iteration() and the next
                     start_match(), the only window the main thread is free.

    Returns
    -------
    overall_mbb : float  — P0's edge over P1 in mBB/hand (positive = P0 wins)
    ci95        : float  — 95% CI half-width (from per-pair variance)
    """
    if num_threads <= 0:
        num_threads = os.cpu_count() or 4

    flags = _make_flags(p0_argmax, p1_argmax, p0_prune, p1_prune)
    chunk_size = report_every if report_every > 0 else pairs

    sb_chunks, bb_chunks = [], []
    filled = 0
    orch = deepcfr.Orchestrator(num_threads)
    t0 = time.perf_counter()
    w = len(f"{pairs:,}")  # field width for progress column

    while filled < pairs:
        chunk = min(chunk_size, pairs - filled)
        orch.start_match(chunk, flags)
        _service_inference(orch, net_p0, net_p1, device, num_threads)
        orch.wait_iteration()

        sb, bb = orch.collect_match_payoffs()
        sb_chunks.append(sb)
        bb_chunks.append(bb)
        filled += len(sb)

        if report_every > 0:
            sb_so_far = np.concatenate(sb_chunks)
            bb_so_far = np.concatenate(bb_chunks)
            pair_totals = sb_so_far + bb_so_far
            n = len(pair_totals)
            edge = pair_totals.mean() / 2 * CHIPS_TO_MBB
            ci = 1.96 * pair_totals.std(ddof=1) / 2 / math.sqrt(n) * CHIPS_TO_MBB
            elapsed = time.perf_counter() - t0
            rate = n / elapsed if elapsed > 0 else 0
            eta = (pairs - n) / rate if rate > 0 else float("inf")
            eta_str = f"{eta:.0f}s" if math.isfinite(eta) else "?"
            print(
                f"  [{n:{w},} / {pairs:,}]  "
                f"edge: {edge:+.2f} ± {ci:.2f} mBB/hand  "
                f"({rate:.0f} pairs/s, ~{eta_str} remaining)"
            )

    sb_payoffs = np.concatenate(sb_chunks)
    bb_payoffs = np.concatenate(bb_chunks)
    pair_totals = sb_payoffs + bb_payoffs
    n = len(pair_totals)
    overall_mbb = pair_totals.mean() / 2 * CHIPS_TO_MBB
    ci95 = 1.96 * pair_totals.std(ddof=1) / 2 / math.sqrt(n) * CHIPS_TO_MBB

    if verbose:
        sb_mbb = sb_payoffs.sum() / n * CHIPS_TO_MBB
        bb_mbb = bb_payoffs.sum() / n * CHIPS_TO_MBB
        print(f"\n{'─' * 60}")
        print(f"  Pairs played           : {n:,}  ({n * 2:,} hands)")
        print(f"  {p0_label} as SB              : {sb_mbb:+.2f} mBB/hand")
        print(f"  {p0_label} as BB              : {bb_mbb:+.2f} mBB/hand")
        print(
            f"  {p0_label} overall edge vs P1 : {overall_mbb:+.2f} ± {ci95:.2f} mBB/hand"
        )

    return overall_mbb, ci95


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="TiltStack head-to-head network evaluation"
    )
    parser.add_argument("--p0", required=True, help="P0 strategy checkpoint")
    parser.add_argument("--p1", required=True, help="P1 strategy checkpoint")
    parser.add_argument("--pairs", type=int, default=100_000)
    parser.add_argument(
        "--threads",
        type=int,
        default=0,
        help="Worker threads (0 = hardware concurrency)",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--p0-argmax", action="store_true")
    parser.add_argument("--p1-argmax", action="store_true")
    parser.add_argument(
        "--report-every",
        type=int,
        default=500_000,
        help="Print rolling edge+CI every N pairs (0 = off; default 50k = 100k hands)",
    )
    args = parser.parse_args()

    if not os.path.isdir(_CLUSTERS):
        sys.exit(f"Error: clusters directory not found at '{_CLUSTERS}'")
    deepcfr.load_tables(_CLUSTERS)

    device = torch.device(args.device)
    print(f"Loading P0 net: {args.p0}")
    net_p0 = load_net_auto(args.p0, device)
    print(f"Loading P1 net: {args.p1}")
    net_p1 = load_net_auto(args.p1, device)

    run_match(
        net_p0,
        net_p1,
        device,
        args.pairs,
        num_threads=args.threads,
        p0_argmax=args.p0_argmax,
        p1_argmax=args.p1_argmax,
        report_every=args.report_every,
    )


if __name__ == "__main__":
    main()
