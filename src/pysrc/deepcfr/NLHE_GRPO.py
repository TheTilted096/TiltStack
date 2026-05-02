"""
NLHE_GRPO.py — TPO training against a fixed target policy.

A single hero policy network is trained from both seats simultaneously.
Each iteration collects samples/2 advantage samples per seat, then trains
the hero network on the combined data for --epochs passes.

Player convention (same as NLHE_Trainer)
-----------------------------------------
  hero=False  ->  player 0  ->  small blind  ->  acts when isButton=True
  hero=True   ->  player 1  ->  big blind    ->  acts when isButton=False
"""

import os
import sys
import time
import argparse
import subprocess
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

import deepcfr
from network_training import (
    DeepCFRNet,
    Reservoir,
    run_inference_loop,
    decode_batch_gpu,
    verify_layout,
    NUM_ACTIONS,
    CONT_DIM,
    NUM_STREETS,
    _ts,
    _fmt,
    _rate,
    _eta,
)
from tb_launch import launch_tb, stop_tb

_GRPO_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_GRPO_DIR, "..", "evaluation"))
from match_runner import run_match, GAME_STRING  # noqa: E402

import pyspiel  # noqa: E402

RESERVOIR_CAPACITY = 100_000_000


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train_tpo_policy(
    net: DeepCFRNet,
    optimizer: torch.optim.Optimizer,
    raw_inputs: np.ndarray,   # (N, INFOSET_BYTES) uint8
    advantages: np.ndarray,   # (N, NUM_ACTIONS)   float32 -- raw milli-chip advantages, NaN for illegal
    old_policy: np.ndarray,   # (N, NUM_ACTIONS)   float32 -- p_old (prior for target distribution q)
    epochs: int,
    eta: float = 1.0,
    batch_size: int = 4096,
    device: torch.device = torch.device("cuda"),
) -> list[float]:
    """Returns losses — one value per epoch."""
    N = (len(raw_inputs) // batch_size) * batch_size  # truncate to whole batches
    raw_pt = torch.from_numpy(raw_inputs[:N])
    adv_pt = torch.from_numpy(advantages[:N])
    pol_pt = torch.from_numpy(old_policy[:N])

    # Single pinned buffers sized to the full (truncated) dataset.
    # Each epoch shuffles the source data into these once; batches are then
    # contiguous pinned slices, so each H2D transfer is a single async DMA.
    raw_pinned = torch.empty(N, raw_inputs.shape[1], dtype=torch.uint8, pin_memory=True)
    adv_pinned = torch.empty(N, NUM_ACTIONS, dtype=torch.float32, pin_memory=True)
    pol_pinned = torch.empty(N, NUM_ACTIONS, dtype=torch.float32, pin_memory=True)

    net.train()
    n_batches = N // batch_size
    losses = []
    for _ in range(epochs):
        perm = torch.randperm(N)
        raw_pinned.copy_(raw_pt[perm])
        adv_pinned.copy_(adv_pt[perm])
        pol_pinned.copy_(pol_pt[perm])

        total_loss = torch.zeros(1, device=device)
        for start in range(0, N, batch_size):
            raw_b = raw_pinned[start : start + batch_size].to(device, non_blocking=True)
            adv_b = adv_pinned[start : start + batch_size].to(device, non_blocking=True)
            pol_b = pol_pinned[start : start + batch_size].to(device, non_blocking=True)
            optimizer.zero_grad()
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                xc, b = decode_batch_gpu(raw_b)
                logits = net(xc, b)

                legal = ~torch.isnan(adv_b)
                adv_clean = adv_b.nan_to_num(0.0)

                # Center advantages per row (per infoset) over legal actions only.
                # This decouples the scale of raw milli-chip values from η so
                # that η is a consistent sharpness parameter regardless of pot size.
                n_legal = legal.float().sum(dim=-1, keepdim=True).clamp(min=1.0)
                adv_mean = (adv_clean * legal.float()).sum(dim=-1, keepdim=True) / n_legal
                u = (adv_clean - adv_mean).masked_fill(~legal, 0.0)

                # Compute target distribution q analytically (fixed constant).
                # q_i ∝ p_old_i * exp(u_i / η)
                log_q = torch.log(pol_b.clamp(min=1e-8)) + u / eta
                log_q = log_q.masked_fill(~legal, float("-inf"))
                q = log_q.softmax(dim=-1)

                # Cross-entropy loss: fit p^θ to q.
                # log_pi is -inf at illegal positions; q is 0 there, but 0 * -inf = NaN
                # in IEEE 754, so mask log_pi to 0 at illegal slots before multiplying.
                log_pi = F.log_softmax(logits.masked_fill(~legal, float("-inf")), dim=-1)
                loss = -(q.detach() * log_pi.masked_fill(~legal, 0.0)).sum(dim=-1).mean()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=0.5)
            optimizer.step()
            total_loss += loss.detach()

        losses.append((total_loss / n_batches).item())
    return losses


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------


def load_policy_net(path: str, device) -> DeepCFRNet:
    ckpt = torch.load(path, map_location=device, weights_only=True)
    net = DeepCFRNet().to(device)
    sd = ckpt["net"]
    if any(k.startswith("_orig_mod.") for k in sd):
        sd = {k.removeprefix("_orig_mod."): v for k, v in sd.items()}
    net.load_state_dict(sd)
    net.eval()
    return net


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="TiltStack TPO training against a fixed target policy"
    )
    parser.add_argument("--target", type=str, required=True)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--samples", type=int, default=10_000)
    parser.add_argument("--batch", type=int, default=16384)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--eta", type=float, default=1.0,
                        help="TPO temperature (lower = q more peaked toward best action)")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--eval_interval", type=int, default=10,
                        help="Evaluate every N iterations (0 to disable)")
    parser.add_argument("--eval_pairs", type=int, default=50_000,
                        help="Duplicate pairs per evaluation match")
    args = parser.parse_args()

    root = Path(__file__).parent.parent.parent
    run_name = time.strftime("%m%d%y_%H%M%S")

    logs_dir = root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / f"tpo-{run_name}.txt"

    _term = sys.stdout
    _log = open(log_path, "w", buffering=1)
    sys.stdout = _log

    _term.write(
        f"TPO training started — output → logs/tpo-{run_name}.txt\n"
        f"  target={args.target}  iters={args.iters}  samples={args.samples:,}  eta={args.eta}\n"
    )

    log_dir = root / "runs" / run_name
    log_dir.mkdir(parents=True, exist_ok=True)
    writer, tb_proc = launch_tb(log_dir)
    _term.write(
        f"  TensorBoard → http://127.0.0.1:6006/?darkMode=false&runFilter={run_name}#timeseries\n\n"
    )
    _term.flush()

    def _shutdown(code: int = 0) -> None:
        writer.close()
        _log.flush()
        _log.close()
        sys.stdout = _term
        stop_tb(tb_proc)
        os._exit(code)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")

    seed = 0xDEADBEEFCAFE1234 if args.seed is None else args.seed

    deepcfr.load_tables("clusters")
    verify_layout(deepcfr.INFOSET_BYTES)

    villain_net = load_policy_net(args.target, device)
    print(f"[{_ts()}]  Villain policy loaded from: {args.target}")

    hero_net = DeepCFRNet().to(device)
    hero_opt = torch.optim.Adam(hero_net.parameters(), lr=args.lr)

    # hero_net = torch.compile(hero_net, fullgraph=True, dynamic=True)
    # villain_net = torch.compile(villain_net, fullgraph=True, dynamic=True)

    osp_game = pyspiel.load_game(GAME_STRING) if args.eval_interval > 0 else None

    adv_res = Reservoir(RESERVOIR_CAPACITY, args.threads, deepcfr.INFOSET_BYTES)
    pol_res = Reservoir(RESERVOIR_CAPACITY, args.threads, deepcfr.INFOSET_BYTES)
    orch = deepcfr.Orchestrator(
        args.threads, adv_res._cpp, adv_res._cpp, pol_res._cpp, seed=seed, grpo=True
    )

    # hero is always nets[0], villain always nets[1] regardless of seat
    nets = [hero_net, villain_net]
    samples_per_seat = args.samples // 2
    iter_times = []

    for t in range(1, args.iters + 1):
        print(f"[{_ts()}] ==> Iteration {t} / {args.iters}")
        iter_start = time.perf_counter()

        # -- Data generation --------------------------------------------------
        adv_res.reset()
        pol_res.reset()
        dg_t0 = time.perf_counter()
        for hero in [False, True]:
            orch.start_iteration(hero, t, samples_per_seat)
            run_inference_loop(orch, device, nets, softmax=True, hero=hero)
            orch.wait_iteration()
            orch.clear_buffers()
            time.sleep(0.5)
        dg_secs = time.perf_counter() - dg_t0

        n = adv_res.size
        print(
            f"\n  [datagen]  {dg_secs:.1f}s"
            f"  ·  samples={_fmt(n)}"
            f"  ·  {_rate(n, dg_secs)} infosets/s"
        )

        # -- Policy training --------------------------------------------------
        train_t0 = time.perf_counter()
        losses = train_tpo_policy(
            hero_net,
            hero_opt,
            adv_res.inputs[:n],
            adv_res.targets[:n],
            pol_res.targets[:n],
            epochs=args.epochs,
            eta=args.eta,
            batch_size=args.batch,
            device=device,
        )
        train_secs = time.perf_counter() - train_t0

        loss_last = losses[-1]
        writer.add_scalar("tpo/loss", loss_last, global_step=t)

        ep_lines = "\n".join(
            f"    ep{i+1}: loss={l:.4f}"
            for i, l in enumerate(losses)
        )
        print(
            f"\n  [training]  {train_secs:.1f}s  ·  epochs={args.epochs}"
            f"\n{ep_lines}"
        )

        # -- Iteration summary ------------------------------------------------
        iter_elapsed = time.perf_counter() - iter_start
        iter_times.append(iter_elapsed)
        remaining = args.iters - t
        eta_str = (
            f"  ETA {_eta(sum(iter_times) / len(iter_times) * remaining)}"
            f"  ({remaining} remaining)"
            if remaining > 0
            else "  (final iteration)"
        )
        print(f"\n  iter {iter_elapsed:.1f}s{eta_str}\n")

        # -- Evaluation -------------------------------------------------------
        if osp_game is not None and args.eval_interval > 0 and t % args.eval_interval == 0:
            print(f"[{_ts()}]  Evaluating hero vs villain ({args.eval_pairs:,} pairs)...")
            mbb, ci95 = run_match(
                hero_net, villain_net, osp_game, device, args.eval_pairs,
                p0_argmax=True, p1_argmax=False, p0_label="hero",
            )
            writer.add_scalar("tpo/hero_edge_mbb", mbb, global_step=t)
            print(f"[{_ts()}]  Hero edge: {mbb:+.2f} +/- {ci95:.2f} mBB/hand")

    ckpt_dir = root / "tpo_checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    date = time.strftime("%m%d%y")
    ckpt_path = ckpt_dir / f"tpo{args.iters:04d}-{date}.pt"
    torch.save({"t": args.iters, "net": hero_net.state_dict()}, ckpt_path)

    msg = f"[{_ts()}] ==> Done.  Hero policy → {ckpt_path.name}"
    print(msg)
    _term.write(f"\n{msg}\n")
    _term.flush()
    _shutdown()


if __name__ == "__main__":
    main()
