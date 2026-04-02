"""
NLHE_Trainer.py — DeepCFR main training loop for TiltStack.

Networks
--------
  adv_net[0]  advantage network for player 0 (small blind)
  adv_net[1]  advantage network for player 1 (big blind / button)
  strat_net   shared strategy network

Training loop (alternating traversal)
--------------------------------------
For t = 1..T:
  For hero in [False, True]:   (player 0 first, then player 1)
    1. start_iteration(hero, t, total_samples)
    2. Inference loop: route each infoset to the acting player's adv_net.
    3. wait_iteration()
    4. collect_into_reservoirs()
    5. clear_buffers()
    6. Train adv_net[hero] on the full adv_res[hero].

After all T iterations:
    Train strat_net once on the full strat_res.

Player convention
-----------------
  hero=False  →  player 0  →  small blind  →  acts when isButton=True
  hero=True   →  player 1  →  big blind    →  acts when isButton=False
"""

import os
import time
import argparse
from pathlib import Path
import numpy as np
import torch

import deepcfr
from network_training import (
    DeepCFRNet, Reservoir, train_advantage, train_policy,
    decode_batch, verify_layout, infoset_dtype, NUM_ACTIONS,
)

RESERVOIR_CAPACITY = 20_000_000


# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------

def _ts() -> str:
    return time.strftime('%H:%M:%S')

def _fmt(n: int) -> str:
    return f"{n:,}"

def _rate(n: int, secs: float) -> str:
    r = n / secs if secs > 0 else 0.0
    if r >= 1e6:
        return f"{r/1e6:.1f}M/s"
    if r >= 1e3:
        return f"{r/1e3:.1f}k/s"
    return f"{r:.0f}/s"

def _eta(secs: float) -> str:
    s = int(secs)
    h, rem = divmod(s, 3600)
    m, s   = divmod(rem, 60)
    if h > 0:
        return f"{h}h {m:02d}m {s:02d}s"
    if m > 0:
        return f"{m}m {s:02d}s"
    return f"{s}s"


# ---------------------------------------------------------------------------
# Inference loop
# ---------------------------------------------------------------------------

def run_inference_loop(orch, adv_nets, device):
    done = 0
    while done < orch.num_threads():
        sched = orch.pop()
        if sched is None:
            done += 1
            continue

        n   = sched.batch_size()
        raw = np.array(sched.input_data(), copy=False)

        x_cont, buckets = decode_batch(raw)
        x_cont  = x_cont .to(device, non_blocking=True)
        buckets = buckets.to(device, non_blocking=True)

        struct       = raw.ravel().view(infoset_dtype)
        is_p0_acting = struct['is_button']

        p0_idx = np.where( is_p0_acting)[0]
        p1_idx = np.where(~is_p0_acting)[0]

        out = np.empty((n, NUM_ACTIONS), dtype=np.float32)

        with torch.no_grad():
            if len(p0_idx) > 0:
                out[p0_idx] = adv_nets[0](
                    x_cont[p0_idx], buckets[p0_idx]).cpu().numpy()
            if len(p1_idx) > 0:
                out[p1_idx] = adv_nets[1](
                    x_cont[p1_idx], buckets[p1_idx]).cpu().numpy()

        sched.output_data()[:] = out
        sched.submit_batch()


# ---------------------------------------------------------------------------
# Data collection
# ---------------------------------------------------------------------------

def collect_into_reservoirs(orch, hero: bool,
                             adv_res: list, strat_res: 'Reservoir') -> None:
    player = int(hero)
    for sched in orch.schedulers:
        if sched.advantage_size() > 0:
            adv_res[player].add(
                np.array(sched.advantage_input_data(),  copy=True),
                np.array(sched.advantage_output_data(), copy=True),
            )
        if sched.policy_size() > 0:
            strat_res.add(
                np.array(sched.policy_input_data(),  copy=True),
                np.array(sched.policy_output_data(), copy=True),
                np.array(sched.policy_weight_data(), copy=True),
            )


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

def save_final_policy(ckpt_dir, t, strat_net):
    """Save only the final policy network."""
    policy_path = os.path.join(ckpt_dir, f"policy{t:04d}.pt")
    torch.save({
        "t": t,
        "net": strat_net.state_dict(),
    }, policy_path)

    return policy_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="TiltStack DeepCFR training loop")
    parser.add_argument("--iters",      type=int,   default=100,
        help="CFR iterations T  (default: 100)")
    parser.add_argument("--threads",    type=int,   default=16)
    parser.add_argument("--samples",    type=int,   default=4_000_000,
        help="Target advantage samples per player per iteration  (default: 4M)")
    parser.add_argument("--batch",      type=int,   default=8192)
    parser.add_argument("--epochs",     type=int,   default=25,
        help="Strategy network training epochs  (default: 25)")
    parser.add_argument("--lr",         type=float, default=3e-4)
    parser.add_argument("--seed",       type=int,   default=None)
    args = parser.parse_args()

    ckpt_dir = Path(__file__).parent.parent.parent / "checkpoints"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- Header -------------------------------------------------------------
    samples_str = (f"{args.samples/1e6:.1f}M" if args.samples >= 1_000_000
                   else _fmt(args.samples))
    print(f"[{_ts()}]  device={device}  threads={args.threads}"
          f"  samples/iter={samples_str}  iters={args.iters}\n")

    # ---- One-time setup -----------------------------------------------------
    deepcfr.load_tables("clusters")
    verify_layout(deepcfr.INFOSET_BYTES)

    adv_nets  = [DeepCFRNet().to(device), DeepCFRNet().to(device)]
    strat_net =  DeepCFRNet().to(device)
    adv_opts  = [torch.optim.Adam(n.parameters(), lr=args.lr) for n in adv_nets]
    strat_opt =  torch.optim.Adam(strat_net.parameters(),     lr=args.lr)

    seed = 0xdeadbeefcafe1234 if args.seed is None else args.seed
    orch = deepcfr.Orchestrator(args.threads, seed)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    adv_res   = [Reservoir(RESERVOIR_CAPACITY, deepcfr.INFOSET_BYTES)
                 for _ in range(2)]
    strat_res =  Reservoir(RESERVOIR_CAPACITY, deepcfr.INFOSET_BYTES,
                           has_weights=True)

    # ---- Training loop ------------------------------------------------------
    iter_times = []

    for t in range(1, args.iters + 1):
        print(f"[{_ts()}] ==> Iteration {t} / {args.iters}")
        iter_start = time.perf_counter()

        for hero in [False, True]:
            player = int(hero)

            # -- Rollout ------------------------------------------------------
            adv_before = adv_res[player].n_seen
            pol_before = strat_res.n_seen

            t0 = time.perf_counter()
            orch.start_iteration(hero, t, args.samples)
            run_inference_loop(orch, adv_nets, device)
            orch.wait_iteration()
            rollout_secs = time.perf_counter() - t0

            rollouts = sum(s.rollout_count() for s in orch.schedulers)
            collect_into_reservoirs(orch, hero, adv_res, strat_res)
            orch.clear_buffers()

            adv_new   = adv_res[player].n_seen - adv_before
            pol_new   = strat_res.n_seen - pol_before
            adv_size  = adv_res[player].size
            pol_size  = strat_res.size
            cap_str   = _fmt(RESERVOIR_CAPACITY)

            print(f"\n  [P{player} rollout]  {rollout_secs:.1f}s"
                  f"  ·  rollouts={_fmt(rollouts)}"
                  f"  ·  {_rate(adv_new + pol_new, rollout_secs)} infosets/s")
            print(f"    advantage  +{_fmt(adv_new):<12}"
                  f"  reservoir  {_fmt(adv_size):>12} / {cap_str}")
            print(f"    policy     +{_fmt(pol_new):<12}"
                  f"  reservoir  {_fmt(pol_size):>12} / {cap_str}")

            # -- Advantage training -------------------------------------------
            n_adv = adv_res[player].size
            if n_adv > 0:
                adv_opts[player].state.clear()
                t0 = time.perf_counter()
                losses = train_advantage(
                    adv_nets[player], adv_opts[player],
                    adv_res[player].inputs [:n_adv],
                    adv_res[player].targets[:n_adv],
                    batch_size=args.batch,
                    epochs=1,
                    device=device,
                )
                train_secs = time.perf_counter() - t0
                loss_str   = "  ".join(f"{l:.5f}" for l in losses)
                print(f"\n  [P{player} advantage]  samples={_fmt(n_adv)}"
                      f"  ·  loss={loss_str}"
                      f"  ·  {train_secs:.1f}s"
                      f"  ·  {_rate(n_adv, train_secs)} samples/s")

        # -- Iteration summary ------------------------------------------------
        iter_elapsed = time.perf_counter() - iter_start
        iter_times.append(iter_elapsed)
        remaining = args.iters - t
        eta_str = (f"  ETA {_eta(sum(iter_times)/len(iter_times) * remaining)}"
                   f"  ({remaining} remaining)" if remaining > 0 else "  (final iteration)")
        print(f"\n  iter {iter_elapsed:.1f}s{eta_str}\n")

    # ---- Strategy network ---------------------------------------------------
    strat_epochs = args.epochs
    n_pol = strat_res.size
    print(f"[{_ts()}] ==> Strategy network\n"
          f"  samples={_fmt(n_pol)}  ·  {strat_epochs} epochs\n")

    def _strat_epoch_cb(ep, loss, secs):
        print(f"    ep {ep:2d} / {strat_epochs}   loss = {loss:.5f}  ·  {secs:.1f}s")

    t0 = time.perf_counter()
    train_policy(
        strat_net, strat_opt,
        strat_res.inputs [:n_pol],
        strat_res.targets[:n_pol],
        strat_res.weights[:n_pol],
        batch_size=args.batch,
        epochs=strat_epochs,
        device=device,
        epoch_callback=_strat_epoch_cb,
    )
    strat_secs = time.perf_counter() - t0
    print(f"\n  {strat_secs:.1f}s  ·  {_rate(n_pol * strat_epochs, strat_secs)} samples/s\n")

    path = save_final_policy(ckpt_dir, args.iters, strat_net)
    print(f"[{_ts()}] ==> Done.  Final policy → {path.split('/')[-1]}")


if __name__ == "__main__":
    main()
