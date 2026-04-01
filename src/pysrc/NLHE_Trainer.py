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

def save_checkpoint(ckpt_dir, t, adv_nets, strat_net, adv_opts, strat_opt):
    path = os.path.join(ckpt_dir, f"ckpt_t{t:04d}.pt")
    torch.save({
        "t":         t,
        "adv_net_0": adv_nets[0].state_dict(),
        "adv_net_1": adv_nets[1].state_dict(),
        "strat_net": strat_net  .state_dict(),
        "adv_opt_0": adv_opts[0].state_dict(),
        "adv_opt_1": adv_opts[1].state_dict(),
        "strat_opt": strat_opt  .state_dict(),
    }, path)
    return path


def load_checkpoint(ckpt_dir, adv_nets, strat_net, adv_opts, strat_opt, device):
    ckpts = sorted(f for f in os.listdir(ckpt_dir) if f.startswith("ckpt_t"))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints found in {ckpt_dir}")
    path = os.path.join(ckpt_dir, ckpts[-1])
    ckpt = torch.load(path, map_location=device)
    adv_nets[0].load_state_dict(ckpt["adv_net_0"])
    adv_nets[1].load_state_dict(ckpt["adv_net_1"])
    strat_net   .load_state_dict(ckpt["strat_net"])
    adv_opts[0] .load_state_dict(ckpt["adv_opt_0"])
    adv_opts[1] .load_state_dict(ckpt["adv_opt_1"])
    strat_opt   .load_state_dict(ckpt["strat_opt"])
    print(f"  Loaded checkpoint: {path}")
    return ckpt["t"] + 1


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="TiltStack DeepCFR training loop")
    parser.add_argument("--iters",      type=int,   default=20,
        help="CFR iterations T  (default: 20)")
    parser.add_argument("--threads",    type=int,   default=8)
    parser.add_argument("--samples",    type=int,   default=1_000_000,
        help="Target advantage samples per player per iteration  (default: 1M)")
    parser.add_argument("--batch",      type=int,   default=4096)
    parser.add_argument("--epochs",     type=int,   default=1,
        help="Advantage training epochs per iteration  (default: 1)")
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--seed",       type=int,   default=None)
    parser.add_argument("--ckpt-dir",   type=str,   default="checkpoints")
    parser.add_argument("--ckpt-every", type=int,   default=10)
    parser.add_argument("--resume",     type=str,   default=None,
        help="Checkpoint directory to resume from")
    args = parser.parse_args()

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

    start_t = 1
    if args.resume:
        start_t = load_checkpoint(
            args.resume, adv_nets, strat_net, adv_opts, strat_opt, device)
        print(f"  Resuming from t={start_t}\n")

    seed = 0xdeadbeefcafe1234 if args.seed is None else args.seed
    orch = deepcfr.Orchestrator(args.threads, seed)
    os.makedirs(args.ckpt_dir, exist_ok=True)

    adv_res   = [Reservoir(RESERVOIR_CAPACITY, deepcfr.INFOSET_BYTES)
                 for _ in range(2)]
    strat_res =  Reservoir(RESERVOIR_CAPACITY, deepcfr.INFOSET_BYTES,
                           has_weights=True)

    # ---- Training loop ------------------------------------------------------
    iter_times = []

    for t in range(start_t, args.iters + 1):
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

            collect_into_reservoirs(orch, hero, adv_res, strat_res)
            orch.clear_buffers()

            adv_new   = adv_res[player].n_seen - adv_before
            pol_new   = strat_res.n_seen - pol_before
            adv_size  = adv_res[player].size
            pol_size  = strat_res.size
            cap_str   = _fmt(RESERVOIR_CAPACITY)

            print(f"\n  [P{player} rollout]  {rollout_secs:.1f}s"
                  f"  ·  {_rate(adv_new + pol_new, rollout_secs)}")
            print(f"    adv  +{_fmt(adv_new):<12}"
                  f"  res  {_fmt(adv_size):>12} / {cap_str}")
            print(f"    pol  +{_fmt(pol_new):<12}"
                  f"  res  {_fmt(pol_size):>12} / {cap_str}")

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
                    epochs=args.epochs,
                    device=device,
                )
                train_secs = time.perf_counter() - t0
                loss_str   = "  ".join(f"{l:.5f}" for l in losses)
                print(f"\n  [P{player} adv]  n={_fmt(n_adv)}"
                      f"  ·  loss={loss_str}"
                      f"  ·  {train_secs:.1f}s"
                      f"  ·  {_rate(n_adv * args.epochs, train_secs)}")

        # -- Iteration summary ------------------------------------------------
        iter_elapsed = time.perf_counter() - iter_start
        iter_times.append(iter_elapsed)
        remaining = args.iters - t
        eta_str = (f"  ETA {_eta(sum(iter_times)/len(iter_times) * remaining)}"
                   f"  ({remaining} remaining)" if remaining > 0 else "  (final iteration)")
        print(f"\n  iter {iter_elapsed:.1f}s{eta_str}\n")

        if t % args.ckpt_every == 0:
            path = save_checkpoint(
                args.ckpt_dir, t,
                adv_nets, strat_net, adv_opts, strat_opt)
            print(f"  checkpoint → {path}\n")

    # ---- Strategy network ---------------------------------------------------
    strat_epochs = 10
    n_pol = strat_res.size
    print(f"[{_ts()}] ==> Strategy network\n"
          f"  n={_fmt(n_pol)}  ·  {strat_epochs} epochs\n")

    t0 = time.perf_counter()
    losses = train_policy(
        strat_net, strat_opt,
        strat_res.inputs [:n_pol],
        strat_res.targets[:n_pol],
        strat_res.weights[:n_pol],
        batch_size=args.batch,
        epochs=strat_epochs,
        device=device,
    )
    strat_secs = time.perf_counter() - t0

    for ep, l in enumerate(losses, 1):
        print(f"    ep {ep:2d} / {strat_epochs}   loss = {l:.5f}")
    print(f"\n  {strat_secs:.1f}s  ·  {_rate(n_pol * strat_epochs, strat_secs)}\n")

    path = save_checkpoint(
        args.ckpt_dir, args.iters,
        adv_nets, strat_net, adv_opts, strat_opt)
    print(f"[{_ts()}] ==> Done.  Final checkpoint → {path}")


if __name__ == "__main__":
    main()
