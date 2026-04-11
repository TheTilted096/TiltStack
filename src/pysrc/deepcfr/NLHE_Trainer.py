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
import sys
import time
import signal
import atexit
import argparse
import subprocess
from pathlib import Path
import numpy as np
import torch
from tb_launch import launch_tb

import deepcfr
from network_training import (
    DeepCFRNet, Reservoir, train_advantage, train_policy,
    decode_batch_gpu, verify_layout, infoset_dtype, NUM_ACTIONS,
    CONT_DIM, NUM_STREETS,
)


RESERVOIR_CAPACITY = 100_000_000


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
        # Block until at least one batch is ready, then drain any additional
        # batches that are already waiting — all get fused into one GPU call.
        first = orch.pop()
        rest  = orch.drain()

        scheds = []
        for item in [first] + list(rest):
            if item is None:
                done += 1
            else:
                scheds.append(item)

        if not scheds:
            continue

        sizes   = [s.batch_size() for s in scheds]
        raws    = [np.array(s.input_data(), copy=False) for s in scheds]
        raw_cat = np.concatenate(raws, axis=0) if len(raws) > 1 else raws[0]

        x_cont, buckets = decode_batch_gpu(
            torch.from_numpy(raw_cat).to(device, non_blocking=True))

        struct       = raw_cat.ravel().view(infoset_dtype)
        is_p0_acting = struct['is_button']

        p0_idx = np.where( is_p0_acting)[0]
        p1_idx = np.where(~is_p0_acting)[0]

        out = np.empty((raw_cat.shape[0], NUM_ACTIONS), dtype=np.float32)

        with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            if len(p0_idx) > 0:
                out[p0_idx] = adv_nets[0](
                    x_cont[p0_idx], buckets[p0_idx]).float().cpu().numpy()
            if len(p1_idx) > 0:
                out[p1_idx] = adv_nets[1](
                    x_cont[p1_idx], buckets[p1_idx]).float().cpu().numpy()

        offset = 0
        for s, sz in zip(scheds, sizes):
            s.output_data()[:] = out[offset:offset + sz]
            s.submit_batch()
            offset += sz


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

def save_final_policy(ckpt_dir, t, strat_net):
    """Save only the final policy network."""
    date = time.strftime('%m%d%y')
    policy_path = os.path.join(ckpt_dir, f"policy{t:04d}-{date}.pt")
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
    parser.add_argument("--iters",      type=int,   default=150,
        help="CFR iterations T  (default: 150)")
    parser.add_argument("--threads",    type=int,   default=16)
    parser.add_argument("--samples",    type=int,   default=10_000_000,
        help="Target advantage samples per player per iteration  (default: 10M)")
    parser.add_argument("--batch",      type=int,   default=16384)
    parser.add_argument("--epochs",     type=int,   default=4,
        help="Strategy network training epochs  (default: 4)")
    parser.add_argument("--adv-step",   type=int,   default=2500,
        help="Mini-batches of advantage training per iteration  (default: 2500)")
    parser.add_argument("--lr",         type=float, default=3e-4)
    parser.add_argument("--seed",       type=int,   default=None)
    args = parser.parse_args()

    ckpt_dir = Path(__file__).parent.parent.parent / "checkpoints"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")

    # ---- Header -------------------------------------------------------------
    samples_str = (f"{args.samples/1e6:.1f}M" if args.samples >= 1_000_000
                   else _fmt(args.samples))
    print(f"[{_ts()}]  device={device}  threads={args.threads}"
          f"  samples/iter={samples_str}  iters={args.iters}\n")

    # ---- TensorBoard --------------------------------------------------------
    # Launched before CUDA init and torch.compile so the stderr suppress
    # window covers both phases, where gRPC triggers its startup noise.
    seed = 0xdeadbeefcafe1234 if args.seed is None else args.seed
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    run_name = time.strftime('%m%d%y_%H%M%S')
    log_dir  = Path(__file__).parent.parent.parent / "runs" / run_name
    log_dir.mkdir(parents=True, exist_ok=True)
    writer   = launch_tb(log_dir)
    atexit.register(lambda: subprocess.run(["pkill", "-f", "tensorboard"], capture_output=True))
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    print(f"[{_ts()}]  TensorBoard → http://127.0.0.1:6006/?darkMode=false&runFilter={run_name}#timeseries\n")

    # ---- One-time setup -----------------------------------------------------
    deepcfr.load_tables("clusters")
    verify_layout(deepcfr.INFOSET_BYTES)

    adv_nets  = [DeepCFRNet().to(device), DeepCFRNet().to(device)]
    strat_net =  DeepCFRNet().to(device)
    adv_opts  = [torch.optim.Adam(n.parameters(), lr=args.lr) for n in adv_nets]
    strat_opt =  torch.optim.Adam(strat_net.parameters(),     lr=args.lr)

    print(f"[{_ts()}]  torch.compile — tracing networks for kernel fusion ...", flush=True)
    _t_compile = time.perf_counter()
    adv_nets  = [torch.compile(net, fullgraph=True) for net in adv_nets]
    strat_net =  torch.compile(strat_net, fullgraph=True)
    _dummy_xc = torch.zeros(args.batch, CONT_DIM,    device=device)
    _dummy_b  = torch.zeros(args.batch, NUM_STREETS, device=device, dtype=torch.long)
    with torch.no_grad():
        for _net in (*adv_nets, strat_net):
            _net(_dummy_xc, _dummy_b)
    print(f"[{_ts()}]  Compiled in {time.perf_counter() - _t_compile:.1f}s\n")

    adv_res   = [Reservoir(RESERVOIR_CAPACITY, args.threads, deepcfr.INFOSET_BYTES)
                 for _ in range(2)]
    strat_res =  Reservoir(RESERVOIR_CAPACITY, args.threads, deepcfr.INFOSET_BYTES,
                           has_weights=True)
    orch = deepcfr.Orchestrator(args.threads,
                                adv_res[0]._cpp, adv_res[1]._cpp, strat_res._cpp,
                                seed)

    # ---- Training loop ------------------------------------------------------
    iter_times = []
    last_t      = 0      # last fully completed CFR iteration
    _interrupted = False

    try:
        for t in range(1, args.iters + 1):
            print(f"[{_ts()}] ==> Iteration {t} / {args.iters}")
            iter_start = time.perf_counter()

            for hero in [False, True]:
                player = int(hero)

                # -- Rollout --------------------------------------------------
                adv_before = adv_res[player].n_seen
                collect_policy = t > 50
                pol_before = strat_res.n_seen if collect_policy else 0

                t0 = time.perf_counter()
                orch.start_iteration(hero, t, args.samples)
                run_inference_loop(orch, adv_nets, device)
                orch.wait_iteration()
                rollout_secs = time.perf_counter() - t0

                rollouts = sum(s.rollout_count() for s in orch.schedulers)
                orch.clear_buffers()

                adv_new  = adv_res[player].n_seen - adv_before
                adv_size = adv_res[player].size
                cap_str  = _fmt(RESERVOIR_CAPACITY)

                print(f"\n  [P{player} rollout]  {rollout_secs:.1f}s"
                      f"  ·  rollouts={_fmt(rollouts)}"
                      f"  ·  {_rate(adv_new, rollout_secs)} infosets/s")
                print(f"    advantage  +{_fmt(adv_new):<12}"
                      f"  reservoir  {_fmt(adv_size):>12} / {cap_str}")
                if collect_policy:
                    pol_new  = strat_res.n_seen - pol_before
                    pol_size = strat_res.size
                    print(f"    policy     +{_fmt(pol_new):<12}"
                          f"  reservoir  {_fmt(pol_size):>12} / {cap_str}")

                # -- Advantage training ---------------------------------------
                n_adv = adv_res[player].size
                if n_adv > 0:
                    adv_nets[player]._orig_mod.apply(
                        lambda m: m.reset_parameters() if hasattr(m, 'reset_parameters') else None)
                    adv_opts[player] = torch.optim.Adam(adv_nets[player].parameters(), lr=args.lr)
                    t0 = time.perf_counter()
                    losses = train_advantage(
                        adv_nets[player], adv_opts[player],
                        adv_res[player].inputs [:n_adv],
                        adv_res[player].targets[:n_adv],
                        batch_size=args.batch,
                        max_steps=args.adv_step,
                        device=device,
                    )
                    train_secs = time.perf_counter() - t0
                    samples_seen = args.adv_step * args.batch
                    writer.add_scalar(f"adv/p{player}", losses[-1], global_step=t)
                    print(f"\n  [P{player} advantage]  samples={_fmt(n_adv)}"
                          f"  ·  steps={args.adv_step}"
                          f"  ·  loss={losses[-1]:.5f}"
                          f"  ·  {train_secs:.1f}s"
                          f"  ·  {_rate(samples_seen, train_secs)} samples/s")

            # -- Iteration summary --------------------------------------------
            iter_elapsed = time.perf_counter() - iter_start
            iter_times.append(iter_elapsed)
            remaining = args.iters - t
            eta_str = (f"  ETA {_eta(sum(iter_times)/len(iter_times) * remaining)}"
                       f"  ({remaining} remaining)" if remaining > 0 else "  (final iteration)")
            print(f"\n  iter {iter_elapsed:.1f}s{eta_str}\n")
            last_t = t

    except KeyboardInterrupt:
        _interrupted = True
        print(f"\n[{_ts()}]  Interrupted after {last_t} iterations — skipping to policy training ...")
        writer.flush()

    # ---- Strategy network ---------------------------------------------------
    # A second Ctrl-C here propagates normally and exits without saving.
    n_pol = strat_res.size
    if n_pol == 0:
        print(f"[{_ts()}]  No policy samples collected — nothing to save.")
        writer.close()
        raise SystemExit(1)

    strat_epochs = args.epochs
    print(f"[{_ts()}] ==> Strategy network\n"
          f"  samples={_fmt(n_pol)}  ·  {strat_epochs} epochs\n")

    t0 = time.perf_counter()
    train_policy(
        strat_net, strat_opt,
        strat_res.inputs [:n_pol],
        strat_res.targets[:n_pol],
        strat_res.weights[:n_pol],
        batch_size=args.batch,
        epochs=strat_epochs,
        device=device,
        epoch_callback=lambda ep, loss, secs:
            print(f"    ep {ep:2d} / {strat_epochs}   loss = {loss:.5f}  ·  {secs:.1f}s"),
        step_callback=lambda step, loss:
            writer.add_scalar("policy/loss", loss, global_step=step),
        step_callback_freq=500,
    )
    strat_secs = time.perf_counter() - t0
    print(f"\n  {strat_secs:.1f}s  ·  {_rate(n_pol * strat_epochs, strat_secs)} samples/s\n")

    writer.close()
    path = save_final_policy(ckpt_dir, last_t, strat_net)
    print(f"[{_ts()}] ==> Done.  Final policy → {path.split('/')[-1]}")

    if _interrupted:
        # C++ worker threads are stuck mid-rollout and will cause the
        # interpreter to hang during shutdown. Force-exit after explicit
        # TensorBoard cleanup (atexit won't run with os._exit).
        subprocess.run(["pkill", "-f", "tensorboard"], capture_output=True)
        os._exit(0)


if __name__ == "__main__":
    main()
