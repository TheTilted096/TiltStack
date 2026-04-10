"""
NLHE_BestResponse.py — Best-response training against a fixed policy network.

Runs the standard DeepCFR traversal, but villain inferences are answered by
the fixed policy network (softmax of its logits) rather than the villain's
own advantage network.  This is valid because getInstantStrat in C++ treats
its input as regrets: it clamps to non-negative, sums over legal actions, and
normalizes.  A softmax output is already non-negative, so the result is
precisely the policy's probability distribution conditioned on legal actions —
no explicit masking is needed on the Python side.

The hero's advantage network trains against this fixed opponent, approximating
the best-response regrets.  Both advantage networks are output as a checkpoint.

Player convention (same as NLHE_Trainer)
-----------------------------------------
  hero=False  →  player 0  →  small blind  →  acts when isButton=True
  hero=True   →  player 1  →  big blind    →  acts when isButton=False
"""

import time
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from tb_launch import launch_tb

import deepcfr
from network_training import (
    DeepCFRNet, Reservoir, train_advantage,
    decode_batch_gpu, verify_layout, infoset_dtype, NUM_ACTIONS,
    CONT_DIM, NUM_STREETS,
)

RESERVOIR_CAPACITY = 100_000_000


# ---------------------------------------------------------------------------
# Logging helpers (identical to NLHE_Trainer)
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

def run_inference_loop(orch, hero: bool, adv_nets, policy_net, device):
    """
    Route each infoset to the correct network:
      hero's turn   → adv_nets[int(hero)]     (logits, as in standard DeepCFR)
      villain's turn → policy_net + softmax    (probability dist as regrets)

    Villain identification: stm == hero iff is_button == (not hero),
    i.e.  is_villain = (is_button == hero)   [bool comparison].
    """
    done = 0
    while done < orch.num_threads():
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

        struct     = raw_cat.ravel().view(infoset_dtype)
        is_button  = struct['is_button']
        is_villain = is_button == hero
        hero_idx    = np.where(~is_villain)[0]
        villain_idx = np.where( is_villain)[0]

        out = np.empty((raw_cat.shape[0], NUM_ACTIONS), dtype=np.float32)

        with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            if len(hero_idx) > 0:
                out[hero_idx] = adv_nets[int(hero)](
                    x_cont[hero_idx], buckets[hero_idx]).float().cpu().numpy()

            if len(villain_idx) > 0:
                logits = policy_net(
                    x_cont[villain_idx], buckets[villain_idx])
                out[villain_idx] = F.softmax(logits, dim=1).float().cpu().numpy()

        offset = 0
        for s, sz in zip(scheds, sizes):
            s.output_data()[:] = out[offset:offset + sz]
            s.submit_batch()
            offset += sz


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

def save_final_advantages(ckpt_dir, t, adv_nets):
    """Save only the final two advantage networks."""
    date = time.strftime('%m%d%y')
    paths = []
    for i in range(2):
        path = ckpt_dir / f"adv{i}_{t:04d}-{date}.pt"
        torch.save({
            "t": t,
            "net": adv_nets[i].state_dict(),
        }, path)
        paths.append(path)
    return paths


def load_policy_net(policy_ckpt: str, device) -> DeepCFRNet:
    ckpt = torch.load(policy_ckpt, map_location=device, weights_only=True)
    net = DeepCFRNet().to(device)
    # Support both old format (strat_net) and new format (net)
    if "net" in ckpt:
        sd = ckpt["net"]
    elif "strat_net" in ckpt:
        sd = ckpt["strat_net"]
    else:
        raise KeyError("Checkpoint must contain either 'net' or 'strat_net' key")
    # Strip _orig_mod. prefix added by torch.compile when saving state_dict
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
        description="TiltStack best-response training against a fixed policy")
    parser.add_argument("--target", type=str, required=True,
        help="Path to a TiltStack checkpoint containing the trained strat_net")
    parser.add_argument("--iters",      type=int,   default=100)
    parser.add_argument("--threads",    type=int,   default=16)
    parser.add_argument("--samples",    type=int,   default=10_000_000,
        help="Target advantage samples per player per iteration  (default: 10M)")
    parser.add_argument("--batch",      type=int,   default=16384)
    parser.add_argument("--adv-step",   type=int,   default=2500,
        help="Mini-batches of advantage training per iteration  (default: 2500)")
    parser.add_argument("--lr",         type=float, default=3e-4)
    parser.add_argument("--seed",       type=int,   default=None)
    args = parser.parse_args()

    ckpt_dir = Path(__file__).parent.parent.parent / "br_checkpoints"

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
    print(f"[{_ts()}]  TensorBoard → http://127.0.0.1:6006/?darkMode=false&runFilter={run_name}#timeseries\n")

    # ---- One-time setup -----------------------------------------------------
    deepcfr.load_tables("clusters")
    verify_layout(deepcfr.INFOSET_BYTES)

    policy_net = load_policy_net(args.target, device)
    print(f"  Policy network loaded from: {args.target}\n")

    adv_nets = [DeepCFRNet().to(device), DeepCFRNet().to(device)]
    adv_opts = [torch.optim.Adam(n.parameters(), lr=args.lr) for n in adv_nets]

    print(f"[{_ts()}]  torch.compile — tracing networks for kernel fusion ...", flush=True)
    _t_compile = time.perf_counter()
    adv_nets   = [torch.compile(net, fullgraph=True) for net in adv_nets]
    policy_net =  torch.compile(policy_net, fullgraph=True)
    _dummy_xc = torch.zeros(args.batch, CONT_DIM,    device=device)
    _dummy_b  = torch.zeros(args.batch, NUM_STREETS, device=device, dtype=torch.long)
    with torch.no_grad():
        for _net in (*adv_nets, policy_net):
            _net(_dummy_xc, _dummy_b)
    print(f"[{_ts()}]  Compiled in {time.perf_counter() - _t_compile:.1f}s\n")

    adv_res = [Reservoir(RESERVOIR_CAPACITY, args.threads, deepcfr.INFOSET_BYTES)
               for _ in range(2)]
    orch = deepcfr.Orchestrator(args.threads,
                                adv_res[0]._cpp, adv_res[1]._cpp,
                                seed=seed)

    # ---- Training loop ------------------------------------------------------
    iter_times = []

    for t in range(1, args.iters + 1):
        print(f"[{_ts()}] ==> Iteration {t} / {args.iters}")
        iter_start = time.perf_counter()

        for hero in [False, True]:
            player = int(hero)

            # -- Rollout ------------------------------------------------------
            adv_before = adv_res[player].n_seen

            t0 = time.perf_counter()
            orch.start_iteration(hero, t, args.samples)
            run_inference_loop(orch, hero, adv_nets, policy_net, device)
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

            # -- Advantage training -------------------------------------------
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

        # -- Iteration summary ------------------------------------------------
        iter_elapsed = time.perf_counter() - iter_start
        iter_times.append(iter_elapsed)
        remaining = args.iters - t
        eta_str = (f"  ETA {_eta(sum(iter_times)/len(iter_times) * remaining)}"
                   f"  ({remaining} remaining)" if remaining > 0
                   else "  (final iteration)")
        print(f"\n  iter {iter_elapsed:.1f}s{eta_str}\n")

    # ---- Final advantage networks -------------------------------------------
    writer.close()
    paths = save_final_advantages(ckpt_dir, args.iters, adv_nets)
    print(f"[{_ts()}] ==> Done.  Final advantage networks → {', '.join(p.name for p in paths)}")


if __name__ == "__main__":
    main()
