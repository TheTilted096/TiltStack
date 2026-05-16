"""
NLHE_Trainer.py — DeepCFR main training loop for TiltStack.

Networks
--------
  adv_net[0]  advantage network for player 0 (small blind)
  adv_net[1]  advantage network for player 1 (big blind)
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
import threading
import argparse
from pathlib import Path
import torch
from tb_launch import launch_tb, stop_tb

import deepcfr
from network_training import (
    DeepCFRNet,
    Reservoir,
    verify_layout,
    CONT_DIM,
    NUM_STREETS,
    _ts,
    _fmt,
    _rate,
    _eta,
)
from rollout import run_player_rollout
from train_loops import cosine_lr, policy_trainer, train_advantage
from training_control import TrainingControl, run_menu


RESERVOIR_CAPACITY = 100_000_000


def save_final_policy(ckpt_dir, t, strat_net):
    """Save only the final policy network."""
    date = time.strftime("%m%d%y")
    policy_path = os.path.join(ckpt_dir, f"policy{t:04d}-{date}.pt")
    torch.save(
        {
            "t": t,
            "net": strat_net.state_dict(),
        },
        policy_path,
    )
    return policy_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="TiltStack DeepCFR training loop")
    parser.add_argument(
        "--iters", type=int, default=150, help="CFR iterations T  (default: 150)"
    )
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument(
        "--samples",
        type=int,
        default=10_000_000,
        help="Target advantage samples per player per iteration  (default: 10M)",
    )
    parser.add_argument("--batch", type=int, default=16384)
    parser.add_argument(
        "--epochs",
        type=int,
        default=4,
        help="Strategy network training epochs  (default: 4)",
    )
    parser.add_argument(
        "--adv-step",
        type=int,
        default=2500,
        help="Mini-batches of advantage training per iteration  (default: 2500)",
    )
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    root = Path(__file__).parent.parent.parent
    ckpt_dir = root / "checkpoints"

    device = torch.device("cuda")
    torch.set_float32_matmul_precision("high")

    # ---- Log file setup -----------------------------------------------------
    run_name = time.strftime("%m%d%y_%H%M%S")
    logs_dir = root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / f"policy-{run_name}.txt"

    _term = sys.stdout
    _log = open(log_path, "w", buffering=1)
    sys.stdout = _log

    _term.write(
        f"Training started — output → logs/policy-{run_name}.txt\n"
        f"  device={device}  threads={args.threads}  iters={args.iters}\n"
    )

    # ---- TensorBoard --------------------------------------------------------
    # Launched before CUDA init and torch.compile so the stderr suppress
    # window covers both phases, where gRPC triggers its startup noise.
    seed = 0xDEADBEEFCAFE1234 if args.seed is None else args.seed
    ckpt_dir.mkdir(parents=True, exist_ok=True)
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

    # ---- Training control + menu thread -------------------------------------
    ctrl = TrainingControl(args.iters, args.epochs)
    menu = threading.Thread(target=run_menu, args=(ctrl, _term), daemon=True)
    menu.start()

    # ---- One-time setup -----------------------------------------------------
    samples_str = (
        f"{args.samples / 1e6:.1f}M"
        if args.samples >= 1_000_000
        else _fmt(args.samples)
    )
    print(
        f"[{_ts()}]  device={device}  threads={args.threads}"
        f"  samples/iter={samples_str}  iters={args.iters}\n"
    )

    deepcfr.load_tables("clusters")
    verify_layout(deepcfr.INFOSET_BYTES)

    adv_nets = [DeepCFRNet().to(device), DeepCFRNet().to(device)]
    strat_net = DeepCFRNet().to(device)
    adv_opts = [torch.optim.Adam(n.parameters(), lr=args.lr) for n in adv_nets]
    strat_opt = torch.optim.Adam(strat_net.parameters(), lr=args.lr)

    print(
        f"[{_ts()}]  torch.compile — tracing networks for kernel fusion ...", flush=True
    )
    _t_compile = time.perf_counter()
    adv_nets = [torch.compile(net, fullgraph=True) for net in adv_nets]
    strat_net = torch.compile(strat_net, fullgraph=True)
    _dummy_xc = torch.zeros(args.batch, CONT_DIM, device=device)
    _dummy_b = torch.zeros(args.batch, NUM_STREETS, device=device, dtype=torch.long)
    with torch.no_grad():
        for _net in (*adv_nets, strat_net):
            _net(_dummy_xc, _dummy_b)
    print(f"[{_ts()}]  Compiled in {time.perf_counter() - _t_compile:.1f}s\n")

    adv_res = [
        Reservoir(RESERVOIR_CAPACITY, args.threads, deepcfr.INFOSET_BYTES)
        for _ in range(2)
    ]
    strat_res = Reservoir(
        RESERVOIR_CAPACITY, args.threads, deepcfr.INFOSET_BYTES, has_weights=True
    )
    orch = deepcfr.Orchestrator(args.threads, seed)

    # ---- Training loop ------------------------------------------------------
    iter_times = []
    last_t = 0

    for t in range(1, args.iters + 1):
        ctrl.current_iter = t
        print(f"[{_ts()}] ==> Iteration {t} / {ctrl.stop_after_iter}")
        iter_start = time.perf_counter()

        for hero in [False, True]:
            player = int(hero)

            # -- Rollout --------------------------------------------------
            collect_policy = t > 50

            rollout = run_player_rollout(
                orch=orch,
                device=device,
                adv_nets=adv_nets,
                adv_res=adv_res,
                strat_res=strat_res,
                hero=hero,
                iteration=t,
                samples=args.samples,
                collect_policy=collect_policy,
            )

            cap_str = _fmt(RESERVOIR_CAPACITY)

            print(
                f"\n  [P{player} rollout]  {rollout.seconds:.1f}s"
                f"  ·  rollouts={_fmt(rollout.rollouts)}"
                f"  ·  {_rate(rollout.adv_new, rollout.seconds)} infosets/s"
            )
            print(
                f"    advantage  +{_fmt(rollout.adv_new):<12}"
                f"  reservoir  {_fmt(rollout.adv_size):>12} / {cap_str}"
            )
            if collect_policy:
                print(
                    f"    policy     +{_fmt(rollout.pol_new):<12}"
                    f"  reservoir  {_fmt(rollout.pol_size):>12} / {cap_str}"
                )

            # -- Advantage training ---------------------------------------
            n_adv = adv_res[player].size
            if n_adv > 0:
                adv_nets[player]._orig_mod.apply(
                    lambda m: (
                        m.reset_parameters() if hasattr(m, "reset_parameters") else None
                    )
                )
                adv_opts[player] = torch.optim.Adam(
                    adv_nets[player].parameters(), lr=args.lr
                )
                t0 = time.perf_counter()
                losses = train_advantage(
                    adv_nets[player],
                    adv_opts[player],
                    adv_res[player].inputs[:n_adv],
                    adv_res[player].targets[:n_adv],
                    batch_size=args.batch,
                    steps=args.adv_step,
                    device=device,
                )
                train_secs = time.perf_counter() - t0
                samples_seen = args.adv_step * min(args.batch, n_adv)
                writer.add_scalar(f"adv/p{player}", losses[-1], global_step=t)
                print(
                    f"\n  [P{player} advantage]  samples={_fmt(n_adv)}"
                    f"  ·  steps={args.adv_step}"
                    f"  ·  loss={losses[-1]:.5f}"
                    f"  ·  {train_secs:.1f}s"
                    f"  ·  {_rate(samples_seen, train_secs)} samples/s"
                )

        # -- Iteration summary --------------------------------------------
        iter_elapsed = time.perf_counter() - iter_start
        iter_times.append(iter_elapsed)
        remaining = ctrl.stop_after_iter - t
        eta_str = (
            f"  ETA {_eta(sum(iter_times) / len(iter_times) * remaining)}"
            f"  ({remaining} remaining)"
            if remaining > 0
            else "  (final iteration)"
        )
        print(f"\n  iter {iter_elapsed:.1f}s{eta_str}\n")
        last_t = t

        if t >= ctrl.stop_after_iter:
            print(f"[{_ts()}]  Stopping after iter {t} (user request).")
            break

    # ---- Strategy network ---------------------------------------------------
    ctrl.phase = "policy"
    _term.write(
        f"\nadv training finished\nenter number of policy epochs ({ctrl.policy_epochs}) > "
    )
    _term.flush()

    n_pol = strat_res.size
    if n_pol == 0:
        print(f"[{_ts()}]  No policy samples collected — nothing to save.")
        _term.write(f"\n[{_ts()}]  No policy samples collected — nothing to save.\n")
        _term.flush()
        _shutdown(1)

    print(
        f"[{_ts()}] ==> Strategy network\n"
        f"  samples={_fmt(n_pol)}  ·  up to {ctrl.policy_epochs} epochs\n"
    )

    # Cosine LR state: lr(ep) = 0.5 * base_lr * (1 + cos(π*(ep-base_ep)/(total-base_ep)))
    sched_base_ep = 0
    sched_base_lr = args.lr
    last_policy_epochs = ctrl.policy_epochs

    # Set LR for epoch 1 before the generator starts.
    for pg in strat_opt.param_groups:
        pg["lr"] = sched_base_lr

    t0 = time.perf_counter()
    for ep, mean_loss, secs in policy_trainer(
        strat_net,
        strat_opt,
        strat_res.inputs[:n_pol],
        strat_res.targets[:n_pol],
        strat_res.weights[:n_pol],
        batch_size=args.batch,
        device=device,
    ):
        ctrl.current_epoch = ep
        n_epochs = ctrl.policy_epochs
        current_lr = strat_opt.param_groups[0]["lr"]

        if n_epochs != last_policy_epochs:
            sched_base_lr = current_lr
            sched_base_ep = ep
            last_policy_epochs = n_epochs
            print(
                f"[{_ts()}]  [ctrl] LR recalibrated: {sched_base_lr:.2e}"
                f" over {n_epochs - ep} remaining epochs"
            )

        writer.add_scalar("policy/lr", current_lr, global_step=ep)
        writer.add_scalar("policy/loss", mean_loss, global_step=ep)
        print(
            f"    ep {ep:2d} / {n_epochs}"
            f"   loss = {mean_loss:.5f}"
            f"  ·  lr = {current_lr:.2e}"
            f"  ·  {secs:.1f}s"
        )
        if ep >= n_epochs:
            break

        next_lr = cosine_lr(ep + 1, sched_base_ep, sched_base_lr, n_epochs)
        for pg in strat_opt.param_groups:
            pg["lr"] = next_lr

    strat_secs = time.perf_counter() - t0
    print(
        f"\n  {strat_secs:.1f}s  ·  {_rate(n_pol * ctrl.current_epoch, strat_secs)} samples/s\n"
    )

    # ---- Save and exit ------------------------------------------------------
    ctrl.phase = "done"
    path = save_final_policy(ckpt_dir, last_t, strat_net)
    msg = f"[{_ts()}] ==> Done.  Final policy → {os.path.basename(path)}"
    print(msg)
    _term.write(f"\n{msg}\n")
    _term.flush()
    _shutdown()


if __name__ == "__main__":
    main()
