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

import math
import os
import sys
import time
import threading
import argparse
import subprocess
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from tb_launch import launch_tb, stop_tb

import deepcfr
from network_training import (
    DeepCFRNet,
    Reservoir,
    run_inference_loop,
    decode_batch_gpu,
    verify_layout,
    infoset_dtype,
    NUM_ACTIONS,
    CONT_DIM,
    NUM_STREETS,
    _ts,
    _fmt,
    _rate,
    _eta,
)


RESERVOIR_CAPACITY = 100_000_000


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train_advantage(
    net: DeepCFRNet,
    optimizer: torch.optim.Optimizer,
    raw_inputs: np.ndarray,
    targets: np.ndarray,
    batch_size: int = 4096,
    epochs: int = 1,
    max_steps: int | None = None,
    device: torch.device = torch.device("cuda"),
) -> list[float]:
    N = len(raw_inputs)
    raw_pt = torch.from_numpy(raw_inputs)
    tgt_pt = torch.from_numpy(targets)

    raw_buf = torch.empty(batch_size, raw_inputs.shape[1], dtype=torch.uint8, pin_memory=True)
    tgt_buf = torch.empty(batch_size, targets.shape[1], dtype=torch.float32, pin_memory=True)

    net.train()

    def _step(idx):
        bs = len(idx)
        raw_buf[:bs].copy_(raw_pt[idx])
        tgt_buf[:bs].copy_(tgt_pt[idx])
        raw_b = raw_buf[:bs].to(device, non_blocking=True)
        t_b = tgt_buf[:bs].to(device, non_blocking=True)
        optimizer.zero_grad()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            xc, b = decode_batch_gpu(raw_b)
            mask = ~torch.isnan(t_b)
            sq_err = ((net(xc, b) - t_b.nan_to_num(0.0)) * mask) ** 2
            loss = sq_err.sum(dim=1).div(mask.sum(dim=1).clamp(min=1)).mean()
        loss.backward()
        optimizer.step()
        return loss.detach()

    if max_steps is not None:
        total = torch.zeros(1, device=device)
        steps_done = 0
        perm = torch.randperm(N)
        pos = 0
        while steps_done < max_steps:
            if pos >= N:
                perm = torch.randperm(N)
                pos = 0
            idx = perm[pos : pos + batch_size]
            pos += batch_size
            total += _step(idx)
            steps_done += 1
        return [(total / steps_done).item()]

    losses = []
    for _ in range(epochs):
        perm = torch.randperm(N)
        total = torch.zeros(1, device=device)
        n_batches = 0
        for start in range(0, N, batch_size):
            idx = perm[start : start + batch_size]
            total += _step(idx)
            n_batches += 1
        losses.append((total / n_batches).item())

    return losses


def policy_trainer(
    net: DeepCFRNet,
    optimizer: torch.optim.Optimizer,
    raw_inputs: np.ndarray,
    targets: np.ndarray,
    weights: np.ndarray,
    batch_size: int = 4096,
    device: torch.device = torch.device("cuda"),
):
    N = len(raw_inputs)
    raw_pt = torch.from_numpy(raw_inputs)
    tgt_pt = torch.from_numpy(targets)
    wt_pt = torch.from_numpy(weights.astype(np.float32))

    raw_buf = torch.empty(batch_size, raw_inputs.shape[1], dtype=torch.uint8, pin_memory=True)
    tgt_buf = torch.empty(batch_size, targets.shape[1], dtype=torch.float32, pin_memory=True)
    wt_buf = torch.empty(batch_size, dtype=torch.float32, pin_memory=True)

    net.train()
    ep = 0
    while True:
        ep += 1
        ep_t0 = time.perf_counter()
        perm = torch.randperm(N)
        total = torch.zeros(1, device=device)
        n_batches = 0
        for start in range(0, N, batch_size):
            idx = perm[start : start + batch_size]
            bs = len(idx)
            raw_buf[:bs].copy_(raw_pt[idx])
            tgt_buf[:bs].copy_(tgt_pt[idx])
            wt_buf[:bs].copy_(wt_pt[idx])
            raw_b = raw_buf[:bs].to(device, non_blocking=True)
            t_b = tgt_buf[:bs].to(device, non_blocking=True)
            wt_b = wt_buf[:bs].to(device, non_blocking=True)
            optimizer.zero_grad()
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                xc, b = decode_batch_gpu(raw_b)
                wt_b = wt_b / wt_b.mean()
                legal = ~torch.isnan(t_b)
                log_probs = F.log_softmax(net(xc, b).masked_fill(~legal, float("-inf")), dim=1)
                per_sample = -(t_b.nan_to_num(0.0) * log_probs).sum(dim=1)
                loss = (wt_b * per_sample).mean()
            loss.backward()
            optimizer.step()
            total += loss.detach()
            n_batches += 1
        yield ep, (total / n_batches).item(), time.perf_counter() - ep_t0


# ---------------------------------------------------------------------------
# LR schedule
# ---------------------------------------------------------------------------


def _cosine_lr(ep: int, base_ep: int, base_lr: float, total_ep: int) -> float:
    """LR for epoch `ep` in a cosine segment that starts at `base_ep` with
    `base_lr` and decays to 0 at `total_ep`."""
    remaining = total_ep - base_ep
    if remaining <= 0:
        return 0.0
    return 0.5 * base_lr * (1.0 + math.cos(math.pi * (ep - base_ep) / remaining))


# ---------------------------------------------------------------------------
# Training control
# ---------------------------------------------------------------------------


class TrainingControl:
    """Shared state between the training loop and the menu thread."""

    def __init__(self, total_iters: int, total_epochs: int) -> None:
        self._lock = threading.Lock()
        self.stop_after_iter = total_iters
        self.policy_epochs = total_epochs
        self.current_iter = 0
        self.current_epoch = 0
        self.phase = "advantage"  # "advantage" | "policy" | "done"


def _run_menu(ctrl: TrainingControl, term) -> None:
    """Daemon thread: display prompts on the terminal and apply user commands."""
    while True:
        with ctrl._lock:
            if ctrl.phase == "done":
                return
            if ctrl.phase == "advantage":
                cur = ctrl.current_iter
                stop = ctrl.stop_after_iter
                prompt = f"\nenter number of adv iters ({stop}) > "
            else:
                cur_ep = ctrl.current_epoch
                epochs = ctrl.policy_epochs
                prompt = f"\nenter number of policy epochs ({epochs}) > "

        term.write(prompt)
        term.flush()

        try:
            line = sys.stdin.readline()
        except OSError:
            return

        if not line:  # EOF
            return

        line = line.strip()
        if not line:
            continue

        try:
            val = int(line)
        except ValueError:
            term.write("  ! not a number\n")
            term.flush()
            continue

        with ctrl._lock:
            if ctrl.phase == "advantage":
                cur = ctrl.current_iter
                if val <= cur:
                    term.write(f"  ! {val} <= current iter {cur}\n")
                else:
                    ctrl.stop_after_iter = val
                    term.write(f"  -> adv iters set to {val}\n")
                    print(f"[{_ts()}]  [ctrl] adv iters set to {val}", flush=True)
            elif ctrl.phase == "policy":
                cur_ep = ctrl.current_epoch
                if val <= cur_ep:
                    term.write(f"  ! {val} <= current epoch {cur_ep}\n")
                else:
                    ctrl.policy_epochs = val
                    term.write(f"  -> policy epochs set to {val}\n")
                    print(f"[{_ts()}]  [ctrl] policy epochs set to {val}", flush=True)
        term.flush()


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------


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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
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
    menu = threading.Thread(target=_run_menu, args=(ctrl, _term), daemon=True)
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
    orch = deepcfr.Orchestrator(
        args.threads, adv_res[0]._cpp, adv_res[1]._cpp, strat_res._cpp, seed
    )

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
            adv_before = adv_res[player].n_seen
            collect_policy = t > 50
            pol_before = strat_res.n_seen if collect_policy else 0

            t0 = time.perf_counter()
            orch.start_iteration(hero, t, args.samples)
            run_inference_loop(orch, device, adv_nets)
            orch.wait_iteration()
            rollout_secs = time.perf_counter() - t0

            rollouts = sum(s.rollout_count() for s in orch.schedulers)
            orch.clear_buffers()

            adv_new = adv_res[player].n_seen - adv_before
            adv_size = adv_res[player].size
            cap_str = _fmt(RESERVOIR_CAPACITY)

            print(
                f"\n  [P{player} rollout]  {rollout_secs:.1f}s"
                f"  ·  rollouts={_fmt(rollouts)}"
                f"  ·  {_rate(adv_new, rollout_secs)} infosets/s"
            )
            print(
                f"    advantage  +{_fmt(adv_new):<12}"
                f"  reservoir  {_fmt(adv_size):>12} / {cap_str}"
            )
            if collect_policy:
                pol_new = strat_res.n_seen - pol_before
                pol_size = strat_res.size
                print(
                    f"    policy     +{_fmt(pol_new):<12}"
                    f"  reservoir  {_fmt(pol_size):>12} / {cap_str}"
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
                    max_steps=args.adv_step,
                    device=device,
                )
                train_secs = time.perf_counter() - t0
                samples_seen = args.adv_step * args.batch
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

        next_lr = _cosine_lr(ep + 1, sched_base_ep, sched_base_lr, n_epochs)
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
