"""
train.py — DeepCFR main training loop for TiltStack.

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
         Workers begin generating rollouts in the background.
    2. Inference loop:
         pop() blocks until a worker needs a network prediction.
         Each infoset is routed to the acting player's advantage network:
           - Player 0 nodes → adv_net[0]  (raw advantage values)
           - Player 1 nodes → adv_net[1]  (raw advantage values)
         C++ calls getInstantStrat() on the returned values, applying
         regret matching (ReLU + normalize) over legal actions.
         A None return from pop() means one thread has finished; count
         until all num_threads() sentinels are received.
    3. wait_iteration()
         Blocks until all worker threads park on the next CV.
    4. collect_data()
         Copy advantage/policy replay buffers out of the C++ Schedulers
         while they are still valid (before clear_buffers()).
    5. clear_buffers()
         Reset per-thread replay buffers for the next iteration.
    6. Train adv_net[hero] on advantage buffer (unweighted MSE).
    7. Train strat_net on policy buffer (iteration-weighted MSE).

Player convention
-----------------
  hero=False  →  hero is player 0  →  acts when isButton=True
  hero=True   →  hero is player 1  →  acts when isButton=False
  isButton encodes "stm == 0" (player 0 to move), so:
    hero_mask = (is_button == (not hero))
"""

import os
import argparse
import numpy as np
import torch

import deepcfr
from network_training import (
    DeepCFRNet, train_advantage, train_policy,
    decode_batch, verify_layout, infoset_dtype, NUM_ACTIONS,
)


# ---------------------------------------------------------------------------
# Inference loop
# ---------------------------------------------------------------------------

def run_inference_loop(orch, hero: bool, adv_nets, device):
    """
    Service the C++ inference queue until every worker thread sends a sentinel.

    Workers push a Scheduler* when they have a full batch awaiting a network
    prediction, and push None (sentinel) when they finish an iteration.

    Both players use their own advantage network during traversal.  The C++
    traversal always calls getInstantStrat() on the returned values, which
    applies regret matching (ReLU + normalize over legal actions), so raw
    network outputs are written back without any transformation.

    Player 0 nodes → adv_nets[0]
    Player 1 nodes → adv_nets[1]

    isButton encodes "stm == 0" (player 0 acting).
    """
    done = 0

    while done < orch.num_threads():
        sched = orch.pop()
        if sched is None:
            done += 1
            continue

        n   = sched.batch_size()
        raw = np.array(sched.input_data(), copy=False)   # zero-copy view

        x_cont, buckets = decode_batch(raw)
        x_cont  = x_cont .to(device, non_blocking=True)
        buckets = buckets.to(device, non_blocking=True)

        struct       = raw.ravel().view(infoset_dtype)
        is_p0_acting = struct['is_button']   # True where player 0 acts

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

def collect_data(orch):
    """
    Copy advantage and policy buffers from every per-thread Scheduler.
    Must be called before clear_buffers() since the C++ buffers are reused.

    Returns
    -------
    adv_inputs  : (N_adv, INFOSET_BYTES) uint8
    adv_targets : (N_adv, NUM_ACTIONS)   float32   regrets
    pol_inputs  : (N_pol, INFOSET_BYTES) uint8
    pol_targets : (N_pol, NUM_ACTIONS)   float32   instant strategy
    pol_weights : (N_pol,)               int32     iteration index t
    """
    adv_inp, adv_tgt            = [], []
    pol_inp, pol_tgt, pol_wgt   = [], [], []

    for sched in orch.schedulers:
        if sched.advantage_size() > 0:
            adv_inp.append(np.array(sched.advantage_input_data(),  copy=True))
            adv_tgt.append(np.array(sched.advantage_output_data(), copy=True))
        if sched.policy_size() > 0:
            pol_inp.append(np.array(sched.policy_input_data(),     copy=True))
            pol_tgt.append(np.array(sched.policy_output_data(),    copy=True))
            pol_wgt.append(np.array(sched.policy_weight_data(),    copy=True))

    def cat(lst, empty_shape, dtype):
        return (np.concatenate(lst, axis=0) if lst
                else np.empty(empty_shape, dtype=dtype))

    adv_inp = cat(adv_inp, (0, deepcfr.INFOSET_BYTES), np.uint8)
    adv_tgt = cat(adv_tgt, (0, NUM_ACTIONS),            np.float32)
    pol_inp = cat(pol_inp, (0, deepcfr.INFOSET_BYTES), np.uint8)
    pol_tgt = cat(pol_tgt, (0, NUM_ACTIONS),            np.float32)
    pol_wgt = cat(pol_wgt, (0,),                        np.int32)

    return adv_inp, adv_tgt, pol_inp, pol_tgt, pol_wgt


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

def save_checkpoint(checkpoint_dir, t, adv_nets, strat_net, adv_opts, strat_opt):
    path = os.path.join(checkpoint_dir, f"ckpt_t{t:04d}.pt")
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


def load_checkpoint(checkpoint_dir, adv_nets, strat_net, adv_opts, strat_opt, device):
    """
    Load the lexicographically last checkpoint in checkpoint_dir.
    Returns the next iteration index (saved_t + 1).
    """
    ckpts = sorted(f for f in os.listdir(checkpoint_dir) if f.startswith("ckpt_t"))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")
    path = os.path.join(checkpoint_dir, ckpts[-1])
    ckpt = torch.load(path, map_location=device)
    adv_nets[0].load_state_dict(ckpt["adv_net_0"])
    adv_nets[1].load_state_dict(ckpt["adv_net_1"])
    strat_net   .load_state_dict(ckpt["strat_net"])
    adv_opts[0] .load_state_dict(ckpt["adv_opt_0"])
    adv_opts[1] .load_state_dict(ckpt["adv_opt_1"])
    strat_opt   .load_state_dict(ckpt["strat_opt"])
    print(f"Loaded checkpoint: {path}")
    return ckpt["t"] + 1


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="TiltStack DeepCFR training loop")
    parser.add_argument("--threads",        type=int,   default=8)
    parser.add_argument("--iterations",     type=int,   default=200,
        help="CFR iterations T")
    parser.add_argument("--total-samples",  type=int,   default=10_000_000,
        help="Target advantage samples per player per iteration")
    parser.add_argument("--batch-size",     type=int,   default=4096,
        help="SGD mini-batch size")
    parser.add_argument("--epochs",         type=int,   default=1,
        help="Training epochs per iteration per network")
    parser.add_argument("--lr",             type=float, default=1e-3)
    parser.add_argument("--seed",           type=int,   default=None,
        help="RNG seed for the C++ worker threads")
    parser.add_argument("--clusters-dir",   type=str,   default="clusters",
        help="Directory with precomputed EHS / cluster-label tables")
    parser.add_argument("--checkpoint-dir", type=str,   default="checkpoints")
    parser.add_argument("--checkpoint-every", type=int, default=10,
        help="Save a checkpoint every N iterations")
    parser.add_argument("--resume",         type=str,   default=None,
        help="Checkpoint directory to resume from")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ---- One-time setup -----------------------------------------------------
    deepcfr.load_tables(args.clusters_dir)
    verify_layout(deepcfr.INFOSET_BYTES)

    adv_nets  = [DeepCFRNet().to(device), DeepCFRNet().to(device)]
    strat_net =  DeepCFRNet().to(device)
    adv_opts  = [torch.optim.Adam(n.parameters(), lr=args.lr) for n in adv_nets]
    strat_opt =  torch.optim.Adam(strat_net.parameters(),     lr=args.lr)

    start_t = 1
    if args.resume:
        start_t = load_checkpoint(
            args.resume, adv_nets, strat_net, adv_opts, strat_opt, device)
        print(f"Resuming from t={start_t}")

    seed = 0xdeadbeefcafe1234 if args.seed is None else args.seed
    orch = deepcfr.Orchestrator(args.threads, seed)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # ---- Training loop ------------------------------------------------------
    for t in range(start_t, args.iterations + 1):
        for hero in [False, True]:
            player = int(hero)
            tag    = f"t={t:3d}  hero=P{player}"

            print(f"[{tag}]  generating samples "
                  f"(target {args.total_samples:,} advantage samples) ...")
            orch.start_iteration(hero, t, args.total_samples)
            run_inference_loop(orch, hero, adv_nets, device)
            orch.wait_iteration()

            adv_inp, adv_tgt, pol_inp, pol_tgt, pol_wgt = collect_data(orch)
            orch.clear_buffers()

            n_adv, n_pol = len(adv_inp), len(pol_inp)
            print(f"[{tag}]  adv={n_adv:,}  pol={n_pol:,}")

            if n_adv > 0:
                loss = train_advantage(
                    adv_nets[player], adv_opts[player],
                    adv_inp, adv_tgt,
                    batch_size=args.batch_size,
                    epochs=args.epochs,
                    device=device,
                )
                print(f"[{tag}]  adv_loss={loss:.5f}")

            if n_pol > 0:
                loss = train_policy(
                    strat_net, strat_opt,
                    pol_inp, pol_tgt, pol_wgt,
                    batch_size=args.batch_size,
                    epochs=args.epochs,
                    device=device,
                )
                print(f"[{tag}]  pol_loss={loss:.5f}")

        if t % args.checkpoint_every == 0:
            path = save_checkpoint(
                args.checkpoint_dir, t,
                adv_nets, strat_net, adv_opts, strat_opt)
            print(f"  checkpoint → {path}")

    path = save_checkpoint(
        args.checkpoint_dir, args.iterations,
        adv_nets, strat_net, adv_opts, strat_opt)
    print(f"Training complete. Final checkpoint → {path}")


if __name__ == "__main__":
    main()
