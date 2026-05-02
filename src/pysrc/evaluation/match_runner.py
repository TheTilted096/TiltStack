"""
match_runner.py — Batched head-to-head evaluation between two TiltStack strategy networks.

1024 games run in parallel.  Each iteration:
  1. Advance all games through chance nodes to the next decision node.
  2. Collect all infosets — CFR is rebuilt from cfr_bets (player-action log)
     instead of replaying OSP history, eliminating the dominant CPU bottleneck.
  3. Run batched GPU forward passes — both networks launched on separate CUDA
     streams for overlap, with CPU mask construction concurrent with GPU work.
  4. Apply sampled actions via batched softmax + torch.multinomial; replace
     finished games with their duplicate (same cards, swapped seats) or a
     fresh pair.

Each deal is played as a duplicate pair: once with P0-net as small blind and
P1-net as big blind, once with seats swapped.  Per-seat payoffs are tracked
separately so positional bias is visible.

Usage:
    cd src
    python pysrc/evaluation/match_runner.py \\
        --p0     checkpoints/policy_v1.pt \\
        --p1     checkpoints/policy_v2.pt \\
        --pairs  100000 \\
        --device cpu
"""

import argparse
import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
import pyspiel

_EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_EVAL_DIR, "..", "deepcfr"))

import deepcfr
from tilt_agents import (
    load_net_auto,
    _pad_to_9,
    _abstract_to_osp,
    _CHECK,
    _CALL,
    STARTING_STACK,
    OSP_MC_SCALE,
)
from network_training import decode_batch, NUM_ACTIONS

# ---------------------------------------------------------------------------
# Game configuration — must match CFRTypes.h training parameters
# ---------------------------------------------------------------------------

OSP_BIG_BLIND = 100
CHIPS_TO_MBB = 1000 / OSP_BIG_BLIND  # = 10.0

GAME_STRING = (
    "universal_poker("
    "betting=nolimit,"
    "numPlayers=2,"
    "numSuits=4,"
    "numRanks=13,"
    "numHoleCards=2,"
    "numRounds=4,"
    "blind=50 100,"
    "maxRaises=99 99 99 99,"
    "numBoardCards=0 3 1 1,"
    "stack=5000 5000,"
    "firstPlayer=1 2 2 2,"
    "bettingAbstraction=fullgame"
    ")"
)

BATCH_SIZE = 8192

# ---------------------------------------------------------------------------
# Per-game slot
# ---------------------------------------------------------------------------


class Slot:
    """State for one game in the parallel batch."""

    __slots__ = (
        "state", "cfr", "cards", "deal_idx", "pair_idx", "is_pass_b",
        "cfr_bets",  # osp_action for every player move so far (replaces OSP history replay)
    )

    def __init__(self, state, cfr, cards, pair_idx, is_pass_b):
        self.state = state
        self.cfr = cfr
        self.cards = cards
        self.deal_idx = 0
        self.pair_idx = pair_idx
        self.is_pass_b = is_pass_b
        self.cfr_bets: list[int] = []


def _make_slot(osp_game, pair_idx, is_pass_b, cards=None):
    if cards is None:
        cards = np.random.permutation(52)[:9].tolist()
    return Slot(osp_game.new_initial_state(), deepcfr.CFRGame(), cards, pair_idx, is_pass_b)


# ---------------------------------------------------------------------------
# Per-step helpers
# ---------------------------------------------------------------------------


def _advance_to_decision(slot):
    """Apply pre-dealt card actions until we reach a decision or terminal node."""
    while not slot.state.is_terminal() and slot.state.is_chance_node():
        slot.state.apply_action(slot.cards[slot.deal_idx])
        slot.deal_idx += 1


def _sync_cfr(slot) -> bool:
    """
    Rebuild slot.cfr using cfr_bets — same semantics as the original _sync_cfr
    but without the expensive OSP state replay.

    Returns False when hole cards have not yet been dealt (slot was just created
    this iteration and hasn't gone through Phase 1 yet).
    """
    if slot.deal_idx < 4:
        return False

    slot.cfr.begin_with_cards(
        STARTING_STACK,
        STARTING_STACK,
        slot.state.current_player() == 0,
        _pad_to_9(slot.cards[:slot.deal_idx]),
    )
    for osp_action in slot.cfr_bets:
        if osp_action == 0:
            slot.cfr.make_move(_CHECK)
        elif osp_action == 1:
            # When a raise was mapped to a call by _abstract_to_osp (e.g. maxRaises
            # hit), osp_action == 1.  Use the same fallback logic as the original.
            legal = slot.cfr.generate_actions()
            slot.cfr.make_move(_CALL if _CALL in legal else _CHECK)
        else:
            stm = slot.cfr.stm
            invested_mc = STARTING_STACK - slot.cfr.stacks[stm]
            slot.cfr.make_bet(osp_action * OSP_MC_SCALE - invested_mc)
    return True


def _build_masks(ds_indices: list[int], decision: list) -> torch.Tensor:
    """Return (N, NUM_ACTIONS) CPU mask: 0 for legal actions, -1e9 for illegal."""
    n = len(ds_indices)
    if n == 0:
        return torch.empty(0, NUM_ACTIONS)
    mask = torch.full((n, NUM_ACTIONS), -1e9)
    for local_i, ds_idx in enumerate(ds_indices):
        _, slot = decision[ds_idx]
        for a in slot.cfr.generate_actions():
            mask[local_i, a] = 0.0
    return mask


# ---------------------------------------------------------------------------
# Core evaluation function
# ---------------------------------------------------------------------------


def run_match(
    net_p0,
    net_p1,
    game,
    device: torch.device,
    pairs: int,
    verbose: bool = True,
    p0_argmax: bool = False,
    p1_argmax: bool = False,
    p0_label: str = "P0",
) -> tuple[float, float]:
    """
    Run a batched duplicate-pair evaluation of net_p0 vs net_p1.

    Parameters
    ----------
    net_p0, net_p1 : DeepCFRNet  — pre-loaded, eval()-mode strategy networks
    game           : pyspiel.Game
    device         : torch.device
    pairs          : number of duplicate pairs to complete
    verbose        : print progress every 1000 pairs

    Returns
    -------
    overall_mbb : float  — P0's edge over P1 in mBB/hand (positive = P0 wins)
    ci95        : float  — 95% confidence interval half-width in mBB/hand
    """
    use_cuda = device.type == "cuda"
    if use_cuda:
        stream0 = torch.cuda.Stream(device)
        stream1 = torch.cuda.Stream(device)

    next_pair_idx = 0
    pending: dict[int, float] = {}  # pair_idx → payoff_a (waiting for pass-B)
    pair_payoffs: list[float] = []
    sb_payoff = 0.0
    bb_payoff = 0.0
    last_report = 0

    slots: list[Slot | None] = [None] * BATCH_SIZE
    for i in range(min(BATCH_SIZE, pairs)):
        slots[i] = _make_slot(game, next_pair_idx, False)
        next_pair_idx += 1

    while any(s is not None for s in slots):

        # Phase 1 — advance every game through chance nodes
        for slot in slots:
            if slot is not None:
                _advance_to_decision(slot)

        # Handle newly terminal games
        for i, slot in enumerate(slots):
            if slot is None or not slot.state.is_terminal():
                continue

            payoff = slot.state.returns()[0]

            if not slot.is_pass_b:
                # Pass A done: store payoff_a and start pass B (same cards)
                pending[slot.pair_idx] = payoff
                slots[i] = _make_slot(game, slot.pair_idx, True, slot.cards)
            else:
                # Pass B done: record the completed pair
                payoff_a = pending.pop(slot.pair_idx)
                payoff_b = payoff
                sb_payoff += payoff_a
                bb_payoff -= payoff_b
                pair_payoffs.append(payoff_a - payoff_b)

                n = len(pair_payoffs)
                if verbose and n % 1000 == 0 and n != last_report:
                    last_report = n
                    hands = n * 2
                    total = sb_payoff + bb_payoff
                    print(
                        f"  [{n:6d}/{pairs}]  "
                        f"{p0_label} overall {total / hands * CHIPS_TO_MBB:+.1f}  "
                        f"as-sb {sb_payoff / n * CHIPS_TO_MBB:+.1f}  "
                        f"as-bb {bb_payoff / n * CHIPS_TO_MBB:+.1f}  mBB/hand"
                    )

                if next_pair_idx < pairs:
                    slots[i] = _make_slot(game, next_pair_idx, False)
                    next_pair_idx += 1
                else:
                    slots[i] = None

        # Gather decision nodes
        decision = [
            (i, s)
            for i, s in enumerate(slots)
            if s is not None and not s.state.is_terminal()
        ]
        if not decision:
            continue

        # Phase 2 — rebuild CFR from cfr_bets and collect infosets by network
        #
        # net_idx = player ^ int(is_pass_b):
        #   pass A, player 0 → 0 ^ 0 = 0 → net_p0
        #   pass A, player 1 → 1 ^ 0 = 1 → net_p1
        #   pass B, player 0 → 0 ^ 1 = 1 → net_p1  (seats swapped)
        #   pass B, player 1 → 1 ^ 1 = 0 → net_p0
        net0_ds: list[int] = []
        net1_ds: list[int] = []
        infosets0: list[np.ndarray] = []
        infosets1: list[np.ndarray] = []

        for ds_idx, (i, slot) in enumerate(decision):
            if not _sync_cfr(slot):
                # Slot was just created this iteration (hasn't gone through Phase 1
                # yet); hole cards not dealt — skip until next iteration.
                continue
            raw = slot.cfr.get_info()  # (1, INFOSET_BYTES)
            net_idx = slot.state.current_player() ^ int(slot.is_pass_b)
            if net_idx == 0:
                net0_ds.append(ds_idx)
                infosets0.append(raw)
            else:
                net1_ds.append(ds_idx)
                infosets1.append(raw)

        # Decode infosets on CPU
        x_cont0 = buckets0 = x_cont1 = buckets1 = None
        if net0_ds:
            x_cont0, buckets0 = decode_batch(np.concatenate(infosets0, axis=0))
        if net1_ds:
            x_cont1, buckets1 = decode_batch(np.concatenate(infosets1, axis=0))

        # Batched forward passes
        # On CUDA: both networks launched on separate streams so their compute
        # can overlap.  CPU mask construction runs concurrently with GPU work.
        logits0 = logits1 = None
        with torch.no_grad():
            if use_cuda:
                if net0_ds:
                    with torch.cuda.stream(stream0):
                        logits0 = net_p0(
                            x_cont0.to(device, non_blocking=True),
                            buckets0.to(device, non_blocking=True),
                        )
                if net1_ds:
                    with torch.cuda.stream(stream1):
                        logits1 = net_p1(
                            x_cont1.to(device, non_blocking=True),
                            buckets1.to(device, non_blocking=True),
                        )
                # Build action masks on CPU while the GPU is computing above.
                mask0 = _build_masks(net0_ds, decision)
                mask1 = _build_masks(net1_ds, decision)
                torch.cuda.synchronize(device)
            else:
                if net0_ds:
                    logits0 = net_p0(x_cont0.to(device), buckets0.to(device))
                if net1_ds:
                    logits1 = net_p1(x_cont1.to(device), buckets1.to(device))
                mask0 = _build_masks(net0_ds, decision)
                mask1 = _build_masks(net1_ds, decision)

        # Merge masked logits from both networks into a single tensor for
        # batched softmax + multinomial — one kernel launch instead of N loops.
        parts: list[torch.Tensor] = []
        all_ds: list[int] = []
        all_use_argmax: list[bool] = []
        if net0_ds:
            parts.append(logits0 + mask0.to(device))
            all_ds.extend(net0_ds)
            all_use_argmax.extend([p0_argmax] * len(net0_ds))
        if net1_ds:
            parts.append(logits1 + mask1.to(device))
            all_ds.extend(net1_ds)
            all_use_argmax.extend([p1_argmax] * len(net1_ds))

        if not parts:
            continue

        all_masked = torch.cat(parts, dim=0)  # (N_total, NUM_ACTIONS)
        with torch.no_grad():
            sampled = torch.multinomial(
                F.softmax(all_masked, dim=1), num_samples=1
            ).squeeze(1)                                  # (N_total,)
            argmax_vals = all_masked.argmax(dim=1)        # (N_total,)

        # Phase 3 — sample actions and step games forward
        for k, ds_idx in enumerate(all_ds):
            _, slot = decision[ds_idx]
            abstract_action = int(
                argmax_vals[k].item() if all_use_argmax[k] else sampled[k].item()
            )
            osp_action = _abstract_to_osp(
                abstract_action, slot.state.legal_actions(), slot.cfr
            )
            slot.cfr_bets.append(osp_action)
            slot.state.apply_action(osp_action)

    n = len(pair_payoffs)
    total_hands = n * 2
    overall_mbb = (sb_payoff + bb_payoff) / total_hands * CHIPS_TO_MBB
    per_hand = np.array(pair_payoffs) / 2.0 * CHIPS_TO_MBB
    ci95 = 1.96 * per_hand.std(ddof=1) / np.sqrt(n)
    return overall_mbb, ci95


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main():
    _clusters = os.path.join(_EVAL_DIR, "..", "..", "clusters")

    parser = argparse.ArgumentParser(
        description="TiltStack head-to-head network evaluation"
    )
    parser.add_argument("--p0", required=True, help="Path to P0 strategy checkpoint")
    parser.add_argument("--p1", required=True, help="Path to P1 strategy checkpoint")
    parser.add_argument(
        "--pairs",
        type=int,
        default=100_000,
        help="Number of duplicate pairs (each played twice, once per seat)",
    )
    parser.add_argument("--device", default="cpu", help="Torch device (cpu or cuda)")
    args = parser.parse_args()

    if not os.path.isdir(_clusters):
        sys.exit(f"Error: clusters directory not found at '{_clusters}'")
    deepcfr.load_tables(_clusters)

    print(f"Loading P0 net: {args.p0}")
    net_p0 = load_net_auto(args.p0, args.device)
    print(f"Loading P1 net: {args.p1}")
    net_p1 = load_net_auto(args.p1, args.device)

    game = pyspiel.load_game(GAME_STRING)
    device = torch.device(args.device)

    overall_mbb, ci95 = run_match(net_p0, net_p1, game, device, args.pairs)

    total_hands = args.pairs * 2
    print(f"\n{'─' * 60}")
    print(f"  Pairs played           : {args.pairs:,}  ({total_hands:,} hands total)")
    print(f"  P0 overall edge vs P1  : {overall_mbb:+.2f} ± {ci95:.2f} mBB/hand  (95% CI)")


if __name__ == "__main__":
    main()
