"""
match_runner.py — Head-to-head evaluation between two TiltStack strategy networks.

Each deal is played as a duplicate pair: once with P0-net as small blind and
P1-net as big blind, once with P1-net as small blind and P0-net as big blind.
Per-seat payoffs are tracked separately so positional bias is visible.

Architecture is detected automatically from checkpoint weight shapes, so the
script handles any DeepCFRNet variant without modification.

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
import pyspiel

_EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_EVAL_DIR, "..", "deepcfr"))

import deepcfr
from tilt_agents import TiltStack_DeepCFR, load_net_auto
from network_training import DeepCFRNet, CONT_DIM, NUM_STREETS

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

# ---------------------------------------------------------------------------
# Match loop
# ---------------------------------------------------------------------------


def deal_cards() -> list:
    """Sample 9 distinct card indices (pre-flop through river) without replacement."""
    return np.random.permutation(52)[:9].tolist()


def play_match(game, bot0, bot1, cards: list) -> float:
    """
    Play one hand with a pre-dealt deck.  Returns Player 0's utility in chips.
    """
    state = game.new_initial_state()
    deal_idx = 0

    while not state.is_terminal():
        if state.is_chance_node():
            action = cards[deal_idx]
            deal_idx += 1
        else:
            current_player = state.current_player()
            action = bot0.step(state) if current_player == 0 else bot1.step(state)
        state.apply_action(action)

    return state.returns()[0]


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    _clusters = os.path.join(_EVAL_DIR, "..", "..", "clusters")

    parser = argparse.ArgumentParser(
        description="TiltStack head-to-head network evaluation"
    )
    parser.add_argument(
        "--p0",
        required=True,
        help="Path to P0 strategy network checkpoint (policy*.pt)",
    )
    parser.add_argument(
        "--p1",
        required=True,
        help="Path to P1 strategy network checkpoint (policy*.pt)",
    )
    parser.add_argument(
        "--pairs",
        type=int,
        default=100_000,
        help="Number of duplicate pairs (each played twice, once per seat)",
    )
    parser.add_argument("--device", default="cpu", help="Torch device (cpu or cuda)")
    args = parser.parse_args()

    # 1. Load cluster tables
    if not os.path.isdir(_clusters):
        sys.exit(f"Error: clusters directory not found at '{_clusters}'")
    deepcfr.load_tables(_clusters)

    # 2. Load networks — architecture detected automatically from checkpoint shapes
    print(f"Loading P0 net: {args.p0}")
    net_p0 = load_net_auto(args.p0, args.device)
    print(f"Loading P1 net: {args.p1}")
    net_p1 = load_net_auto(args.p1, args.device)

    # 3. Initialise OpenSpiel game
    game = pyspiel.load_game(GAME_STRING)

    # 4. Create agents — TiltStack_DeepCFR samples from softmax over legal actions
    agent_p0 = TiltStack_DeepCFR(net_p0, osp_game=game, device=args.device)
    agent_p1 = TiltStack_DeepCFR(net_p1, osp_game=game, device=args.device)

    # 5. Duplicate evaluation loop.
    #    Pass A: P0-net is SB (P0), P1-net is BB (P1) → payoff_a = P0-net's P0 utility
    #    Pass B: P1-net is SB (P0), P0-net is BB (P1) → payoff_b = P1-net's P0 utility
    #    P0-net's BB payoff = -payoff_b; combined per-pair edge = payoff_a - payoff_b.
    sb_payoff = 0.0
    bb_payoff = 0.0
    pair_payoffs = []  # per-pair combined payoff for CI calculation

    for i in range(args.pairs):
        cards = deal_cards()
        payoff_a = play_match(game, agent_p0, agent_p1, cards)
        payoff_b = play_match(game, agent_p1, agent_p0, cards)
        sb_payoff += payoff_a
        bb_payoff -= payoff_b
        pair_payoffs.append(payoff_a - payoff_b)

        if (i + 1) % 1000 == 0:
            hands_so_far = (i + 1) * 2
            total_so_far = sb_payoff + bb_payoff
            print(
                f"  [{i + 1:6d}/{args.pairs}]  "
                f"P0 overall {total_so_far / hands_so_far * CHIPS_TO_MBB:+.1f}  "
                f"sb {sb_payoff / (i + 1) * CHIPS_TO_MBB:+.1f}  "
                f"bb {bb_payoff / (i + 1) * CHIPS_TO_MBB:+.1f}  mBB/hand"
            )

    total_hands = args.pairs * 2
    overall_mbb = (sb_payoff + bb_payoff) / total_hands * CHIPS_TO_MBB

    per_hand = np.array(pair_payoffs) / 2.0 * CHIPS_TO_MBB
    ci95 = 1.96 * per_hand.std(ddof=1) / np.sqrt(len(per_hand))

    print(f"\n{'─' * 60}")
    print(f"  Pairs played           : {args.pairs:,}  ({total_hands:,} hands total)")
    print(
        f"  P0 edge as SB          : {sb_payoff / args.pairs * CHIPS_TO_MBB:+.2f} mBB/hand"
    )
    print(
        f"  P0 edge as BB          : {bb_payoff / args.pairs * CHIPS_TO_MBB:+.2f} mBB/hand"
    )
    print(
        f"  P0 overall edge vs P1  : {overall_mbb:+.2f} ± {ci95:.2f} mBB/hand  (95% CI)"
    )


if __name__ == "__main__":
    main()
