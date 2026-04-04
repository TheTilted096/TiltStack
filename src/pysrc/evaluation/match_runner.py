"""
match_runner.py — Duplicate match evaluation between TiltStack agents.

Each deal is played twice: once with TiltStack in the small blind seat and
once with TiltStack in the big blind seat.  Per-seat payoffs are tracked
separately so positional bias is visible alongside the aggregate result.

Usage:
    cd src
    python pysrc/evaluation/match_runner.py \\
        --strat-net  ../checkpoints/policy0050.pt \\
        --br-net0    ../br_checkpoints/br_adv0_0040.pt \\
        --br-net1    ../br_checkpoints/br_adv1_0040.pt \\
        --clusters   ../clusters \\
        --num-games  10000 \\
        --device     cpu

OpenSpiel game parameters are tuned to match the training setup:
    STARTING_STACK = 40,000 milli-chips = 20 BB
    BIG_BLIND      = 2,000  milli-chips  →  chips unit = 100 milli-chips
    Stack in chips = 2000,  Blinds = 50/100
"""

import argparse
import os
import sys
import numpy as np
import torch
import pyspiel

_EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_EVAL_DIR, '..', 'deepcfr'))

import deepcfr
from tilt_agents import TiltStack_DeepCFR, Anti_TiltStack_NBR
from network_training import DeepCFRNet

# ---------------------------------------------------------------------------
# Game configuration — must match CFRTypes.h training parameters
# ---------------------------------------------------------------------------

# Payoff conversion — play_match returns OpenSpiel chip units.
# 1 BB = OSP_BIG_BLIND chips  →  1 mBB = OSP_BIG_BLIND / 1000 chips
# chips_to_mbb = 1000 / OSP_BIG_BLIND
OSP_BIG_BLIND  = 100          # chips (must match blind= in GAME_STRING)
CHIPS_TO_MBB   = 1000 / OSP_BIG_BLIND   # = 10.0

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
    "stack=2000 2000,"
    "firstPlayer=1 2 2 2,"  # <-- Explicitly sets HUNL action order
    "bettingAbstraction=fullgame"
    ")"
)

# ---------------------------------------------------------------------------
# Match loop
# ---------------------------------------------------------------------------

def deal_cards() -> list:
    """
    Sample a fresh 9-card deal: [p0h0, p0h1, p1h0, p1h1, f0, f1, f2, turn, river].

    Returns a list of 9 distinct card indices drawn without replacement from
    the 52-card deck.  Each index is in [0, 51] matching OpenSpiel's encoding.
    """
    return np.random.permutation(52)[:9].tolist()



def play_match(game, bot0, bot1, cards: list) -> float:
    """
    Play a single hand between bot0 (P0) and bot1 (P1) with a pre-dealt deck.

    Chance nodes are resolved deterministically from `cards` in deal order
    [p0h0, p0h1, p1h0, p1h1, f0, f1, f2, turn, river], eliminating any
    independent randomness between the two passes of a duplicate pair.

    Returns Player 0's utility in chips (positive = P0 wins).
    """
    state    = game.new_initial_state()
    deal_idx = 0

    while not state.is_terminal():
        if state.is_chance_node():
            action    = cards[deal_idx]
            deal_idx += 1
        else:
            current_player = state.current_player()
            action = bot0.step(state) if current_player == 0 else bot1.step(state)

        state.apply_action(action)

    return state.returns()[0]   # Player 0's utility


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="TiltStack match evaluation")
    parser.add_argument('--strat-net',  required=True,
                        help='Path to strategy network checkpoint (policy*.pt)')
    parser.add_argument('--br-net0',    required=True,
                        help='Path to best-response advantage network for P0')
    parser.add_argument('--br-net1',    required=True,
                        help='Path to best-response advantage network for P1')
    parser.add_argument('--clusters',   default=None,
                        help='Path to clusters/ directory (for EHS + bucket lookups)')
    parser.add_argument('--num-games',  type=int, default=100_000,
                        help='Number of deals (each played twice, once per seat)')
    parser.add_argument('--device',     default='cpu',
                        help='Torch device (cpu or cuda)')
    args = parser.parse_args()

    # 1. Load cluster tables into CFRGame's global gEHS / gLabels arrays.
    #    Must happen before constructing any CFRGame (including inside agents).
    if args.clusters:
        deepcfr.load_tables(args.clusters)

    # 2. Load networks
    def _load_net(path):
        ckpt = torch.load(path, map_location=args.device, weights_only=True)
        net  = DeepCFRNet()
        net.load_state_dict(ckpt['net'])
        return net.to(args.device).eval()

    strat_net = _load_net(args.strat_net)
    br_net0   = _load_net(args.br_net0)
    br_net1   = _load_net(args.br_net1)

    # 3. Initialise OpenSpiel game
    game = pyspiel.load_game(GAME_STRING)

    # 4. Create agents — each receives the shared game object so _split_history
    #    can replay from a fresh initial state when syncing the CFRGame member.
    #    Each agent reads its current seat from state.current_player() at step
    #    time, so a single instance plays correctly from either seat.
    tiltstack = TiltStack_DeepCFR(strat_net, osp_game=game, device=args.device)
    nbr       = Anti_TiltStack_NBR(br_net0, br_net1, osp_game=game, device=args.device)

    # 5. Evaluation loop — every deal is played exactly twice.
    #    Pass A: TiltStack as small blind (P0), NBR as big blind (P1).
    #    Pass B: NBR as small blind (P0), TiltStack as big blind (P1).
    #    play_match() returns P0's utility, so TiltStack's BB payoff = -payoff_b.
    sb_payoff  = 0.0   # TiltStack cumulative payoff as small blind
    bb_payoff  = 0.0   # TiltStack cumulative payoff as big blind
    deal_payoffs = []  # per-deal combined payoff (both passes) for CI calculation

    for i in range(args.num_games):
        cards     = deal_cards()
        payoff_a  = play_match(game, tiltstack, nbr, cards)
        payoff_b  = play_match(game, nbr, tiltstack, cards)
        sb_payoff += payoff_a
        bb_payoff -= payoff_b   # flip: payoff_b is NBR's P0 utility
        deal_payoffs.append(payoff_a - payoff_b)

        if (i + 1) % 1000 == 0:
            hands_so_far = (i + 1) * 2
            total_so_far = sb_payoff + bb_payoff
            print(f"  [{i+1:6d}/{args.num_games}]  "
                  f"TiltStack overall {total_so_far / hands_so_far * CHIPS_TO_MBB:+.1f}  "
                  f"sb {sb_payoff / (i+1) * CHIPS_TO_MBB:+.1f}  "
                  f"bb {bb_payoff / (i+1) * CHIPS_TO_MBB:+.1f}  mBB/hand")

    total_hands = args.num_games * 2
    overall_mbb = (sb_payoff + bb_payoff) / total_hands * CHIPS_TO_MBB

    # 95% CI: each deal contributes 2 hands; per-hand payoff = deal_payoff / 2.
    # SE of the per-hand mean = std(deal_payoff/2) / sqrt(n_deals).
    per_hand = np.array(deal_payoffs) / 2.0 * CHIPS_TO_MBB
    ci95 = 1.96 * per_hand.std(ddof=1) / np.sqrt(len(per_hand))

    print(f"\n{'─'*60}")
    print(f"  Deals played             : {args.num_games:,}  ({total_hands:,} hands total)")
    print(f"  TiltStack edge as SB     : {sb_payoff / args.num_games * CHIPS_TO_MBB:+.2f} mBB/hand")
    print(f"  TiltStack edge as BB     : {bb_payoff / args.num_games * CHIPS_TO_MBB:+.2f} mBB/hand")
    print(f"  TiltStack overall edge   : {overall_mbb:+.2f} ± {ci95:.2f} mBB/hand  (95% CI)")


if __name__ == "__main__":
    main()
