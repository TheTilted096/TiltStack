"""
preflop_query.py — Show the strategy network's preflop policy for a given hand.

Usage:
    cd src
    python pysrc/evaluation/preflop_query.py --net ../checkpoints/policy.pt 99 SB
    python pysrc/evaluation/preflop_query.py --net ../checkpoints/policy.pt A6o BB
    python pysrc/evaluation/preflop_query.py --net ../checkpoints/policy.pt 56s SB

Hand notation:
    Pairs   : 99, TT, AA          (two cards same rank, different suits)
    Suited  : AKs, 56s, T9s       (both cards same suit)
    Offsuit : A6o, KJo, 72o       (cards different suits)

Position:
    SB  small blind / button (acts first preflop)
    BB  big blind             (SB assumed to call, then BB acts)
"""

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

_EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_EVAL_DIR, "..", "deepcfr"))

import deepcfr
from network_training import DeepCFRNet, decode_batch, NUM_ACTIONS
from tilt_agents import load_net_auto

# ---------------------------------------------------------------------------
# Constants (must match CFRTypes.h)
# ---------------------------------------------------------------------------

STARTING_STACK = 100_000  # milli-chips
SMALL_BLIND = 1_000
BIG_BLIND = 2_000

_FOLD = 0  # CHECK when to_call > 0
_CALL = 1
_BET33 = 2
_BET50 = 3
_BET75 = 4
_BET100 = 5
_BET150 = 6
_BET200 = 7
_BET300 = 8
_ALLIN = 9

_ACTION_NAMES = [
    "Fold / Check",  # slot 0: Fold when to_call > 0, Check when to_call == 0
    "Call",
    "Raise ~1/3 pot",
    "Raise ~1/2 pot",
    "Raise ~3/4 pot",
    "Raise  1x  pot",
    "Raise 1.5x pot",
    "Raise  2x  pot",
    "Raise  3x  pot",
    "All-in",
]

# ---------------------------------------------------------------------------
# Card parsing
# ---------------------------------------------------------------------------

_RANK_CHARS = "23456789TJQKA"
_SUIT_CHARS = "cdhs"

# card index = rank * 4 + suit  (matches hand-isomorphism deck.h)
_RANK = {c: i for i, c in enumerate(_RANK_CHARS)}
_SUIT = {c: i for i, c in enumerate(_SUIT_CHARS)}


def _card(rank_char: str, suit_char: str) -> int:
    return _RANK[rank_char.upper()] * 4 + _SUIT[suit_char.lower()]


def parse_hand(hand: str) -> tuple[int, int]:
    """
    Parse a hand string and return two card indices.

    '99'   → pocket 9s (9c, 9d)
    'AKs'  → ace-king suited (Ac, Kc)
    'A6o'  → ace-six offsuit (Ac, 6d)
    """
    hand = hand.strip()
    if len(hand) == 2:
        r1, r2 = hand[0].upper(), hand[1].upper()
        if r1 not in _RANK or r2 not in _RANK:
            raise ValueError(f"Unknown rank in '{hand}'")
        if r1 != r2:
            raise ValueError(
                f"Two-character hand '{hand}' must be a pair. "
                "Use three characters (e.g. AKs / AKo) for non-pairs."
            )
        return _card(r1, "c"), _card(r2, "d")

    if len(hand) == 3:
        r1, r2, suf = hand[0].upper(), hand[1].upper(), hand[2].lower()
        if r1 not in _RANK or r2 not in _RANK:
            raise ValueError(f"Unknown rank in '{hand}'")
        if r1 == r2:
            raise ValueError(f"Pairs don't have a suit suffix — use '{r1}{r2}'.")
        if suf == "s":
            return _card(r1, "c"), _card(r2, "c")
        if suf == "o":
            return _card(r1, "c"), _card(r2, "d")
        raise ValueError(f"Unknown suit suffix '{suf}' in '{hand}'. Use 's' or 'o'.")

    raise ValueError(f"Cannot parse hand '{hand}'. Expected format: 99 / AKs / A6o.")


def _dummy_cards(used: set, n: int) -> list[int]:
    """Return n card indices not in `used`."""
    result = []
    for c in range(52):
        if c not in used:
            result.append(c)
        if len(result) == n:
            break
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _query(game: "deepcfr.CFRGame", net, device, hand: str, pos: str) -> None:
    try:
        c1, c2 = parse_hand(hand)
    except ValueError as e:
        print(f"  Error: {e}")
        return

    # cards9: [p0h0, p0h1, p1h0, p1h1, f0, f1, f2, turn, river]
    # Need 7 dummies: 2 for opponent hole cards + 5 board cards
    dummies = _dummy_cards({c1, c2}, 7)
    if pos == "SB":
        cards9 = [c1, c2] + dummies[:2] + dummies[2:]
    else:
        cards9 = dummies[:2] + [c1, c2] + dummies[2:]

    game.begin_with_cards(STARTING_STACK, STARTING_STACK, True, cards9)
    if pos == "BB":
        game.make_move(_CALL)

    raw = game.get_info()
    x_cont, buckets = decode_batch(raw)
    with torch.no_grad():
        logits = net(x_cont.to(device), buckets.to(device))[0]

    legal = game.generate_actions()
    facing_bet = game.to_call > 0
    mask = torch.full((NUM_ACTIONS,), -1e9)
    for a in legal:
        mask[a] = 0.0
    probs = F.softmax(logits.cpu() + mask, dim=0).numpy()

    action_names = list(_ACTION_NAMES)
    action_names[0] = "Fold" if facing_bet else "Check"

    hand_str = hand.upper() if len(hand) == 2 else hand[:2].upper() + hand[2].lower()
    print(f"\n  {hand_str}  {pos}  —  preflop strategy\n")
    print(f"  {'#':<2}  {'Action':<22}  {'Prob':>6}  Bar")
    print(f"  {'-' * 2}  {'-' * 22}  {'-' * 6}  {'-' * 20}")
    for a in range(NUM_ACTIONS):
        label = action_names[a]
        p = probs[a]
        if a not in legal:
            print(f"  {a:<2}  {label:<22}  {'—':>6}")
            continue
        bar = "█" * int(p * 20)
        print(f"  {a:<2}  {label:<22}  {p * 100:5.1f}%  {bar}")
    print(f"\n  distribution: [{', '.join(f'{p:.4f}' for p in probs)}]\n")


def main():
    _clusters = os.path.join(_EVAL_DIR, "..", "..", "clusters")

    parser = argparse.ArgumentParser(description="Preflop strategy query (interactive)")
    parser.add_argument("--net", required=True, help="Path to strategy network (.pt)")
    parser.add_argument(
        "--clusters", default=_clusters, help="Path to clusters/ directory"
    )
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    if not os.path.isdir(args.clusters):
        sys.exit(f"Error: clusters directory not found at '{args.clusters}'")

    print("Loading tables...", flush=True)
    deepcfr.load_tables(args.clusters)

    device = torch.device(args.device)
    net = load_net_auto(args.net, device)

    game = deepcfr.CFRGame()

    print("Ready. Enter queries as '<hand> <SB|BB>', or 'quit' to exit.")
    print("Examples: AA SB  /  A6o BB  /  56s SB\n")

    while True:
        try:
            line = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not line:
            continue
        if line.lower() == "quit":
            break

        parts = line.split()
        if len(parts) != 2:
            print("  Usage: <hand> <SB|BB>")
            continue

        hand, pos_raw = parts
        pos = pos_raw.upper()
        if pos not in ("SB", "BB"):
            print("  Position must be SB or BB.")
            continue

        _query(game, net, device, hand, pos)


if __name__ == "__main__":
    main()
