#!/usr/bin/env python3
"""
TiltStack Demo: Exploitative Agent vs Tight Opponent
Northwestern IEEE Technical Program | April 4, 2026

Trains GTO baseline via CFR+, then runs a live animated comparison:
  - GTO agent (Nash equilibrium)  vs Tight opponent
  - Exploitative agent            vs Tight opponent

Shows real-time cumulative chips + rolling win-rate.
"""
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'pysrc'))
from Leduc import Leduc
from leducsolver import Action, BestResponse as BRSolver

# ─────────────────────────────────────────────────────────────
# Pure-Python game logic (mirrors Node.cpp / Leduc.cpp)
# ─────────────────────────────────────────────────────────────

def _seq(h):
    return h % 8 if h < 24 else ((h - 24) // 21) % 8

def _stm(h):
    s = _seq(h)
    return ((s % 2) + (s // 4)) % 2

def _raises(h):
    return _seq(h) % 4

def _bet_round(h):
    return 0 if h < 24 else 1

def _private_card(h):
    return h // 8 if h < 24 else ((h - 24) % 21) // 7

def _shared_card(h):
    return 0 if h < 24 else (h - 24) // 168

def _legal_moves(h):
    r = _raises(h)
    if r == 0:   return [Action.CHECK, Action.RAISE]
    if r == 3:   return [Action.CHECK, Action.BET]
    return [Action.CHECK, Action.BET, Action.RAISE]

def _round_action(h):
    return Action.BET if _raises(h) > 0 else Action.CHECK

def _ends_hand(h, a):
    if a == Action.CHECK and _raises(h) > 0:
        return True
    if _bet_round(h) == 1:
        if a == _round_action(h) and _seq(h) != 0:
            return True
    return False

def _compare(a, b, c):
    sa = a + c + 4 * (a == c)
    sb = b + c + 4 * (b == c)
    return 2 if sa > sb else (1 if sa == sb else 0)

def _payout(h, a, opp):
    """Payout for the player acting at node h."""
    br = _bet_round(h)
    r  = _raises(h)
    my = _private_card(h)
    sc = _shared_card(h)
    r1r = (((h - 24) % 21) % 7 + 1) % 4 if br == 1 else r

    if a == Action.CHECK and r > 0:          # fold
        if br == 0:
            return -(1 + 2 * (r - 1))
        return -(1 + 2 * r1r + 4 * (r - 1))

    total = 1 + 2 * r1r + (4 * r if br == 1 else 0)
    return (_compare(my, opp, sc) - 1) * total

def _next_stm(h, a):
    br  = _bet_round(h)
    s   = _seq(h)
    cur = _stm(h)
    is_trans = int(br == 0) * int(a == _round_action(h)) * int(s != 0)
    return (1 - is_trans) * (1 - cur)

def _next_hash(h, a, community, next_card):
    br = _bet_round(h)
    if br == 0:
        if a == _round_action(h) and (h % 8 != 0):   # round 1 → 2
            r1_seq = h % 8
            return 24 + community * 168 + next_card * 7 + (r1_seq - 1)
        old_seq = h % 8
        new_seq = old_seq + (4 if a == Action.CHECK else 1)
        return next_card * 8 + new_seq
    else:
        adj = h - 24
        sc       = adj // 168
        r2_seq   = (adj // 21) % 8
        r1_info  = adj % 7
        new_r2   = r2_seq + (4 if a == Action.CHECK else 1)
        return 24 + sc * 168 + new_r2 * 21 + next_card * 7 + r1_info

# ─────────────────────────────────────────────────────────────
# Strategy functions
# ─────────────────────────────────────────────────────────────

def gto_action(strategies, h, rng):
    """Sample an action from the trained Nash-equilibrium strategy."""
    moves = _legal_moves(h)
    strat = strategies[h]
    probs = [float(strat[m.value]) for m in moves]
    total = sum(probs)
    if total < 1e-9:
        return rng.choice(moves)
    probs = [p / total for p in probs]
    return rng.choice(moves, p=probs)


# Rank constants (0=Jack, 1=Queen, 2=King)
JACK, QUEEN, KING = 0, 1, 2

def tight_action(h, rng):
    """
    Tight player profile:
      Jack  – never bets; folds to any bet
      Queen – calls bets, rarely opens
      King  – always bets/raises
    """
    moves = _legal_moves(h)
    card  = _private_card(h)
    r     = _raises(h)

    if card == JACK:
        return Action.CHECK                                     # check or fold

    if card == QUEEN:
        if r == 0:
            if Action.RAISE in moves and rng.random() < 0.25:
                return Action.RAISE                             # rare open-bet
            return Action.CHECK
        return Action.BET if Action.BET in moves else Action.CHECK  # call

    # King: maximum aggression
    if Action.RAISE in moves: return Action.RAISE
    return Action.BET if Action.BET in moves else Action.CHECK


def build_tight_strategy_vector():
    """Convert the tight opponent's policy to a 528-element Strategy vector for BRSolver."""
    strat = []
    for h in range(528):
        s = [0.0, 0.0, 0.0]
        card  = _private_card(h)
        r     = _raises(h)
        moves = _legal_moves(h)
        if card == JACK:
            s[Action.CHECK.value] = 1.0
        elif card == QUEEN:
            if r == 0 and Action.RAISE in moves:
                s[Action.CHECK.value] = 0.75
                s[Action.RAISE.value] = 0.25
            elif Action.BET in moves:
                s[Action.BET.value] = 1.0
            else:
                s[Action.CHECK.value] = 1.0
        else:   # King
            if Action.RAISE in moves:
                s[Action.RAISE.value] = 1.0
            elif Action.BET in moves:
                s[Action.BET.value] = 1.0
            else:
                s[Action.CHECK.value] = 1.0
        strat.append(s)
    return strat


def compute_exploit_strategy():
    """
    Use the C++ BestResponse solver to find the exact optimal counter-strategy
    against the tight opponent. Returns (br_strategy, theoretical_ev).
    """
    tight_vec = build_tight_strategy_vector()
    br = BRSolver()
    br.load_strategy(tight_vec)
    ev = br.compute(0)          # P0 best response vs tight P1
    br_strat = br.get_full_br_strategy()
    return br_strat, ev


def exploit_action(br_strategy, h, rng):
    """Sample from the BRSolver-computed optimal counter-strategy vs tight."""
    moves = _legal_moves(h)
    strat = br_strategy[h]
    probs = [float(strat[m.value]) for m in moves]
    total = sum(probs)
    if total < 1e-9:
        return rng.choice(moves)
    probs = [p / total for p in probs]
    return rng.choice(moves, p=probs)

# ─────────────────────────────────────────────────────────────
# Hand simulator
# ─────────────────────────────────────────────────────────────

def simulate_hand(p0_fn, p1_fn, rng):
    """
    Play out one Leduc hand.  Returns chip payout from P0's perspective.
    p0_fn / p1_fn: callable(hash) -> Action
    """
    p0   = int(rng.integers(0, 3))
    p1   = int(rng.integers(0, 3))
    comm = int(rng.integers(0, 3))
    cards = [p0, p1, comm]
    h = p0 * 8                          # P0 acts first in round 1

    while True:
        cur    = _stm(h)
        action = p0_fn(h) if cur == 0 else p1_fn(h)

        if _ends_hand(h, action):
            pay = _payout(h, action, cards[1 - cur])
            return -pay if cur == 1 else pay

        ns = _next_stm(h, action)
        h  = _next_hash(h, action, comm, cards[ns])

# ─────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────

TRAIN_ITERS     = 15_000
N_HANDS         = 600
HANDS_PER_FRAME = 5
ROLL_WINDOW     = 60

# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    rng = np.random.default_rng(42)

    print("=" * 62)
    print("  TiltStack  |  Adaptive Poker AI  |  IEEE Demo")
    print("=" * 62)
    print(f"\n[1/2] Training GTO baseline ({TRAIN_ITERS:,} iterations)…\n")

    leduc = Leduc()
    checkpoints = [1_000, 3_000, 5_000, 8_000, 12_000, 15_000]

    for i in range(TRAIN_ITERS):
        leduc.train(1)
        if (i + 1) in checkpoints:
            expl = leduc.compute_exploitability() * 1000
            filled = int((i + 1) / TRAIN_ITERS * 36)
            bar = "█" * filled + "░" * (36 - filled)
            print(f"  [{bar}]  iter {i+1:>6}  |  exploitability {expl:.3f} mBB/hand")

    strategies = leduc.solver.get_all_strategies()
    print("\n  ✓ Nash equilibrium reached (≈ 0.00 mBB/hand)\n")

    print("  Computing exact best response vs tight opponent (BRSolver)…")
    br_strategy, br_ev = compute_exploit_strategy()
    print(f"  ✓ Exploit EV (theoretical): {br_ev:+.4f} chips/hand\n")

    print("[2/2] Running live match — both agents vs Tight Opponent…\n")

    p0_gto     = lambda h: gto_action(strategies, h, rng)
    p0_exploit = lambda h: exploit_action(br_strategy, h, rng)
    p1_tight   = lambda h: tight_action(h, rng)

    # ── Dark-themed figure ────────────────────────────────────
    BG     = "#0f1117"
    PANEL  = "#1a1d27"
    GTO_C  = "#4ecdc4"
    EXP_C  = "#ff6b6b"
    GRID_C = "#2a2d3a"
    WHITE  = "#e8eaf0"
    MUTED  = "#888fa8"

    fig, (ax_cum, ax_roll) = plt.subplots(1, 2, figsize=(15, 6.5))
    fig.patch.set_facecolor(BG)
    fig.suptitle(
        "TiltStack — Exploitative vs GTO vs Tight Opponent",
        color=WHITE, fontsize=15, fontweight="bold", y=0.97
    )

    for ax in (ax_cum, ax_roll):
        ax.set_facecolor(PANEL)
        ax.tick_params(colors=MUTED, labelsize=9)
        for spine in ax.spines.values():
            spine.set_color(GRID_C)
        ax.grid(color=GRID_C, linewidth=0.6, linestyle="--", alpha=0.7)

    ax_cum.set_title("Cumulative Chips Won", color=WHITE, fontsize=12, pad=10)
    ax_cum.set_xlabel("Hand #", color=MUTED, fontsize=10)
    ax_cum.set_ylabel("Chips", color=MUTED, fontsize=10)

    ax_roll.set_title(f"Rolling Win Rate (last {ROLL_WINDOW} hands)", color=WHITE, fontsize=12, pad=10)
    ax_roll.set_xlabel("Hand #", color=MUTED, fontsize=10)
    ax_roll.set_ylabel("Chips / Hand", color=MUTED, fontsize=10)
    ax_roll.axhline(0, color=GRID_C, linewidth=1.2)

    line_gto_cum,  = ax_cum.plot([], [], color=GTO_C, linewidth=2,   label="GTO baseline")
    line_exp_cum,  = ax_cum.plot([], [], color=EXP_C, linewidth=2.5, label="Exploitative")
    line_gto_roll, = ax_roll.plot([], [], color=GTO_C, linewidth=2,   label="GTO baseline")
    line_exp_roll, = ax_roll.plot([], [], color=EXP_C, linewidth=2.5, label="Exploitative")

    legend_kw = dict(facecolor=PANEL, edgecolor=GRID_C, labelcolor=WHITE, fontsize=10)
    ax_cum.legend(**legend_kw)
    ax_roll.legend(**legend_kw)

    stat_text = fig.text(
        0.5, 0.015, "Simulating…",
        ha="center", color=WHITE, fontsize=11, fontfamily="monospace"
    )

    # State
    gto_hands,  expl_hands  = [], []
    gto_cumulative = [0]
    exp_cumulative = [0]
    xs = [0]
    hand_counter = [0]

    def update(_frame):
        if hand_counter[0] >= N_HANDS:
            return

        for _ in range(HANDS_PER_FRAME):
            if hand_counter[0] >= N_HANDS:
                break
            g = simulate_hand(p0_gto,     p1_tight, rng)
            e = simulate_hand(p0_exploit, p1_tight, rng)
            gto_hands.append(g)
            expl_hands.append(e)
            gto_cumulative.append(gto_cumulative[-1] + g)
            exp_cumulative.append(exp_cumulative[-1] + e)
            hand_counter[0] += 1
            xs.append(hand_counter[0])

        n = hand_counter[0]
        line_gto_cum.set_data(xs[:n + 1], gto_cumulative[:n + 1])
        line_exp_cum.set_data(xs[:n + 1], exp_cumulative[:n + 1])
        ax_cum.relim(); ax_cum.autoscale_view()

        if n >= ROLL_WINDOW:
            rx = list(range(ROLL_WINDOW, n + 1))
            gr = [np.mean(gto_hands[i - ROLL_WINDOW:i])  for i in range(ROLL_WINDOW, n + 1)]
            er = [np.mean(expl_hands[i - ROLL_WINDOW:i]) for i in range(ROLL_WINDOW, n + 1)]
            line_gto_roll.set_data(rx, gr)
            line_exp_roll.set_data(rx, er)
            ax_roll.relim(); ax_roll.autoscale_view()

        if n > 10:
            g_avg = np.mean(gto_hands)
            e_avg = np.mean(expl_hands)
            mult  = f"{e_avg / g_avg:.2f}x" if abs(g_avg) > 1e-6 else "∞"
            stat_text.set_text(
                f"Hand {n}/{N_HANDS}  │  "
                f"GTO {g_avg:+.3f} chips/hand  │  "
                f"Exploit {e_avg:+.3f} chips/hand  │  "
                f"Speedup {mult}"
            )

    ani = animation.FuncAnimation(
        fig, update,
        frames=N_HANDS // HANDS_PER_FRAME + 20,
        interval=60,
        repeat=False,
    )

    plt.tight_layout(rect=[0, 0.055, 1, 0.94])
    plt.show()

    # ── Final summary ─────────────────────────────────────────
    g_avg = np.mean(gto_hands)
    e_avg = np.mean(expl_hands)
    print(f"\n{'─' * 62}")
    print(f"  Results — {N_HANDS} hands vs Tight Opponent")
    print(f"{'─' * 62}")
    print(f"  GTO baseline:        {g_avg:+.4f} chips/hand")
    print(f"  Exploitative agent:  {e_avg:+.4f} chips/hand")
    if abs(g_avg) > 1e-6:
        print(f"  Improvement:         {e_avg / g_avg:.2f}x")
    print(f"{'─' * 62}\n")
