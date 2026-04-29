#!/usr/bin/env python3
"""
poker_live.py — Play NLHE against TiltStack DeepCFR in the terminal.

The board is redrawn in-place (curses) after every action.
Seats are randomised each hand so you play both positions.

Usage
-----
    cd src
    python pysrc/evaluation/poker_live.py \\
        --net       ../checkpoints/policy0050.pt \\
        --clusters  ../clusters \\
        --device    cpu

Keybindings
-----------
  During your turn:
    f / F    fold
    c / C    call or check
    r / R    raise (type chip amount, Enter to confirm, Esc to cancel)
    a / A    all-in

  Any time:
    x / X    toggle cheat mode  (bot's hole cards + last inference probs)
    n / N    deal next hand     (only after hand ends)
    q / Q    quit

OpenSpiel / CFRGame conventions match match_runner.py exactly:
    STARTING_STACK = 100,000 milli-chips  =  5,000 chips  =  50 BB
    BIG_BLIND      =   2,000 milli-chips  =    100 chips
    1 chip = 10 mBB
"""

import argparse
import curses
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
import pyspiel

_EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_EVAL_DIR, "..", "deepcfr"))

import deepcfr
from tilt_agents import (
    TiltStack_DeepCFR,
    load_net_auto,
    _abstract_to_osp,
)
from network_training import DeepCFRNet, NUM_ACTIONS, infoset_dtype

# ---------------------------------------------------------------------------
# Game configuration — must match match_runner.py / CFRTypes.h
# ---------------------------------------------------------------------------

OSP_BIG_BLIND = 100  # chips  (blind=50 100 in game string)
OSP_STACK = 5_000  # chips per player

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
    "firstPlayer=1 2 2 2,"  # <-- Explicitly sets HUNL action order
    "bettingAbstraction=fullgame"
    ")"
)

# ---------------------------------------------------------------------------
# Card rendering
# ---------------------------------------------------------------------------

RANKS = ["2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A"]
SUIT_CHARS = ["s", "h", "d", "c"]  # matches hand-isomorphism SUIT_TO_CHAR "shdc"
SUIT_GLYPHS = ["♠", "♥", "♦", "♣"]  # spades hearts diamonds clubs


def _card_str(idx: int) -> str:
    return RANKS[idx // 4] + SUIT_GLYPHS[idx % 4]


def _is_red(idx: int) -> bool:
    """True for hearts (suit 1) and diamonds (suit 2)."""
    return idx % 4 in (1, 2)


# ---------------------------------------------------------------------------
# PokerLive
# ---------------------------------------------------------------------------


class PokerLive:
    """Manages one interactive NLHE session spanning multiple hands."""

    def __init__(self, net_path: str, clusters_path: str | None, device: str):
        self.device = torch.device(device)

        # Load strategy network
        self.model = load_net_auto(net_path, self.device)

        # OpenSpiel game (stateless, shared)
        self.osp_game = pyspiel.load_game(GAME_STRING)

        # Bot agent — owns a deepcfr.CFRGame for feature extraction / inference
        self.bot = TiltStack_DeepCFR(self.model, self.osp_game, device=device)

        # Session totals
        self.hand_num = 0
        self.session_bb = 0.0
        self.session_hands = 0

        # Per-hand state  (initialised by _new_hand)
        self.state = None
        self.raw_deal = []  # 9 card indices (full future deal, pre-shuffled)
        self.deal_idx = 0  # how many chance cards have been applied so far
        self.human_player = 0  # 0 or 1
        self.invested = [0, 0]  # cumulative chips each player has put into the pot

        # Action log: list of (text, is_street_marker)
        self.action_log: list[tuple[str, bool]] = []

        self.hand_over = False
        self.result_msg = ""

        # Cheat-mode state
        self.cheat_mode = False
        self.last_bot_probs = None  # np.ndarray (NUM_ACTIONS,) or None
        self.last_bot_abstract = -1  # abstract action index bot chose last
        self.last_bot_legal: set = set()  # set of legal abstract action indices
        self.last_bot_raw_info = None  # raw uint8 InfoSet buffer (1, 168) or None

        # Raise input state (None = not active, str = digits typed so far)
        self.raise_input: str | None = None

    # ------------------------------------------------------------------
    # Hand lifecycle
    # ------------------------------------------------------------------

    def _new_hand(self):
        self.human_player = int(np.random.randint(0, 2))
        self.raw_deal = np.random.permutation(52)[:9].tolist()
        self.deal_idx = 0
        self.state = self.osp_game.new_initial_state()
        self.hand_num += 1
        self.hand_over = False
        self.result_msg = ""
        self.invested = [50, 100]  # blinds are auto-posted before betting
        self.last_bot_probs = None
        self.last_bot_abstract = -1
        self.last_bot_legal = set()
        self.last_bot_raw_info = None
        self.raise_input = None

        hp, bp = self.human_player, 1 - self.human_player
        sb_tag = f"{'YOU' if hp == 0 else 'bot'}(SB)"
        bb_tag = f"{'YOU' if hp == 1 else 'bot'}(BB)"
        self.action_log = [(f"{sb_tag} post 0.50 BB   {bb_tag} post 1.00 BB", False)]

        self._advance()

    def _advance(self):
        """
        Drive the OpenSpiel state forward until the human must act or the hand
        ends.  Chance nodes (card deals) and bot turns are resolved immediately.
        """
        while not self.state.is_terminal():
            if self.state.is_chance_node():
                card = self.raw_deal[self.deal_idx]
                self.deal_idx += 1
                self.state.apply_action(card)
                # Announce board streets in the action log
                if self.deal_idx == 7:
                    self.action_log.append(("────── Flop ──────", True))
                elif self.deal_idx == 8:
                    self.action_log.append(("────── Turn ──────", True))
                elif self.deal_idx == 9:
                    self.action_log.append(("────── River ──────", True))

            elif self.state.current_player() == self.human_player:
                break  # hand off to the human

            else:
                # Bot's turn — sync CFRGame, run forward pass, capture probs
                synced = self.bot._sync_game(self.state)
                if not synced:
                    osp_action = 1  # fallback: call/check
                else:
                    self.last_bot_raw_info = self.bot.game.get_info().copy()
                    masked_logits, legal_abstract = self.bot._forward(self.bot.model)
                    probs = F.softmax(torch.from_numpy(masked_logits), dim=0).numpy()
                    self.last_bot_probs = probs.copy()
                    self.last_bot_legal = set(legal_abstract)
                    self.last_bot_abstract = int(np.random.choice(NUM_ACTIONS, p=probs))
                    osp_action = _abstract_to_osp(
                        self.last_bot_abstract,
                        self.state.legal_actions(),
                        self.bot.game,
                    )
                self._apply_logged(self.state.current_player(), osp_action)

        if self.state.is_terminal():
            self._end_hand()

    def _apply_logged(self, player: int, osp_action: int):
        """Apply osp_action to self.state, update invested, and log the action."""
        to_call = max(self.invested) - self.invested[player]
        is_you = player == self.human_player
        pos = "SB" if player == 0 else "BB"
        who = f"{pos}{'(you)' if is_you else '(bot)'}"

        if osp_action == 0:  # fold
            msg = f"{who}: fold"
        elif osp_action == 1:  # call / check
            if to_call == 0:
                msg = f"{who}: check"
            else:
                msg = f"{who}: call  +{to_call / OSP_BIG_BLIND:.2f} BB"
            self.invested[player] = max(self.invested)
        else:  # raise — osp_action = total invested
            added = osp_action - self.invested[player]
            msg = f"{who}: raise +{added / OSP_BIG_BLIND:.2f} BB"
            self.invested[player] = osp_action

        self.action_log.append((msg, False))
        self.state.apply_action(osp_action)

    def _end_hand(self):
        self.hand_over = True
        self.session_hands += 1
        returns = self.state.returns()
        human_chips = returns[self.human_player]
        human_bb = human_chips / OSP_BIG_BLIND
        self.session_bb += human_bb

        if human_chips > 0:
            outcome = f"WON  +{human_bb:.2f} BB"
        elif human_chips < 0:
            outcome = f"LOST  {human_bb:.2f} BB"
        else:
            outcome = "CHOPPED  (0.00 BB)"

        self.result_msg = f"{outcome}     [N] deal next hand"

    # ------------------------------------------------------------------
    # Human-action helpers
    # ------------------------------------------------------------------

    def _sync_human_perspective(self):
        """
        Sync bot.game from the human's (current player's) perspective so that
        CFRGame bet-amount helpers are valid from the human's seat.
        Must only be called when it is the human's turn.
        """
        self.bot._sync_game(self.state)

    def _resolve_raise(self) -> int | None:
        """Snap the typed chip amount to the nearest legal OSP raise action."""
        if not self.raise_input:
            return None
        try:
            added = int(self.raise_input)
        except ValueError:
            return None
        if added <= 0:
            return None
        legal = self.state.legal_actions()
        raises = sorted(a for a in legal if a > 1)
        if not raises:
            return None
        target = self.invested[self.human_player] + added
        return min(raises, key=lambda a: abs(a - target))

    # ------------------------------------------------------------------
    # Rendering helpers
    # ------------------------------------------------------------------

    def _infoset_lines(self) -> list[str]:
        """
        Parse self.last_bot_raw_info into human-readable display lines.

        Scalars are normalised by STARTING_STACK=100 000 milli-chips in the C++
        struct.  To recover chips: raw × 100 000 / OSP_MC_SCALE = raw × 5 000.
        """
        # raw × STARTING_STACK_mc / mc_per_chip = raw × 100000 / 20 = raw × 5000
        NORM_TO_CHIPS = 100_000 / 20  # = 5000

        if self.last_bot_raw_info is None:
            return ["  │  InfoSet: (no data yet)"]

        rec = self.last_bot_raw_info.ravel().view(infoset_dtype)[0]

        street_names = ["PREFLOP", "FLOP", "TURN", "RIVER"]
        se = list(rec["street_embed"])
        street = street_names[se.index(True)] if True in se else "?"
        se_str = " ".join("1" if b else "0" for b in se)  # one-hot bits
        pos = "BTN" if bool(rec["is_button"]) else "BB"

        def _bkt(v):
            return str(int(v)) if v else "—"

        bkts = rec["street_bucket"]
        bkt_str = f"Fl:{_bkt(bkts[0])}  Tu:{_bkt(bkts[1])}  Rv:{_bkt(bkts[2])}"

        my_norm = float(rec["my_stack"])
        opp_norm = float(rec["opp_stack"])
        pot_norm = float(rec["pot_size"])
        tc_norm = float(rec["to_call"])
        spr = float(rec["explicit_spr"])

        def _mask_to_cards(mask: int) -> str:
            if mask == 0:
                return "?"
            cards = []
            m = mask
            while m:
                lsb = m & (-m)
                cards.append(lsb.bit_length() - 1)
                m &= m - 1
            return " ".join(str(c) for c in sorted(cards))

        hole_str = f"[{_mask_to_cards(int(rec['hole']))}]"
        flop_str = f"[{_mask_to_cards(int(rec['flop']))}]"
        turn_str = f"[{_mask_to_cards(int(rec['turn']))}]"
        river_str = f"[{_mask_to_cards(int(rec['river']))}]"
        cards_str = f"{hole_str} {flop_str} {turn_str} {river_str}"

        bh = rec["bet_hist"]  # shape (4, 6), normalised fractions
        bhmask = int(rec["bet_hist_mask"])
        bh_lines = []
        MAX_ACTIONS = 6
        for rnd, rname in enumerate(["PF  ", "Flop", "Turn", "Riv "]):
            slots = []
            for slot in range(MAX_ACTIONS):
                v = bh[rnd][slot]
                if bhmask & (1 << (rnd * MAX_ACTIONS + slot)):
                    slots.append(f"[{v:.3f}]")  # used: bracketed
                else:
                    slots.append(f" {v:.3f} ")  # unused: same width, no brackets
            bh_lines.append(f"  │    {rname}  " + " ".join(slots))

        return [
            f"  │  Street: {street:<7}  [{se_str}]  Pos: {pos}",
            f"  │  Cards:  {cards_str}",
            f"  │  Buckets: {bkt_str}",
            f"  │  Stack:  Me:{my_norm:.4f}  Opp:{opp_norm:.4f}  (norm)",
            f"  │  Pot:{pot_norm:.4f}  ToCall:{tc_norm:.4f}  SPR:{spr:.4f}  (norm)",
            "  │  BetHist:",
            *bh_lines,
        ]

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def draw(self, stdscr, h: int, w: int):
        stdscr.erase()

        # Curses color-pair aliases (initialised in run())
        BOLD = curses.A_BOLD
        DIM = curses.A_DIM
        C_RED = curses.color_pair(2)  # red suit glyphs
        C_GRN = curses.color_pair(3)  # green  — wins / headers
        C_CYN = curses.color_pair(4)  # cyan   — bot
        C_YLW = curses.color_pair(5)  # yellow — street markers

        def put(r: int, c: int, text: str, attr: int = curses.A_NORMAL):
            if r >= h or c >= w:
                return
            try:
                stdscr.addstr(r, c, str(text)[: max(0, w - c)], attr)
            except curses.error:
                pass

        def hline(r: int):
            put(r, 0, "─" * w)

        row = 0

        # ── Header ──────────────────────────────────────────────────────
        cheat_tag = "   ★ CHEAT ★" if self.cheat_mode else ""
        session_str = (
            f"Session: {self.session_bb:+.2f} BB"
            f"  ({self.session_hands} hands){cheat_tag}"
        )
        put(row, 0, f"  Hand #{self.hand_num}", BOLD)
        put(row, 20, session_str, C_GRN | BOLD)
        row += 1
        hline(row)
        row += 1

        # ── Parse deal ──────────────────────────────────────────────────
        nd = self.deal_idx
        p0_cards = self.raw_deal[0:2] if nd >= 4 else []
        p1_cards = self.raw_deal[2:4] if nd >= 4 else []

        # In cheat mode, reveal the full pre-shuffled board; future cards
        # (not yet officially dealt) are wrapped in parentheses.
        cheat_board = self.cheat_mode and nd >= 4
        flop = self.raw_deal[4:7] if (nd >= 7 or cheat_board) else []
        turn_c = [self.raw_deal[7]] if (nd >= 8 or cheat_board) else []
        river_c = [self.raw_deal[8]] if (nd >= 9 or cheat_board) else []

        hp = self.human_player
        bp = 1 - hp
        hc = p0_cards if hp == 0 else p1_cards  # human cards
        bc = p0_cards if bp == 0 else p1_cards  # bot cards

        h_pos = "SB" if hp == 0 else "BB"
        b_pos = "SB" if bp == 0 else "BB"

        pot = sum(self.invested)
        stacks = [OSP_STACK - self.invested[0], OSP_STACK - self.invested[1]]
        h_stack = stacks[hp]
        b_stack = stacks[bp]

        # ── Community cards ─────────────────────────────────────────────
        # deal_offset[i] = index into raw_deal for the i-th board card
        deal_offsets = [4, 5, 6, 7, 8]
        board = flop + turn_c + river_c
        b_parts: list[str] = []
        for i, c in enumerate(board):
            if i == 3:
                b_parts.append("|")
            if i == 4:
                b_parts.append("|")
            future = cheat_board and deal_offsets[i] >= nd
            s = _card_str(c)
            b_parts.append(f"({s})" if future else s)
        board_display = " ".join(b_parts) if b_parts else "(preflop)"

        put(row, 0, f"  Board: {board_display}", BOLD)
        put(row, 42, f"Pot: {pot / OSP_BIG_BLIND:.2f} BB", BOLD)
        row += 1
        hline(row)
        row += 1

        # ── Bot row ──────────────────────────────────────────────────────
        if bc:
            bc_str = (
                " ".join(_card_str(c) for c in bc) if self.cheat_mode else "[??] [??]"
            )
        else:
            bc_str = "..."

        put(
            row,
            0,
            f"  BOT ({b_pos})  Stack: {b_stack / OSP_BIG_BLIND:.2f} BB  Cards: {bc_str}",
            C_CYN | BOLD,
        )
        row += 1

        # Inference + InfoSet rows (cheat mode only)
        if self.cheat_mode:
            if self.last_bot_probs is not None:
                labels = [
                    "F/Chk",
                    "Call ",
                    "B33% ",
                    "B50% ",
                    "B75% ",
                    "B100%",
                    "B150%",
                    "B200%",
                    "B300%",
                    "A-in ",
                ]
                probs = self.last_bot_probs
                chosen = self.last_bot_abstract
                """
                COL = 8
                label_line = "".join(
                    f"{'►' if i == chosen else ' '}{lb:<{COL - 1}}"
                    for i, lb in enumerate(labels)
                )
                prob_line = "".join(f"{p * 100:{COL}.0f}%" for p in probs)
                put(row, 0, "  └── " + label_line, C_CYN)
                row += 1
                put(row, 0, prob_line, C_CYN)
                """
                prefix = "   └── "
                COL_WIDTH = 9
                inner = COL_WIDTH - 2

                put(row, 0, prefix, C_CYN)
                for i, (label, p) in enumerate(zip(labels, probs)):
                    col = len(prefix) + i * COL_WIDTH
                    illegal = i not in self.last_bot_legal
                    attr = (
                        C_CYN | BOLD
                        if i == chosen
                        else C_CYN | DIM
                        if illegal
                        else C_CYN
                    )
                    val_str = f"{p * 100:.0f}%"
                    if i == chosen:
                        put(row, col, f"[{label:^{inner}}]", attr)
                        put(row + 1, col, f"[{val_str:^{inner}}]", attr)
                    else:
                        put(row, col, f" {label:^{inner}} ", attr)
                        put(row + 1, col, f" {val_str:^{inner}} ", attr)
                row += 2
            else:
                put(row, 0, "  └── (no bot action yet)", C_CYN | DIM)
                row += 1
            row += 1  # blank line before InfoSet
            for line in self._infoset_lines():
                put(row, 0, line, C_CYN | DIM)
                row += 1

        hline(row)
        row += 1

        # ── Human row ───────────────────────────────────────────────────
        hc_str = " ".join(_card_str(c) for c in hc) if hc else "..."
        put(
            row,
            0,
            f"  YOU ({h_pos})  Stack: {h_stack / OSP_BIG_BLIND:.2f} BB  Cards: {hc_str}",
            BOLD,
        )
        row += 1
        hline(row)
        row += 1

        # ── Action log ──────────────────────────────────────────────────
        FOOTER_ROWS = 4  # hline + control + hline + global-keys
        log_area = max(0, h - row - FOOTER_ROWS)
        visible = self.action_log[-log_area:] if log_area else []
        for text, is_marker in visible:
            attr = C_YLW | BOLD if is_marker else curses.A_NORMAL
            put(row, 0, f"  {text}", attr)
            row += 1

        # ── Footer ──────────────────────────────────────────────────────
        footer_row = h - FOOTER_ROWS
        row = max(row, footer_row)
        hline(row)
        row += 1

        if self.hand_over:
            won = "+" in self.result_msg or "WON" in self.result_msg
            put(row, 0, f"  {self.result_msg}", (C_GRN if won else C_RED) | BOLD)

        elif (
            self.state is not None
            and not self.state.is_terminal()
            and self.state.current_player() == self.human_player
        ):
            legal = self.state.legal_actions()
            to_call = max(self.invested) - self.invested[self.human_player]
            raises = sorted(a for a in legal if a > 1)
            hp_invested = self.invested[self.human_player]

            if self.raise_input is not None:
                resolved = self._resolve_raise()
                preview = (
                    f"  +{(resolved - hp_invested) / OSP_BIG_BLIND:.2f} BB"
                    if resolved is not None
                    else "  (invalid)"
                )
                put(
                    row,
                    0,
                    f"  Raise: {self.raise_input}_{preview}   [Enter] confirm   [Esc] cancel",
                    BOLD,
                )
            else:
                parts: list[str] = []
                if 0 in legal:
                    parts.append("[F]old")
                if 1 in legal:
                    parts.append(
                        f"[C]all +{to_call / OSP_BIG_BLIND:.2f} BB"
                        if to_call > 0
                        else "[C]heck"
                    )
                if raises:
                    parts.append("[R]aise")
                    allin_added = raises[-1] - hp_invested
                    parts.append(f"[A]ll-in +{allin_added / OSP_BIG_BLIND:.2f} BB")
                put(row, 0, "  → " + "   ".join(parts), BOLD)

        else:
            put(row, 0, "  (bot acting...)", DIM)

        row += 1
        hline(row)
        row += 1
        put(row, 0, "  [X] cheat   [N] next hand   [Q] quit", DIM)

        stdscr.refresh()

    # ------------------------------------------------------------------
    # Main curses loop
    # ------------------------------------------------------------------

    def run(self, stdscr):
        curses.curs_set(0)
        stdscr.nodelay(True)
        curses.start_color()
        curses.use_default_colors()
        curses.init_pair(1, curses.COLOR_WHITE, -1)  # default
        curses.init_pair(2, curses.COLOR_RED, -1)  # red suits / losses
        curses.init_pair(3, curses.COLOR_GREEN, -1)  # session / wins
        curses.init_pair(4, curses.COLOR_CYAN, -1)  # bot
        curses.init_pair(5, curses.COLOR_YELLOW, -1)  # street markers

        self._new_hand()

        while True:
            h, w = stdscr.getmaxyx()
            self.draw(stdscr, h, w)

            key = stdscr.getch()

            if key in (ord("q"), ord("Q")):
                break

            elif key in (ord("x"), ord("X")):
                self.cheat_mode = not self.cheat_mode

            elif key in (ord("n"), ord("N")) and self.hand_over:
                self._new_hand()

            elif (
                not self.hand_over
                and self.state is not None
                and not self.state.is_terminal()
                and self.state.current_player() == self.human_player
            ):
                legal = self.state.legal_actions()
                raises = sorted(a for a in legal if a > 1)

                if self.raise_input is not None:
                    if key in (10, 13, curses.KEY_ENTER):  # confirm
                        action = self._resolve_raise()
                        self.raise_input = None
                        if action is not None:
                            self._apply_logged(self.human_player, action)
                            self._advance()
                    elif key == 27:  # Esc — cancel
                        self.raise_input = None
                    elif key in (curses.KEY_BACKSPACE, 127, 8):
                        self.raise_input = self.raise_input[:-1]
                    elif 48 <= key <= 57:  # digit
                        self.raise_input += chr(key)
                else:
                    if key in (ord("f"), ord("F")) and 0 in legal:
                        self._apply_logged(self.human_player, 0)
                        self._advance()
                    elif key in (ord("c"), ord("C")) and 1 in legal:
                        self._apply_logged(self.human_player, 1)
                        self._advance()
                    elif key in (ord("r"), ord("R")) and raises:
                        self.raise_input = ""
                    elif key in (ord("a"), ord("A")) and raises:
                        self._apply_logged(self.human_player, raises[-1])
                        self._advance()

            time.sleep(0.10)


# ---------------------------------------------------------------------------
# Deal-order test
# ---------------------------------------------------------------------------


def _test_deal_order():
    """
    Verify which player receives each successive chance-node card.

    Applies cards 0..3 one at a time and checks whose information-state
    string changes after each deal, revealing whether OpenSpiel uses
    grouped order (p0c0, p0c1, p1c0, p1c1) or interleaved order
    (p0c0, p1c0, p0c1, p1c1).
    """
    game = pyspiel.load_game(GAME_STRING)
    state = game.new_initial_state()

    prev_info = [state.information_state_string(p) for p in range(2)]
    deal_step = 0

    print("Deal-order probe — applying cards 0,1,2,3 to successive chance nodes:\n")
    while deal_step < 4:
        assert state.is_chance_node(), f"Expected chance node at step {deal_step}"
        card = deal_step  # use card indices 0,1,2,3 as sentinels
        state.apply_action(card)
        deal_step += 1

        curr_info = [state.information_state_string(p) for p in range(2)]
        changed = [curr_info[p] != prev_info[p] for p in range(2)]
        prev_info = curr_info

        recipients = [f"P{p}" for p in range(2) if changed[p]]
        print(
            f"  card slot {deal_step}: visible to {', '.join(recipients) if recipients else '(nobody yet)'}"
        )

    print()
    # Summarise the assumed layout used by this codebase
    print("Codebase assumption:  cards[0,1]=P0 hole, cards[2,3]=P1 hole  (grouped)")
    print("If the output above shows P0,P0,P1,P1 the assumption is correct.")
    print("If it shows P0,P1,P0,P1 the deal is interleaved — cards must be reordered.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Play NLHE against TiltStack DeepCFR in the terminal.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--net",
        default=None,
        help="Path to strategy network checkpoint (policy*.pt)",
    )
    parser.add_argument(
        "--clusters",
        default="clusters",
        help="Path to clusters/ directory (required for bucket lookups)",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Torch device string: cpu or cuda  (default: cpu)",
    )
    parser.add_argument(
        "--test-deal",
        action="store_true",
        help="Run the deal-order probe and exit (no --net required).",
    )
    args = parser.parse_args()

    if args.test_deal:
        _test_deal_order()
        return

    if args.net is None:
        parser.error("--net is required unless --test-deal is set")

    if not os.path.isdir(args.clusters):
        parser.error(
            f"clusters directory not found: {args.clusters!r}  "
            f"(pass --clusters <path> or run from the src/ directory)"
        )
    deepcfr.load_tables(args.clusters)

    game = PokerLive(args.net, args.clusters, args.device)
    curses.wrapper(game.run)


if __name__ == "__main__":
    main()
