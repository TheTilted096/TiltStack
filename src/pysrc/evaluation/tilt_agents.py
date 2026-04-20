"""
tilt_agents.py — OpenSpiel Bot wrappers for TiltStack evaluation.

Each agent owns a deepcfr.CFRGame instance.  The CFRGame handles all feature
extraction — card bitmasks, EHS lookup, cluster bucket lookup, stack/pot
normalisation, and betting history — so the agents contain no parsing logic.

Workflow inside step():
  1. _sync_game(state)  — walk state.history(), separate card deals from bets,
                          call game.begin_with_cards() then replay every bet
  2. game.get_info()    — returns the 168-byte InfoSet buffer directly
  3. decode_batch()     — convert to (x_cont, buckets) tensors
  4. network forward    — get action logits
  5. _select_action()   — mask to CFRGame-legal abstract actions, pick one,
                          then map back to an OpenSpiel action integer

Prerequisites:
  • deepcfr.load_tables(clusters_dir) called once before constructing agents
  • deepcfr module compiled (cd src && pip install -e . --no-build-isolation)

Two agents are provided:

  TiltStack_DeepCFR   — GTO strategy network (policy*.pt).
                        Samples from softmax over legal abstract actions.

  Anti_TiltStack_NBR  — Neural best-response (br_adv{0,1}_*.pt).
                        Greedy argmax over legal abstract actions.

OpenSpiel / CFRGame conventions:
  Player 0  = small blind = button in HUNL
  Player 1  = big blind
  STARTING_STACK = 40,000 milli-chips = 20 BB   (CFRGame training constant)
  OSP_STACK      = 2,000  chips                 (recommended OpenSpiel config)
  OSP_MC_SCALE   = STARTING_STACK / OSP_STACK = 20  (chips → milli-chips)
"""

import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
import pyspiel

_EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_EVAL_DIR, "..", "deepcfr"))

import deepcfr
from network_training import DeepCFRNet, decode_batch, NUM_ACTIONS

# ---------------------------------------------------------------------------
# Action index constants (must match CFRTypes.h Action enum)
# ---------------------------------------------------------------------------

_CHECK = 0  # also FOLD when to_call > 0
_CALL = 1
_BET50 = 2
_BET100 = 3
_ALLIN = 4

# ---------------------------------------------------------------------------
# Scale factor: 1 OpenSpiel chip == OSP_MC_SCALE CFRGame milli-chips.
# Derived from STARTING_STACK=40000 / OSP_STACK=2000.
# Update if the OpenSpiel game string uses different stack sizes.
# ---------------------------------------------------------------------------

STARTING_STACK = 40_000  # milli-chips (CFRGame constant)
OSP_STACK = 2_000  # chips (must match stack= in the game string)
OSP_MC_SCALE = STARTING_STACK // OSP_STACK  # = 20


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------


def load_net_auto(path: str, device) -> DeepCFRNet:
    """Load a DeepCFRNet checkpoint."""
    ckpt = torch.load(path, map_location=device, weights_only=True)
    sd = ckpt["net"]
    if any(k.startswith("_orig_mod.") for k in sd):
        sd = {k.removeprefix("_orig_mod."): v for k, v in sd.items()}
    net = DeepCFRNet()
    net.load_state_dict(sd)
    return net.to(device).eval()


# ---------------------------------------------------------------------------
# History helpers
# ---------------------------------------------------------------------------


def _split_history(osp_game, state):
    """
    Walk state.history() from a fresh initial state to separate card-deal
    actions (chance nodes) from player betting actions.

    OpenSpiel universal_poker action encoding:
      0        — fold
      1        — call / check
      K … M   — raise; the action integer is the TOTAL chips invested by the
                 acting player this hand (cumulative, not the amount added)

    Returns
    -------
    deal_cards : list[int]             — card indices in deal order
    bet_seq    : list[(player, action)] — player who acted and OpenSpiel action
    """
    tmp = osp_game.new_initial_state()
    deal_cards = []
    bet_seq = []

    for action in state.history():
        if tmp.is_chance_node():
            deal_cards.append(action)
        else:
            bet_seq.append((tmp.current_player(), action))
        tmp.apply_action(action)

    return deal_cards, bet_seq


def _pad_to_9(deal_cards):
    """
    Extend deal_cards to exactly 9 entries using the first unused card indices.

    hand_index_all() requires all 7 cards (per player) even for partial boards,
    but the EHS / bucket values for unreached streets are gated by currentRound
    inside CFRGame::getInfo() and never exposed to the network.
    """
    used = set(deal_cards[:9])
    result = list(deal_cards[:9])
    for c in range(52):
        if len(result) >= 9:
            break
        if c not in used:
            result.append(c)
            used.add(c)
    return result


# ---------------------------------------------------------------------------
# CFRGame action helpers
# ---------------------------------------------------------------------------


def _cfr_bet_amounts(game):
    """
    Return the milli-chip cost of each abstract raise action in the current
    CFRGame state, mirroring CFRGame::makeMove / generateActions.

    Returns a dict {action_int: milli_chips} for BET50, BET100, ALLIN.
    CHECK and CALL are not raises so they are omitted.
    """
    pot = game.pot
    to_call = game.to_call
    stm = game.stm
    stacks = game.stacks

    return {
        _BET50: to_call + (pot + to_call) // 2,
        _BET100: to_call + (pot + to_call),
        _ALLIN: min(stacks[stm], to_call + stacks[1 - stm]),
    }


def _abstract_to_osp(abstract_action, legal_osp, game):
    """
    Map a CFRGame abstract action to an OpenSpiel action integer.

    For fold/call the mapping is exact.  For raises, the CFRGame milli-chip
    bet amount is converted to chips and the nearest legal OpenSpiel action
    (which is itself the chip amount) is returned.
    """
    fold_action = 0 if 0 in legal_osp else None
    call_action = 1 if 1 in legal_osp else (legal_osp[0] if legal_osp else 0)

    if abstract_action == _CHECK:
        # to_call > 0 means we face a live bet — CHECK is a fold in CFRGame.
        return (
            fold_action
            if (game.to_call > 0 and fold_action is not None)
            else call_action
        )

    if abstract_action == _CALL:
        return call_action

    # Raise: convert the CFRGame milli-chip bet amount to total chips invested,
    # then find the closest legal OpenSpiel action (which is total chips invested).
    raise_osp = sorted(a for a in legal_osp if a > 1)
    if not raise_osp:
        return call_action

    amount_added_mc = _cfr_bet_amounts(game)[abstract_action]
    invested_mc = STARTING_STACK - game.stacks[game.stm]
    target_total_chips = (invested_mc + amount_added_mc) / OSP_MC_SCALE
    return min(raise_osp, key=lambda a: abs(a - target_total_chips))


# ---------------------------------------------------------------------------
# Shared base logic (mixin — not a standalone pyspiel.Bot)
# ---------------------------------------------------------------------------


class _CFRBotMixin:
    """
    Common setup and step logic shared by TiltStack_DeepCFR and Anti_TiltStack_NBR.

    Subclasses must set:
        self.game        : deepcfr.CFRGame
        self.osp_game    : pyspiel.Game
        self.player_id   : int
        self.device      : torch.device
    """

    def _sync_game(self, state):
        """
        Bring self.game in sync with the OpenSpiel state:
          1. Extract dealt cards from the chance-node history.
          2. begin_with_cards() — initialises CFRGame with those cards.
          3. Replay every subsequent betting action via make_move().
        """
        deal_cards, bet_seq = _split_history(self.osp_game, state)

        if len(deal_cards) < 4:
            return False  # hole cards not yet dealt

        cards9 = _pad_to_9(deal_cards)
        self.game.begin_with_cards(
            STARTING_STACK,
            STARTING_STACK,
            state.current_player() == 0,  # hero = True for player 0 (small blind)
            cards9,
        )

        for player, osp_action in bet_seq:
            if osp_action == 0:  # fold
                self.game.make_move(_CHECK)
            elif osp_action == 1:  # call / check
                legal = self.game.generate_actions()
                self.game.make_move(_CALL if _CALL in legal else _CHECK)
            else:  # raise — action is total chips invested
                stm = self.game.stm
                invested_mc = STARTING_STACK - self.game.stacks[stm]
                amount_added_mc = osp_action * OSP_MC_SCALE - invested_mc
                self.game.make_bet(amount_added_mc)

        return True

    def _forward(self, model):
        """
        Run one forward pass and return masked logits over all NUM_ACTIONS slots.

        Illegal actions (those not returned by game.generate_actions()) receive
        logit −1e9, which drives their softmax probability to zero and ensures
        argmax never selects them.  The caller decides how to use the logits
        (softmax+sample for TiltStack_DeepCFR, argmax for Anti_TiltStack_NBR).

        Returns
        -------
        masked_logits : np.ndarray, shape (NUM_ACTIONS,), dtype float32
        legal_abstract : list[int]
        """
        raw = self.game.get_info()  # (1, 168) uint8
        x_cont, buckets = decode_batch(raw)
        with torch.no_grad():
            logits = model(x_cont.to(self.device), buckets.to(self.device))

        legal_abstract = self.game.generate_actions()
        mask = torch.full((NUM_ACTIONS,), -1e9, device=self.device)
        for a in legal_abstract:
            mask[a] = 0.0

        return (logits[0] + mask).cpu().numpy(), legal_abstract


# ---------------------------------------------------------------------------
# Agents
# ---------------------------------------------------------------------------


class TiltStack_DeepCFR(_CFRBotMixin, pyspiel.Bot):
    """
    GTO-approximating agent backed by the trained strategy network (policy*.pt).

    Samples actions from the softmax probability distribution over legal
    CFRGame abstract actions.  A single instance plays correctly from either
    seat — the current seat is read from state.current_player() at step time.

    Parameters
    ----------
    model    : DeepCFRNet  — pre-loaded strategy network (.eval() already called)
    osp_game : pyspiel.Game — the same Game object used by match_runner
    device   : str         — 'cpu' or 'cuda'
    """

    def __init__(self, model: DeepCFRNet, osp_game, device: str = "cpu"):
        pyspiel.Bot.__init__(self)
        self.model = model.to(device).eval()
        self.osp_game = osp_game
        self.device = torch.device(device)
        self.game = deepcfr.CFRGame()

    def step(self, state) -> int:
        if not self._sync_game(state):
            legal = state.legal_actions()
            return 1 if 1 in legal else legal[0]

        masked_logits, _ = self._forward(self.model)
        probs = F.softmax(torch.from_numpy(masked_logits), dim=0).numpy()
        our_action = int(np.random.choice(NUM_ACTIONS, p=probs))
        return _abstract_to_osp(our_action, state.legal_actions(), self.game)


class Anti_TiltStack_NBR(_CFRBotMixin, pyspiel.Bot):
    """
    Neural best-response agent backed by per-player advantage networks
    (br_adv{0,1}_*.pt produced by NLHE_BestResponse.py).

    Selects the highest-probability action (greedy argmax) since it represents
    the approximate best response against TiltStack_DeepCFR.  A single instance
    plays correctly from either seat — the current seat is read from
    state.current_player() at step time to select the matching model.

    Parameters
    ----------
    model_p0 : DeepCFRNet  — best-response advantage network for player 0
    model_p1 : DeepCFRNet  — best-response advantage network for player 1
    osp_game : pyspiel.Game
    device   : str
    """

    def __init__(
        self, model_p0: DeepCFRNet, model_p1: DeepCFRNet, osp_game, device: str = "cpu"
    ):
        pyspiel.Bot.__init__(self)
        self.model_p0 = model_p0.to(device).eval()
        self.model_p1 = model_p1.to(device).eval()
        self.osp_game = osp_game
        self.device = torch.device(device)
        self.game = deepcfr.CFRGame()

    def step(self, state) -> int:
        if not self._sync_game(state):
            legal = state.legal_actions()
            return 1 if 1 in legal else legal[0]

        model = self.model_p0 if state.current_player() == 0 else self.model_p1
        masked_logits, _ = self._forward(model)
        our_action = int(np.argmax(masked_logits))
        return _abstract_to_osp(our_action, state.legal_actions(), self.game)
