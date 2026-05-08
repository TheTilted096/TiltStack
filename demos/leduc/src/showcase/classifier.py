"""
TiltStack — Opponent Style Classifier (Layer 2)
================================================
Trained on ACPC 2011 2-player No-Limit Hold'em match data (150k hands, 18 bots).

Classifies a human judge's play style from their casino-table action history:
  0  Tight/Balanced  — selective, value-bet focused
  1  Loose-Passive   — calls wide, rarely raises (calling station)
  2  Aggressive      — bets/raises frequently regardless of hand strength

Usage
-----
    from classifier import OpponentClassifier, ActionTracker
    clf = OpponentClassifier.load_or_train(hands_csv, cache_pkl)
    tracker = ActionTracker()

    # per-hand lifecycle:
    tracker.start_hand()
    tracker.record_action("Raise", raises_before=0, round_id=1)
    tracker.end_hand(went_to_showdown=True)

    result = clf.predict(tracker)
    # → ("Aggressive", 0.87, array([0.04, 0.09, 0.87]))
"""

from __future__ import annotations

import os
import csv
import pickle
from pathlib import Path
from typing import Optional

import numpy as np

# ── Labels & display metadata ──────────────────────────────────────────────────

LABEL_NAMES  = ["Tight/Balanced", "Loose-Passive", "Aggressive"]
LABEL_COLORS = ["#4ecdc4",        "#f5d46d",        "#ff6b6b"]
LABEL_ICONS  = ["🎯",             "🐢",              "🔥"]

# Maps each opposing profile in app.py to the label that best exploits it
EXPLOIT_MAP: dict[str, str] = {
    "Tight/Balanced": "Tight",        # mirror tight → GTO is fine
    "Loose-Passive":  "Loose-Passive",
    "Aggressive":     "Aggressive",
}

# ── Per-player style labels derived from players.csv stats ────────────────────
#
#   Loose-Passive : vpip ≥ 75  AND  aggression ≤ 25  AND  pfr ≤ 30
#   Aggressive    : aggression ≥ 45  OR  (pfr ≥ 40 and vpip < 70)
#   Tight/Balanced: everything else

_PLAYER_LABELS: dict[str, int] = {
    # 0 — Tight / Balanced
    "Hyperborean-2011-2p-nolimit-iro": 0,
    "Hyperborean-2011-2p-nolimit-tbr": 0,
    "hyperborean":   0,
    "little_rock":   0,
    "sartre":        0,
    "tartanian5":    0,
    # 1 — Loose-Passive
    "Lucky7":        1,
    "POMPEIA":       1,
    "Rembrant":      1,
    "SartreNL":      1,
    "uni_mb_poker":  1,
    # 2 — Aggressive
    "azure_sky":     2,
    "dcubot":        2,
    "hugh":          2,
    "lucky7_12":     2,
    "neo_poker_lab": 2,
    "player_kappa_nl": 2,
    "spewy_louie":   2,
}

# ── Feature extraction ─────────────────────────────────────────────────────────

_FEATURES = [
    "aggression_pct",   # ratio: raises / (raises + calls + folds)  [0-1]
    "fold_rate",        # ratio: folds / total_voluntary            [0-1]
    "call_rate",        # ratio: calls / total_voluntary            [0-1]
    "raise_rate",       # ratio: raises / total_voluntary           [0-1]
    "pfr",              # 0/1: raised in first street
]
# Using ratios makes features comparable between NLHE (10+ actions/hand)
# and Leduc (4-6 actions/hand).


def _row_features(row: dict, side: str) -> Optional[list[float]]:
    """5-dim ratio-based feature vector for one player-side of an ACPC hand row.

    All features are in [0, 1] so the distributions are comparable with
    the shorter Leduc hands recorded by ActionTracker.
    """
    try:
        raises = float(row[f"raises_{side}"])
        calls  = float(row[f"calls_{side}"])
        folds  = float(row[f"folds_{side}"])
        pfr    = float(row[f"pfr_{side}"])
        agg    = float(row[f"aggression_pct_{side}"]) / 100.0

        total = raises + calls + folds
        if total < 0.5:
            return None   # skip hands where this player made no voluntary actions

        return [
            agg,                    # aggression_pct already a fraction
            folds  / total,         # fold_rate
            calls  / total,         # call_rate
            raises / total,         # raise_rate
            pfr,                    # preflop raise flag
        ]
    except (KeyError, ValueError):
        return None


def _build_training_data(hands_csv: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Parse hands.csv → (X, y) arrays ready for sklearn."""
    X: list[list[float]] = []
    y: list[int] = []

    with open(hands_csv, newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            for side, name_col in [("p1", "player1"), ("p2", "player2")]:
                player = row.get(name_col, "").strip()
                label  = _PLAYER_LABELS.get(player)
                if label is None:
                    continue
                feat = _row_features(row, side)
                if feat is None:
                    continue
                X.append(feat)
                y.append(label)

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)


# ── Training ───────────────────────────────────────────────────────────────────

def train_classifier(hands_csv: str | Path):
    """Train a GradientBoostingClassifier pipeline on ACPC data."""
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline

    X, y = _build_training_data(hands_csv)
    counts = np.bincount(y, minlength=3)
    print(f"[TiltStack] Training classifier on {len(X):,} samples "
          f"(TB={counts[0]:,}  LP={counts[1]:,}  Agg={counts[2]:,})…")

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", GradientBoostingClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.8,
            random_state=42,
        )),
    ])
    pipe.fit(X, y)
    acc = pipe.score(X, y)
    print(f"[TiltStack] Classifier ready — train accuracy {acc:.3f}")
    return pipe


# ── Heuristic fallback (no training data available) ───────────────────────────

def _heuristic_predict(fv: np.ndarray) -> tuple[str, float, np.ndarray]:
    """Rule-based fallback classifier calibrated for ratio-based features.

    fv shape: (1, 5) — [aggression_pct, fold_rate, call_rate, raise_rate, pfr]
    All values are in [0, 1].
    """
    agg, fold_rate, call_rate, raise_rate, pfr = fv[0]

    if agg > 0.40 or raise_rate > 0.35:
        # Aggressive: high fraction of actions are raises
        conf  = min(0.55 + agg * 0.35, 0.90)
        probs = np.array([1 - conf - 0.05, 0.05, conf])
    elif agg < 0.20 and call_rate > 0.40:
        # Loose-Passive: calls dominate, few raises
        conf  = min(0.55 + (1 - agg) * 0.20 + call_rate * 0.15, 0.88)
        probs = np.array([1 - conf - 0.05, conf, 0.05])
    elif fold_rate > 0.40 and raise_rate < 0.20:
        # Tight: folds often, rarely raises
        probs = np.array([0.72, 0.18, 0.10])
    else:
        # Balanced: unclear signal
        probs = np.array([0.55, 0.25, 0.20])

    probs = np.clip(probs, 0.01, 1.0)
    probs /= probs.sum()
    idx = int(np.argmax(probs))
    return LABEL_NAMES[idx], float(probs[idx]), probs


# ── ActionTracker ──────────────────────────────────────────────────────────────

class ActionTracker:
    """Accumulates per-hand action stats as a judge plays the casino table."""

    def __init__(self) -> None:
        self.hands: list[list[float]] = []
        self._cur: Optional[dict] = None

    # ── Hand lifecycle ─────────────────────────────────────────────────────────

    def start_hand(self) -> None:
        """Call at the start of each new casino hand."""
        self._cur = {
            "raises": 0, "calls": 0, "folds": 0,
            "vpip": 0, "pfr": 0,
            "aggression_pct": 0.0,
            "showdown": 0,
            "_round1_done": False,
        }

    def record_action(
        self,
        action_label: str,   # "Fold" | "Check" | "Call" | "Raise"
        raises_before: int,  # raises already in the current street
        round_id: int,       # 1 = first betting round, 2 = second
    ) -> None:
        """Record one of the judge's in-hand decisions."""
        if self._cur is None:
            self.start_hand()
        c = self._cur

        if action_label == "Raise":
            c["raises"] += 1
            c["vpip"]    = 1
            if round_id == 1 and not c["_round1_done"]:
                c["pfr"] = 1
        elif action_label == "Call":
            c["calls"] += 1
            c["vpip"]   = 1
        elif action_label == "Fold":
            c["folds"]  = 1
        # "Check" — passive, no counters change

        if not c["_round1_done"] and round_id == 2:
            c["_round1_done"] = True

    def end_hand(self, went_to_showdown: bool) -> None:
        """Call when the casino hand resolves.

        Builds the same 5-dim ratio feature vector as _row_features() so
        inference is in-distribution with the ACPC-trained model.
        """
        if self._cur is None:
            return
        c = self._cur

        raises = float(c["raises"])
        calls  = float(c["calls"])
        folds  = float(c["folds"])
        pfr    = float(c["pfr"])
        total  = raises + calls + folds

        if total < 0.5:
            # Judge made no voluntary actions this hand — skip
            self._cur = None
            return

        agg        = raises / total
        fold_rate  = folds  / total
        call_rate  = calls  / total
        raise_rate = raises / total

        self.hands.append([
            agg,
            fold_rate,
            call_rate,
            raise_rate,
            pfr,
        ])
        self._cur = None

    # ── Inference helpers ──────────────────────────────────────────────────────

    @property
    def n_hands(self) -> int:
        return len(self.hands)

    def feature_vector(self) -> Optional[np.ndarray]:
        """Mean feature vector (shape 1×7) over completed hands, or None if < 3."""
        if len(self.hands) < 3:
            return None
        arr = np.array(self.hands, dtype=np.float32)
        return arr.mean(axis=0).reshape(1, -1)

    def reset(self) -> None:
        self.hands = []
        self._cur  = None


# ── OpponentClassifier ─────────────────────────────────────────────────────────

class OpponentClassifier:
    """Thin wrapper around a fitted sklearn pipeline (or the heuristic fallback)."""

    def __init__(self, model=None) -> None:
        self._model = model      # None → use heuristic

    # ── Prediction ─────────────────────────────────────────────────────────────

    def predict(
        self, tracker: ActionTracker
    ) -> Optional[tuple[str, float, np.ndarray]]:
        """
        Returns (label_name, confidence, probs) after ≥ 3 completed hands,
        or None if not enough data yet.

        Blends the heuristic (calibrated for Leduc action distributions) with
        the ACPC-trained ML model (60/40 weight).  The heuristic anchors
        out-of-distribution feature vectors that the NLHE-trained model
        hasn't seen; the ML model adds signal from learned ACPC patterns.
        """
        fv = tracker.feature_vector()
        if fv is None:
            return None

        # Heuristic — always stable, calibrated for Leduc feature ranges
        _, _, h_probs = _heuristic_predict(fv)

        if self._model is None:
            probs = h_probs
        else:
            ml_probs = self._model.predict_proba(fv)[0]
            # 60% heuristic, 40% ML — heuristic dominates OOD regions
            probs = 0.60 * h_probs + 0.40 * ml_probs
            probs /= probs.sum()

        idx = int(np.argmax(probs))
        return LABEL_NAMES[idx], float(probs[idx]), probs

    # ── Loading / training ─────────────────────────────────────────────────────

    @classmethod
    def load_or_train(
        cls,
        hands_csv: Optional[str | Path] = None,
        cache_pkl: Optional[str | Path] = None,
    ) -> "OpponentClassifier":
        """
        Priority order:
          1. Load cached pickle if present.
          2. Train on hands_csv if provided and pickle not found.
          3. Fall back to heuristic classifier.
        """
        # 1 — cached pickle
        if cache_pkl and Path(cache_pkl).exists():
            with open(cache_pkl, "rb") as fh:
                model = pickle.load(fh)
            print(f"[TiltStack] Loaded classifier from {cache_pkl}")
            return cls(model)

        # 2 — train from ACPC data
        if hands_csv and Path(hands_csv).exists():
            try:
                model = train_classifier(hands_csv)
                if cache_pkl:
                    Path(cache_pkl).parent.mkdir(parents=True, exist_ok=True)
                    with open(cache_pkl, "wb") as fh:
                        pickle.dump(model, fh)
                    print(f"[TiltStack] Saved classifier to {cache_pkl}")
                return cls(model)
            except Exception as exc:
                print(f"[TiltStack] Training failed ({exc}), using heuristic fallback")

        # 3 — heuristic fallback (always works, no deps beyond numpy)
        print("[TiltStack] No training data found — using rule-based classifier")
        return cls(model=None)


# ── Convenience loader for app.py ─────────────────────────────────────────────

def _resolve_paths() -> tuple[Optional[Path], Optional[Path]]:
    """Locate hands.csv and cache.pkl relative to this file."""
    here      = Path(__file__).parent
    data_dir  = here.parent.parent / "data"          # demos/leduc/data/
    hands_csv = data_dir / "hands.csv"
    cache_pkl = data_dir / "classifier_model.pkl"
    return (
        hands_csv if hands_csv.exists() else None,
        cache_pkl,
    )


def load_classifier() -> OpponentClassifier:
    """One-call loader used by app.py."""
    hands_csv, cache_pkl = _resolve_paths()
    return OpponentClassifier.load_or_train(hands_csv, cache_pkl)
