"""
TiltStack — Interactive Demo App
Northwestern IEEE Technical Program | April 4, 2026

Run with:  streamlit run src/showcase/app.py
"""
import sys
import os
import time
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_here, '..', 'pysrc'))  # leducsolver, Leduc
sys.path.insert(0, _here)                                # demo_vs_tight
from Leduc import Leduc
from leducsolver import Action, BestResponse as BRSolver
from demo_vs_tight import (
    _stm, _raises, _bet_round, _private_card, _legal_moves,
    _ends_hand, _payout, _next_stm, _next_hash,
    gto_action, simulate_hand,
    build_tight_strategy_vector, compute_exploit_strategy,
    JACK, QUEEN, KING,
)
from classifier import (
    OpponentClassifier, ActionTracker,
    LABEL_NAMES, LABEL_COLORS, LABEL_ICONS,
    load_classifier,
)

# ─────────────────────────────────────────────────────────────
# Opponent profiles
# ─────────────────────────────────────────────────────────────

def tight_action(h, rng):
    """Tight: folds Jack to bets, rarely bets Queen, always bets King."""
    moves = _legal_moves(h)
    card  = _private_card(h)
    r     = _raises(h)
    if card == JACK:
        return Action.CHECK
    if card == QUEEN:
        if r == 0 and Action.RAISE in moves and rng.random() < 0.25:
            return Action.RAISE
        return Action.BET if (r > 0 and Action.BET in moves) else Action.CHECK
    if Action.RAISE in moves: return Action.RAISE
    return Action.BET if Action.BET in moves else Action.CHECK

def loose_action(h, rng):
    """Loose-passive: calls almost everything, rarely raises."""
    moves = _legal_moves(h)
    r     = _raises(h)
    if r == 0:
        return Action.RAISE if Action.RAISE in moves and rng.random() < 0.3 else Action.CHECK
    return Action.BET if Action.BET in moves else Action.CHECK  # always call

def aggressive_action(h, rng):
    """Aggressive: bets and raises frequently regardless of hand."""
    moves = _legal_moves(h)
    r     = _raises(h)
    if Action.RAISE in moves and rng.random() < 0.7:
        return Action.RAISE
    if Action.BET in moves and rng.random() < 0.6:
        return Action.BET
    return Action.CHECK

def _br_action(br_strategy, h, rng):
    """Sample from BRSolver-computed optimal counter-strategy."""
    moves = _legal_moves(h)
    strat = br_strategy[h]
    probs = [float(strat[m.value]) for m in moves]
    total = sum(probs)
    if total < 1e-9:
        return rng.choice(moves)
    probs = [p / total for p in probs]
    return rng.choice(moves, p=probs)


OPPONENT_FNS = {
    "Tight":      tight_action,
    "Loose-Passive": loose_action,
    "Aggressive": aggressive_action,
}

OPPONENT_DESC = {
    "Tight":         "Folds weak hands, only bets Kings. Classic over-folder.",
    "Loose-Passive": "Calls everything, rarely raises. Classic calling station.",
    "Aggressive":    "Bets and raises constantly regardless of hand strength.",
}

CARD_LABELS = {
    JACK: "J",
    QUEEN: "Q",
    KING: "K",
}

def build_opponent_vector(profile: str):
    if profile == "Tight":
        return build_tight_strategy_vector()
    # For other profiles, approximate with a random strategy vector
    # (BRSolver still gives optimal counter)
    strat = []
    rng_b = np.random.default_rng(0)
    for h in range(528):
        moves = _legal_moves(h)
        s = [0.0, 0.0, 0.0]
        if profile == "Loose-Passive":
            r = _raises(h)
            if r == 0 and Action.RAISE in moves:
                s[Action.CHECK.value] = 0.7
                s[Action.RAISE.value] = 0.3
            elif Action.BET in moves:
                s[Action.BET.value] = 1.0
            else:
                s[Action.CHECK.value] = 1.0
        else:  # Aggressive — matches aggressive_action():
            # P(RAISE)=0.70, P(BET)=0.30*0.60=0.18, P(CHECK)=0.30*0.40=0.12
            if Action.RAISE in moves:
                s[Action.RAISE.value] = 0.70
                s[Action.BET.value]   = 0.18
                s[Action.CHECK.value] = 0.12
            elif Action.BET in moves:   # max raises (r=3): {CHECK, BET}
                s[Action.BET.value]   = 0.6
                s[Action.CHECK.value] = 0.4
            else:
                s[Action.CHECK.value] = 1.0
        strat.append(s)
    return strat

# ─────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="TiltStack Demo",
    page_icon="🃏",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
/* Dark background */
[data-testid="stAppViewContainer"] { background-color: #0f1117; }
[data-testid="stSidebar"]          { background-color: #1a1d27; }
section.main > div                 { padding-top: 1.5rem; }

/* Metric cards */
[data-testid="metric-container"] {
    background: linear-gradient(160deg, #161a24 0%, #131722 100%);
    border: 1px solid #2f3548;
    border-radius: 12px;
    padding: 1rem 1.2rem;
    box-shadow: 0 6px 24px rgba(0, 0, 0, 0.25);
}
[data-testid="stMetricValue"]  { color: #e8eaf0 !important; font-size: 2rem !important; }
[data-testid="stMetricLabel"]  { color: #9ca3bc !important; }
[data-testid="stMetricDelta"]  { font-size: 1rem !important; }

/* Progress bar color */
[data-testid="stProgress"] > div > div { background-color: #4ecdc4; }

/* Button */
.stButton > button {
    background: linear-gradient(135deg, #d4af37 0%, #f5d46d 100%);
    color: #11141e;
    font-weight: 700; border: none; border-radius: 8px;
    padding: 0.6rem 2rem; font-size: 1rem;
    width: 100%;
    box-shadow: 0 4px 14px rgba(212, 175, 55, 0.28);
}
.stButton > button:hover { filter: brightness(0.96); }

h1 { color: #e8eaf0 !important; }
h2, h3 { color: #c8cad8 !important; }
p, li  { color: #888fa8 !important; }
label  { color: #c8cad8 !important; }

.casino-table {
    background: radial-gradient(circle at top, #214233 0%, #173328 42%, #111b18 100%);
    border: 1px solid #785d1b;
    border-radius: 18px;
    padding: 1.1rem 1rem 0.9rem 1rem;
    margin-bottom: 0.9rem;
    box-shadow: inset 0 0 0 1px rgba(245, 212, 109, 0.12), 0 14px 30px rgba(0, 0, 0, 0.35);
}
.casino-title {
    color: #f5d46d;
    font-size: 0.92rem;
    letter-spacing: 0.12em;
    font-weight: 700;
    text-transform: uppercase;
    margin-bottom: 0.5rem;
}
.card-pill {
    display: inline-block;
    min-width: 2.2rem;
    text-align: center;
    padding: 0.2rem 0.55rem;
    border-radius: 999px;
    margin-left: 0.35rem;
    font-weight: 700;
    background: #f7f8fb;
    color: #1a1d27;
}
.chip-pill {
    display: inline-block;
    padding: 0.22rem 0.6rem;
    border-radius: 999px;
    margin-right: 0.35rem;
    margin-top: 0.25rem;
    border: 1px solid #9a7a25;
    color: #f5d46d;
    background: rgba(10, 13, 20, 0.42);
    font-size: 0.84rem;
}
.dealer-log {
    background: #141a26;
    border: 1px solid #2f3548;
    border-radius: 12px;
    padding: 0.65rem 0.75rem;
    min-height: 8rem;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🃏 TiltStack")
    st.markdown("*Adaptive Poker AI*")
    st.divider()

    st.markdown("### Opponent Profile")
    opponent = st.selectbox("Select opponent type", list(OPPONENT_FNS.keys()), index=0)
    st.caption(OPPONENT_DESC[opponent])

    st.divider()
    st.markdown("### Parameters")
    train_iters = st.slider("GTO training iterations", 5_000, 30_000, 15_000, step=1_000,
                            help="More iterations = closer to Nash equilibrium")
    n_hands = st.slider("Hands to simulate", 200, 1000, 600, step=100)
    roll_window = st.slider("Rolling average window", 20, 100, 50, step=10)

    st.divider()
    st.markdown("### About")
    st.caption("Layer 1: CFR+ Nash equilibrium\nLayer 2: Opponent classification\nLayer 3: BRSolver exploit strategy")

# ─────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────

st.markdown("# TiltStack — Adaptive Poker AI Demo")
st.markdown(f"**Opponent:** {opponent} · **Hands:** {n_hands} · **GTO iters:** {train_iters:,}")
st.divider()

# ─────────────────────────────────────────────────────────────
# Session state
# ─────────────────────────────────────────────────────────────

if "trained" not in st.session_state:
    st.session_state.trained      = False
    st.session_state.strategies   = None
    st.session_state.br_strategy  = None
    st.session_state.br_ev        = None
    st.session_state.gto_hands    = []
    st.session_state.expl_hands   = []
    st.session_state.sim_done     = False
    st.session_state.last_opponent = None
    st.session_state.play_bankroll = 0
    st.session_state.play_hands    = 0
    st.session_state.play_wins     = 0
    st.session_state.play_hand_active = False
    st.session_state.play_h        = None
    st.session_state.play_cards    = None
    st.session_state.play_log      = []
    st.session_state.play_last_delta  = 0
    st.session_state.table_rng_seed   = 0
    st.session_state.last_showdown    = None   # persists after hand ends
    # Layer 2: opponent classifier
    st.session_state.clf            = None   # OpponentClassifier (loaded lazily)
    st.session_state.tracker        = ActionTracker()
    st.session_state.clf_result     = None   # (label, confidence, probs) or None
    st.session_state.auto_opponent  = None   # detected profile driving dealer strategy
    st.session_state.detected_br    = None   # cached BRSolver strategy for detected profile

# Reset if opponent changed
if st.session_state.last_opponent != opponent:
    st.session_state.trained     = False
    st.session_state.sim_done    = False
    st.session_state.gto_hands   = []
    st.session_state.expl_hands  = []
    st.session_state.br_strategy = None
    st.session_state.last_opponent = opponent


def _new_table_hand(rng):
    # Deal 3 distinct ranks (same rank can't appear twice — no pair without community match)
    ranks = rng.choice(3, size=3, replace=False).tolist()
    p0, p1, comm = int(ranks[0]), int(ranks[1]), int(ranks[2])
    st.session_state.play_h     = p0 * 8
    st.session_state.play_cards = [p0, p1, comm]
    st.session_state.play_log   = ["New hand dealt."]
    st.session_state.play_hand_active = True
    # Advance the seed so each deal is fresh
    st.session_state.table_rng_seed += 1
    # Start tracking this hand for the classifier
    st.session_state.tracker.start_hand()


def _move_label(h, action):
    if action == Action.CHECK:
        return "Fold" if _raises(h) > 0 else "Check"
    if action == Action.BET:
        return "Call"
    return "Raise"


def _apply_table_action(action):
    h     = st.session_state.play_h
    cards = st.session_state.play_cards
    cur   = _stm(h)
    actor = "You" if cur == 0 else "Dealer"
    label = _move_label(h, action)
    st.session_state.play_log.append(f"{actor}: {label}")

    # Record the judge's action for classifier training
    if cur == 0:
        st.session_state.tracker.record_action(
            action_label=label,
            raises_before=_raises(h),
            round_id=_bet_round(h) + 1,
        )

    if _ends_hand(h, action):
        pay = _payout(h, action, cards[1 - cur])
        user_delta = pay if cur == 0 else -pay
        st.session_state.play_last_delta = int(user_delta)
        st.session_state.play_bankroll  += int(user_delta)
        st.session_state.play_hands     += 1
        if user_delta > 0:
            st.session_state.play_wins  += 1
        # Showdown = hand ends without a fold
        went_to_showdown = (label != "Fold")
        st.session_state.tracker.end_hand(went_to_showdown)
        # Re-classify after every completed hand
        _update_classification()
        st.session_state.play_hand_active = False
        st.session_state.play_log.append(
            f"Hand over. You {'won' if user_delta > 0 else 'lost' if user_delta < 0 else 'pushed'} {user_delta:+d} chips."
        )
        # Persist showdown info so it stays visible after rerun
        st.session_state.last_showdown = {
            "cards": list(cards),   # [p0, p1, comm]
            "delta": int(user_delta),
            "showdown": went_to_showdown,
        }
        return

    ns = _next_stm(h, action)
    st.session_state.play_h = _next_hash(h, action, cards[2], cards[ns])


def _update_classification() -> None:
    """Run the classifier after a hand completes and cache the result."""
    clf = st.session_state.get("clf")
    if clf is None:
        # Lazy-load (blocks once, ~1–2 s on first hand)
        clf = load_classifier()
        st.session_state.clf = clf
    result = clf.predict(st.session_state.tracker)
    if result is not None:
        label, conf, probs = result
        st.session_state.clf_result    = (label, conf, probs)
        # After 5 hands with ≥ 70 % confidence, auto-switch dealer opponent profile.
        # Recompute the BRSolver for the new detected profile only when it changes.
        if st.session_state.tracker.n_hands >= 5 and conf >= 0.70:
            prev = st.session_state.get("auto_opponent")
            st.session_state.auto_opponent = label
            if label != prev and st.session_state.br_strategy is not None:
                opp_vec = build_opponent_vector(label)
                br_live = BRSolver()
                br_live.load_strategy(opp_vec)
                br_live.compute(0)
                st.session_state.detected_br = br_live.get_full_br_strategy()

# ─────────────────────────────────────────────────────────────
# Phase 1: Train GTO
# ─────────────────────────────────────────────────────────────

col_train, col_gap, col_br = st.columns([5, 1, 4])

with col_train:
    st.markdown("### Step 1 — Train GTO Baseline")
    train_btn = st.button("Train CFR+ (Nash Equilibrium)", disabled=st.session_state.trained)

    train_status = st.empty()
    train_bar    = st.empty()
    train_metric = st.empty()

    if st.session_state.trained:
        expl_val = st.session_state.get("final_expl", 0.0)
        train_status.success(f"✓ Nash equilibrium reached · exploitability ≈ {expl_val:.3f} mBB/hand")

with col_br:
    st.markdown("### Step 2 — Compute Exploit Strategy")
    br_status = st.empty()
    if st.session_state.br_strategy is not None:
        br_status.success(f"✓ BRSolver complete · theoretical EV {st.session_state.br_ev:+.4f} chips/hand")
    else:
        br_status.info("Waiting for GTO training…")

if train_btn and not st.session_state.trained:
    rng = np.random.default_rng(42)
    leduc = Leduc()
    checkpoints = list(range(train_iters // 10, train_iters + 1, train_iters // 10))

    train_status.info("Training CFR+…")
    for i in range(train_iters):
        leduc.train(1)
        if (i + 1) in checkpoints:
            expl = leduc.compute_exploitability() * 1000
            pct  = (i + 1) / train_iters
            train_bar.progress(pct, text=f"Iter {i+1:,}/{train_iters:,} · exploitability {expl:.3f} mBB/hand")

    st.session_state.strategies  = leduc.solver.get_all_strategies()
    st.session_state.final_expl  = expl
    st.session_state.trained     = True
    train_bar.empty()
    train_status.success(f"✓ Nash equilibrium reached · exploitability ≈ {expl:.3f} mBB/hand")

    # Compute BR immediately after training
    with br_status:
        with st.spinner("Computing best response via BRSolver…"):
            opp_vec = build_opponent_vector(opponent)
            br      = BRSolver()
            br.load_strategy(opp_vec)
            ev      = br.compute(0)
            br_strat = br.get_full_br_strategy()
        st.session_state.br_strategy = br_strat
        st.session_state.br_ev       = ev
    br_status.success(f"✓ BRSolver complete · theoretical EV {ev:+.4f} chips/hand")
    st.rerun()

# ─────────────────────────────────────────────────────────────
# Phase 2: Simulate
# ─────────────────────────────────────────────────────────────

st.divider()
st.markdown("### Step 3 — Live Simulation")

if not st.session_state.trained:
    st.info("Complete Step 1 first.")
else:
    not_ready = st.session_state.sim_done or not st.session_state.trained or st.session_state.br_strategy is None
    sim_btn = st.button("▶  Run Live Demo", disabled=not_ready)

    chart_placeholder = st.empty()
    stats_placeholder = st.columns(4)

    if st.session_state.sim_done and st.session_state.gto_hands:
        # Redraw final chart
        g = st.session_state.gto_hands
        e = st.session_state.expl_hands
        n = len(g)
        xs = list(range(1, n + 1))
        g_cum = list(np.cumsum(g))
        e_cum = list(np.cumsum(e))

        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=("Cumulative Chips Won", f"Rolling Win Rate (last {roll_window} hands)"),
        )
        fig.add_trace(go.Scatter(x=xs, y=g_cum, name="GTO baseline",
                                 line=dict(color="#4ecdc4", width=2)), row=1, col=1)
        fig.add_trace(go.Scatter(x=xs, y=e_cum, name="Exploitative",
                                 line=dict(color="#ff6b6b", width=2.5)), row=1, col=1)

        if n >= roll_window:
            rx = list(range(roll_window, n + 1))
            fig.add_trace(go.Scatter(
                x=rx,
                y=[float(np.mean(g[i-roll_window:i])) for i in range(roll_window, n+1)],
                name="GTO baseline", line=dict(color="#4ecdc4", width=2),
                showlegend=False), row=1, col=2)
            fig.add_trace(go.Scatter(
                x=rx,
                y=[float(np.mean(e[i-roll_window:i])) for i in range(roll_window, n+1)],
                name="Exploitative", line=dict(color="#ff6b6b", width=2.5),
                showlegend=False), row=1, col=2)
        fig.add_hline(y=0, line_color="#444", line_dash="dash", row=1, col=2)

        _style_fig(fig) if False else None
        fig.update_layout(
            paper_bgcolor="#0f1117", plot_bgcolor="#1a1d27",
            font=dict(color="#e8eaf0"),
            legend=dict(bgcolor="#1a1d27", bordercolor="#2a2d3a"),
            height=420, margin=dict(t=40, b=20, l=20, r=20),
        )
        for axis in ["xaxis", "yaxis", "xaxis2", "yaxis2"]:
            fig.update_layout({axis: dict(gridcolor="#2a2d3a", color="#888fa8")})
        chart_placeholder.plotly_chart(fig, use_container_width=True)

        g_avg = float(np.mean(g))
        e_avg = float(np.mean(e))
        mult  = e_avg / g_avg if abs(g_avg) > 1e-6 else float("inf")
        stats_placeholder[0].metric("GTO chips/hand",         f"{g_avg:+.3f}")
        stats_placeholder[1].metric("Exploit chips/hand",     f"{e_avg:+.3f}")
        stats_placeholder[2].metric("Speedup",                f"{mult:.2f}x",
                                    delta=f"+{(mult-1)*100:.0f}% vs GTO")
        stats_placeholder[3].metric("Theoretical exploit EV", f"{st.session_state.br_ev:+.3f}")

    if sim_btn and not st.session_state.sim_done:
        rng = np.random.default_rng(42)
        strategies  = st.session_state.strategies
        br_strategy = st.session_state.br_strategy
        opp_fn      = OPPONENT_FNS[opponent]

        p0_gto     = lambda h: gto_action(strategies, h, rng)
        p0_exploit = lambda h: _br_action(br_strategy, h, rng)
        p1_opp     = lambda h: opp_fn(h, rng)

        gto_hands  = []
        expl_hands = []
        BATCH = 10

        prog = st.progress(0, text="Simulating hands…")

        for start in range(0, n_hands, BATCH):
            end = min(start + BATCH, n_hands)
            for _ in range(end - start):
                gto_hands.append(simulate_hand(p0_gto,    p1_opp, rng))
                expl_hands.append(simulate_hand(p0_exploit, p1_opp, rng))

            n = len(gto_hands)
            pct = n / n_hands
            prog.progress(pct, text=f"Hand {n}/{n_hands}")

            # Live chart update every 50 hands
            if n % 50 == 0 or n == n_hands:
                xs    = list(range(1, n + 1))
                g_cum = list(np.cumsum(gto_hands))
                e_cum = list(np.cumsum(expl_hands))

                fig = make_subplots(
                    rows=1, cols=2,
                    subplot_titles=("Cumulative Chips Won",
                                    f"Rolling Win Rate (last {roll_window} hands)"),
                )
                fig.add_trace(go.Scatter(x=xs, y=g_cum, name="GTO baseline",
                                         line=dict(color="#4ecdc4", width=2)), row=1, col=1)
                fig.add_trace(go.Scatter(x=xs, y=e_cum, name="Exploitative",
                                         line=dict(color="#ff6b6b", width=2.5)), row=1, col=1)

                if n >= roll_window:
                    rx = list(range(roll_window, n + 1))
                    fig.add_trace(go.Scatter(
                        x=rx,
                        y=[float(np.mean(gto_hands[i-roll_window:i]))
                           for i in range(roll_window, n+1)],
                        name="GTO baseline", line=dict(color="#4ecdc4", width=2),
                        showlegend=False), row=1, col=2)
                    fig.add_trace(go.Scatter(
                        x=rx,
                        y=[float(np.mean(expl_hands[i-roll_window:i]))
                           for i in range(roll_window, n+1)],
                        name="Exploitative", line=dict(color="#ff6b6b", width=2.5),
                        showlegend=False), row=1, col=2)
                fig.add_hline(y=0, line_color="#444", line_dash="dash", row=1, col=2)
                fig.update_layout(
                    paper_bgcolor="#0f1117", plot_bgcolor="#1a1d27",
                    font=dict(color="#e8eaf0"),
                    legend=dict(bgcolor="#1a1d27", bordercolor="#2a2d3a"),
                    height=420, margin=dict(t=40, b=20, l=20, r=20),
                )
                for axis in ["xaxis", "yaxis", "xaxis2", "yaxis2"]:
                    fig.update_layout({axis: dict(gridcolor="#2a2d3a", color="#888fa8")})
                chart_placeholder.plotly_chart(fig, use_container_width=True)

                g_avg = float(np.mean(gto_hands))
                e_avg = float(np.mean(expl_hands))
                mult  = e_avg / g_avg if abs(g_avg) > 1e-6 else float("inf")
                stats_placeholder[0].metric("GTO chips/hand",     f"{g_avg:+.3f}")
                stats_placeholder[1].metric("Exploit chips/hand", f"{e_avg:+.3f}")
                stats_placeholder[2].metric("Speedup",            f"{mult:.2f}x",
                                            delta=f"+{(mult-1)*100:.0f}% vs GTO")
                stats_placeholder[3].metric("Theoretical exploit EV",
                                            f"{st.session_state.br_ev:+.3f}")

        prog.empty()
        st.session_state.gto_hands   = gto_hands
        st.session_state.expl_hands  = expl_hands
        st.session_state.sim_done    = True
        st.rerun()

# ─────────────────────────────────────────────────────────────
# Phase 3: Play at the table
# ─────────────────────────────────────────────────────────────
st.divider()

# ── Section header with architecture flow ─────────────────────
st.markdown("""
<div style='margin-bottom:0.6rem'>
  <span style='color:#e8eaf0;font-size:1.3rem;font-weight:700'>🃏 Play the Dealer — Watch TiltStack Adapt</span><br>
  <span style='color:#888fa8;font-size:0.88rem'>
    Every hand you play is fed into the 3-layer pipeline in real time.
  </span>
</div>
""", unsafe_allow_html=True)

if not st.session_state.trained:
    st.info("Complete Step 1 (Train CFR+) first to unlock table play.")
else:
    rng_table   = np.random.default_rng(st.session_state.table_rng_seed)
    dealer_mode = "Exploit" if st.session_state.br_strategy is not None else "GTO"

    clf_result    = st.session_state.get("clf_result")
    auto_opponent = st.session_state.get("auto_opponent")
    n_tracked     = st.session_state.tracker.n_hands
    LOCK_THRESHOLD = 5   # hands before strategy locks in

    # ── Architecture pipeline display ─────────────────────────────────────────
    # Always visible — shows what stage TiltStack is at right now.
    L1_done  = st.session_state.trained
    L2_done  = clf_result is not None
    L2_locked = auto_opponent is not None
    L3_done  = L2_locked

    def _pipe_node(num, label, sublabel, done, locked=False, color="#4ecdc4"):
        if locked:
            bg, border, tc = "rgba(78,44,132,0.35)", "#7c3aed", "#c4b5fd"
        elif done:
            bg, border, tc = f"rgba(0,0,0,0.25)", f"{color}55", color
        else:
            bg, border, tc = "rgba(0,0,0,0.15)", "#2f3548", "#555d75"
        icon = "✓" if (done and not locked) else ("🔒" if locked else "○")
        return (
            f"<div style='flex:1;background:{bg};border:1px solid {border};"
            f"border-radius:10px;padding:0.6rem 0.8rem;text-align:center'>"
            f"<div style='color:{tc};font-size:1.1rem;font-weight:700'>{icon} L{num}</div>"
            f"<div style='color:{tc};font-size:0.82rem;font-weight:600;margin-top:2px'>{label}</div>"
            f"<div style='color:#555d75;font-size:0.72rem;margin-top:2px'>{sublabel}</div>"
            f"</div>"
        )

    l2_sublabel = (
        f"Locked: {auto_opponent}" if L2_locked
        else (f"Reading… {n_tracked}/{LOCK_THRESHOLD} hands" if L2_done else f"{n_tracked}/3 hands")
    )
    l3_sublabel = f"Exploiting {auto_opponent}" if L3_done else "Waiting for L2"

    pipe_html = (
        "<div style='display:flex;gap:0.5rem;align-items:stretch;margin-bottom:0.9rem'>"
        + _pipe_node(1, "CFR+ Nash",   "Equilibrium trained",      L1_done,  color="#4ecdc4")
        + "<div style='display:flex;align-items:center;color:#2f3548;font-size:1.2rem;padding:0 2px'>→</div>"
        + _pipe_node(2, "LSTM Classify", l2_sublabel,              L2_done,  L2_locked, color="#f5d46d")
        + "<div style='display:flex;align-items:center;color:#2f3548;font-size:1.2rem;padding:0 2px'>→</div>"
        + _pipe_node(3, "BRSolver",    l3_sublabel,                L3_done,  color="#ff6b6b")
        + "</div>"
    )
    st.markdown(pipe_html, unsafe_allow_html=True)

    # ── Classifier confidence panel (shows from hand 1) ───────────────────────
    if n_tracked == 0 and not st.session_state.play_hand_active:
        st.markdown(
            "<div style='background:rgba(245,212,109,0.06);border:1px solid #3a3010;"
            "border-radius:10px;padding:0.65rem 1rem;margin-bottom:0.7rem;"
            "color:#888fa8;font-size:0.88rem'>"
            "🧠 <strong style='color:#c8cad8'>Layer 2 will read your play style</strong> — "
            "raise aggressively, call everything, or play tight. "
            "After 3 hands it produces a live classification. After 5 hands it locks in and "
            "Layer 3 (BRSolver) recomputes an exploit strategy specifically for you."
            "</div>",
            unsafe_allow_html=True,
        )
    elif clf_result is not None:
        det_label, det_conf, det_probs = clf_result
        det_idx   = LABEL_NAMES.index(det_label)
        det_color = LABEL_COLORS[det_idx]
        det_icon  = LABEL_ICONS[det_idx]

        # Build confidence bar rows
        bars_html = ""
        for i, (lname, lcolor) in enumerate(zip(LABEL_NAMES, LABEL_COLORS)):
            pct  = det_probs[i] * 100
            bold = "font-weight:700;" if i == det_idx else ""
            bars_html += (
                f"<div style='display:flex;align-items:center;gap:0.5rem;margin-bottom:4px'>"
                f"<div style='color:{lcolor};font-size:0.75rem;width:4.5rem;{bold}'>{lname}</div>"
                f"<div style='flex:1;background:#1e2130;border-radius:4px;height:8px'>"
                f"<div style='width:{pct:.0f}%;background:{lcolor};height:8px;border-radius:4px'></div>"
                f"</div>"
                f"<div style='color:{lcolor};font-size:0.75rem;width:2.5rem;text-align:right;{bold}'>{pct:.0f}%</div>"
                f"</div>"
            )

        status_line = (
            f"<span style='color:#7c3aed;font-weight:700'>🔒 LOCKED IN — Strategy updated for you</span>"
            if L2_locked else
            f"<span style='color:#888fa8'>Confidence building… {n_tracked}/{LOCK_THRESHOLD} hands ({det_conf*100:.0f}% confidence needed: 70%)</span>"
        )

        clf_html = (
            f"<div style='background:rgba(0,0,0,0.3);border:1px solid {det_color}55;"
            f"border-radius:12px;padding:0.8rem 1rem;margin-bottom:0.8rem'>"
            f"<div style='display:flex;align-items:center;gap:0.5rem;margin-bottom:0.5rem'>"
            f"<span style='font-size:1.6rem'>{det_icon}</span>"
            f"<div>"
            f"<div style='color:#9ca3bc;font-size:0.7rem;text-transform:uppercase;letter-spacing:0.1em'>Layer 2 · LSTM Classifier</div>"
            f"<div style='color:{det_color};font-size:1.15rem;font-weight:700'>Reading you as: {det_label}</div>"
            f"</div>"
            f"<div style='margin-left:auto;font-size:0.78rem'>{status_line}</div>"
            f"</div>"
            f"{bars_html}"
            f"</div>"
        )
        st.markdown(clf_html, unsafe_allow_html=True)

        # Dramatic lock-in banner (fires once when auto_opponent first set)
        if L2_locked and st.session_state.get("_lock_banner_shown") != auto_opponent:
            st.session_state["_lock_banner_shown"] = auto_opponent
            st.balloons()
            st.success(
                f"⚡ TiltStack identified you as **{auto_opponent}** — "
                f"BRSolver recomputed. The dealer is now exploiting your specific tendencies."
            )
    else:
        # 1–2 hands in: show a progress bar so judges see something happening
        pct_done = n_tracked / 3
        st.markdown(
            f"<div style='background:rgba(0,0,0,0.25);border:1px solid #2f3548;"
            f"border-radius:10px;padding:0.65rem 1rem;margin-bottom:0.7rem'>"
            f"<div style='color:#9ca3bc;font-size:0.72rem;text-transform:uppercase;"
            f"letter-spacing:0.08em;margin-bottom:6px'>Layer 2 · Building profile…</div>"
            f"<div style='background:#1e2130;border-radius:4px;height:10px;margin-bottom:6px'>"
            f"<div style='width:{pct_done*100:.0f}%;background:linear-gradient(90deg,#f5d46d,#d4af37);"
            f"height:10px;border-radius:4px;transition:width 0.4s'></div></div>"
            f"<div style='color:#888fa8;font-size:0.8rem'>"
            f"{n_tracked} / 3 hands · Play more to unlock classification</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

    # ── Metrics row ────────────────────────────────────────────────────────────
    top_a, top_b, top_c, top_d = st.columns([2, 2, 2, 2])
    with top_a:
        delta_chips = st.session_state.play_last_delta
        st.metric("Your Bankroll",
                  f"{st.session_state.play_bankroll:+d} chips",
                  delta=f"{delta_chips:+d}" if st.session_state.play_hands > 0 else None)
    with top_b:
        st.metric("Hands Played", f"{st.session_state.play_hands}")
    with top_c:
        win_rate = (st.session_state.play_wins / st.session_state.play_hands * 100
                    if st.session_state.play_hands > 0 else 0.0)
        st.metric("Win Rate", f"{win_rate:.0f}%")
    with top_d:
        active_delta = "BRSolver exploit" if dealer_mode == "Exploit" else "Nash equilibrium"
        if auto_opponent is not None:
            active_delta = f"Exploiting **{auto_opponent}**"
        st.metric("Dealer strategy", dealer_mode, delta=active_delta)

    # ── Persistent last showdown result ──────────────────────────────────────
    # Shown above the deal button so judges always see the last result
    # and know to press Deal again. Stays visible even after rerun.
    last_sd = st.session_state.get("last_showdown")
    if last_sd and not st.session_state.play_hand_active:
        delta         = last_sd["delta"]
        p_card        = CARD_LABELS[last_sd["cards"][0]]
        d_card        = CARD_LABELS[last_sd["cards"][1]]
        b_card        = CARD_LABELS[last_sd["cards"][2]]
        outcome_color = "#4ecdc4" if delta > 0 else ("#ef4444" if delta < 0 else "#888fa8")
        outcome_word  = "WIN ✓" if delta > 0 else ("LOSS ✗" if delta < 0 else "PUSH")
        st.markdown(
            f"<div style='background:rgba(0,0,0,0.3);border:1px solid {outcome_color}55;"
            f"border-radius:10px;padding:0.8rem 1rem;margin-bottom:0.6rem;"
            f"display:flex;align-items:center;justify-content:space-between'>"
            f"<div>"
            f"<div style='color:#9ca3bc;font-size:0.72rem;text-transform:uppercase;"
            f"letter-spacing:0.08em'>Last hand</div>"
            f"<div style='color:#e8eaf0;font-size:0.95rem;margin-top:2px'>"
            f"You <strong>{p_card}</strong> · Dealer <strong>{d_card}</strong> · Board <strong>{b_card}</strong>"
            f"</div>"
            f"</div>"
            f"<div style='text-align:right'>"
            f"<div style='color:{outcome_color};font-size:1.3rem;font-weight:900'>{outcome_word}</div>"
            f"<div style='color:{outcome_color};font-size:0.95rem'>{delta:+d} chips</div>"
            f"</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

    # ── Deal / Reset — always visible ────────────────────────────────────────
    col_deal, col_reset = st.columns([4, 1])
    with col_deal:
        hand_active = st.session_state.play_hand_active
        btn_label   = "⏳  Hand in progress…" if hand_active else "🃏  Deal New Hand"
        if st.button(btn_label, disabled=hand_active, use_container_width=True):
            _new_table_hand(rng_table)
            st.rerun()
    with col_reset:
        if st.button("↺  Reset", use_container_width=True, disabled=hand_active):
            st.session_state.play_bankroll    = 0
            st.session_state.play_hands       = 0
            st.session_state.play_wins        = 0
            st.session_state.play_last_delta  = 0
            st.session_state.play_hand_active = False
            st.session_state.table_rng_seed   = 0
            st.session_state.last_showdown    = None
            st.session_state.tracker.reset()
            st.session_state.clf_result       = None
            st.session_state.auto_opponent    = None
            st.session_state.detected_br      = None
            st.session_state.pop("_lock_banner_shown", None)
            st.rerun()

    # ── Active hand ────────────────────────────────────────────────────────────
    if st.session_state.play_hand_active:
        h         = st.session_state.play_h
        cards     = st.session_state.play_cards
        round_id  = _bet_round(h) + 1
        pot_chips = 2 + 2 * _raises(h) + (4 if _bet_round(h) == 1 else 0)

        table_left, table_right = st.columns([3, 2])
        with table_left:
            board_card = CARD_LABELS[cards[2]] if _bet_round(h) == 1 else "—"
            dealer_tag = (
                f"<span style='font-size:0.7rem;color:#7c3aed;margin-left:6px'>"
                f"🔒 exploit</span>"
                if auto_opponent else ""
            )
            st.markdown(
                f"<div class='casino-table'>"
                f"<div class='casino-title'>Main Table{dealer_tag}</div>"
                f"<span class='chip-pill'>Round {round_id} of 2</span>"
                f"<span class='chip-pill'>Pot: {pot_chips} chips</span>"
                f"<div style='margin-top:0.7rem;display:flex;gap:1.2rem;align-items:center'>"
                f"<div style='text-align:center'>"
                f"<div style='color:#9ca3bc;font-size:0.72rem;margin-bottom:3px'>YOUR CARD</div>"
                f"<div style='background:#f7f8fb;color:#1a1d27;font-size:1.8rem;font-weight:900;"
                f"width:2.6rem;height:2.6rem;border-radius:8px;display:flex;"
                f"align-items:center;justify-content:center'>{CARD_LABELS[cards[0]]}</div>"
                f"</div>"
                f"<div style='text-align:center'>"
                f"<div style='color:#9ca3bc;font-size:0.72rem;margin-bottom:3px'>BOARD</div>"
                f"<div style='background:#f7f8fb;color:#1a1d27;font-size:1.8rem;font-weight:900;"
                f"width:2.6rem;height:2.6rem;border-radius:8px;display:flex;"
                f"align-items:center;justify-content:center'>{board_card}</div>"
                f"</div>"
                f"<div style='text-align:center'>"
                f"<div style='color:#9ca3bc;font-size:0.72rem;margin-bottom:3px'>DEALER</div>"
                f"<div style='background:#2a2d3a;color:#555d75;font-size:1.8rem;font-weight:900;"
                f"width:2.6rem;height:2.6rem;border-radius:8px;border:1px dashed #3a3d4a;"
                f"display:flex;align-items:center;justify-content:center'>?</div>"
                f"</div>"
                f"</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

        with table_right:
            st.markdown(
                "<div class='dealer-log'>"
                "<div style='color:#9ca3bc;font-size:0.75rem;font-weight:600;"
                "text-transform:uppercase;letter-spacing:0.08em;margin-bottom:6px'>"
                "Action Log</div>",
                unsafe_allow_html=True,
            )
            for entry in st.session_state.play_log[-7:]:
                icon  = "▶" if entry.startswith("You") else ("◀" if entry.startswith("Dealer") else "·")
                color = "#e8eaf0" if entry.startswith("You") else (
                    "#ff6b6b" if entry.startswith("Dealer") else "#555d75")
                st.markdown(
                    f"<div style='color:{color};font-size:0.82rem;padding:2px 0'>{icon} {entry}</div>",
                    unsafe_allow_html=True,
                )
            st.markdown("</div>", unsafe_allow_html=True)

        # Dealer auto-plays until judge's turn (or hand ends)
        guard = 0
        while st.session_state.play_hand_active and _stm(st.session_state.play_h) == 1 and guard < 6:
            guard += 1
            h_bot       = st.session_state.play_h
            detected_br = st.session_state.get("detected_br")
            if detected_br is not None:
                bot_action = _br_action(detected_br, h_bot, rng_table)
            elif st.session_state.br_strategy is not None:
                bot_action = _br_action(st.session_state.br_strategy, h_bot, rng_table)
            else:
                bot_action = gto_action(st.session_state.strategies, h_bot, rng_table)
            _apply_table_action(bot_action)
            st.session_state.table_rng_seed += 1
            rng_table = np.random.default_rng(st.session_state.table_rng_seed)

        # If the dealer's action ended the hand, rerun immediately so the
        # "Last hand" result and re-enabled Deal button render correctly.
        # Without this, Streamlit keeps the stale "Hand in progress…" UI.
        if not st.session_state.play_hand_active:
            st.rerun()

        # Judge's move buttons
        if st.session_state.play_hand_active and _stm(st.session_state.play_h) == 0:
            h_user = st.session_state.play_h
            legal  = _legal_moves(h_user)
            st.markdown(
                "<div style='color:#f5d46d;font-weight:600;font-size:0.9rem;"
                "margin:0.5rem 0 0.3rem'>Your move:</div>",
                unsafe_allow_html=True,
            )
            cols = st.columns(len(legal))
            action_labels = {
                Action.CHECK: ("Fold" if _raises(h_user) > 0 else "Check"),
                Action.BET:   ("Call"  if _raises(h_user) > 0 else "Bet"),
                Action.RAISE: "Raise",
            }
            for idx, action in enumerate(legal):
                lbl = action_labels[action]
                if cols[idx].button(lbl, key=f"ua_{idx}_{h_user}", use_container_width=True):
                    _apply_table_action(action)
                    st.rerun()
