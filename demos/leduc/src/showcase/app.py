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
    background: #1a1d27;
    border: 1px solid #2a2d3a;
    border-radius: 10px;
    padding: 1rem 1.2rem;
}
[data-testid="stMetricValue"]  { color: #e8eaf0 !important; font-size: 2rem !important; }
[data-testid="stMetricLabel"]  { color: #888fa8 !important; }
[data-testid="stMetricDelta"]  { font-size: 1rem !important; }

/* Progress bar color */
[data-testid="stProgress"] > div > div { background-color: #4ecdc4; }

/* Button */
.stButton > button {
    background: #4ecdc4; color: #0f1117;
    font-weight: 700; border: none; border-radius: 8px;
    padding: 0.6rem 2rem; font-size: 1rem;
    width: 100%;
}
.stButton > button:hover { background: #38b2ac; }

h1 { color: #e8eaf0 !important; }
h2, h3 { color: #c8cad8 !important; }
p, li  { color: #888fa8 !important; }
label  { color: #c8cad8 !important; }
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

# Reset if opponent changed
if st.session_state.last_opponent != opponent:
    st.session_state.trained     = False
    st.session_state.sim_done    = False
    st.session_state.gto_hands   = []
    st.session_state.expl_hands  = []
    st.session_state.br_strategy = None
    st.session_state.last_opponent = opponent

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
