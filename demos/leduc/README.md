# Leduc Hold'em — CFR+ Solver & Adaptive Demo

Self-contained C++/Python implementation of CFR+ for Leduc Hold'em, plus an
interactive Streamlit demo showing how an exploitative agent outperforms a
Nash-equilibrium (GTO) baseline against predictable opponents.

## Quick Start

```bash
cd demos/leduc

# 1. Install Python demo dependencies (streamlit, plotly, rlcard, numpy, matplotlib)
make install

# 2. Compile the C++ extension (requires CMake 3.20+, C++20 compiler)
make

# 3. Launch the interactive Streamlit demo
make demo          # opens localhost:8501

# or run the static matplotlib animation
python src/showcase/demo_vs_tight.py

# 4. Run CFR+ solver standalone (100k iterations, ~27s)
make test
```

## Requirements

| Tool | Version | Notes |
|------|---------|-------|
| Python | 3.11+ | detected automatically by Makefile |
| CMake | 3.20+ | fetches pybind11 automatically |
| C++ compiler | g++ 10+ / clang++ 12+ | C++20 required |

No conda, no GPU, no CUDA needed for the demo.

---

## Demo: GTO vs Exploitative Agent

The Streamlit UI (`src/showcase/app.py`) walks through the three layers of
TiltStack in real time:

1. **Train GTO** — CFR+ converges to Nash equilibrium in ~15k iterations
   (exploitability → 0.00 mBB/hand)
2. **Compute exploit strategy** — BRSolver finds the mathematically optimal
   counter-strategy for the selected opponent profile
3. **Live simulation** — watch chips accumulate hand-by-hand: exploitative
   vs GTO, both playing the same opponent

### Opponent Profiles & Typical Results

| Opponent | Description | GTO chips/hand | Exploit chips/hand | Speedup |
|----------|-------------|:--------------:|:------------------:|:-------:|
| Tight | Folds weak hands, only bets Kings | ~+0.46 | ~+1.40 | ~3× |
| Loose-Passive | Calls everything, rarely raises | ~+0.30 | ~+0.65 | ~2× |
| Aggressive | Bets/raises constantly | ~+0.25 | ~+1.80 | ~6–8× |

Run with:
```bash
make demo
# or, from demos/leduc/:
python -m streamlit run src/showcase/app.py
```

---

## Solver Performance

| Metric | Value |
|--------|-------|
| Exploitability | 0.00 mBB/hand (Nash equilibrium) |
| Convergence | ~14k iterations |
| Speed | ~3,810 iterations/sec |
| Time to 100k iters | ~26.7s |

---

## Repository Structure

```
demos/leduc/
├── CMakeLists.txt          # Fetches pybind11, builds leducsolver extension
├── Makefile                # build / install / test / demo / clean
├── README.md
├── docs/
│   ├── IMPLEMENTATION.md   # CFR+ algorithm details and bug-fix notes
│   └── PERFORMANCE.md      # Benchmarks and profiling breakdown
├── graphs/
│   ├── results_bar.png         # Static bar chart (slide asset)
│   └── results_comparison.png  # Cumulative chips comparison
└── src/
    ├── cppsrc/             # C++ CFR+ solver (compiled to leducsolver.so)
    │   ├── Node.cpp/h          # CFR node: regrets, strategy accumulation
    │   ├── Leduc.cpp/h         # Game tree traversal
    │   ├── BestResponse.cpp/h  # Exact exploitability / BR strategy
    │   └── bindings.cpp        # pybind11 interface
    ├── pysrc/              # Python training wrapper
    │   ├── Leduc.py            # CFR+ orchestrator, exploitability helper
    │   └── leducsolver.so      # Built by CMake (gitignored)
    ├── showcase/           # Interactive demo
    │   ├── app.py              # Streamlit UI — 3-step GTO → BR → simulate
    │   ├── demo_vs_tight.py    # Matplotlib animation + pure-Python game logic
    │   ├── opponents.py        # Opponent bot classes
    │   ├── exploitative_agent.py
    │   ├── gto_agent.py
    │   └── evaluate.py         # Benchmark runner (rlcard-based)
    └── tests/
        ├── test_cfr.py         # CFR+ convergence unit tests
        └── test_rlcard.py      # rlcard environment smoke tests
```

---

## pybind11 API

```python
from leducsolver import LeducSolver, BestResponse, Action, Rank

# Train CFR+
solver = LeducSolver()
solver.cfr(cards, root_hash, weights, iteration, player, accumulate)
solver.flush_regrets()
strategies = solver.get_all_strategies()   # list[list[float]], shape (528, 3)

# Compute best response
br = BestResponse()
br.load_strategy(opponent_strategy_vector)  # 528-node list[list[float]]
ev       = br.compute(player=0)             # EV for the BR player
br_strat = br.get_full_br_strategy()        # optimal counter, shape (528, 3)
```

Actions: `Action.CHECK`, `Action.BET`, `Action.RAISE`  
Ranks: `Rank.JACK` (0), `Rank.QUEEN` (1), `Rank.KING` (2)

---

## Game Rules: Leduc Hold'em

- **Deck**: 6 cards — J, Q, K in 2 suits
- **Players**: 2
- **Rounds**: 2 (private card → shared card revealed)
- **Bet sizes**: Round 1 = 2 chips, Round 2 = 4 chips; max 2 raises/round
- **Showdown**: Pair beats high card; higher rank wins ties
- **Game states**: 528 information-set nodes (integer hash encoding)

---

## Adding a New Opponent Profile

1. **Write an action function** in `src/showcase/app.py`:
   ```python
   def my_opponent_action(h, rng):
       moves = _legal_moves(h)
       # h encodes game state; return one Action from moves
       return moves[0]
   ```

2. **Build a 528-node strategy vector** in `build_opponent_vector()` so
   BRSolver can compute the exact optimal counter. Each entry is
   `[p_check, p_bet, p_raise]` summing to 1 over legal moves at node `h`.

3. **Register it**:
   ```python
   OPPONENT_FNS["MyOpponent"]  = my_opponent_action
   OPPONENT_DESC["MyOpponent"] = "One-line description shown in the UI."
   ```

The `simulate_hand(p0_fn, p1_fn, rng)` helper runs a full hand and returns
the P0 payout. `gto_action(strategies, h, rng)` samples from the trained GTO
strategy vector.

---

## References

- Tammelin, O. (2014). "Solving Large Imperfect Information Games Using CFR+"
- Zinkevich et al. (2007). "Regret Minimization in Games with Incomplete Information"
