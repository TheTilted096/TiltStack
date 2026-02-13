# Leduc Poker CFR+ Solver

High-performance C++ implementation of Counterfactual Regret Minimization Plus (CFR+) for solving Leduc Hold'em poker.

## Features

- **CFR+ Algorithm**: Alternating updates, batched regret flooring, linear strategy weighting
- **Double Precision**: Prevents floating-point accumulation errors at high iteration counts
- **Optimal Performance**: Converges to Nash equilibrium (0.00 mBB exploitability) in 14k iterations
- **Fast Execution**: 3,810 iterations/second (~26.7s for 100k iterations)
- **C++/Python**: C++ core with Python bindings via pybind11

## Quick Start

### Installation

```bash
# Install dependencies
make install

# Build C++ extension
make build

# Run solver (100k iterations)
make test
```

### Requirements

- Python 3.8+
- C++ compiler with C++20 support
- pybind11
- matplotlib (for plotting)

```bash
pip install -r requirements.txt
```

## Performance

| Metric | Value |
|--------|-------|
| **Exploitability** | 0.00 mBB/hand (Nash equilibrium) |
| **Convergence** | 14k iterations to optimal |
| **Speed** | 3,810 iterations/sec |
| **vs Vanilla CFR** | 56x better solution quality |

See `docs/PERFORMANCE.md` for detailed benchmarks.

## Repository Structure

```
leduc/
├── src/
│   ├── cppsrc/          # C++ implementation
│   │   ├── Node.cpp/h        # CFR node (regrets, strategy)
│   │   ├── Leduc.cpp/h       # Game tree traversal
│   │   ├── BestResponse.cpp  # Exploitability computation
│   │   └── bindings.cpp      # pybind11 interface
│   └── pysrc/           # Python interface
│       └── Leduc.py          # Training loop, visualization
├── output/              # Results (CSV, plots, analysis)
├── docs/               # Documentation
├── Makefile            # Build targets
├── setup.py            # pybind11 build config
└── requirements.txt    # Python dependencies
```

## Algorithm Details

### CFR+ Implementation

1. **Alternating Updates**: Players update regrets separately to avoid simultaneous update bugs
2. **Batched Regret Flooring**: Accumulate deltas during traversal, apply and floor after
3. **Linear Strategy Weighting**: `weight = max(0, iteration - 500)` with 500-iteration delay
4. **Double Precision**: Accumulated strategy uses `double` to maintain precision at billions of operations

### Key Improvements Over Vanilla CFR

- **Regret flooring** (CFR+): Prevents negative regrets from slowing convergence
- **Alternating updates**: Ensures clean separation of player updates
- **Linear weighting**: Emphasizes recent (more refined) iterations
- **Delay period**: Filters out noisy early iterations (500-iteration warmup)

## Usage

### Basic Training

```python
from leducsolver import LeducSolver

solver = LeducSolver()

# Train for 10k iterations
for i in range(1, 10001):
    # P0 pass
    for p0 in range(3):
        for p1 in range(3):
            for c in range(3):
                if p0 == p1 == c:
                    continue
                weight = 4.0 if (p0 == p1 or p0 == c or p1 == c) else 8.0
                solver.cfr([p0, p1, c], p0 * 8, [weight, weight], i, 0, True)

    solver.flush_regrets()

    # P1 pass
    for p0 in range(3):
        for p1 in range(3):
            for c in range(3):
                if p0 == p1 == c:
                    continue
                weight = 4.0 if (p0 == p1 or p0 == c or p1 == c) else 8.0
                solver.cfr([p0, p1, c], p0 * 8, [weight, weight], i, 1, False)

    solver.flush_regrets()

# Get final strategy
strategies = solver.get_all_strategies()
```

### Computing Exploitability

```python
from leducsolver import BestResponse

br = BestResponse()
br.load_strategy(strategies)

ev0 = br.compute(0)
ev1 = br.compute(1)

exploitability = (ev0 + ev1) / 2.0
print(f"Exploitability: {exploitability:.4f} mBB/hand")
```

## Output

Training generates:

- `output/exploitability.png` - Convergence graph
- `output/exploitability.csv` - Iteration-by-iteration exploitability
- `output/leduc_strategy.csv` - Final Nash equilibrium strategy
- `output/leduc_results.txt` - Human-readable strategy breakdown
- `output/br_results.txt` - Best response analysis

## Game Rules: Leduc Hold'em

- **Deck**: 3 ranks (Jack, Queen, King), 2 suits (6 cards total)
- **Players**: 2 players
- **Betting**: 2 rounds, 2-bet limit per round
- **Structure**:
  - Round 1: Each player dealt 1 private card, bet size = 2
  - Round 2: 1 community card revealed, bet size = 4
- **Showdown**: Best hand wins (pair > high card)

## References

- Tammelin, O. (2014). "Solving Large Imperfect Information Games Using CFR+"
- Zinkevich, M. et al. (2007). "Regret Minimization in Games with Incomplete Information"

## License

MIT License
