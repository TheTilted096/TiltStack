# Evaluation Methodology

This document describes how TiltStack agents are evaluated against each other and against exploitable baselines. Evaluation is critical for monitoring convergence to equilibrium and measuring exploit-ability.

## Overview

TiltStack uses **OpenSpiel's universal poker engine** to run duplicate matches between trained agents. Each evaluation produces two types of data:

1. **Head-to-head win rates** (chips/hand, split by position)
2. **Exploitability** (best-response edge, approximated during training)

## Game Configuration

### OpenSpiel Game String

```python
GAME_STRING = (
    "universal_poker("
    "betting=nolimit,"
    "numPlayers=2,"
    "numSuits=4,"
    "numRanks=13,"
    "numHoleCards=2,"
    "numRounds=4,"
    "blind=50 100,"  # Small/Big blind
    "maxRaises=99 99 99 99,"  # No raise cap
    "numBoardCards=0 3 1 1,"  # 0 (preflop), 3 (flop), 1 (turn), 1 (river)
    "stack=2000 2000,"  # Starting stacks
    "firstPlayer=1 2 2 2,"  # Action order (SB acts first preflop)
    "bettingAbstraction=fullgame"
    ")"
)
```

**Parameter Mapping to CFRTypes.h:**

| Game Param | CFRTypes.h | Unit | Notes |
|------------|-----------|------|-------|
| `blind=50 100` | SB/BB | chips (1/100 mBB) | 50 chips = 500 mBB; 100 chips = 1000 mBB (1 BB) |
| `stack=2000 2000` | STARTING_STACK | chips | 2000 chips = 40 BB (must match `STARTING_STACK = 40,000` milli-chips) |
| `numRounds=4` | NUM_ROUNDS | streets | preflop, flop, turn, river |

**Payoff Conversion:**

```python
OSP_BIG_BLIND = 100  # chips (game param)
CHIPS_TO_MBB = 1000 / OSP_BIG_BLIND  # = 10.0

# Example: OpenSpiel returns +50 chips for P0
payoff_mbb = 50 * CHIPS_TO_MBB = 500 mBB
```

## Duplicate Match Protocol

### Standard Evaluation (10,000 hands)

```python
# Run 500,000 duplicate pairs
for pair in range(500000):
    cards = deal_cards()  # 9 distinct random cards
    
    # Pass 1: TiltStack as Small Blind (P0)
    p0_ev_sb = play_match(game, tiltstack, opponent, cards)
    
    # Pass 2: TiltStack as Big Blind (P1) — same cards, swapped seats
    p0_ev_bb = play_match(game, opponent, tiltstack, cards)
    
    total_ev = (p0_ev_sb + p0_ev_bb) / 2
    record(pair, p0_ev_sb, p0_ev_bb)
```

**Why duplicate pairs?**

Poker is high-variance. A single pair (one deal) gives P0 an unfair edge. Running both seats cancels deal-specific variance:

$$\text{Agent edge} = \frac{EV_{sb} + EV_{bb}}{2}$$

Expected value is symmetric under perfect play (EV ≈ 0).

## Test Run Results

Recent evaluation (500,000 deals / 1,000,000 hands):

```
TiltStack edge as SB:   -162.76 mBB/hand
TiltStack edge as BB:   -252.43 mBB/hand
TiltStack overall edge: -207.60 ± 10.20 mBB/hand
```

**Interpretation:**

- **Negative edge** indicates the agent is exploitable
- **Positional disparity** (SB -163 vs BB -252): Agent plays worse out of position, a common poker weakness
- **Standard error ±10.20**: Result is significant

This suggests a decent convergence given the current network arch. Also consider that Neural Best Response is far more punishing than a balanced opponent.

## Match Runner Architecture

The `match_runner.py` script (`src/pysrc/evaluation/match_runner.py`):

1. **Loads networks**: Policy network (`strat_net`), advantage networks (`adv_net[0]`, `adv_net[1]`)
2. **Initializes game**: Creates OpenSpiel game instance with the game string
3. **Spawns agents**:
   - `TiltStack_DeepCFR`: Inference agent (queries networks during play)
   - `Anti_TiltStack_NBR`: Neural Best Response network counter agent
4. **Runs duplicate pairs**: Each pair plays two matches (swapped seats)
5. **Records results**: Payoff per match, aggregated statistics, confidence intervals

### Agent Integration

```python
class TiltStack_DeepCFR:
    def __init__(self, net_dict, clusters_dir):
        self.strat_net = DeepCFRNet(...)
        self.adv_net = [DeepCFRNet(...), DeepCFRNet(...)]
        self.clusters = load_clusters(clusters_dir)  # Labels and centroids
        self.game = CFRGame(...)  # C++ game instance
    
    def step(self, osp_state) -> int:
        """Convert OpenSpiel state → InfoSet → logits → action."""
        # Convert OpenSpiel state to CFRGame representation
        cards = extract_cards(osp_state)
        bucket_state = compute_buckets(cards, self.clusters)
        
        # Construct InfoSet
        info_set = self.game.getInfo(bucket_state)
        
        # Query strategy network
        logits = self.strat_net(info_set)
        
        # Sample action (or take argmax for best play)
        action = sample_action(logits)
        
        # Convert back to OpenSpiel action
        return osp_to_action(action)
```

## Exploitability: Best-Response Approximation

During training, exploitability is estimated using a **best-response agent** (`NLHE_BestResponse.py`).

This is a **one-sided approximation**: BR plays optimally against our current strategy, but *assumes that strategy strategy is fixed* (not co-evolving).

## Baseline Agents

### Best-Response (Strong Baseline)

```python
br_agent = Anti_TiltStack_NBR(opponent_policy)
```

Learns the optimal counterstrategy. Strong, but unrealistic (requires solving).

### Random Agent (Weak Baseline)

```python
def step(self, state):
    legal_actions = state.legal_actions()
    return random.choice(legal_actions)
```

Useful for sanity checks (should lose >500 mBB/hand).