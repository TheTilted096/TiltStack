# DeepCFR Traversal Algorithm

DeepCFR is a neural-network-accelerated variant of Counterfactual Regret Minimization (CFR) for solving imperfect-information games. This document describes the game traversal algorithm, the coroutine-based architecture that enables efficient GPU batching, and how neural networks integrate into the CFR loop.

## Overview

The core idea is to replace full game-tree CFR with a learned approximation: instead of computing regrets from scratch at every information set, we use neural networks to predict them. This reduces the effective traversal cost while maintaining convergence guarantees under self-play.

**Key differences from vanilla CFR:**

- **Advantage networks** (`adv_net[0]`, `adv_net[1]`): Predict regrets at information sets during traversal
- **Strategy network** (`strat_net`): Learns opponent's strategies from self-play samples
- **Alternating updates**: Players are updated separately (P0 on `hero=False` passes, P1 on `hero=True` passes)
- **Coroutine-based rollouts**: Game traversals yield at neural inference points, allowing batched GPU execution

## Game State Representation

The CFRGame class (`cppsrc/deepcfr/CFRGame.h`) represents a single path through the game tree:

### Mutable State (Modified by makeMove/unmakeMove)

```cpp
int ply;                           // Current depth in game tree
int currentRound;                  // Street: PREFLOP, FLOP, TURN, RIVER
std::array<int, 2> stacks;         // Remaining stack sizes (milli-chips)
std::array<uint8_t, NUM_ROUNDS> actionCount;  // Actions taken per street
float betHist[NUM_ROUNDS][MAX_ACTIONS];  // Bet amounts (for pot-fraction abstraction)
```

### Immutable State (Set at begin/beginWithCards, Read-Only After)

```cpp
std::array<Card, 9> hole cards and board cards;  // Indexed once via CFRGame::indexCards()
hand_index_t streetIDs[NUM_ROUNDS][2];  // Isomorphic hand indices (precomputed)
uint16_t streetBucket[NUM_ROUNDS][2];   // Cluster labels for each street (precomputed)
float streetEHS[NUM_ROUNDS][2];         // Per-state EHS values (precomputed)
```

## Information Set Encoding

At each decision point, `CFRGame::getInfo()` returns an `InfoSet` — a compact binary representation of the current state:

```cpp
struct InfoSet {
    uint8_t data[INFOSET_BYTES];  // Layout: card bits | action history | pot state
};
```

The C++ layout is mirrored exactly in Python via a NumPy structured dtype. This allows the network to consume raw bytes without any overhead.

**Fields (108 bytes total):**

| Field | Size | Meaning |
|-------|------|---------|
| `private_cards` | 4 bytes × 2 | Bitmask of hole cards (player-specific) |
| `board_cards` | 2 bytes | Bitmask of 5 board cards (if reached) |
| `action_history` | 5 bytes × NUM_ROUNDS | Action labels per street |
| `street_buckets` | 2 bytes × NUM_ROUNDS | Cluster indices (1-indexed; 0 = unreached) |
| `pot_state` | 24 bytes | Stack sizes, committed amounts, action sequence |

Python receives this as a `(N, 108)` numpy array of uint8, decoded via structured dtype.

## Traversal: The Core Loop

The traversal algorithm (`DeepCFR::rollout()`, `DeepCFR::traverse()`) is implemented as C++20 coroutines:

```cpp
Task<float> DeepCFR::rollout(CFRGame game, bool hero, int t, Scheduler &sched)
{
    // game is owned by the coroutine frame; traverse() receives a reference
    co_return co_await DeepCFR::traverse(game, hero, t, sched);
}

Task<float> DeepCFR::traverse(CFRGame &game, bool hero, int t, Scheduler &sched)
{
    // Terminal state: return payoff immediately
    if (game.isTerminal) {
        co_return game.payout();
    }

    InfoSet input = game.getInfo();
    bool isHero = (game.stm() == hero);

    if (isHero) {
        // =========== HERO (learning player) ===========
        // 1. Request regrets from advantage network
        Regrets regrets = co_await InferenceAwaitable{input, sched};

        // 2. Convert regrets → strategy (CFR+ updates)
        Strategy strat = getInstantStrat(regrets, game.moves, game.numMoves);

        // 3. Sample action from strategy
        Action action = sampleAction(strat, game.moves, game.numMoves);

        // 4. Recursively evaluate child nodes
        game.makeMove(action);
        float childEV = co_await DeepCFR::traverse(game, hero, t, sched);
        game.unmakeMove();

        // 5. Return child EV as payoff (regrets are updated during training)
        co_return childEV;

    } else {
        // =========== OPPONENT (instant strategy) ===========
        // 1. Request opponent's strategy from strategy network
        Strategy strat = co_await InferenceAwaitable{input, sched};  // treated as logits

        // 2. Sample from opponent's strategy
        Action action = sampleAction(strat, game.moves, game.numMoves);

        // 3. Recursively evaluate (no regret backprop needed)
        game.makeMove(action);
        float childEV = co_await DeepCFR::traverse(game, hero, t, sched);
        game.unmakeMove();

        co_return childEV;
    }
}
```

## The Coroutine Architecture

C++20 coroutines enable **asynchronous game traversal with automatic batching**:

1. **Suspension**: When `co_await InferenceAwaitable` is encountered, the coroutine yields control and registers its InfoSet with the scheduler.
2. **Batching**: The scheduler collects many suspended coroutines, concatenates their InfoSets, and sends a single batch to the GPU.
3. **Resumption**: Once the GPU returns logits/regrets, each coroutine is resumed in order, reads its result, and continues traversal.

This is fundamentally more efficient than:
- **Naive batching** (collect decisions, freeze at batch size, process) — forces games into lockstep
- **Per-decision queuing** (one network call per decision point) — massive GPU underutilization

**Pool allocation** (`Coroutine.h`): Coroutine frames are allocated from a pre-warmed pool of ~1000 frames per thread. When a frame is destroyed, it's linked back onto the free list — amortized O(1) allocation/deallocation with no locks.

## Regrets and Strategy

At each information set, the network predicts **regrets** `[r_1, ..., r_k]` for each action.

The **positive regret** (CFR+ style):

$$\text{regret}_i^+ = \max(0, r_i)$$

is normalized to form a probability distribution:

$$\sigma_i = \frac{(\text{regret}_i^+)^\alpha}{\sum_j (\text{regret}_j^+)^\alpha}$$

where `alpha` controls how much the strategy favors high-regret actions. In practice:
- `alpha = 1`: proportional to regrets (standard CFR+)
- `alpha = 2`: stronger emphasis on high-regret actions

The **instant strategy** (what the opponent plays during a single rollout) is sampled once at each decision point. This differs from accumulated strategy (which averages strategies across all iterations), enabling faster convergence.

## Training Loop Integration

Each DeepCFR iteration (`iteration t`):

1. **P0 rollout pass** (`hero=False`):
   - Spawn POOL_SIZE coroutines, each running a game
   - P0 (hero) rollouts collect **advantage data**: (InfoSet, regrets used, action sampled, outcome)
   - P0's opponent (P1) samples from `strat_net`
   - Coroutines suspend at every decision point; scheduler batches and queries the networks

2. **Train advantage network for P0**:
   - Minimize MSE loss between predicted regrets and realized counter-factual values
   - This is a bootstrapped loss: the value comes from downstream nodes (children), not a separate value network

3. **Flush P0 regrets** to accumulator:
   - Regrets are summed across rollouts for each unique InfoSet
   - Accumulated regrets drive the next iteration's strategy

4. **P1 rollout pass** (`hero=True`):
   - Mirror of P0 pass; now P1 is the learning player

5. **Train strategy network**:
   - Minimize cross-entropy between predicted logits and opponent's instant sampled actions
   - Opponent data is collected during *both* P0 and P1 rollout passes

## Performance Characteristics

- **Traversal depth**: ~4 ply per rollout (HUNL structure: preflop, flop, turn, river)
- **Branching factor**: ~5 actions per decision point (CHECK, CALL, BET50, BET100, ALLIN)
- **Coroutine overhead**: < 5% of total GPU time (frame allocation/deallocation is negligible)
- **GPU utilization**: ~95% (batching minimizes gaps between GPU kernel launches)
- **Time Per Iteration**: 3 minutes (depends on hardware and net size)

