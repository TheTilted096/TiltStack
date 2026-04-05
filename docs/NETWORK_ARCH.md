# Network Architecture

This document describes the neural network architectures used in DeepCFR for predicting regrets (advantage networks) and opponent strategies (strategy network). All three networks share the same architecture (`DeepCFRNet`).

## Overview

**Three networks, one architecture:**

| Network | Player | Purpose | Loss | Labels |
|---------|--------|---------|------|--------|
| `adv_net[0]` | P0 (Small Blind) | Predict regrets at P0's information sets | MSE | Counter-factual values |
| `adv_net[1]` | P1 (Big Blind) | Predict regrets at P1's information sets | MSE | Counter-factual values |
| `strat_net` | Shared | Predict opponent's instant strategy | Cross-entropy | Opponent action samples |

All networks output **5 logits** (one per action: CHECK, CALL, BET50, BET100, ALLIN).

## Input: InfoSet Encoding → 338 Floats

Networks consume a 160-byte `InfoSet` struct (from `CFRTypes.h`), which is decoded into **338 float features**:

```cpp
struct InfoSet {
    uint64_t hole, flop, turn, river;           // 32 bytes: card bitmasks
    float myStack, oppStack, potSize, toCall;   // 16 bytes: normalized scalars
    float currentEHS;                           // 4 bytes: equity strength
    float betHist[NUM_ROUNDS][MAX_ACTIONS];     // 96 bytes: betting history (4 rounds × 6 actions)
    std::array<uint16_t, NUM_ROUNDS - 1> streetBucket;  // 6 bytes: cluster indices (flop/turn/river)
    std::array<bool, NUM_ROUNDS> streetEmbed;   // 4 bytes: current street one-hot
    bool isButton;                              // 1 byte: position flag
};  // Total: 160 bytes (padded)
```

### Continuous Features: 242 Floats

Decoded from the C++ InfoSet struct into continuous floats:

| Feature | Encoding | Floats | Calculation |
|---------|----------|--------|-------------|
| Card bitmasks | 4 × uint64 → binary | **208** | hole, flop, turn, river: 4 × 52 bits each |
| Normalized scalars | 5 × float32 | **5** | myStack, oppStack, potSize, toCall, currentEHS |
| Betting history | 4×6 float32 | **24** | betHist[NUM_ROUNDS][MAX_ACTIONS] flattened |
| Street embedding | 4 × bool → float | **4** | streetEmbed (one-hot current round) |
| Button flag | 1 × bool → float | **1** | isButton |
| **Continuous subtotal** | — | **242** | `CONT_DIM = 208 + 5 + 24 + 4 + 1` |

### Street Bucket Embeddings: 96 Floats

Three street-specific embedding tables convert discrete cluster indices to dense 32-dimensional vectors:

| Street | Vocab Size | Embed Dim | Floats |
|--------|-----------|-----------|--------|
| Flop | 2,049 (0=unused, 1–2048=labels) | 32 | 32 |
| Turn | 8,193 (0=unused, 1–8192=labels) | 32 | 32 |
| River | 8,193 (0=unused, 1–8192=labels) | 32 | 32 |
| **Embedding subtotal** | — | — | **96** |

Index 0 is marked `padding_idx=0` in PyTorch, so unreached streets (bucket=0) embed as zero vectors, reducing information leakage.

### Total Network Input: 338 Floats

```python
# From network_training.py
CARD_BITS = 52
CONT_DIM = 4 * CARD_BITS + 5 + 4 * 6 + 4 + 1
         = 208 + 5 + 24 + 4 + 1
         = 242                          # Continuous features

EMBED_DIM = 32                          # Per-street embedding dimension
NUM_STREETS = 3                         # Flop, turn, river

INPUT_DIM = CONT_DIM + NUM_STREETS * EMBED_DIM
          = 242 + 3 * 32
          = 242 + 96
          = 338                         # Total network input floats
```

The **338 float32 values** are concatenated before passing to the first dense layer.

## Architecture: DeepCFRNet

A **feedforward network with street-specific embeddings**:

```
InfoSet (160 bytes) → Decode continuous features (242 floats)
                     + Embed street buckets (96 floats)
                            ↓
                    Concatenate → 338-dim vector
                            ↓
                    Dense (338 → 512) + ReLU
                            ↓
                    Dense (512 → 256) + ReLU
                            ↓
                    Dense (256 → 256) + ReLU
                            ↓
                    Dense (256 → 5) [logits]
                            ↓
                    Output (5 logits)
```

### Embedding Tables

Three separate embedding tables (one per street):

| Street | Clusters | Embedding Dim | Total Parameters |
|--------|----------|---------------|------------------|
| Flop | 2,049 (0=unused, 1–2048=labels) | 32 | 65,568 |
| Turn | 8,193 (0=unused, 1–8192=labels) | 32 | 262,176 |
| River | 8,193 (0=unused, 1–8192=labels) | 32 | 262,176 |

**Index 0 is marked `padding_idx=0`** in PyTorch, so unreached streets (where bucket=0) are embedded as all-zero vectors. This reduces information leakage and improves generalization.

### Continuous Feature Extraction

Raw InfoSet bytes are decoded into continuous floats via the `decode_batch()` function in `network_training.py`:

```python
# Card bitmasks: each uint64 → 52 binary floats
for field in ('hole', 'flop', 'turn', 'river'):
    bits = ((masks >> arange(52)) & 1).astype(float32)  # (N, 52) each

# Normalized scalars (already float32)
scalars = [myStack, oppStack, potSize, toCall, currentEHS]  # (N, 5)

# Betting history: flattened
bet_hist = betHist[4, 6].reshape(N, 24)  # (N, 24)

# Street encoding and button
street_embed = streetEmbed.astype(float32)              # (N, 4)
is_button = isButton.astype(float32).reshape(N, 1)     # (N, 1)

# Concatenate: 52×4 + 5 + 24 + 4 + 1 = 242 floats per sample
x_cont = np.concatenate([bits_hole, bits_flop, bits_turn, bits_river,
                         scalars, bet_hist, street_embed, is_button], axis=1)
# Shape: (N, 242)

# Street buckets for embedding lookup
buckets = batch['street_bucket'].astype(int64)  # (N, 3)
embeddings = embedding_table(buckets)           # (N, 3, 32) → (N, 96)

# Final input
x = torch.cat([x_cont, embeddings], dim=1)  # (N, 338)
```

### Hidden Layers

Two hidden layers (512 → 256 → 256), each with:
- ReLU activation
- No dropout (small network; low overfitting risk on self-play data)
- No batch normalization (coroutine batches are ephemeral; norm statistics don't stabilize)

### Output

**5 logits** (one per action), no softmax. During training, raw logits feed into MSE loss (advantage networks) or cross-entropy loss (strategy network). During inference, logits are exponentiated to recover action probabilities.

## Training Details

### Advantage Network Loss

**Input:** InfoSet (current player's private view)  
**Label:** Counter-factual value (CFV) — the expected payoff from this state, discounted by reach probability

$$\text{CFV}_i = P(\text{reach}) \times \text{(outcome of rollout)}$$

**Loss:** MSE

$$\mathcal{L}_{\text{adv}} = \frac{1}{N} \sum_{n=1}^N \| \text{logits}_n - \text{CFV}_n \|_2^2$$

Regrets are derived post-hoc as `regrets = logits - value_of_best_action`.

### Strategy Network Loss

**Input:** InfoSet (opponent's private view)  
**Label:** Action sampled by opponent during this rollout

**Loss:** Cross-entropy

$$\mathcal{L}_{\text{strat}} = \frac{1}{N} \sum_{n=1}^N -\log P(\text{action}_n | \text{logits}_n)$$

where $P(a | \text{logits})$ is the softmax over logits.

### Optimization

- **Optimizer:** Adam (β₁=0.9, β₂=0.999, ε=1e-8)
- **Learning rate:** 1e-3 (fixed; no scheduling)
- **Batch size:** ~2000 samples per epoch
- **Epochs per iteration:** 5–10 (until convergence or max epochs reached)
- **Device:** CUDA GPU (required; CPU training is impractical)

## Inference Integration

During rollout, the coroutine suspends at each decision point, passing the InfoSet to the scheduler. The scheduler:

1. Collects raw InfoSet buffers from many suspended coroutines (typically 100–1000)
2. Decodes into continuous features (242 floats) and street bucket embeddings (96 floats)
3. Concatenates into a tensor of shape `(batch_size, 338)`
4. Passes through the appropriate network (`adv_net[hero]` or `strat_net`)
5. Receives 5 logits per InfoSet
6. Distributes results back to resuming coroutines

This batching is **transparent to the game traversal logic** and achieves ~95% GPU utilization. Decoding (`decode_batch()`) is implemented in NumPy and runs on CPU while the GPU processes the network forward pass.

## Model Size

**Embedding tables:**
- Flop: 2,049 × 32 = 65,568 parameters
- Turn: 8,193 × 32 = 262,176 parameters
- River: 8,193 × 32 = 262,176 parameters
- **Subtotal:** ~590k parameters

**Dense layers:**
- 338 → 512: 338 × 512 + 512 = 173,568 parameters
- 512 → 256: 512 × 256 + 256 = 131,328 parameters
- 256 → 256: 256 × 256 + 256 = 65,792 parameters
- 256 → 5: 256 × 5 + 5 = 1,285 parameters
- **Subtotal:** ~372k parameters

**Total per network:** ~962k parameters
**Total for all three networks:** ~2.9M parameters

Modest size enables:
- Rapid training (~5–10 epochs per iteration)
- Low memory footprint (fits in GPU VRAM alongside batch)
- Negligible overfitting risk on self-play data

## Initialization

- **Embeddings:** Uniform random (±√k where k is embedding dim)
- **Dense layers:** Kaiming uniform (automatically scales to layer width)
- **Biases:** Zero

Networks are initialized afresh before each training run; no pre-training or transfer learning.

