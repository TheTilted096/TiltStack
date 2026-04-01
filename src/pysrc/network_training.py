"""
network_training.py — DeepCFR model definition and per-iteration training.

Three networks share the same architecture (DeepCFRNet):
  - adv_net[0]: advantage network for player 0, trained when hero=False
  - adv_net[1]: advantage network for player 1, trained when hero=True
  - strat_net:  single shared strategy network, trained on opponent samples
                collected during whichever player's traversal just ran.

Each iteration is run for one hero. The traversing player's regrets go to
adv_net[hero]; the opponent's instant strategies go to strat_net. The full
training loop therefore alternates hero=False / hero=True across iterations.

InfoSet decoding
----------------
The raw buffer from C++ is uint8 of shape (N, INFOSET_BYTES). Fields are
extracted using a numpy structured dtype that mirrors the C++ layout exactly.

Street bucket indexing
----------------------
Each street has a separate embedding table (flop: 2049 entries, turn/river:
8193 entries each). Index 0 means "unused street" (padding_idx=0, embedded as
all-zeros); indices 1–N mean actual cluster labels. The FAISS clustering
pipeline assigns labels 0-indexed, so CFRGame stores `label + 1` for reached
streets and leaves unreached streets at their zero-initialised default.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# ---------------------------------------------------------------------------
# Constants — must match CFRTypes.h
# ---------------------------------------------------------------------------

NUM_ACTIONS  = 5
CARD_BITS    = 52   # bits used in each card bitmask
EMBED_DIM    = 32   # street-bucket embedding dimension per street
NUM_STREETS  = 3    # flop, turn, river (no bucket for preflop)

# Each street has its own embedding table.  Flop isomorphism classes are a
# subset of turn classes (the hand-indexer collapses more symmetries at the
# flop), so the flop table is smaller.  Valid labels are 1-indexed (0=unused).
FLOP_BUCKETS  = 2049   # 0 = unused, 1–2048 = flop cluster labels
TURN_BUCKETS  = 8193   # 0 = unused, 1–8192 = turn cluster labels
RIVER_BUCKETS = 8193   # 0 = unused, 1–8192 = river cluster labels

CONT_DIM  = 4 * CARD_BITS + 5 + 4 * 6 + 4 + 1   # 208+5+24+4+1 = 242
INPUT_DIM = CONT_DIM + NUM_STREETS * EMBED_DIM     # 242 + 96     = 338

# ---------------------------------------------------------------------------
# InfoSet structured dtype — mirrors the C++ struct layout byte-for-byte.
# Offsets are verified against sizeof(InfoSet) = INFOSET_BYTES at import time.
# ---------------------------------------------------------------------------

infoset_dtype = np.dtype({
    'names': [
        'hole', 'flop', 'turn', 'river',
        'my_stack', 'opp_stack', 'pot_size', 'to_call', 'current_ehs',
        'bet_hist',
        'street_bucket', 'street_embed', 'is_button',
    ],
    'formats': [
        '<u8', '<u8', '<u8', '<u8',           # uint64 card bitmasks
        '<f4', '<f4', '<f4', '<f4', '<f4',    # normalized scalars + EHS
        ('<f4', (4, 6)),                        # betHist[NUM_ROUNDS][MAX_ACTIONS]
        ('<u2', 3),                             # streetBucket[3]
        ('?',   4),                             # streetEmbed[4]  (bool)
        '?',                                    # isButton         (bool)
    ],
    'offsets': [
         0,  8, 16, 24,   # hole, flop, turn, river
        32, 36, 40, 44,   # my_stack, opp_stack, pot_size, to_call
        48,               # current_ehs
        52,               # bet_hist (96 bytes)
       148,               # street_bucket (6 bytes)
       154,               # street_embed (4 bytes)
       158,               # is_button (1 byte) — struct padded to 160
    ],
    'itemsize': 160,
})


def verify_layout(infoset_bytes: int) -> None:
    """Call once with deepcfr.INFOSET_BYTES to catch C++/Python layout drift."""
    assert infoset_dtype.itemsize == infoset_bytes, (
        f"InfoSet size mismatch: C++ reports {infoset_bytes} bytes, "
        f"Python dtype is {infoset_dtype.itemsize} bytes. "
        "Update infoset_dtype offsets or itemsize."
    )


# ---------------------------------------------------------------------------
# InfoSet decoding
# ---------------------------------------------------------------------------

def decode_batch(raw: np.ndarray):
    """
    Decode a raw InfoSet buffer into tensors ready for the network.

    Parameters
    ----------
    raw : np.ndarray, shape (N, INFOSET_BYTES), dtype uint8

    Returns
    -------
    x_cont  : torch.Tensor, shape (N, CONT_DIM),   dtype float32  (CPU)
    buckets : torch.Tensor, shape (N, NUM_STREETS), dtype int64    (CPU)
    """
    N = raw.shape[0]
    # Zero-copy reinterpretation of the raw bytes as structured records.
    batch = raw.ravel().view(infoset_dtype)  # shape (N,)

    parts = []

    # Card bitmasks: each uint64 → 52 binary floats
    shifts = np.arange(CARD_BITS, dtype=np.uint64)
    for field in ('hole', 'flop', 'turn', 'river'):
        masks = batch[field].astype(np.uint64)                      # (N,)
        bits  = ((masks[:, None] >> shifts[None, :]) & np.uint64(1)).astype(np.float32)
        parts.append(bits)                                          # (N, 52)

    # Normalized scalars
    for field in ('my_stack', 'opp_stack', 'pot_size', 'to_call', 'current_ehs'):
        parts.append(batch[field].astype(np.float32).reshape(N, 1))

    # Betting history: (N, 4, 6) → (N, 24)
    parts.append(batch['bet_hist'].astype(np.float32).reshape(N, 24))

    # Street encoding and button flag
    parts.append(batch['street_embed'].astype(np.float32))          # (N, 4)
    parts.append(batch['is_button'].astype(np.float32).reshape(N, 1))

    x_cont = torch.from_numpy(np.concatenate(parts, axis=1))        # (N, CONT_DIM)

    # Street buckets for the embedding layer (values 0–8192, 0 = unused)
    buckets = torch.from_numpy(batch['street_bucket'].astype(np.int64))  # (N, 3)

    return x_cont, buckets


# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------

class DeepCFRNet(nn.Module):
    """
    Shared architecture for both the advantage and strategy networks.

        Input (338) → Linear(256) → ReLU → Linear(256) → ReLU → Linear(5)

    Street buckets are passed as integer indices (N, 3) — columns are flop,
    turn, river — and embedded into EMBED_DIM-dimensional vectors before
    concatenation with the continuous features.  Each street has its own
    embedding table so the flop table can be smaller (2049 vs 8193 entries).
    Index 0 is treated as a padding token (all-zeros, not trained) in each
    table, representing an unreached street.
    """

    def __init__(self, embed_dim: int = EMBED_DIM):
        super().__init__()
        self.flop_embed  = nn.Embedding(FLOP_BUCKETS,  embed_dim, padding_idx=0)
        self.turn_embed  = nn.Embedding(TURN_BUCKETS,  embed_dim, padding_idx=0)
        self.river_embed = nn.Embedding(RIVER_BUCKETS, embed_dim, padding_idx=0)
        self.net = nn.Sequential(
            nn.Linear(INPUT_DIM, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, NUM_ACTIONS),
        )

    def forward(self, x_cont: torch.Tensor, buckets: torch.Tensor) -> torch.Tensor:
        """
        x_cont  : (N, CONT_DIM)   float32
        buckets : (N, NUM_STREETS) int64   — columns: [flop, turn, river]
        returns : (N, NUM_ACTIONS) float32
        """
        e_flop  = self.flop_embed (buckets[:, 0])  # (N, 32)
        e_turn  = self.turn_embed (buckets[:, 1])  # (N, 32)
        e_river = self.river_embed(buckets[:, 2])  # (N, 32)
        embeds  = torch.cat([e_flop, e_turn, e_river], dim=1)  # (N, 96)
        x = torch.cat([x_cont, embeds], dim=1)                 # (N, 338)
        return self.net(x)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_advantage(
    net: DeepCFRNet,
    optimizer: torch.optim.Optimizer,
    raw_inputs: np.ndarray,    # (N, INFOSET_BYTES) uint8
    targets: np.ndarray,       # (N, NUM_ACTIONS)   float32 — true regrets
    batch_size: int = 4096,
    epochs: int = 1,
    device: torch.device = torch.device('cuda'),
) -> float:
    """
    Train the advantage network with MSE loss on the regret targets.
    Returns the mean loss over the final epoch.
    """
    x_cont, buckets = decode_batch(raw_inputs)
    y = torch.from_numpy(targets)

    dataset = TensorDataset(x_cont, buckets, y)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                         pin_memory=True, num_workers=0)
    net.train()
    last_loss = 0.0

    for _ in range(epochs):
        total = 0.0
        for xc, b, t in loader:
            xc, b, t = xc.to(device), b.to(device), t.to(device)
            optimizer.zero_grad()
            # Illegal actions are stored as NaN; exclude them from the loss so
            # the network does not learn a spurious target of 0 for those slots.
            mask = ~torch.isnan(t)                         # (N, A) bool
            sq_err = ((net(xc, b) - t.nan_to_num(0.0)) * mask) ** 2
            loss = sq_err.sum(dim=1).div(mask.sum(dim=1).clamp(min=1)).mean()
            loss.backward()
            optimizer.step()
            total += loss.item()
        last_loss = total / len(loader)

    return last_loss


def train_policy(
    net: DeepCFRNet,
    optimizer: torch.optim.Optimizer,
    raw_inputs: np.ndarray,    # (N, INFOSET_BYTES) uint8
    targets: np.ndarray,       # (N, NUM_ACTIONS)   float32 — instant strategy
    weights: np.ndarray,       # (N,)               int32   — iteration t
    batch_size: int = 4096,
    epochs: int = 1,
    device: torch.device = torch.device('cuda'),
) -> float:
    """
    Train the strategy network with iteration-weighted cross-entropy loss.

    Targets are probability distributions (instant strategies) produced by
    regret matching in C++.  Cross-entropy H(target, pred) = -sum_a target(a)
    * log softmax(pred(a)) is more appropriate than MSE for distribution
    targets.

    Each sample i contributes weight w_i = weights[i] (the iteration at which
    it was collected). This follows the DeepCFR paper's linear weighting scheme,
    which up-weights more recent strategy samples.

    Returns the mean loss over the final epoch.
    """
    x_cont, buckets = decode_batch(raw_inputs)
    y = torch.from_numpy(targets)
    w = torch.from_numpy(weights.astype(np.float32))

    dataset = TensorDataset(x_cont, buckets, y, w)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                         pin_memory=True, num_workers=0)
    net.train()
    last_loss = 0.0

    for _ in range(epochs):
        total = 0.0
        for xc, b, t, wt in loader:
            xc, b, t, wt = xc.to(device), b.to(device), t.to(device), wt.to(device)
            optimizer.zero_grad()
            # Cross-entropy with soft targets: -sum_a target(a) * log_softmax(pred(a))
            log_probs  = F.log_softmax(net(xc, b), dim=1)
            per_sample = -(t * log_probs).sum(dim=1)
            loss = (wt * per_sample).mean()
            loss.backward()
            optimizer.step()
            total += loss.item()
        last_loss = total / len(loader)

    return last_loss
