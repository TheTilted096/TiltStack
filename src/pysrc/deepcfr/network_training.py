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

import deepcfr
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Constants — must match CFRTypes.h
# ---------------------------------------------------------------------------

NUM_ACTIONS = 5
CARD_BITS = 52  # bits used in each card bitmask
EMBED_DIM = 32  # street-bucket embedding dimension per street
NUM_STREETS = 3  # flop, turn, river (no bucket for preflop)

# Each street has its own embedding table.  Flop isomorphism classes are a
# subset of turn classes (the hand-indexer collapses more symmetries at the
# flop), so the flop table is smaller.  Valid labels are 1-indexed (0=unused).
FLOP_BUCKETS = 2049  # 0 = unused, 1–2048 = flop cluster labels
TURN_BUCKETS = 8193  # 0 = unused, 1–8192 = turn cluster labels
RIVER_BUCKETS = 8193  # 0 = unused, 1–8192 = river cluster labels

CONT_DIM = 4 * CARD_BITS + 5 + 4 * 6 + 4 * 6 + 4 + 1  # 208+5+24+24+4+1 = 266
INPUT_DIM = CONT_DIM + NUM_STREETS * EMBED_DIM  # 266 + 96        = 362

# ---------------------------------------------------------------------------
# InfoSet structured dtype — mirrors the C++ struct layout byte-for-byte.
# Offsets are verified against sizeof(InfoSet) = INFOSET_BYTES at import time.
# ---------------------------------------------------------------------------

infoset_dtype = np.dtype(
    {
        "names": [
            "hole",
            "flop",
            "turn",
            "river",
            "my_stack",
            "opp_stack",
            "pot_size",
            "to_call",
            "current_ehs",
            "bet_hist",
            "bet_hist_mask",
            "street_bucket",
            "street_embed",
            "is_button",
        ],
        "formats": [
            "<u8",
            "<u8",
            "<u8",
            "<u8",  # uint64 card bitmasks
            "<f4",
            "<f4",
            "<f4",
            "<f4",
            "<f4",  # normalized scalars + EHS
            ("<f4", (4, 6)),  # betHist[NUM_ROUNDS][MAX_ACTIONS]
            "<u4",  # betHistMask (uint32, 24 LSBs used)
            ("<u2", 3),  # streetBucket[3]
            ("?", 4),  # streetEmbed[4]  (bool)
            "?",  # isButton         (bool)
        ],
        "offsets": [
            0,
            8,
            16,
            24,  # hole, flop, turn, river
            32,
            36,
            40,
            44,  # my_stack, opp_stack, pot_size, to_call
            48,  # current_ehs
            52,  # bet_hist (96 bytes)
            148,  # bet_hist_mask (4 bytes)
            152,  # street_bucket (6 bytes)
            158,  # street_embed (4 bytes)
            162,  # is_button (1 byte) — struct padded to 168
        ],
        "itemsize": 168,
    }
)


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
    for field in ("hole", "flop", "turn", "river"):
        masks = batch[field].astype(np.uint64)  # (N,)
        bits = ((masks[:, None] >> shifts[None, :]) & np.uint64(1)).astype(np.float32)
        parts.append(bits)  # (N, 52)

    # Normalized scalars
    for field in ("my_stack", "opp_stack", "pot_size", "to_call", "current_ehs"):
        parts.append(batch[field].astype(np.float32).reshape(N, 1))

    # Betting history: (N, 4, 6) → (N, 24)
    parts.append(batch["bet_hist"].astype(np.float32).reshape(N, 24))

    # Betting history mask: unpack 24 LSBs of uint32 → (N, 24) float
    mask = batch["bet_hist_mask"].astype(np.uint32)  # (N,)
    shifts = np.arange(24, dtype=np.uint32)
    parts.append(((mask[:, None] >> shifts[None, :]) & np.uint32(1)).astype(np.float32))

    # Street encoding and button flag
    parts.append(batch["street_embed"].astype(np.float32))  # (N, 4)
    parts.append(batch["is_button"].astype(np.float32).reshape(N, 1))

    x_cont = torch.from_numpy(np.concatenate(parts, axis=1))  # (N, CONT_DIM)

    # Street buckets for the embedding layer (values 0–8192, 0 = unused)
    buckets = torch.from_numpy(batch["street_bucket"].astype(np.int64))  # (N, 3)

    return x_cont, buckets


# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------

HIDDEN_DIM = 1024
NUM_RES_BLOCKS = 4


class ResidualBlock(nn.Module):
    """A standard 2-layer residual block for MLPs."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
        self.fc2 = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # The internal non-linear transformation
        out = F.leaky_relu(self.fc1(x))
        out = self.fc2(out)

        # The skip connection adds the original input back IN FRONT of the final activation
        return F.leaky_relu(out + x)


class DeepCFRNet(nn.Module):
    """
    Shared architecture for both the advantage and strategy networks.

        Input (INPUT_DIM) → Linear(HIDDEN_DIM) → LeakyReLU
                          → ResidualBlock × NUM_RES_BLOCKS
                          → Linear(HIDDEN_DIM → NUM_ACTIONS)

    Street buckets are passed as integer indices (N, 3) — columns are flop,
    turn, river — and embedded into EMBED_DIM-dimensional vectors before
    concatenation with the continuous features.  Each street has its own
    embedding table so the flop table can be smaller (2049 vs 8193 entries).
    Index 0 is treated as a padding token (all-zeros, not trained) in each
    table, representing an unreached street.
    """

    def __init__(self):
        super().__init__()
        self.flop_embed = nn.Embedding(FLOP_BUCKETS, EMBED_DIM, padding_idx=0)
        self.turn_embed = nn.Embedding(TURN_BUCKETS, EMBED_DIM, padding_idx=0)
        self.river_embed = nn.Embedding(RIVER_BUCKETS, EMBED_DIM, padding_idx=0)
        self.linear1 = nn.Linear(INPUT_DIM, HIDDEN_DIM)
        for i in range(1, NUM_RES_BLOCKS + 1):
            setattr(self, f"res_block{i}", ResidualBlock())
        self.output = nn.Linear(HIDDEN_DIM, NUM_ACTIONS)

    def forward(self, x_cont: torch.Tensor, buckets: torch.Tensor) -> torch.Tensor:
        """
        x_cont  : (N, CONT_DIM)   float32
        buckets : (N, NUM_STREETS) int64   — columns: [flop, turn, river]
        returns : (N, NUM_ACTIONS) float32
        """
        e_flop = self.flop_embed(buckets[:, 0])
        e_turn = self.turn_embed(buckets[:, 1])
        e_river = self.river_embed(buckets[:, 2])
        x = torch.cat([x_cont, e_flop, e_turn, e_river], dim=1)

        x = F.leaky_relu(self.linear1(x))
        for i in range(1, NUM_RES_BLOCKS + 1):
            x = getattr(self, f"res_block{i}")(x)
        return self.output(x)


# ---------------------------------------------------------------------------
# Reservoir
# ---------------------------------------------------------------------------


class Reservoir:
    """
    Fixed-capacity reservoir backed by a C++ deepcfr.Reservoir.

    Python allocates the numpy buffers (for zero-copy PyTorch access).
    Worker threads write into them directly via Reservoir.insert() during
    flushBatch(), overlapping data collection with GPU inference and
    eliminating the serial post-iteration collection phase entirely.

    Parameters
    ----------
    capacity     : maximum number of samples to retain
    num_threads  : number of Orchestrator worker threads (determines slices)
    infoset_bytes: width of the raw InfoSet buffer (== deepcfr.INFOSET_BYTES)
    has_weights  : if True, allocate and maintain an int32 weight column
    """

    def __init__(
        self,
        capacity: int,
        num_threads: int,
        infoset_bytes: int,
        has_weights: bool = False,
    ):
        self.capacity = capacity
        self.inputs = np.empty((capacity, infoset_bytes), dtype=np.uint8)
        self.targets = np.empty((capacity, NUM_ACTIONS), dtype=np.float32)
        self.weights = np.empty(capacity, dtype=np.int32) if has_weights else None
        self._cpp = deepcfr.Reservoir(
            capacity, num_threads, self.inputs, self.targets, self.weights
        )

    @property
    def n_seen(self) -> int:
        """Total items ever offered (not clamped to capacity)."""
        return self._cpp.n_seen

    @property
    def size(self) -> int:
        """Number of valid (filled) entries: min(n_seen, capacity)."""
        return self._cpp.size()


def decode_batch_gpu(raw_gpu: torch.Tensor):
    """Decodes a raw (N, 168) uint8 tensor natively on the GPU."""
    N = raw_gpu.shape[0]

    # 1. Cards (Bytes 0-31): 4x 64-bit ints
    cards_i64 = raw_gpu.view(torch.int64)[:, 0:4]
    shifts = torch.arange(52, device=raw_gpu.device, dtype=torch.int64)
    bits = ((cards_i64.unsqueeze(-1) >> shifts) & 1).float()
    bits = bits.view(N, 208)

    # 2. Floats (Bytes 32-147): 29x 32-bit floats
    floats = raw_gpu.view(torch.float32)[:, 8:37]

    # 3. betHistMask (Bytes 148-151): uint32 → unpack 24 LSBs → (N, 24) float
    mask_i32 = raw_gpu.view(torch.int32)[:, 37]  # (N,)
    shifts = torch.arange(24, device=raw_gpu.device, dtype=torch.int32)
    mask_bits = ((mask_i32.unsqueeze(1) >> shifts) & 1).float()  # (N, 24)

    # 4. Bools (Bytes 158-162): streetEmbed[4] + isButton[1]
    bools = raw_gpu[:, 158:163].float()

    # 5. Buckets (Bytes 152-157): 3x 16-bit ints
    buckets = raw_gpu.view(torch.int16)[:, 76:79].long()

    x_cont = torch.cat([bits, floats, mask_bits, bools], dim=1)

    return x_cont, buckets


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train_advantage(
    net: DeepCFRNet,
    optimizer: torch.optim.Optimizer,
    raw_inputs: np.ndarray,  # (N, INFOSET_BYTES) uint8
    targets: np.ndarray,  # (N, NUM_ACTIONS)   float32 — true regrets
    batch_size: int = 4096,
    epochs: int = 1,
    max_steps: int | None = None,
    device: torch.device = torch.device("cuda"),
) -> list[float]:
    """
    Train the advantage network with MSE loss on the regret targets.
    Returns a list of per-epoch mean losses (epoch mode) or a single-element
    list of mean loss over all steps (max_steps mode).

    max_steps: if given, train exactly this many mini-batches, reshuffling
    the dataset whenever it is exhausted.  Ignores the epochs argument.

    Samples are shuffled via torch.randperm each epoch and sliced directly
    from zero-copy numpy views into pre-allocated pinned staging buffers,
    avoiding DataLoader multiprocessing overhead. Loss is accumulated as a
    GPU tensor and synced once per epoch rather than once per batch.
    """
    N = len(raw_inputs)
    raw_pt = torch.from_numpy(raw_inputs)
    tgt_pt = torch.from_numpy(targets)

    # Pinned staging buffers — allocated once, reused every batch.
    raw_buf = torch.empty(
        batch_size, raw_inputs.shape[1], dtype=torch.uint8, pin_memory=True
    )
    tgt_buf = torch.empty(
        batch_size, targets.shape[1], dtype=torch.float32, pin_memory=True
    )

    net.train()

    def _step(idx):
        bs = len(idx)
        raw_buf[:bs].copy_(raw_pt[idx])
        tgt_buf[:bs].copy_(tgt_pt[idx])
        raw_b = raw_buf[:bs].to(device, non_blocking=True)
        t_b = tgt_buf[:bs].to(device, non_blocking=True)
        optimizer.zero_grad()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            xc, b = decode_batch_gpu(raw_b)
            # Illegal actions are stored as NaN; exclude them from the loss so
            # the network does not learn a spurious target of 0 for those slots.
            mask = ~torch.isnan(t_b)
            sq_err = ((net(xc, b) - t_b.nan_to_num(0.0)) * mask) ** 2
            loss = sq_err.sum(dim=1).div(mask.sum(dim=1).clamp(min=1)).mean()
        loss.backward()
        optimizer.step()
        return loss.detach()

    if max_steps is not None:
        total = torch.zeros(1, device=device)
        steps_done = 0
        perm = torch.randperm(N)
        pos = 0
        while steps_done < max_steps:
            if pos >= N:
                perm = torch.randperm(N)
                pos = 0
            idx = perm[pos : pos + batch_size]
            pos += batch_size
            total += _step(idx)
            steps_done += 1
        return [(total / steps_done).item()]

    losses = []
    for _ in range(epochs):
        perm = torch.randperm(N)
        total = torch.zeros(1, device=device)
        n_batches = 0
        for start in range(0, N, batch_size):
            idx = perm[start : start + batch_size]
            total += _step(idx)
            n_batches += 1
        losses.append((total / n_batches).item())

    return losses


def train_policy(
    net: DeepCFRNet,
    optimizer: torch.optim.Optimizer,
    raw_inputs: np.ndarray,  # (N, INFOSET_BYTES) uint8
    targets: np.ndarray,  # (N, NUM_ACTIONS)   float32 — instant strategy
    weights: np.ndarray,  # (N,)               int32   — iteration t
    batch_size: int = 4096,
    epochs: int = 1,
    device: torch.device = torch.device("cuda"),
    epoch_callback=None,  # called as epoch_callback(ep, loss, secs) after each epoch
    step_callback=None,  # called as step_callback(global_step, loss) every step_callback_freq steps
    step_callback_freq: int = 100,
) -> list[float]:
    """
    Train the strategy network with iteration-weighted cross-entropy loss.

    Targets are probability distributions (instant strategies) produced by
    regret matching in C++.  Cross-entropy H(target, pred) = -sum_a target(a)
    * log softmax(pred(a)) is more appropriate than MSE for distribution
    targets.

    Each sample i contributes weight w_i = weights[i] (the iteration at which
    it was collected). This follows the DeepCFR paper's linear weighting scheme,
    which up-weights more recent strategy samples.

    Returns per-epoch mean losses.
    """
    import time as _time

    N = len(raw_inputs)
    raw_pt = torch.from_numpy(raw_inputs)
    tgt_pt = torch.from_numpy(targets)
    wt_pt = torch.from_numpy(weights.astype(np.float32))

    # Pinned staging buffers — allocated once, reused every batch.
    raw_buf = torch.empty(
        batch_size, raw_inputs.shape[1], dtype=torch.uint8, pin_memory=True
    )
    tgt_buf = torch.empty(
        batch_size, targets.shape[1], dtype=torch.float32, pin_memory=True
    )
    wt_buf = torch.empty(batch_size, dtype=torch.float32, pin_memory=True)

    net.train()
    losses = []
    global_step = 0

    for ep in range(1, epochs + 1):
        ep_t0 = _time.perf_counter()
        perm = torch.randperm(N)
        total = torch.zeros(1, device=device)
        n_batches = 0
        for start in range(0, N, batch_size):
            idx = perm[start : start + batch_size]
            bs = len(idx)
            raw_buf[:bs].copy_(raw_pt[idx])
            tgt_buf[:bs].copy_(tgt_pt[idx])
            wt_buf[:bs].copy_(wt_pt[idx])
            raw_b = raw_buf[:bs].to(device, non_blocking=True)
            t_b = tgt_buf[:bs].to(device, non_blocking=True)
            wt_b = wt_buf[:bs].to(device, non_blocking=True)
            optimizer.zero_grad()
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                xc, b = decode_batch_gpu(raw_b)
                # Normalise weights within the mini-batch so they sum to 1,
                # preventing the loss magnitude from scaling with iteration number.
                wt_b = wt_b / wt_b.mean()
                # Cross-entropy with soft targets: -sum_a target(a) * log_softmax(pred(a))
                log_probs = F.log_softmax(net(xc, b), dim=1)
                per_sample = -(t_b * log_probs).sum(dim=1)
                loss = (wt_b * per_sample).mean()
            loss.backward()
            optimizer.step()
            total += loss.detach()
            n_batches += 1
            global_step += 1
            if step_callback is not None and global_step % step_callback_freq == 0:
                step_callback(global_step, loss.item())
        mean_loss = (total / n_batches).item()
        losses.append(mean_loss)
        if epoch_callback is not None:
            epoch_callback(ep, mean_loss, _time.perf_counter() - ep_t0)

    return losses
