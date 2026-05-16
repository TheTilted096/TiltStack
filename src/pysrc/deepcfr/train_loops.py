"""
Training kernels for DeepCFR.

This module intentionally contains only optimizer-facing loops and scheduler
math. Process orchestration, rollout generation, logging, and user control live
elsewhere so the same kernels can later be used from single-process or DDP
workers.
"""

import math
import time

import numpy as np
import torch
import torch.nn.functional as F

from network_training import DeepCFRNet, decode_batch_gpu


SHUFFLE_BLOCK_SIZE = 1_048_576


def shuffled_minibatches(N: int, batch_size: int):
    block_size = max(batch_size, min(SHUFFLE_BLOCK_SIZE, N))
    n_blocks = math.ceil(N / block_size)
    for block in torch.randperm(n_blocks):
        start = int(block) * block_size
        end = min(start + block_size, N)
        perm = torch.randperm(end - start).add_(start)
        for pos in range(0, end - start, batch_size):
            yield perm[pos : pos + batch_size]


def train_advantage(
    net: DeepCFRNet,
    optimizer: torch.optim.Optimizer,
    raw_inputs: np.ndarray,
    targets: np.ndarray,
    batch_size: int = 4096,
    steps: int = 2500,
    device: torch.device = torch.device("cuda"),
) -> list[float]:
    N = len(raw_inputs)
    if N == 0:
        raise ValueError("train_advantage requires at least one sample")
    if steps <= 0:
        raise ValueError("steps must be positive")

    raw_pt = torch.from_numpy(raw_inputs)
    tgt_pt = torch.from_numpy(targets)

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
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            xc, b = decode_batch_gpu(raw_b)
            mask = ~torch.isnan(t_b)
            sq_err = ((net(xc, b) - t_b.nan_to_num(0.0)) * mask) ** 2
            loss = sq_err.sum(dim=1).div(mask.sum(dim=1).clamp(min=1)).mean()
        loss.backward()
        optimizer.step()
        return loss.detach()

    total = torch.zeros(1, device=device)
    steps_done = 0
    while steps_done < steps:
        for idx in shuffled_minibatches(N, batch_size):
            total += _step(idx)
            steps_done += 1
            if steps_done >= steps:
                break

    return [(total / steps).item()]


def policy_trainer(
    net: DeepCFRNet,
    optimizer: torch.optim.Optimizer,
    raw_inputs: np.ndarray,
    targets: np.ndarray,
    weights: np.ndarray,
    batch_size: int = 4096,
    device: torch.device = torch.device("cuda"),
):
    N = len(raw_inputs)
    raw_pt = torch.from_numpy(raw_inputs)
    tgt_pt = torch.from_numpy(targets)
    wt_pt = torch.from_numpy(np.asarray(weights, dtype=np.float32))

    raw_buf = torch.empty(
        batch_size, raw_inputs.shape[1], dtype=torch.uint8, pin_memory=True
    )
    tgt_buf = torch.empty(
        batch_size, targets.shape[1], dtype=torch.float32, pin_memory=True
    )
    wt_buf = torch.empty(batch_size, dtype=torch.float32, pin_memory=True)

    net.train()
    ep = 0
    while True:
        ep += 1
        ep_t0 = time.perf_counter()
        total = torch.zeros(1, device=device)
        n_batches = 0
        for idx in shuffled_minibatches(N, batch_size):
            bs = len(idx)
            raw_buf[:bs].copy_(raw_pt[idx])
            tgt_buf[:bs].copy_(tgt_pt[idx])
            wt_buf[:bs].copy_(wt_pt[idx])
            raw_b = raw_buf[:bs].to(device, non_blocking=True)
            t_b = tgt_buf[:bs].to(device, non_blocking=True)
            wt_b = wt_buf[:bs].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                xc, b = decode_batch_gpu(raw_b)
                wt_b = wt_b / wt_b.mean()
                legal = ~torch.isnan(t_b)
                log_probs = F.log_softmax(
                    net(xc, b).masked_fill(~legal, float("-inf")), dim=1
                )
                # Illegal slots have target 0 and log_prob -inf; mask log_probs
                # before multiplying to avoid IEEE 0 * -inf -> NaN.
                per_sample = -(
                    t_b.nan_to_num(0.0) * log_probs.masked_fill(~legal, 0.0)
                ).sum(dim=1)
                loss = (wt_b * per_sample).mean()
            loss.backward()
            optimizer.step()
            total += loss.detach()
            n_batches += 1
        yield ep, (total / n_batches).item(), time.perf_counter() - ep_t0


def cosine_lr(ep: int, base_ep: int, base_lr: float, total_ep: int) -> float:
    """LR for epoch `ep` in a cosine segment that decays to 0 at `total_ep`."""
    remaining = total_ep - base_ep
    if remaining <= 0:
        return 0.0
    return 0.5 * base_lr * (1.0 + math.cos(math.pi * (ep - base_ep) / remaining))
