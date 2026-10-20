"""Closed-loop decoder adaptation (CLDA) in the simulator, ReFIT-style (Gilja et al. 2012).

1. Run closed-loop sessions with the current decoder.
2. Re-label: the user's intended velocity is assumed to point from the cursor to the target, so
   each decoded velocity is rotated toward the target (speed kept) and set to zero while the
   cursor is on the target (Gilja et al. 2012's "intention estimation"). No access to the
   simulator's true intention is used.
3. Re-fit (linear decoders) or fine-tune (neural decoders) on the re-labelled data.
4. Repeat for a few rounds, accumulating data.

For neural decoders the fine-tuning is deliberately small: the GRU fine-tunes all weights with a
small learning rate; SLD fine-tunes only its day adapter (A_k, b_k) of the simulated day, which is
the part designed to absorb day- and context-specific changes, with an L2 penalty toward its
starting value. Both optionally rehearse a fraction of the offline calibration data so that
closed-loop adaptation does not erase offline accuracy.
"""
from __future__ import annotations

import copy
import dataclasses

import numpy as np
import torch
import torch.nn.functional as F

from .models import StableLatentDecoder
from .train import TorchDecoder


def refit_labels(runs: list, side_mm: float) -> list[tuple[np.ndarray, np.ndarray]]:
    """ReFIT intention estimate for each closed-loop session -> (counts, velocity) segments."""
    segs = []
    for r in runs:
        to_t = r.target_pos - r.cursor
        dist = np.linalg.norm(to_t, axis=1, keepdims=True)
        speed = np.linalg.norm(r.decoded, axis=1, keepdims=True)
        y = speed * to_t / np.maximum(dist, 1e-6)
        cell = side_mm / r.grid
        y[np.max(np.abs(to_t), axis=1) < cell / 2] = 0
        segs.append((r.counts, y.astype(np.float32)))
    return segs


@dataclasses.dataclass
class CLDAConfig:
    rounds: int = 3
    sessions_per_round: int = 4
    seconds: float = 60.0
    steps: int = 200
    batch: int = 16
    window: int = 64
    warmup_bins: int = 24
    lr_gru: float = 3e-4
    lr_adapter: float = 1e-3
    adapter_l2: float = 0.01
    rehearsal: float = 0.5  # fraction of each batch drawn from offline calibration data
    sld_mode: str = "adapter"  # "adapter" (adapter only) or "adapter_head" (+ velocity head)
    seed: int = 0


def finetune_closed_loop(dec: TorchDecoder, segs: list[tuple[np.ndarray, np.ndarray]], cfg: CLDAConfig,
                         offline: list[tuple[np.ndarray, np.ndarray]] | None = None) -> TorchDecoder:
    """Fine-tune a neural decoder on re-labelled closed-loop segments (plus optional rehearsal).

    ``offline`` are (counts, velocity) segments of the decoder's own calibration day, normalised
    with the decoder's statistics like the closed-loop counts. Returns a new decoder."""
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    m = copy.deepcopy(dec.model)
    is_sld = isinstance(m, StableLatentDecoder)
    if is_sld:
        names = ["adapt_w", "adapt_b"] + (["vel_head.weight", "vel_head.bias"] if cfg.sld_mode == "adapter_head" else [])
        for n, p in m.named_parameters():
            p.requires_grad = n in names
        A0, b0 = m.adapt_w.data[dec.day].clone(), m.adapt_b.data[dec.day].clone()
        lr = cfg.lr_adapter
    else:
        lr = cfg.lr_gru
    params = [p for p in m.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=lr)
    m.eval()  # dropout off: tiny fine-tuning, deterministic
    def prep(S):
        return [(((X - dec.mu) / dec.sd).astype(np.float32), (Y / dec.vscale).astype(np.float32)) for X, Y in S
                if len(X) > cfg.window]
    cl, off = prep(segs), prep(offline or [])
    w = cfg.warmup_bins
    day = torch.full((cfg.batch,), dec.day, dtype=torch.long)
    for _ in range(cfg.steps):
        xs, ys = [], []
        for i in range(cfg.batch):
            pool = off if (off and rng.random() < cfg.rehearsal) else cl
            X, Y = pool[rng.integers(len(pool))]
            s = rng.integers(0, len(X) - cfg.window)
            xs.append(X[s : s + cfg.window])
            ys.append(Y[s : s + cfg.window])
        x, y = torch.as_tensor(np.stack(xs)), torch.as_tensor(np.stack(ys))
        if is_sld:
            vel, _, _ = m(x, day)
            loss = F.mse_loss(vel[:, w:], y[:, w:])
            loss = loss + cfg.adapter_l2 * (((m.adapt_w[dec.day] - A0) ** 2).sum() + ((m.adapt_b[dec.day] - b0) ** 2).sum())
        else:
            vel, _ = m(x)
            loss = F.mse_loss(vel[:, w:], y[:, w:])
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
    for p in m.parameters():
        p.requires_grad = True
    return TorchDecoder(m.eval(), dec.day, dec.mu, dec.sd, dec.vscale)
