"""Recurrent / attention decoders with per-day input "read-in" layers.

``SeqDecoder`` = [day-specific linear read-in] -> shared core (GRU or causal
Transformer) -> linear readout to 7-D velocity.

* With ``per_day_readin=False`` this is a standard single-read-in GRU/Transformer decoder.
* With ``per_day_readin=True`` each training day gets its own read-in matrix, so the core
  learns a day-invariant latent space ("stitching", as in LFADS multi-session models and
  NDT2/POYO context tokens). A new day is handled by fitting only a fresh read-in on a
  few seconds of labelled recalibration data, with the core frozen.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn


def pick_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CausalTransformerCore(nn.Module):
    """Small causal Transformer encoder with learned positional embeddings."""

    def __init__(self, d: int, layers: int, heads: int, context: int, dropout: float):
        super().__init__()
        self.context = context
        self.pos = nn.Parameter(torch.zeros(1, context, d))
        layer = nn.TransformerEncoderLayer(d, heads, 2 * d, dropout, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, layers)
        self.register_buffer("mask", torch.triu(torch.full((context, context), float("-inf")), 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = x.shape[1]
        return self.enc(x + self.pos[:, -T:], mask=self.mask[:T, :T])


class SeqDecoder(nn.Module):
    """Read-in -> core -> readout velocity decoder."""

    def __init__(self, n_channels: int, n_out: int, n_readins: int = 1, d_in: int = 64,
                 hidden: int = 256, layers: int = 2, core: str = "gru", dropout: float = 0.2,
                 heads: int = 4, context: int = 50):
        super().__init__()
        self.core_type = core
        self.readins = nn.ModuleList([nn.Linear(n_channels, d_in) for _ in range(n_readins)])
        self.drop = nn.Dropout(dropout)
        if core == "gru":
            self.core = nn.GRU(d_in, hidden, layers, batch_first=True, dropout=dropout if layers > 1 else 0.0)
            d_core = hidden
        elif core == "transformer":
            self.proj = nn.Linear(d_in, hidden)
            self.core = CausalTransformerCore(hidden, layers, heads, context, dropout)
            d_core = hidden
        else:
            raise ValueError(core)
        self.readout = nn.Linear(d_core, n_out)

    def embed(self, x: torch.Tensor, readin: int | nn.Module | torch.Tensor) -> torch.Tensor:
        """Apply a read-in: a day index, an explicit layer, or a (B,) tensor of day indices."""
        if isinstance(readin, torch.Tensor):  # per-sample read-ins in one batched matmul
            W = torch.stack([r.weight for r in self.readins])[readin]  # (B, d_in, C)
            b = torch.stack([r.bias for r in self.readins])[readin]  # (B, d_in)
            return self.drop(torch.einsum("btc,bdc->btd", x, W) + b[:, None])
        layer = self.readins[readin] if isinstance(readin, int) else readin
        return self.drop(layer(x))

    def forward(self, x: torch.Tensor, readin: int | nn.Module | torch.Tensor = 0) -> torch.Tensor:
        h = self.embed(x, readin)
        if self.core_type == "gru":
            h, _ = self.core(h)
        else:
            h = self.core(self.proj(h))
        return self.readout(self.drop(h))

    @torch.no_grad()
    def predict(self, z: np.ndarray, readin: int | nn.Module = 0) -> np.ndarray:
        """Causal prediction over a whole block (streaming GRU / sliding-window Transformer)."""
        self.eval()
        dev = next(self.parameters()).device
        x = torch.as_tensor(z, dtype=torch.float32, device=dev)[None]
        if self.core_type == "gru":
            return self(x, readin)[0].cpu().numpy()
        ctx = self.core.context
        T = x.shape[1]
        xp = torch.cat([torch.zeros(1, ctx - 1, x.shape[2], device=dev), x], 1)
        out = []
        for s in range(0, T, 512):  # batch of sliding windows, keep last position
            idx = torch.arange(s, min(s + 512, T), device=dev)
            win = xp[0, idx[:, None] + torch.arange(ctx, device=dev)[None]]
            out.append(self(win, readin)[:, -1])
        return torch.cat(out).cpu().numpy()


@dataclass
class Augment:
    """Nonstationarity augmentation on z-scored inputs (Sussillo et al. 2016 style)."""

    gain_sd: float = 0.0
    offset_sd: float = 0.0
    channel_drop: float = 0.0
    noise_sd: float = 0.0

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        B, _, C = x.shape
        if self.gain_sd > 0:
            x = x * torch.exp(self.gain_sd * torch.randn(B, 1, C, device=x.device))
        if self.offset_sd > 0:
            x = x + self.offset_sd * torch.randn(B, 1, C, device=x.device)
        if self.channel_drop > 0:
            x = x * (torch.rand(B, 1, C, device=x.device) > self.channel_drop)
        if self.noise_sd > 0:
            x = x + self.noise_sd * torch.randn_like(x)
        return x


@dataclass
class TrainCfg:
    window: int = 100
    batch: int = 64
    steps: int = 3000
    lr: float = 2e-3
    weight_decay: float = 1e-2
    eval_every: int = 250
    patience: int = 6
    augment: Augment = field(default_factory=Augment)


def _sample_batch(zs, ys, ms, rid, window, batch, rng, device):
    lens = np.array([len(z) for z in zs], dtype=np.float64)
    which = rng.choice(len(zs), size=batch, p=lens / lens.sum())
    xb, yb, mb, rb = [], [], [], []
    for i in which:
        s = rng.integers(0, max(1, len(zs[i]) - window))
        xb.append(zs[i][s:s + window]); yb.append(ys[i][s:s + window])
        mb.append(ms[i][s:s + window]); rb.append(rid[i])
    t = lambda a, dt=torch.float32: torch.as_tensor(np.stack(a), dtype=dt, device=device)
    return t(xb), t(yb), t(mb, torch.bool), np.array(rb)


def masked_mse(pred: torch.Tensor, y: torch.Tensor, m: torch.Tensor, burn_in: int = 0) -> torch.Tensor:
    m = m.clone()
    m[:, :burn_in] = False
    return ((pred - y) ** 2)[m].mean()


def _forward_multi(model: SeqDecoder, x, rids):
    if len(model.readins) == 1:
        return model(x, 0)
    return model(x, torch.as_tensor(rids, dtype=torch.long, device=x.device))


def train_model(model: SeqDecoder, train: tuple, val: tuple, cfg: TrainCfg, seed: int = 0,
                log=print) -> dict:
    """Train with masked MSE; early-stop on validation loss (streaming evaluation).

    ``train``/``val`` are tuples of lists ``(zs, ys, masks, readin_ids)``.
    """
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    dev = next(model.parameters()).device
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / 200) * 0.5 * (1 + math.cos(math.pi * min(s, cfg.steps) / cfg.steps)))
    best, best_state, bad, hist = float("inf"), None, 0, []
    burn = min(20, cfg.window // 5)
    for step in range(1, cfg.steps + 1):
        model.train()
        x, y, m, r = _sample_batch(*train, cfg.window, cfg.batch, rng, dev)
        # bfloat16 autocast on CUDA (AMP); full precision on CPU
        with torch.autocast(device_type=dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            pred = _forward_multi(model, cfg.augment(x), r)
        loss = masked_mse(pred.float(), y, m, burn)
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % cfg.eval_every == 0 or step == cfg.steps:
            vl = val_loss(model, val)
            hist.append((step, float(loss), vl))
            log(f"  step {step:5d} train {loss.item():.4f} val {vl:.4f}")
            if vl < best - 1e-4:
                best, best_state, bad = vl, copy.deepcopy(model.state_dict()), 0
            else:
                bad += 1
                if bad >= cfg.patience:
                    break
    if best_state is not None:
        model.load_state_dict(best_state)
    return {"best_val": best, "history": hist}


@torch.no_grad()
def val_loss(model: SeqDecoder, val: tuple, readin_override=None) -> float:
    zs, ys, ms, rid = val
    num, den = 0.0, 0
    for z, y, m, r in zip(zs, ys, ms, rid):
        p = model.predict(z, readin_override if readin_override is not None else int(r))
        num += float(((p - y) ** 2)[m].sum())
        den += int(m.sum()) * y.shape[1]
    return num / max(den, 1)


def fit_new_readin(model: SeqDecoder, zs, ys, ms, steps: int = 400, lr: float = 3e-3,
                   window: int = 100, batch: int = 32, anchor: float = 1e-2, seed: int = 0,
                   init: str = "mean") -> nn.Linear:
    """Few-shot recalibration: learn a new read-in for an unseen day, core frozen.

    The read-in is initialised to the average of the trained read-ins and L2-anchored to
    that initialisation, which keeps it well-posed with only seconds of data.
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    dev = next(model.parameters()).device
    new = copy.deepcopy(model.readins[0])
    with torch.no_grad():
        if init == "mean":
            new.weight.copy_(torch.stack([r.weight for r in model.readins]).mean(0))
            new.bias.copy_(torch.stack([r.bias for r in model.readins]).mean(0))
    w0, b0 = new.weight.detach().clone(), new.bias.detach().clone()
    for p in model.parameters():
        p.requires_grad_(False)
    opt = torch.optim.Adam(new.parameters(), lr=lr)
    rid = [0] * len(zs)
    for _ in range(steps):
        window = min(window, min(len(z) for z in zs))
        model.train()
        x, y, m, _ = _sample_batch(zs, ys, ms, rid, window, batch, rng, dev)
        loss = masked_mse(model(x, new), y, m, min(20, window // 5))
        loss = loss + anchor * (((new.weight - w0) ** 2).sum() + ((new.bias - b0) ** 2).sum())
        opt.zero_grad()
        loss.backward()
        opt.step()
    for p in model.parameters():
        p.requires_grad_(True)
    model.eval()
    return new


def finetune_all(model: SeqDecoder, zs, ys, ms, steps: int = 400, lr: float = 5e-4,
                 window: int = 100, batch: int = 32, seed: int = 0) -> SeqDecoder:
    """Baseline recalibration: fine-tune every parameter of a copy of the model."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    m2 = copy.deepcopy(model)
    dev = next(m2.parameters()).device
    opt = torch.optim.AdamW(m2.parameters(), lr=lr, weight_decay=1e-2)
    window = min(window, min(len(z) for z in zs))
    rid = [0] * len(zs)
    for _ in range(steps):
        m2.train()
        x, y, m, _ = _sample_batch(zs, ys, ms, rid, window, batch, rng, dev)
        loss = masked_mse(m2(x, 0), y, m, min(20, window // 5))
        opt.zero_grad()
        loss.backward()
        opt.step()
    m2.eval()
    return m2
