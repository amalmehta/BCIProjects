"""Neural-network decoders.

* :class:`GRUDecoder`  - recurrent baseline (one shared input layer for all days).
* :class:`StableLatentDecoder` (SLD) - the proposed model: a per-day *channel adapter*
  (identity-initialised C x C affine map) feeds a shared read-in and a causal backbone (GRU by
  default, banded transformer as an option) that models latent dynamics; two heads read out
  (i) intended velocity and (ii) per-day Poisson log-rates for masked-channel reconstruction
  (self-supervised). Only the adapter and the read-out gain/offset are day-specific, so a new
  day is handled by fitting those, with or without labels.

All models are strictly causal: the output at bin t depends only on bins <= t.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalConv(nn.Module):
    """Depth-preserving causal 1-D convolution over time on (B, T, D) tensors."""

    def __init__(self, d: int, k: int):
        super().__init__()
        self.k = k
        self.conv = nn.Conv1d(d, d, k)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(x.transpose(1, 2), (self.k - 1, 0))
        return self.conv(x).transpose(1, 2)


def alibi_band_bias(T: int, n_heads: int, window: int, device=None) -> torch.Tensor:
    """Additive attention bias (H, T, T): causal, limited to ``window`` past bins, with
    ALiBi linear distance penalties (Press et al. 2022) so any sequence length works."""
    i = torch.arange(T, device=device)
    dist = (i[:, None] - i[None, :]).float()
    slopes = torch.tensor([2 ** (-8 * (h + 1) / n_heads) for h in range(n_heads)], device=device)
    bias = -slopes[:, None, None] * dist[None]
    allowed = (dist >= 0) & (dist < window)
    return bias.masked_fill(~allowed[None], float("-inf"))


class BandedAttentionBlock(nn.Module):
    """Pre-norm transformer block with banded causal ALiBi self-attention."""

    def __init__(self, d: int, n_heads: int, ff: int, dropout: float):
        super().__init__()
        self.h = n_heads
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.ff = nn.Sequential(nn.Linear(d, ff), nn.GELU(), nn.Linear(ff, d))
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        q, k, v = self.qkv(self.ln1(x)).view(B, T, 3, self.h, D // self.h).permute(2, 0, 3, 1, 4)
        att = (q @ k.transpose(-1, -2)) / math.sqrt(D // self.h) + bias
        y = (att.softmax(-1) @ v).transpose(1, 2).reshape(B, T, D)
        x = x + self.drop(self.proj(y))
        return x + self.drop(self.ff(self.ln2(x)))


class StableLatentDecoder(nn.Module):
    """Proposed decoder (see module docstring).

    Input path for day k:  x  ->  A_k x + b_k  (per-day *channel adapter*, initialised to the
    identity)  ->  shared read-in (linear + GELU)  ->  causal backbone (GRU or banded
    transformer)  ->  latent h_t.  Heads: velocity (shared) and Poisson log-rates
    ``g_k * (W h_t) + c_k`` (shared W, per-day gain/offset) for masked-channel reconstruction.

    Because inputs are z-scored per day and the adapter starts at the identity, a new day
    starts from the "re-centred" decoder; alignment can only move away from it as far as the
    identity penalty allows.

    Parameters
    ----------
    n_ch: electrodes; n_days: training days (adapter slots); d: latent width;
    backbone: "gru" or "transformer"; window: attention span per layer (transformer only).
    """

    def __init__(self, n_ch: int, n_days: int, d: int = 128, n_layers: int = 1, n_heads: int = 4,
                 ff: int = 256, window: int = 16, conv_k: int = 3, dropout: float = 0.1,
                 backbone: str = "gru", adapters: bool = True):
        super().__init__()
        self.backbone, self.use_adapters = backbone, adapters
        self.n_ch, self.d, self.window, self.n_layers, self.conv_k = n_ch, d, window, n_layers, conv_k
        self.n_heads = n_heads
        self.adapt_w = nn.Parameter(torch.eye(n_ch).repeat(n_days, 1, 1))
        self.adapt_b = nn.Parameter(torch.zeros(n_days, n_ch))
        self.out_g = nn.Parameter(torch.ones(n_days, n_ch))
        self.out_c = nn.Parameter(torch.zeros(n_days, n_ch))
        self.readin = nn.Linear(n_ch, d)
        self.in_drop = nn.Dropout(dropout)
        if backbone == "transformer":
            self.conv = CausalConv(d, conv_k)
            self.blocks = nn.ModuleList([BandedAttentionBlock(d, n_heads, ff, dropout) for _ in range(n_layers)])
        elif backbone == "gru":
            self.rnn = nn.GRU(d, d, n_layers, batch_first=True)
        else:
            raise ValueError(backbone)
        self.ln = nn.LayerNorm(d)
        self.vel_head = nn.Linear(d, 2)
        self.rate_head = nn.Linear(d, n_ch)

    DAY_PARAMS = ("adapt_w", "adapt_b", "out_g", "out_c")

    @property
    def receptive_field(self) -> int:
        """Bins of history the transformer sees (GRU: used only as a warm-up length)."""
        return self.n_layers * (self.window - 1) + self.conv_k if self.backbone == "transformer" else 64

    @property
    def n_days(self) -> int:
        return self.adapt_w.shape[0]

    def add_day(self, init: torch.Tensor | None = None) -> int:
        """Append a day slot: identity adapter and mean read-out gain/offset of existing days
        (or of the ``init`` day indices). Returns the new day index."""
        idx = torch.arange(self.n_days) if init is None else init
        with torch.no_grad():
            new = {"adapt_w": torch.eye(self.n_ch)[None], "adapt_b": torch.zeros(1, self.n_ch),
                   "out_g": self.out_g.data[idx].mean(0, keepdim=True),
                   "out_c": self.out_c.data[idx].mean(0, keepdim=True)}
        for n in self.DAY_PARAMS:
            p = getattr(self, n)
            p.data = torch.cat([p.data, new[n]], 0)
        return self.n_days - 1

    def adapter_penalty(self, day: torch.Tensor) -> torch.Tensor:
        """Mean squared deviation of the selected days' adapters from the identity."""
        I = torch.eye(self.n_ch, device=self.adapt_w.device)
        return ((self.adapt_w[day] - I) ** 2).sum((1, 2)).mean() + (self.adapt_b[day] ** 2).sum(1).mean()

    def latents(self, x: torch.Tensor, day: torch.Tensor, state=None, return_state: bool = False):
        """x: (B, T, C) normalised counts; day: (B,) long -> latent states (B, T, d).

        ``state`` / ``return_state`` are only used by the GRU backbone (streaming)."""
        if self.use_adapters:
            x = torch.einsum("btc,bce->bte", x, self.adapt_w[day]) + self.adapt_b[day][:, None]
        z = self.in_drop(F.gelu(self.readin(x)))
        new_state = None
        if self.backbone == "transformer":
            z = z + F.gelu(self.conv(z))
            bias = alibi_band_bias(x.shape[1], self.n_heads, self.window, x.device)
            for blk in self.blocks:
                z = blk(z, bias)
        else:
            y, new_state = self.rnn(z, state)
            z = z + y
        z = self.ln(z)
        return (z, new_state) if return_state else z

    def rates(self, h: torch.Tensor, day: torch.Tensor) -> torch.Tensor:
        """Per-day Poisson log-rates from latents."""
        return self.out_g[day][:, None] * self.rate_head(h) + self.out_c[day][:, None]

    def forward(self, x: torch.Tensor, day: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.latents(x, day)
        return self.vel_head(h), self.rates(h, day), h


class GRUDecoder(nn.Module):
    """Recurrent baseline: shared input projection -> GRU -> linear velocity readout."""

    def __init__(self, n_ch: int, hidden: int = 128, n_layers: int = 1, dropout: float = 0.1):
        super().__init__()
        self.inp = nn.Linear(n_ch, hidden)
        self.drop = nn.Dropout(dropout)
        self.gru = nn.GRU(hidden, hidden, n_layers, batch_first=True)
        self.out = nn.Linear(hidden, 2)

    def forward(self, x: torch.Tensor, day: torch.Tensor | None = None, h0=None):
        y, h = self.gru(self.drop(F.gelu(self.inp(x))), h0)
        return self.out(y), h
