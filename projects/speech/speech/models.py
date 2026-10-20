"""Neural-to-phoneme CTC decoders.

* :class:`GRUDecoder` re-implements the published Brain-to-Text baseline
  (Willett et al., 2023; Card et al., 2024): day-specific affine input layers
  with a softsign, Gaussian smoothing, "patching" (concatenating 14 bins with a
  stride of 4), a stack of GRU layers and a linear phoneme read-out.
* :class:`ConformerDecoder` is our proposed encoder: the same day-specific input
  layers, a factorised patch embedding, Conformer blocks (Gulati et al., 2020)
  that can run causally (masked attention + left-padded convolutions) for
  streaming, and intermediate / self-conditioned CTC (Lee & Watanabe, 2021;
  Nozaki & Komatsu, 2021).

Both return log-probabilities of shape (B, T', 41) and output lengths T'.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .augment import gauss_smooth, gaussian_kernel
from .phonemes import N_CLASSES


class DayInputLayers(nn.Module):
    """One affine map + softsign per recording day, initialised to identity.

    Electrode impedances, unit drop-out and baseline firing rates drift between
    sessions; a per-day linear map lets a shared decoder see a stabilised input
    space (Willett et al., 2023). With ``per_day=False`` a single shared layer is
    used instead (ablation).
    """

    def __init__(self, n_days: int, dim: int, dropout: float, per_day: bool = True):
        super().__init__()
        self.per_day = per_day
        n = n_days if per_day else 1
        self.weight = nn.Parameter(torch.eye(dim).repeat(n, 1, 1))
        self.bias = nn.Parameter(torch.zeros(n, 1, dim))
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, day: torch.Tensor) -> torch.Tensor:
        idx = day if self.per_day else torch.zeros_like(day)
        x = torch.bmm(x, self.weight[idx]) + self.bias[idx]
        return self.drop(F.softsign(x))


def init_blank_bias(linear: nn.Linear, blank_bias: float | None) -> None:
    """Initialise the CTC read-out so the blank starts with a chosen logit offset.

    With zero weights' contribution the initial posterior of blank is
    ``e^b / (e^b + C - 1)``; a positive ``b`` starts the model near the frame-level
    blank prior (most CTC frames are blank), so early gradients go into the
    label logits rather than into first learning to emit blanks.
    """
    if blank_bias is None:
        return
    with torch.no_grad():
        linear.bias.zero_()
        linear.bias[0] = float(blank_bias)


def patch(x: torch.Tensor, size: int, stride: int) -> torch.Tensor:
    """Concatenate ``size`` consecutive bins every ``stride`` bins: (B,T,F)->(B,T',size*F)."""
    x = x.unfold(1, size, stride)  # (B, T', F, size)
    return x.permute(0, 1, 3, 2).reshape(x.shape[0], x.shape[1], -1)


def out_lengths(x_len: torch.Tensor, size: int, stride: int) -> torch.Tensor:
    """Number of patches produced for inputs of length ``x_len``."""
    return torch.clamp((x_len - size) // stride + 1, min=1)


class GRUDecoder(nn.Module):
    """Published-style GRU-CTC phoneme decoder (scaled for CPU via ``hidden``/``layers``)."""

    def __init__(
        self,
        n_days: int,
        n_features: int = 512,
        hidden: int = 256,
        layers: int = 3,
        patch_size: int = 14,
        patch_stride: int = 4,
        dropout: float = 0.4,
        input_dropout: float = 0.2,
        bidirectional: bool = False,
        per_day: bool = True,
        smooth_std: float = 2.0,
        blank_bias: float | None = None,
    ):
        super().__init__()
        self.patch_size, self.patch_stride = patch_size, patch_stride
        self.register_buffer("kernel", gaussian_kernel(smooth_std) if smooth_std > 0 else torch.ones(1))
        self.day = DayInputLayers(n_days, n_features, input_dropout, per_day)
        self.gru = nn.GRU(
            n_features * patch_size,
            hidden,
            num_layers=layers,
            dropout=dropout,
            batch_first=True,
            bidirectional=bidirectional,
        )
        for name, p in self.gru.named_parameters():
            if "weight_hh" in name:
                nn.init.orthogonal_(p)
            elif "weight_ih" in name:
                nn.init.xavier_uniform_(p)
        self.out = nn.Linear(hidden * (2 if bidirectional else 1), N_CLASSES)
        init_blank_bias(self.out, blank_bias)
        self.causal = not bidirectional

    def features(self, x: torch.Tensor, day: torch.Tensor) -> torch.Tensor:
        """Smoothing, day layer and patching (shared by full and streaming inference)."""
        x = gauss_smooth(x, self.kernel)
        x = self.day(x, day)
        return patch(x, self.patch_size, self.patch_stride)

    def forward(self, x, x_len, day):
        h, _ = self.gru(self.features(x, day))
        logits = self.out(h)
        return {"log_probs": logits.log_softmax(-1), "lengths": out_lengths(x_len, self.patch_size, self.patch_stride)}


class FeedForward(nn.Module):
    def __init__(self, d: int, mult: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d), nn.Linear(d, d * mult), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(d * mult, d), nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class ConvModule(nn.Module):
    """Conformer convolution module; left-only padding when causal."""

    def __init__(self, d: int, kernel: int, dropout: float, causal: bool):
        super().__init__()
        self.causal, self.kernel = causal, kernel
        self.ln = nn.LayerNorm(d)
        self.pw1 = nn.Linear(d, 2 * d)
        self.dw = nn.Conv1d(d, d, kernel, groups=d)
        self.ln2 = nn.LayerNorm(d)  # LayerNorm instead of BatchNorm: streaming-friendly
        self.pw2 = nn.Linear(d, d)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, pad_mask):
        y = F.glu(self.pw1(self.ln(x)), dim=-1)
        y = y.masked_fill(pad_mask.unsqueeze(-1), 0.0).transpose(1, 2)
        left = self.kernel - 1 if self.causal else (self.kernel - 1) // 2
        right = 0 if self.causal else self.kernel - 1 - left
        y = self.dw(F.pad(y, (left, right))).transpose(1, 2)
        return self.drop(self.pw2(F.silu(self.ln2(y))))


class ConformerBlock(nn.Module):
    def __init__(self, d: int, heads: int, ff_mult: int, conv_kernel: int, dropout: float, causal: bool):
        super().__init__()
        self.ff1 = FeedForward(d, ff_mult, dropout)
        self.ln_att = nn.LayerNorm(d)
        self.att = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.conv = ConvModule(d, conv_kernel, dropout, causal)
        self.ff2 = FeedForward(d, ff_mult, dropout)
        self.ln_out = nn.LayerNorm(d)

    def forward(self, x, pad_mask, att_mask):
        x = x + 0.5 * self.ff1(x)
        h = self.ln_att(x)
        h, _ = self.att(h, h, h, key_padding_mask=pad_mask, attn_mask=att_mask, need_weights=False)
        x = x + self.drop(h)
        x = x + self.conv(x, pad_mask)
        x = x + 0.5 * self.ff2(x)
        return self.ln_out(x)


class ConformerDecoder(nn.Module):
    """Day layers -> factorised patch embedding -> Conformer blocks -> CTC head.

    ``inter_ctc_layers`` lists block indices (1-based) after which an
    intermediate CTC loss is computed with the shared output head; when
    ``self_condition`` is set, the intermediate posteriors are projected back
    and added to the residual stream (self-conditioned CTC).
    """

    def __init__(
        self,
        n_days: int,
        n_features: int = 512,
        d_model: int = 192,
        layers: int = 6,
        heads: int = 4,
        ff_mult: int = 4,
        conv_kernel: int = 15,
        patch_size: int = 14,
        patch_stride: int = 4,
        bottleneck: int = 128,
        dropout: float = 0.2,
        input_dropout: float = 0.2,
        causal: bool = True,
        per_day: bool = True,
        inter_ctc_layers: tuple[int, ...] = (3,),
        self_condition: bool = True,
        smooth_std: float = 2.0,
        blank_bias: float | None = None,
    ):
        super().__init__()
        self.patch_size, self.patch_stride = patch_size, patch_stride
        self.causal = causal
        self.register_buffer("kernel", gaussian_kernel(smooth_std) if smooth_std > 0 else torch.ones(1))
        self.day = DayInputLayers(n_days, n_features, input_dropout, per_day)
        # optional per-bin projection before patching (0 = patch the raw 512-d features, as the GRU does)
        self.reduce = nn.Linear(n_features, bottleneck) if bottleneck else nn.Identity()
        bottleneck = bottleneck or n_features
        # No absolute positional encoding: the causal mask and the depthwise convolutions
        # already provide order information, and an un-scaled sinusoid swamped the
        # small-magnitude patch embeddings early in training (slow escape from the blank plateau).
        self.embed = nn.Sequential(nn.Linear(bottleneck * patch_size, d_model), nn.LayerNorm(d_model), nn.Dropout(dropout))
        self.blocks = nn.ModuleList(
            ConformerBlock(d_model, heads, ff_mult, conv_kernel, dropout, causal) for _ in range(layers)
        )
        self.out = nn.Linear(d_model, N_CLASSES)
        init_blank_bias(self.out, blank_bias)
        self.inter = set(inter_ctc_layers)
        self.self_condition = self_condition
        if self_condition:
            self.cond = nn.Linear(N_CLASSES, d_model)

    def forward(self, x, x_len, day):
        x = gauss_smooth(x, self.kernel)
        x = self.reduce(self.day(x, day))
        x = self.embed(patch(x, self.patch_size, self.patch_stride))
        B, T, D = x.shape
        lengths = out_lengths(x_len, self.patch_size, self.patch_stride)
        pad = torch.arange(T, device=x.device)[None, :] >= lengths[:, None]
        att = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), 1) if self.causal else None
        inter = []
        for i, blk in enumerate(self.blocks, start=1):
            x = blk(x, pad, att)
            if i in self.inter and i < len(self.blocks):
                lp = self.out(x).log_softmax(-1)
                inter.append(lp)
                if self.self_condition:
                    x = x + self.cond(lp.exp())
        return {"log_probs": self.out(x).log_softmax(-1), "lengths": lengths, "inter_log_probs": inter}


def build_model(cfg: dict, n_days: int) -> nn.Module:
    """Instantiate a decoder from a config dict ``{"type": "gru"|"conformer", ...}``."""
    kw = {k: v for k, v in cfg.items() if k != "type"}
    if "inter_ctc_layers" in kw:
        kw["inter_ctc_layers"] = tuple(kw["inter_ctc_layers"])
    if cfg["type"] == "gru":
        return GRUDecoder(n_days, **kw)
    if cfg["type"] == "conformer":
        return ConformerDecoder(n_days, **kw)
    raise ValueError(f"unknown model type {cfg['type']}")
