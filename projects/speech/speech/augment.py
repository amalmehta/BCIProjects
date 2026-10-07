"""Training-time augmentations for binned intracortical features.

The first three follow the published Brain-to-Text baselines (Willett et al.,
2023; Card et al., 2024): additive white noise, a per-trial constant offset on
every feature (models slow baseline drift between blocks/days), and a random
cut of the first few bins. The last two are our additions, motivated by
SpecAugment (Park et al., 2019) and by electrode loss in chronic implants:
time masking and electrode dropout (zeroing both features of a random subset
of electrodes for the whole trial).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter1d


@dataclass
class AugmentConfig:
    """Augmentation strengths (all disabled when zero)."""

    white_noise_std: float = 0.0
    constant_offset_std: float = 0.0
    random_cut: int = 0
    time_mask_num: int = 0
    time_mask_max_bins: int = 0
    electrode_dropout: float = 0.0


def augment_batch(x: torch.Tensor, x_len: torch.Tensor, cfg: AugmentConfig, n_electrodes: int = 256):
    """Apply augmentations to a padded batch ``x`` of shape (B, T, F).

    Returns the augmented batch and the (possibly shortened) lengths.
    """
    B, T, Fdim = x.shape
    if cfg.white_noise_std > 0:
        x = x + cfg.white_noise_std * torch.randn_like(x)
    if cfg.constant_offset_std > 0:
        x = x + cfg.constant_offset_std * torch.randn(B, 1, Fdim, device=x.device)
    if cfg.electrode_dropout > 0:
        keep = (torch.rand(B, 1, n_electrodes, device=x.device) >= cfg.electrode_dropout).float()
        x = x * keep.repeat(1, 1, Fdim // n_electrodes)
    if cfg.time_mask_num > 0 and cfg.time_mask_max_bins > 0:
        for b in range(B):
            L = int(x_len[b])
            for _ in range(cfg.time_mask_num):
                w = int(torch.randint(0, cfg.time_mask_max_bins + 1, (1,)))
                if w == 0 or L <= w:
                    continue
                s = int(torch.randint(0, L - w, (1,)))
                x[b, s : s + w] = 0.0
    if cfg.random_cut > 0:
        cut = int(torch.randint(0, cfg.random_cut + 1, (1,)))
        if cut:
            x = x[:, cut:]
            x_len = x_len - cut
    return x, x_len


def gaussian_kernel(std: float, size: int = 100) -> torch.Tensor:
    """Normalised 1-D Gaussian kernel, truncated where it falls below 1% (as in the baseline)."""
    inp = np.zeros(size, dtype=np.float32)
    inp[size // 2] = 1
    k = gaussian_filter1d(inp, std)
    k = k[k > 0.01]
    return torch.tensor(k / k.sum(), dtype=torch.float32)


def gauss_smooth(x: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """Smooth (B, T, F) features along time with a depthwise 'same' convolution."""
    B, T, C = x.shape
    w = kernel.to(x).view(1, 1, -1).repeat(C, 1, 1)
    y = F.conv1d(x.transpose(1, 2), w, padding="same", groups=C)
    return y.transpose(1, 2)
