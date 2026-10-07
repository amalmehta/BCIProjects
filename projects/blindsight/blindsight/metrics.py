"""Image-similarity metrics (per image, images in [0, 1])."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def _gaussian_1d(size: int = 11, sigma: float = 1.5) -> torch.Tensor:
    t = torch.arange(size, dtype=torch.float32) - (size - 1) / 2
    g = torch.exp(-t ** 2 / (2 * sigma ** 2))
    return g / g.sum()


def ssim(x: torch.Tensor, y: torch.Tensor, data_range: float = 1.0, size: int = 11, sigma: float = 1.5) -> torch.Tensor:
    """Structural similarity (Wang et al. 2004), Gaussian 11x11 window (sigma 1.5), 'valid' region; returns (B,).

    The window is applied separably to the five local statistics at once (grouped conv).
    """
    g = _gaussian_1d(size, sigma).to(x)
    c1, c2 = (0.01 * data_range) ** 2, (0.03 * data_range) ** 2
    stats = torch.cat([x, y, x * x, y * y, x * y], 1)  # (B, 5, H, W)
    stats = F.conv2d(stats, g.view(1, 1, 1, -1).repeat(5, 1, 1, 1), groups=5)
    mx, my, exx, eyy, exy = F.conv2d(stats, g.view(1, 1, -1, 1).repeat(5, 1, 1, 1), groups=5).unbind(1)
    sxx, syy, sxy = exx - mx ** 2, eyy - my ** 2, exy - mx * my
    s = ((2 * mx * my + c1) * (2 * sxy + c2)) / ((mx ** 2 + my ** 2 + c1) * (sxx + syy + c2))
    return s.flatten(1).mean(1)


def psnr(x: torch.Tensor, y: torch.Tensor, data_range: float = 1.0) -> torch.Tensor:
    """Peak signal-to-noise ratio in dB; returns (B,)."""
    mse = ((x - y) ** 2).flatten(1).mean(1).clamp_min(1e-10)
    return 10 * torch.log10(data_range ** 2 / mse)


def ssim_lowpass(x: torch.Tensor, y: torch.Tensor, sigma_px: float = 2.0) -> torch.Tensor:
    """SSIM after blurring both images with a Gaussian (sigma 2 px = 0.375 deg at the default scale).

    Phosphene vision cannot convey detail finer than the phosphene spacing, so this
    variant scores the coarse layout of the scene rather than dot texture.
    """
    from .encoders import gaussian_blur

    return ssim(gaussian_blur(x, sigma_px), gaussian_blur(y, sigma_px))
