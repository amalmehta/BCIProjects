"""Stimulus encoders: image (B, 1, H, W) in [0, 1] -> amplitudes (B, N) in microamps.

Every encoder output goes through the same :class:`SafetyLayer` (amplitude ceiling
and a limit on simultaneously active electrodes) before reaching the simulator.

Baselines (all non-learned, all given the patient's calibrated thresholds and dead
electrodes when ``calibrated=True``, i.e. after a clinical threshold-mapping session):

* :class:`IntensityEncoder` - blur, sample image intensity at each phosphene centre,
  map intensity linearly from the electrode's threshold current up to the ceiling.
  This is the standard "pixelised" encoding (e.g. dynaphos' ``sample_stimulus``).
* :class:`EdgeEncoder` - the same, applied to the Sobel gradient magnitude (edge-based
  encodings are the usual default in simulated cortical prosthetic vision).
* :class:`SalientEdgeEncoder` - scene simplification: Sobel edges weighted by spectral
  residual saliency (Hou & Zhang 2007), keeping contours of salient regions only.

Proposed: :class:`LearnedEncoder`, a small fully convolutional network with coordinate
channels whose output map is read out at each electrode's phosphene location, followed
by a per-electrode affine (gain, bias) "patient adapter" and a sigmoid scaled to the
safe current ceiling.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class EncoderContext:
    """What an encoder is allowed to know about the implant."""

    x_deg: torch.Tensor  # nominal phosphene centres (from the implant plan / mapping)
    y_deg: torch.Tensor
    fov_deg: float
    res: int
    threshold_ua: torch.Tensor  # per-electrode threshold current used by the baselines
    alive: torch.Tensor  # known-working electrodes (1) vs known-dead (0)
    max_current_ua: float


def bilinear_taps(x_deg: torch.Tensor, y_deg: torch.Tensor, fov_deg: float, res: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Indices (N, 4) into a flattened res x res map and bilinear weights (N, 4) for each location.

    Pixel (row i, col j) has its centre at x = (j + 0.5) * fov / res - fov / 2,
    y = fov / 2 - (i + 0.5) * fov / res.
    """
    col = (x_deg + fov_deg / 2) / fov_deg * res - 0.5
    row = (fov_deg / 2 - y_deg) / fov_deg * res - 0.5
    col = col.clamp(0, res - 1 - 1e-4)
    row = row.clamp(0, res - 1 - 1e-4)
    c0, r0 = col.floor(), row.floor()
    fc, fr = col - c0, row - r0
    c0, r0 = c0.long(), r0.long()
    c1, r1 = (c0 + 1).clamp(max=res - 1), (r0 + 1).clamp(max=res - 1)
    idx = torch.stack([r0 * res + c0, r0 * res + c1, r1 * res + c0, r1 * res + c1], 1)
    w = torch.stack([(1 - fr) * (1 - fc), (1 - fr) * fc, fr * (1 - fc), fr * fc], 1)
    return idx, w


def sample_at(maps: torch.Tensor, idx: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """Bilinear read-out of (B, 1, H, W) maps at precomputed taps -> (B, N). Uses Gather only (ONNX friendly)."""
    flat = maps.flatten(1)  # (B, H*W)
    vals = flat[:, idx.reshape(-1)].reshape(flat.shape[0], idx.shape[0], 4)
    return (vals * w).sum(-1)


def gaussian_blur(x: torch.Tensor, sigma_px: float) -> torch.Tensor:
    """Separable Gaussian blur with reflect padding (radius = ceil(3 sigma))."""
    if sigma_px <= 0:
        return x
    r = max(1, int(math.ceil(3 * sigma_px)))
    t = torch.arange(-r, r + 1, dtype=x.dtype, device=x.device)
    k = torch.exp(-t ** 2 / (2 * sigma_px ** 2))
    k = k / k.sum()
    x = F.conv2d(F.pad(x, (r, r, 0, 0), mode="reflect"), k.view(1, 1, 1, -1))
    return F.conv2d(F.pad(x, (0, 0, r, r), mode="reflect"), k.view(1, 1, -1, 1))


def sobel_magnitude(x: torch.Tensor) -> torch.Tensor:
    """Sobel gradient magnitude (replicate padding), not normalised."""
    kx = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]], dtype=x.dtype, device=x.device)
    xp = F.pad(x, (1, 1, 1, 1), mode="replicate")
    gx = F.conv2d(xp, kx.view(1, 1, 3, 3))
    gy = F.conv2d(xp, kx.t().contiguous().view(1, 1, 3, 3))
    return torch.sqrt(gx ** 2 + gy ** 2 + 1e-12)


def spectral_residual_saliency(x: torch.Tensor, blur_px: float = 2.0) -> torch.Tensor:
    """Spectral residual saliency (Hou & Zhang 2007) on a 32x32 downsampled copy, normalised to [0, 1]."""
    small = F.interpolate(x, size=(32, 32), mode="bilinear", align_corners=False)
    f = torch.fft.fft2(small)
    log_amp = torch.log(f.abs() + 1e-8)
    phase = torch.angle(f)
    avg = F.avg_pool2d(F.pad(log_amp, (1, 1, 1, 1), mode="replicate"), 3, stride=1)
    residual = log_amp - avg
    sal = torch.fft.ifft2(torch.exp(residual + 1j * phase)).abs() ** 2
    sal = gaussian_blur(sal, blur_px)
    sal = F.interpolate(sal, size=x.shape[-2:], mode="bilinear", align_corners=False)
    lo = sal.amin(dim=(2, 3), keepdim=True)
    hi = sal.amax(dim=(2, 3), keepdim=True)
    return (sal - lo) / (hi - lo + 1e-8)


class SafetyLayer(nn.Module):
    """Clamp amplitudes to [0, I_max] and keep at most ``max_active`` electrodes per frame.

    The top-k mask is applied multiplicatively, so in training gradients flow to the
    selected electrodes only (the selection itself is treated as constant).
    """

    def __init__(self, max_current_ua: float, max_active: int | None):
        super().__init__()
        self.max_current_ua = max_current_ua
        self.max_active = max_active

    def forward(self, amp: torch.Tensor) -> torch.Tensor:
        amp = amp.clamp(0.0, self.max_current_ua)
        if self.max_active is not None and self.max_active < amp.shape[1]:
            kth = torch.topk(amp.detach(), self.max_active, dim=1).values[:, -1:]
            amp = amp * (amp.detach() >= kth).float()
        return amp


class _SamplingBaseline(nn.Module):
    """Shared logic: feature map in [0, 1] -> sample at phosphenes -> threshold-calibrated linear map."""

    def __init__(self, ctx: EncoderContext, blur_px: float = 1.0, cutoff: float = 0.1):
        super().__init__()
        idx, w = bilinear_taps(ctx.x_deg, ctx.y_deg, ctx.fov_deg, ctx.res)
        self.register_buffer("idx", idx)
        self.register_buffer("w", w)
        self.register_buffer("thr", ctx.threshold_ua.clone())
        self.register_buffer("alive", ctx.alive.clone())
        self.max_current_ua = ctx.max_current_ua
        self.blur_px, self.cutoff = blur_px, cutoff

    def feature(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        v = sample_at(self.feature(x), self.idx, self.w).clamp(0, 1)
        thr = self.thr.clamp(max=self.max_current_ua)
        amp = thr + v * (self.max_current_ua - thr)
        return torch.where(v > self.cutoff, amp, torch.zeros_like(amp)) * self.alive


class IntensityEncoder(_SamplingBaseline):
    """Blurred image intensity sampled at each phosphene centre."""

    def feature(self, x: torch.Tensor) -> torch.Tensor:
        return gaussian_blur(x, self.blur_px)


class EdgeEncoder(_SamplingBaseline):
    """Sobel edge magnitude (per-image max-normalised) sampled at each phosphene centre."""

    def feature(self, x: torch.Tensor) -> torch.Tensor:
        e = sobel_magnitude(gaussian_blur(x, self.blur_px))
        return e / (e.amax(dim=(2, 3), keepdim=True) + 1e-6)


class SalientEdgeEncoder(_SamplingBaseline):
    """Edges gated by spectral-residual saliency: a light-weight scene simplification."""

    def feature(self, x: torch.Tensor) -> torch.Tensor:
        e = sobel_magnitude(gaussian_blur(x, self.blur_px))
        e = e * spectral_residual_saliency(x)
        return e / (e.amax(dim=(2, 3), keepdim=True) + 1e-6)


def coord_channels(res: int, fov_deg: float) -> torch.Tensor:
    """(1, 3, res, res) channels: x/half_fov, y/half_fov, eccentricity/half_fov."""
    c = (torch.arange(res, dtype=torch.float32) + 0.5) / res * fov_deg - fov_deg / 2
    yy, xx = torch.meshgrid(-c, c, indexing="ij")
    half = fov_deg / 2
    return torch.stack([xx / half, yy / half, torch.sqrt(xx ** 2 + yy ** 2) / half]).unsqueeze(0)


class LearnedEncoder(nn.Module):
    """Fully convolutional encoder with a retinotopic read-out and per-electrode adapter.

    image (B,1,H,W) + coordinate channels [x, y, ecc]
      -> small U-Net (H/2, H/4, back to H/2) -> stimulation map (B,1,H/2,W/2)
      -> bilinear read-out at the nominal phosphene centres -> u_n
      -> amplitude_n = I_max * sigmoid(gain_n * u_n + bias_n).
    The per-electrode (gain, bias) pair is the "patient adapter" tuned in few-shot adaptation.
    """

    def __init__(self, ctx: EncoderContext, width: int = 32):
        super().__init__()
        self.register_buffer("coords", coord_channels(ctx.res, ctx.fov_deg))
        idx, w = bilinear_taps(ctx.x_deg, ctx.y_deg, ctx.fov_deg, ctx.res // 2)
        self.register_buffer("idx", idx)
        self.register_buffer("w", w)
        c = width
        self.down1 = nn.Sequential(nn.Conv2d(4, c // 2, 3, stride=2, padding=1), nn.ReLU(),
                                   nn.Conv2d(c // 2, c, 3, padding=1), nn.ReLU())  # H/2
        self.down2 = nn.Sequential(nn.Conv2d(c, c, 3, stride=2, padding=1), nn.ReLU(),
                                   nn.Conv2d(c, c, 3, padding=1), nn.ReLU(),
                                   nn.Conv2d(c, c, 3, padding=1), nn.ReLU())  # H/4
        self.up1 = nn.Sequential(nn.Conv2d(2 * c + 4, c // 2, 3, padding=1), nn.ReLU(), nn.Conv2d(c // 2, 1, 1))  # H/2
        n = ctx.x_deg.shape[0]
        self.gain = nn.Parameter(torch.ones(n))
        # Start every electrode near 50 uA whatever the ceiling (bias 0 when I_max = 100 uA), so
        # variants with a higher ceiling do not start in the saturated, all-bright regime.
        p0 = min(0.5, 50.0 / ctx.max_current_ua)
        self.bias = nn.Parameter(torch.full((n,), math.log(p0 / (1 - p0))))
        self.max_current_ua = ctx.max_current_ua

    def stimulation_map(self, x: torch.Tensor) -> torch.Tensor:
        """Stimulation map at half the input resolution, (B, 1, H/2, W/2)."""
        h = torch.cat([x, self.coords.expand(x.shape[0], -1, -1, -1)], 1)
        d1 = self.down1(h)
        d2 = self.down2(d1)
        skip = F.avg_pool2d(h, 2)
        return self.up1(torch.cat([d1, F.interpolate(d2, scale_factor=2.0, mode="nearest"), skip], 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = sample_at(self.stimulation_map(x), self.idx, self.w)
        return self.max_current_ua * torch.sigmoid(self.gain * u + self.bias)


class Decoder(nn.Module):
    """Percept (B,1,H,W) -> reconstructed image; stands in for the brain's interpretation (end-to-end loss)."""

    def __init__(self, width: int = 32):
        super().__init__()
        c = width
        self.net = nn.Sequential(
            nn.Conv2d(1, c // 2, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(c // 2, c, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(c, c, 3, padding=1), nn.ReLU(),
            nn.Upsample(scale_factor=2.0, mode="nearest"),
            nn.Conv2d(c, 1, 3, padding=1),
            nn.Upsample(scale_factor=2.0, mode="bilinear", align_corners=False), nn.Sigmoid())

    def forward(self, p: torch.Tensor) -> torch.Tensor:
        return self.net(p)


class PerceptClassifier(nn.Module):
    """Small CNN trained on simulated percepts: a proxy for what a user could learn to recognise.

    Percepts are average-pooled 2x first (64 -> 32 px, 0.375 deg/px), which is finer than
    the phosphene spacing outside the central degree.
    """

    def __init__(self, n_classes: int, width: int = 32):
        super().__init__()
        c = width
        self.net = nn.Sequential(
            nn.AvgPool2d(2),
            nn.Conv2d(1, c, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(c, 2 * c, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(2 * c, 2 * c, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2), nn.Flatten(),
            nn.Dropout(0.3), nn.Linear(2 * c * 16, 128), nn.ReLU(), nn.Linear(128, n_classes))

    def forward(self, p: torch.Tensor) -> torch.Tensor:
        return self.net(p)
