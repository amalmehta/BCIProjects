"""Differentiable cortical phosphene simulator (PyTorch).

A re-implementation (from the published equations, not the GPL code) of the
biologically plausible simulator of van der Grinten et al. (2024, eLife,
doi:10.7554/eLife.85812), restricted to what we need and made fast on CPU:

1. Electrode -> phosphene location: inverse wedge-dipole map (``cortex.py``).
2. Effective current (Eq. 7):  I_eff = max(0, (I - I0 - B) * PW * f),  I0 = rheobase.
3. Tissue activation: leaky integration of I_eff with time constant tau_act, while the
   habituation trace B grows with the delivered charge. For a single pulse train of
   duration T applied from rest (the static mode used for training) the coupled system
   has a closed form, A = (I - I0) PW f g(T) (see ``SimParams.activation_gain``).
4. Brightness (Eq. 10): b = sigmoid(lambda * (A - A50)), shown only if A > A_thr,n,
   where per-electrode thresholds A_thr,n ~ N(theta50, sd^2) (fitted by the dynaphos
   authors to Fernandez et al. 2021).
5. Size (Tehovnik & Slocum 2007 current spread): activated radius r = sqrt(I / K) mm,
   phosphene diameter P = 2 r / M(ecc) degrees, Gaussian sigma = P / 2 (two standard
   deviations equal the phosphene size, as in the eLife text).
6. Rendering: peak-normalised isotropic Gaussians summed and clipped to [0, 1]. We add a
   fixed display blur sigma_floor in quadrature, sigma_tot = sqrt(sigma^2 + sigma_floor^2),
   so sub-pixel phosphenes do not alias at the coarse training resolution. The Gaussian
   is separable, so the render is a batched matrix product (B, H, N) @ (B, N, W), which
   keeps memory at O(B N (H + W)) instead of O(B N H W).

Patient variability (dead electrodes, threshold spread, positional jitter of the
cortical sites) is drawn by :func:`sample_patient`.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict

import numpy as np
import torch
from torch import nn

from .cortex import WedgeDipole, hemifield_to_cartesian
from .layouts import Layout


@dataclass
class SimParams:
    """All simulator constants (SI units unless stated). Defaults follow dynaphos."""

    fov_deg: float = 12.0  # full width of the rendered square field of view
    resolution: int = 64  # rendered pixels per side
    pulse_width_s: float = 170e-6
    frequency_hz: float = 300.0
    train_s: float = 0.166  # pulse-train duration of one static "frame" of stimulation
    rheobase_a: float = 23.9e-6  # leak current I0
    tau_act_s: float = 1.0 / 9.000  # activation decay: -ln(1.234e-4 per second) = 9.0 / s
    trace_increase_rate: float = 13.95528162  # per second (habituation memory trace)
    trace_decay_per_second: float = 0.99949191
    a50: float = 1.057631326853325e-07  # activation at half-maximum brightness
    brightness_slope: float = 19152642.500946816
    threshold_mean: float = 9.141886000943878e-08
    threshold_sd: float = 6.715877869833961e-08
    current_spread_a_per_mm2: float = 675e-6  # K in r = sqrt(I / K)
    sigma_floor_deg: float = 0.15  # display blur added in quadrature
    max_current_ua: float = 100.0  # hard safety limit on amplitude
    soft_gate_width: float = 0.1  # width of the surrogate threshold gate, in units of a50
    model: WedgeDipole = field(default_factory=WedgeDipole)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["model"] = asdict(self.model)
        return d

    @property
    def activation_gain(self) -> float:
        """Activation per unit of initial effective current, A(T) / ((I - I0) * PW * f), for one train from rest.

        Solves the coupled leaky integrator and habituation trace over the train
        (trace decay, tau_trace ~ 2000 s, is neglected within 166 ms):
            dB/dt = kappa * I_eff,  I_eff = (I - I0 - B) d   =>   I_eff(t) = (I - I0) d exp(-kappa d t)
            dA/dt = -A / tau + I_eff  =>  A(T) = (I - I0) d (exp(-kappa d T) - exp(-T / tau)) / (1 / tau - kappa d).
        """
        kd = self.trace_increase_rate * self.pulse_width_s * self.frequency_hz
        inv_tau = 1.0 / self.tau_act_s
        return (math.exp(-kd * self.train_s) - math.exp(-self.train_s * inv_tau)) / (inv_tau - kd)

    def threshold_current_ua(self, a_thr: np.ndarray | float) -> np.ndarray | float:
        """Current (uA) at which a static train reaches activation ``a_thr``."""
        duty = self.pulse_width_s * self.frequency_hz
        return (self.rheobase_a + np.asarray(a_thr) / (duty * self.activation_gain)) * 1e6


@dataclass
class Patient:
    """Per-electrode state of one (simulated) implanted person."""

    x_deg: torch.Tensor  # (N,) phosphene centres in the visual field
    y_deg: torch.Tensor
    ecc_deg: torch.Tensor
    threshold: torch.Tensor  # (N,) activation threshold A_thr
    alive: torch.Tensor  # (N,) 1 = working electrode, 0 = dead / open circuit
    name: str = "nominal"

    def threshold_current_ua(self, params: SimParams) -> torch.Tensor:
        thr = params.threshold_current_ua(self.threshold.detach().cpu().numpy())
        return torch.as_tensor(thr, dtype=torch.float32, device=self.threshold.device)

    def to(self, device: torch.device | str) -> "Patient":
        """Move all per-electrode tensors to ``device`` (in place); returns self."""
        for k in ("x_deg", "y_deg", "ecc_deg", "threshold", "alive"):
            setattr(self, k, getattr(self, k).to(device))
        return self


def _positions(layout: Layout, params: SimParams, dx: np.ndarray | None = None, dy: np.ndarray | None = None):
    cx = layout.cortex_x_mm + (0 if dx is None else dx)
    cy = layout.cortex_y_mm + (0 if dy is None else dy)
    ecc, theta = params.model.to_visual_field(cx, cy)
    x, y = hemifield_to_cartesian(ecc, theta, layout.hemisphere)
    return x, y, ecc


def nominal_patient(layout: Layout, params: SimParams) -> Patient:
    """Patient with no jitter, all electrodes alive and every threshold at the mean."""
    x, y, ecc = _positions(layout, params)
    f = lambda a: torch.as_tensor(np.asarray(a), dtype=torch.float32)
    return Patient(f(x), f(y), f(ecc), torch.full((layout.n,), params.threshold_mean), torch.ones(layout.n), "nominal")


def sample_patient(layout: Layout, params: SimParams, seed: int, dropout: float = 0.1,
                   jitter_mm: float = 0.4, threshold_sd_scale: float = 1.0) -> Patient:
    """Draw a simulated patient: dead electrodes, threshold spread and cortical position jitter."""
    rng = np.random.default_rng(seed)
    dx = rng.normal(0, jitter_mm, layout.n)
    dy = rng.normal(0, jitter_mm, layout.n)
    x, y, ecc = _positions(layout, params, dx, dy)
    thr = np.clip(rng.normal(params.threshold_mean, params.threshold_sd * threshold_sd_scale, layout.n), 0, None)
    alive = (rng.random(layout.n) >= dropout).astype(np.float32)
    f = lambda a: torch.as_tensor(np.asarray(a), dtype=torch.float32)
    return Patient(f(x), f(y), f(ecc), f(thr), f(alive), f"patient_{seed}")


class PhospheneSimulator(nn.Module):
    """Static (single pulse-train) phosphene renderer; differentiable w.r.t. amplitudes."""

    def __init__(self, params: SimParams):
        super().__init__()
        self.p = params
        n = params.resolution
        half = params.fov_deg / 2
        # Pixel centres in degrees: columns left->right (x), rows top->bottom (y descending).
        centres = (torch.arange(n, dtype=torch.float32) + 0.5) / n * params.fov_deg - half
        self.register_buffer("px", centres)
        self.register_buffer("py", -centres)

    # --- per-electrode quantities -------------------------------------------------------
    def activation(self, amp_ua: torch.Tensor, trace_a: torch.Tensor | float = 0.0) -> torch.Tensor:
        """Tissue activation after one static train from rest (Eq. 7-9 with the trace, closed form)."""
        p = self.p
        i_eff = torch.relu((amp_ua * 1e-6 - p.rheobase_a - trace_a) * p.pulse_width_s * p.frequency_hz)
        return i_eff * p.activation_gain

    def brightness(self, activation: torch.Tensor, threshold: torch.Tensor, hard: bool = True) -> torch.Tensor:
        """Sigmoidal brightness gated by the per-electrode detection threshold.

        With ``hard=False`` the forward pass is still the hard gate, but gradients flow
        through a sigmoid surrogate (straight-through estimator), so training sees the
        same percept that evaluation does.
        """
        p = self.p
        b = torch.sigmoid(p.brightness_slope * (activation - p.a50))
        gate = (activation > threshold).float()
        if not hard:
            soft = torch.sigmoid((activation - threshold) / (p.soft_gate_width * p.a50))
            gate = gate + soft - soft.detach()
        return b * gate

    def sigma_deg(self, amp_ua: torch.Tensor, ecc_deg: torch.Tensor) -> torch.Tensor:
        """Rendered Gaussian sigma (deg): current-spread size in quadrature with display blur."""
        p = self.p
        r_mm = torch.sqrt(torch.clamp(amp_ua, min=0.0) * 1e-6 / p.current_spread_a_per_mm2 + 1e-12)
        mag = p.model.k * (1.0 / (ecc_deg + p.model.a) - 1.0 / (ecc_deg + p.model.b))
        sigma = r_mm / mag  # = (2 r / M) / 2
        return torch.sqrt(sigma ** 2 + p.sigma_floor_deg ** 2)

    # --- rendering ----------------------------------------------------------------------
    def render(self, brightness: torch.Tensor, sigma: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Sum of peak-normalised Gaussians. brightness/sigma: (B, N); x, y: (N,) -> (B, 1, H, W)."""
        inv = 1.0 / (2.0 * sigma ** 2)  # (B, N)
        dx2 = (self.px[:, None] - x[None, :]) ** 2  # (W, N)
        dy2 = (self.py[:, None] - y[None, :]) ** 2  # (H, N)
        gx = torch.exp(-dx2[None] * inv[:, None, :])  # (B, W, N)
        gy = torch.exp(-dy2[None] * inv[:, None, :])
        # Zero negligible tails: denormal floats make CPU matmuls an order of magnitude slower.
        gx = torch.where(gx > 1e-6, gx, torch.zeros_like(gx))
        gy = torch.where(gy > 1e-6, gy, torch.zeros_like(gy)) * brightness[:, None, :]  # (B, H, N)
        img = torch.bmm(gy, gx.transpose(1, 2))  # (B, H, W)
        return img.clamp(0.0, 1.0).unsqueeze(1)

    def _basis(self, patient: Patient) -> tuple[torch.Tensor, torch.Tensor]:
        """Gaussian maps (2, N, H*W) at the two size anchors used by the fast renderer (cached)."""
        key = id(patient)
        if getattr(self, "_basis_key", None) != key:
            maps = []
            for amp in (self.p.threshold_current_ua(self.p.threshold_mean), self.p.max_current_ua):
                sig = self.sigma_deg(torch.full_like(patient.ecc_deg, float(amp)), patient.ecc_deg)
                gx = torch.exp(-((self.px[None, :] - patient.x_deg[:, None]) ** 2) / (2 * sig[:, None] ** 2))  # (N, W)
                gy = torch.exp(-((self.py[None, :] - patient.y_deg[:, None]) ** 2) / (2 * sig[:, None] ** 2))  # (N, H)
                g = (gy[:, :, None] * gx[:, None, :]).reshape(len(sig), -1)
                maps.append(torch.where(g > 1e-6, g, torch.zeros_like(g)))  # avoid denormals
            self._basis_cache, self._basis_key = torch.stack(maps), key
        return self._basis_cache[0], self._basis_cache[1]

    def forward(self, amp_ua: torch.Tensor, patient: Patient, hard: bool = True, fast: bool = False) -> torch.Tensor:
        """Amplitudes (B, N) in uA -> percept (B, 1, H, W) in [0, 1].

        ``fast=True`` (training) replaces the exact amplitude-dependent Gaussian by a
        linear blend of two precomputed Gaussians at the size anchors I_lo (mean threshold
        current) and I_max, turning the render into two (B, N) @ (N, H*W) products. Over
        the supra-threshold range the phosphene sigma changes by < 20 %, and the blend's
        error is measured in the tests; evaluation always uses the exact renderer.
        """
        amp_ua = amp_ua * patient.alive
        act = self.activation(amp_ua)
        b = self.brightness(act, patient.threshold, hard=hard) * patient.alive
        if not fast:
            sig = self.sigma_deg(amp_ua, patient.ecc_deg)
            return self.render(b, sig, patient.x_deg, patient.y_deg)
        g_lo, g_hi = self._basis(patient)
        i_lo = float(self.p.threshold_current_ua(self.p.threshold_mean))
        t = ((amp_ua - i_lo) / (self.p.max_current_ua - i_lo)).clamp(0.0, 1.0)
        img = (b * (1 - t)) @ g_lo + (b * t) @ g_hi
        n = self.p.resolution
        return img.clamp(0.0, 1.0).view(-1, 1, n, n)


class TemporalSimulator:
    """Frame-by-frame dynamics with the habituation memory trace (Eq. 7-9 with trace B).

    Used to validate the temporal behaviour (response attenuation under repeated
    stimulation, Schmidt et al. 1996); the encoders are trained in the static mode.
    """

    def __init__(self, params: SimParams, n: int, dt_s: float, substeps: int = 10):
        self.p, self.dt, self.k = params, dt_s, substeps
        self.act = torch.zeros(n)
        self.trace = torch.zeros(n)

    def step(self, amp_ua: torch.Tensor, on_fraction: float = 1.0) -> torch.Tensor:
        """Advance one frame of duration dt with stimulation on for ``on_fraction`` of it. Returns activation."""
        p, h = self.p, self.dt / self.k
        decay_trace = -math.log(p.trace_decay_per_second)
        for _ in range(self.k):
            i_eff = torch.relu((amp_ua * 1e-6 - p.rheobase_a - self.trace) * p.pulse_width_s * p.frequency_hz) * on_fraction
            self.act = self.act + (-self.act / p.tau_act_s + i_eff) * h
            self.trace = self.trace + (-self.trace * decay_trace + p.trace_increase_rate * i_eff) * h
        return self.act
