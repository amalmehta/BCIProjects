"""Closed-loop WebGrid simulator.

This is a *simulator*, not a human (or monkey) result. It closes the loop

    simulated user  --intended velocity-->  neural encoding model  --spike counts-->
    decoder  --decoded velocity-->  cursor  --delayed visual feedback-->  simulated user

so that decoders trained on real data can be compared on the metric the product cares about:
bits per second on a grid-selection task (Neuralink's Webgrid; Nuyujukian et al. 2015).

Components
----------
* :class:`EncodingModel` - per-day neural encoding model fit to real data: a small MLP maps
  the recent history of intended velocity (and cursor position) to per-electrode Poisson
  log-rates. Shared, temporally correlated latent noise (AR(1), low rank, loadings from
  factor analysis of residuals) is added to the log-rates; its scale is calibrated so that a
  real-data Wiener filter achieves the same R^2 on simulated spikes as on held-out real spikes.
* :class:`UserModel` - delayed feedback controller in the spirit of optimal-feedback-control
  user models (Willett et al. 2019): it sees the cursor with a visual delay, predicts the
  present cursor position with an efference copy of its own recent intentions (a forward
  model), and commands a velocity toward the target with a saturating speed-distance profile
  fit to the monkey's own reaches, filtered by a first-order motor lag, with signal-dependent
  motor noise.
* :func:`run_webgrid` - an n x n grid; one cell is the target; selection by simulated click
  (with miss and false-click rates) or by dwell. Bitrate ``log2(N-1) max(Sc-Si,0)/t``.
"""
from __future__ import annotations

import dataclasses
import math

import numpy as np
import torch
import torch.nn as nn
from scipy.special import erf
from sklearn.decomposition import FactorAnalysis

from .features import DayData
from .linear import Decoder, WienerFilter
from .metrics import r2_score, webgrid_bitrate

DT = 0.02  # seconds per bin


# ----------------------------------------------------------------------------------------
# Neural encoding model
# ----------------------------------------------------------------------------------------
class _EncNet(nn.Module):
    def __init__(self, n_in: int, n_ch: int, hidden: int = 128):
        super().__init__()
        if hidden > 0:
            self.net = nn.Sequential(nn.Linear(n_in, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU(),
                                     nn.Linear(hidden, n_ch))
        else:  # Poisson GLM: log-linear in the features
            self.net = nn.Sequential(nn.Linear(n_in, n_ch))

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        return self.net(u)


@dataclasses.dataclass
class EncodingModel:
    """Maps intended-velocity history (and optionally position) to spike counts."""

    net: _EncNet
    hist: int  # bins of intended-velocity history
    fut: int  # bins of "future" intention (neural activity leads the decoded target)
    vscale: float
    pos_center: np.ndarray
    pos_scale: float
    U: np.ndarray  # (C, r) latent-noise loadings in log-rate space
    noise_scale: float = 1.0
    tau: float = 0.2  # latent noise time constant (s)
    use_pos: bool = False  # arm posture is absent in a paralysed user, so off by default

    @property
    def n_lags(self) -> int:
        return self.hist + self.fut

    @property
    def n_features(self) -> int:
        return 3 * self.n_lags + (2 if self.use_pos else 0)

    def features(self, V: np.ndarray, P: np.ndarray) -> np.ndarray:
        """Rows t: velocity lags [V[t+fut] ... V[t-hist+1]], speed lags |V| (same order) and,
        if ``use_pos``, position P[t]; all normalised and zero padded."""
        T = len(V)
        Vp = np.concatenate([np.zeros((self.hist - 1, 2)), V / self.vscale, np.zeros((self.fut, 2))])
        lags = [Vp[self.hist - 1 - k : self.hist - 1 - k + T] for k in range(-self.fut, self.hist)]
        cols = lags + [np.linalg.norm(v, axis=1, keepdims=True) for v in lags]
        if self.use_pos:
            cols.append((P - self.pos_center) / self.pos_scale)
        return np.concatenate(cols, 1).astype(np.float32)

    def step_features(self, hist: np.ndarray, p: np.ndarray) -> np.ndarray:
        """Features for one bin from the newest-first normalised intention buffer ``hist``."""
        cols = [hist.reshape(-1), np.linalg.norm(hist, axis=1)]
        if self.use_pos:
            cols.append((p - self.pos_center) / self.pos_scale)
        return np.concatenate(cols).astype(np.float32)

    @torch.no_grad()
    def log_rates(self, F: np.ndarray) -> np.ndarray:
        return self.net(torch.as_tensor(F)).numpy()

    def sample(self, V: np.ndarray, P: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Open-loop sampling of counts for a whole intended-velocity / position trajectory."""
        L = self.log_rates(self.features(V, P))
        eps = self.latent_noise(len(V), rng)
        return rng.poisson(np.exp(np.clip(L + eps, -10, 4))).astype(np.float32)

    def latent_noise(self, T: int, rng: np.random.Generator) -> np.ndarray:
        r = self.U.shape[1]
        rho = math.exp(-DT / self.tau)
        z = np.zeros((T, r))
        e = rng.standard_normal((T, r)) * math.sqrt(1 - rho**2)
        z[0] = rng.standard_normal(r)
        for t in range(1, T):
            z[t] = rho * z[t - 1] + e[t]
        return self.noise_scale * z @ self.U.T


def fit_encoding_model(day: DayData, vscale: float, hist: int = 8, fut: int = 5, steps: int = 1500,
                       rank: int = 4, hidden: int = 0, use_pos: bool = False, seed: int = 0) -> EncodingModel:
    """Fit the encoding model (Poisson NLL) and residual latent-noise loadings on a day's train
    block. ``hidden=0`` (default) is a Poisson GLM, which extrapolates sensibly to the sustained
    intentions that occur in closed loop; ``hidden>0`` is an MLP."""
    torch.manual_seed(seed)
    tr = day.train
    center = np.array([5.0, 55.0], np.float32)
    enc = EncodingModel(None, hist, fut, vscale, center, 65.0, np.zeros((day.X.shape[1], rank)), use_pos=use_pos)
    enc.net = _EncNet(enc.n_features, day.X.shape[1], hidden)
    F = torch.as_tensor(enc.features(day.Y[tr], day.pos[tr]))
    C = torch.as_tensor(day.X[tr])
    opt = torch.optim.Adam(enc.net.parameters(), lr=3e-3, weight_decay=1e-4)
    rng = np.random.default_rng(seed)
    for _ in range(steps):
        idx = torch.as_tensor(rng.integers(0, len(F), 2048))
        loss = nn.functional.poisson_nll_loss(enc.net(F[idx]), C[idx], log_input=True)
        opt.zero_grad()
        loss.backward()
        opt.step()
    # Latent noise loadings: FA on smoothed relative residuals ~ log-rate fluctuations.
    lam = np.exp(enc.log_rates(F.numpy()))
    k = 10
    cs = np.cumsum(np.concatenate([np.zeros((k, C.shape[1])), day.X[tr] - lam]), 0)
    cl = np.cumsum(np.concatenate([np.zeros((k, C.shape[1])), lam]), 0)
    rel = (cs[k:] - cs[:-k]) / np.maximum(cl[k:] - cl[:-k], 0.5)
    fa = FactorAnalysis(rank, random_state=seed).fit(rel)
    enc.U = fa.components_.T.astype(np.float32)
    return enc


def calibrate_noise(enc: EncodingModel, day: DayData, wiener: WienerFilter, grid=(0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0),
                    seed: int = 0) -> tuple[float, dict]:
    """Choose the latent-noise scale so a real-data Wiener filter's R^2 on simulated spikes
    (driven by the real intended velocities of the test block) matches its real test R^2."""
    te = day.test
    real = r2_score(day.Y[te], wiener.predict(day.X)[te])
    out = {}
    for s in grid:
        enc.noise_scale = s
        Xs = enc.sample(day.Y[te], day.pos[te], np.random.default_rng(seed))
        out[s] = r2_score(day.Y[te], wiener.predict(Xs))
    best = min(grid, key=lambda s: abs(out[s] - real))
    enc.noise_scale = best
    return best, {"real_r2": real, "sim_r2_by_scale": out}


# ----------------------------------------------------------------------------------------
# Simulated user
# ----------------------------------------------------------------------------------------
@dataclasses.dataclass
class UserModel:
    """Delayed feedback controller producing intended velocity (mm/s)."""

    vmax: float = 120.0  # asymptotic speed (mm/s)
    d0: float = 25.0  # distance scale of the speed profile (mm)
    delay_bins: int = 5  # visual feedback delay (100 ms)
    tau_motor: float = 0.08  # first-order motor lag (s)
    motor_noise: float = 0.1  # signal-dependent noise (fraction of speed)
    efference: float = 1.0  # weight of the efference-copy forward model (0 = no prediction)

    def command(self, target: np.ndarray, p_hat: np.ndarray, stop_radius: float) -> np.ndarray:
        e = target - p_hat
        dist = float(np.linalg.norm(e))
        if dist < stop_radius:
            return np.zeros(2)
        return self.vmax * (1 - math.exp(-dist / self.d0)) * e / dist


def fit_user_speed_profile(days: list[DayData]) -> tuple[float, float]:
    """Fit ``speed_toward_target = vmax (1 - exp(-dist/d0))`` to the monkey's reaches."""
    from scipy.optimize import curve_fit

    d_all, s_all = [], []
    for d in days:
        e = d.target[d.train] - d.pos[d.train]
        dist = np.linalg.norm(e, axis=1)
        ok = dist > 2
        s = (d.Y[d.train][ok] * e[ok]).sum(1) / dist[ok]
        d_all.append(dist[ok])
        s_all.append(s)
    dist, sp = np.concatenate(d_all), np.concatenate(s_all)
    (vmax, d0), _ = curve_fit(lambda x, a, b: a * (1 - np.exp(-x / b)), dist, sp, p0=(100.0, 20.0),
                              bounds=([10, 1], [1000, 500]))
    return float(vmax), float(d0)


# ----------------------------------------------------------------------------------------
# WebGrid task
# ----------------------------------------------------------------------------------------
@dataclasses.dataclass
class WebGridConfig:
    """Task and selection parameters."""

    grid: int = 35
    side_mm: float = 130.0
    center: tuple[float, float] = (5.0, 55.0)
    seconds: float = 60.0
    selection: str = "click"  # "click" or "dwell"
    dwell_s: float = 0.5
    click_latency_bins: int = 2
    click_speed_frac: float = 0.25  # user clicks when perceived speed < frac * vmax ...
    click_hold_s: float = 0.1  # ... and the cursor has looked inside the target this long
    click_refractory_s: float = 0.3
    click_miss_prob: float = 0.05
    false_click_rate_hz: float = 0.05


@dataclasses.dataclass
class WebGridResult:
    bps: float
    n_correct: int
    n_incorrect: int
    seconds: float
    grid: int
    cursor: np.ndarray  # (T, 2)
    targets: list  # list of (t_seconds, (row, col))
    selections: list  # list of (t_seconds, (row, col), correct)
    intent: np.ndarray  # (T, 2) the simulated user's intended velocity
    decoded: np.ndarray  # (T, 2) raw decoder output
    counts: np.ndarray  # (T, C) simulated spike counts
    target_pos: np.ndarray  # (T, 2) centre of the current target

    def summary(self) -> dict:
        return {"bps": self.bps, "n_correct": self.n_correct, "n_incorrect": self.n_incorrect,
                "seconds": self.seconds, "grid": self.grid}


def run_webgrid(decoder: Decoder | None, enc: EncodingModel, user: UserModel, cfg: WebGridConfig, gain: float = 1.0,
                smooth: float = 0.0, seed: int = 0) -> WebGridResult:
    """Simulate one closed-loop WebGrid session and return its bitrate and full trace.

    The cursor moves with ``gain`` x an exponentially smoothed decoded velocity
    (``v_s <- smooth v_s + (1 - smooth) v_dec``); gain and smoothing are the two standard
    closed-loop tuning knobs (Willett et al. 2019) and are tuned per decoder.
    """
    rng = np.random.default_rng(seed)
    n, side = cfg.grid, cfg.side_mm
    cell = side / n
    origin = np.asarray(cfg.center) - side / 2
    T = int(cfg.seconds / DT)

    def cell_of(p):
        ij = np.floor((p - origin) / cell).astype(int)
        return int(np.clip(ij[1], 0, n - 1)), int(np.clip(ij[0], 0, n - 1))  # (row, col)

    def center_of(rc):
        return origin + (np.array([rc[1], rc[0]]) + 0.5) * cell

    def new_target(prev):
        while True:
            rc = (int(rng.integers(n)), int(rng.integers(n)))
            if rc != prev:
                return rc

    if decoder is not None:
        decoder.reset()
    noise = enc.latent_noise(T + 1, rng)
    # Intention buffer, newest first. The encoding model's "present" lags the newest intention
    # by ``enc.fut`` bins, so neural activity leads the decoded target as in the real data.
    hist = np.zeros((enc.n_lags, 2), np.float32)
    p = np.asarray(cfg.center, np.float64).copy()
    path = np.zeros((T, 2))
    rec_int, rec_dec = np.zeros((T, 2)), np.zeros((T, 2))
    rec_counts, rec_tgt = np.zeros((T, enc.U.shape[0]), np.float32), np.zeros((T, 2))
    v_int = np.zeros(2)
    v_s = np.zeros(2)
    tgt = new_target(None)
    targets, selections = [(0.0, tgt)], []
    n_c = n_i = 0
    dwell_cell, dwell_t = None, 0
    pending_click, last_click, inside = -1, -10**9, 0
    stop_r = 0.35 * cell
    for t in range(T):
        # --- user: perceive delayed cursor, extrapolate, command velocity
        td = max(0, t - user.delay_bins)
        p_seen = path[td] if t > 0 else p
        v_seen = (path[td] - path[max(0, td - 1)]) / DT if t > 1 else np.zeros(2)
        # Internal forward model (efference copy): add the user's own recent intentions that
        # cannot yet be visible, i.e. those issued within the visual delay + neural lead.
        n_unseen = min(t, user.delay_bins + enc.fut)
        p_hat = p_seen + DT * user.efference * rec_int[t - n_unseen : t].sum(0)
        cmd = user.command(center_of(tgt), p_hat, stop_r)
        v_int = v_int + (DT / user.tau_motor) * (cmd - v_int)
        v_out = v_int + user.motor_noise * np.linalg.norm(v_int) * rng.standard_normal(2)
        rec_int[t] = v_out
        # --- neural encoding of the intention
        hist = np.roll(hist, 1, axis=0)
        hist[0] = v_out / enc.vscale
        feat = enc.step_features(hist, p)
        lr = _fast_mlp(enc, feat)
        counts = rng.poisson(np.exp(np.clip(lr + noise[t], -10, 4))).astype(np.float32)
        # --- decode and move the cursor
        # Oracle: perfect decoding at the same latency as a real decoder (which estimates the
        # intention ``enc.fut`` bins in the past, see EncodingModel.features).
        v_dec = rec_int[max(0, t - enc.fut)] if decoder is None else decoder.step(counts)
        v_s = smooth * v_s + (1 - smooth) * v_dec
        p = np.clip(p + gain * DT * v_s, origin, origin + side)
        path[t] = p
        rec_dec[t], rec_counts[t], rec_tgt[t] = v_dec, counts, center_of(tgt)
        # --- selection
        sel = None
        if cfg.selection == "dwell":
            c = cell_of(p)
            dwell_t = dwell_t + 1 if c == dwell_cell else 1
            dwell_cell = c
            if dwell_t * DT >= cfg.dwell_s:
                sel, dwell_t = c, 0
        else:
            # The user clicks only when the cursor *looks* inside the target (delayed view) and
            # their forward model agrees it is still there.
            in_tgt = max(np.max(np.abs(p_hat - center_of(tgt))), np.max(np.abs(p_seen - center_of(tgt)))) < 0.5 * cell
            inside = inside + 1 if in_tgt else 0
            slow = np.linalg.norm(v_seen) < cfg.click_speed_frac * user.vmax
            if inside * DT >= cfg.click_hold_s and slow and pending_click < 0 and (t - last_click) * DT > cfg.click_refractory_s:
                last_click = t
                if rng.random() > cfg.click_miss_prob:
                    pending_click = t + cfg.click_latency_bins
            if pending_click < 0 and rng.random() < cfg.false_click_rate_hz * DT:
                pending_click = t + cfg.click_latency_bins
            if pending_click >= 0 and t >= pending_click:
                sel, pending_click = cell_of(p), -1
        if sel is not None:
            ok = sel == tgt
            selections.append((round((t + 1) * DT, 3), sel, bool(ok)))
            if ok:
                n_c += 1
                tgt = new_target(tgt)
                targets.append((round((t + 1) * DT, 3), tgt))
            else:
                n_i += 1
    bps = webgrid_bitrate(n * n, n_c, n_i, cfg.seconds)
    return WebGridResult(bps, n_c, n_i, cfg.seconds, n, path, targets, selections, rec_int, rec_dec, rec_counts,
                         rec_tgt)


def _fast_mlp(enc: EncodingModel, feat: np.ndarray) -> np.ndarray:
    """Numpy forward pass of the encoding MLP (much faster than torch for one sample)."""
    if not hasattr(enc, "_np_layers"):
        enc._np_layers = [(m.weight.detach().numpy().T.copy(), m.bias.detach().numpy().copy())
                          for m in enc.net.net if isinstance(m, nn.Linear)]
    h = feat
    for i, (W, b) in enumerate(enc._np_layers):
        h = h @ W + b
        if i < len(enc._np_layers) - 1:
            h = 0.5 * h * (1 + erf(h / math.sqrt(2)))  # exact GELU, as in torch
    return h
