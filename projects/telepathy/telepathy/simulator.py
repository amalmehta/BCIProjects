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
    """Simulate one closed-loop WebGrid session (a batch of one, see :func:`run_webgrid_batch`).

    The cursor moves with ``gain`` x an exponentially smoothed decoded velocity
    (``v_s <- smooth v_s + (1 - smooth) v_dec``); gain and smoothing are the two standard
    closed-loop tuning knobs (Willett et al. 2019) and are tuned per decoder.
    """
    return run_webgrid_batch(decoder, enc, user, cfg, [seed], gain, smooth)[0]


@dataclasses.dataclass
class SessionNoise:
    """Decoder-independent random numbers of one simulated session (common random numbers).

    Every decoder evaluated with the same seed sees the same target sequence, latent neural
    noise, motor noise and click randomness, so bitrate differences between decoders are paired.
    Only the Poisson spike draws depend on the decoder (through the rates), and they come from a
    separate per-seed stream."""

    targets: np.ndarray  # (K, 2) sequence of (row, col) cells, consecutive cells differ
    latent: np.ndarray  # (T + 1, C) latent log-rate noise
    motor: np.ndarray  # (T, 2) standard normals
    u_miss: np.ndarray  # (T,) uniforms for missed clicks
    u_false: np.ndarray  # (T,) uniforms for spurious clicks
    pois: np.random.Generator

    @classmethod
    def draw(cls, seed: int, enc: EncodingModel, n: int, T: int) -> "SessionNoise":
        rng = np.random.default_rng(seed)
        K = T // 10 + 2  # >= max selections possible (refractory 0.3 s)
        cells = [(int(rng.integers(n)), int(rng.integers(n)))]
        while len(cells) < K:
            rc = (int(rng.integers(n)), int(rng.integers(n)))
            if rc != cells[-1]:
                cells.append(rc)
        latent = enc.latent_noise(T + 1, rng)
        return cls(np.array(cells), latent, rng.standard_normal((T, 2)), rng.random(T), rng.random(T),
                   np.random.default_rng([seed, 7919]))


def _vec(x, B: int) -> np.ndarray:
    return np.broadcast_to(np.asarray(x, np.float64), (B,)).copy()


def run_webgrid_batch(decoder: Decoder | None, enc: EncodingModel, user: UserModel, cfg: WebGridConfig,
                      seeds, gain=1.0, smooth=0.0, hold=None) -> list[WebGridResult]:
    """Simulate ``len(seeds)`` independent closed-loop sessions in lock-step (vectorised over sessions).

    ``gain``, ``smooth`` and ``hold`` (click hold time, default ``cfg.click_hold_s``) may be scalars
    or one value per session, so a whole tuning grid runs as one batch. Session ``b`` draws its
    decoder-independent randomness from ``SessionNoise.draw(seeds[b])`` (common random numbers),
    so the same seed gives the same task and noise for every decoder, and a session's result does
    not depend on which other sessions share the batch. The decoder (if any) must implement
    ``reset_batch``/``step_batch`` or ``reset``/``step`` (then it is deep-copied per session).
    """
    import copy as _copy

    seeds = list(seeds)
    B = len(seeds)
    n, side = cfg.grid, cfg.side_mm
    cell = side / n
    origin = np.asarray(cfg.center, np.float64) - side / 2
    T = int(cfg.seconds / DT)
    gain, smooth = _vec(gain, B), _vec(smooth, B)
    hold = _vec(cfg.click_hold_s if hold is None else hold, B)
    noise = [SessionNoise.draw(s, enc, n, T) for s in seeds]
    tgt_seq = np.stack([z.targets for z in noise])  # (B, K, 2)
    latent = np.stack([z.latent for z in noise])  # (B, T+1, C)
    motor = np.stack([z.motor for z in noise], 1)  # (T, B, 2)
    u_miss = np.stack([z.u_miss for z in noise], 1)
    u_false = np.stack([z.u_false for z in noise], 1)
    batched = decoder is not None and hasattr(decoder, "step_batch")
    copies = None
    if decoder is not None:
        if batched:
            decoder.reset_batch(B)
        else:
            copies = [_copy.deepcopy(decoder) for _ in range(B)]
            for c in copies:
                c.reset()
    C = enc.U.shape[0]
    hist = np.zeros((B, enc.n_lags, 2), np.float32)
    p = np.tile(np.asarray(cfg.center, np.float64), (B, 1))
    path = np.zeros((B, T, 2))
    rec_int, rec_dec = np.zeros((B, T, 2)), np.zeros((B, T, 2))
    rec_counts, rec_tgt = np.zeros((B, T, C), np.float32), np.zeros((B, T, 2))
    v_int, v_s = np.zeros((B, 2)), np.zeros((B, 2))
    k_tgt = np.zeros(B, int)  # index into each session's target sequence
    t_start = [[0.0] for _ in range(B)]
    selections: list[list] = [[] for _ in range(B)]
    n_c, n_i = np.zeros(B, int), np.zeros(B, int)
    dwell_cell, dwell_t = np.full((B, 2), -1), np.zeros(B, int)
    pending, last_click, inside = np.full(B, -1), np.full(B, -10**9), np.zeros(B, int)
    stop_r = 0.35 * cell
    ar = np.arange(B)

    def centers(rc):  # (B, 2) (row, col) -> (B, 2) mm
        return origin + (rc[:, ::-1] + 0.5) * cell

    def cells_of(pp):
        ij = np.clip(np.floor((pp - origin) / cell).astype(int), 0, n - 1)
        return ij[:, ::-1]  # (row, col)

    for t in range(T):
        tc = centers(tgt_seq[ar, k_tgt])
        # --- user: perceive delayed cursor, extrapolate with an efference copy, command velocity
        td = max(0, t - user.delay_bins)
        p_seen = path[:, td] if t > 0 else p
        v_seen = (path[:, td] - path[:, max(0, td - 1)]) / DT if t > 1 else np.zeros((B, 2))
        n_unseen = min(t, user.delay_bins + enc.fut)
        p_hat = p_seen + DT * user.efference * rec_int[:, t - n_unseen : t].sum(1)
        e = tc - p_hat
        dist = np.linalg.norm(e, axis=1, keepdims=True)
        cmd = user.vmax * (1 - np.exp(-dist / user.d0)) * e / np.maximum(dist, 1e-9)
        cmd[dist[:, 0] < stop_r] = 0
        v_int = v_int + (DT / user.tau_motor) * (cmd - v_int)
        v_out = v_int + user.motor_noise * np.linalg.norm(v_int, axis=1, keepdims=True) * motor[t]
        rec_int[:, t] = v_out
        # --- neural encoding of the intention
        hist = np.roll(hist, 1, axis=1)
        hist[:, 0] = v_out / enc.vscale
        cols = [hist.reshape(B, -1), np.linalg.norm(hist, axis=2)]
        if enc.use_pos:
            cols.append((p - enc.pos_center) / enc.pos_scale)
        lam = np.exp(np.clip(_fast_mlp(enc, np.concatenate(cols, 1).astype(np.float32)) + latent[:, t], -10, 4))
        counts = np.stack([z.pois.poisson(l) for z, l in zip(noise, lam)]).astype(np.float32)
        # --- decode and move the cursor (oracle: true intention at a decoder's latency)
        if decoder is None:
            v_dec = rec_int[:, max(0, t - enc.fut)]
        elif batched:
            if hasattr(decoder, "set_cursor"):  # decoders that use the known cursor position (ReFIT-KF)
                decoder.set_cursor(p)
            v_dec = np.asarray(decoder.step_batch(counts), np.float64)
        else:
            v_dec = np.stack([c.step(x) for c, x in zip(copies, counts)]).astype(np.float64)
        v_s = smooth[:, None] * v_s + (1 - smooth[:, None]) * v_dec
        p = np.clip(p + gain[:, None] * DT * v_s, origin, origin + side)
        path[:, t] = p
        rec_dec[:, t], rec_counts[:, t], rec_tgt[:, t] = v_dec, counts, tc
        # --- selection
        fire = np.zeros(B, bool)
        if cfg.selection == "dwell":
            c = cells_of(p)
            same = np.all(c == dwell_cell, axis=1)
            dwell_t = np.where(same, dwell_t + 1, 1)
            dwell_cell = c
            fire = dwell_t * DT >= cfg.dwell_s
            dwell_t[fire] = 0
        else:
            in_tgt = np.maximum(np.max(np.abs(p_hat - tc), 1), np.max(np.abs(p_seen - tc), 1)) < 0.5 * cell
            inside = np.where(in_tgt, inside + 1, 0)
            slow = np.linalg.norm(v_seen, axis=1) < cfg.click_speed_frac * user.vmax
            can = (inside * DT >= hold - 1e-9) & slow & (pending < 0) & ((t - last_click) * DT > cfg.click_refractory_s)
            last_click[can] = t
            pending[can & (u_miss[t] > cfg.click_miss_prob)] = t + cfg.click_latency_bins
            spurious = (pending < 0) & (u_false[t] < cfg.false_click_rate_hz * DT)
            pending[spurious] = t + cfg.click_latency_bins
            fire = (pending >= 0) & (t >= pending)
            pending[fire] = -1
        if fire.any():
            sel = cells_of(p)
            for b in np.nonzero(fire)[0]:
                rc = (int(sel[b, 0]), int(sel[b, 1]))
                cur = tuple(int(v) for v in tgt_seq[b, k_tgt[b]])
                ok = rc == cur
                selections[b].append((round((t + 1) * DT, 3), rc, bool(ok)))
                if ok:
                    n_c[b] += 1
                    k_tgt[b] += 1
                    t_start[b].append(round((t + 1) * DT, 3))
                else:
                    n_i[b] += 1
    out = []
    for b in range(B):
        targets = [(ts, tuple(int(v) for v in tgt_seq[b, k])) for k, ts in enumerate(t_start[b])]
        out.append(WebGridResult(webgrid_bitrate(n * n, int(n_c[b]), int(n_i[b]), cfg.seconds), int(n_c[b]),
                                 int(n_i[b]), cfg.seconds, n, path[b], targets, selections[b], rec_int[b], rec_dec[b],
                                 rec_counts[b], rec_tgt[b]))
    return out


def _fast_mlp(enc: EncodingModel, feat: np.ndarray) -> np.ndarray:
    """Numpy forward pass of the encoding MLP (much faster than torch for one sample)."""
    if not hasattr(enc, "_np_layers"):
        enc._np_layers = [(m.weight.detach().numpy().T.copy(), m.bias.detach().numpy().copy())
                          for m in enc.net.net if isinstance(m, nn.Linear)]
    h = feat  # (F,) or (B, F)
    for i, (W, b) in enumerate(enc._np_layers):
        h = h @ W + b
        if i < len(enc._np_layers) - 1:
            h = 0.5 * h * (1 + erf(h / math.sqrt(2)))  # exact GELU, as in torch
    return h
