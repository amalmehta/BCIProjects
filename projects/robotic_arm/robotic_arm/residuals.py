"""Closed-loop decoder noise taken from *real* FALCON H1 decoder errors.

The synthetic closed loop (``encoding`` + ``OnlineDecoder``) produces decoder errors from
a Poisson encoding model whose noise level is matched to the real decoder only in
offline R². Here the decoder is instead replaced by an empirical model of a real
decoder's output given the cued (intended) velocity, fitted to held-out H1 predictions:

    z_hat_t = sum_{k=0}^{K-1} H_k z_{t-k} + b + s(|z_t|) * r_t

* ``H`` (a causal FIR from intended to decoded velocity, ridge-fitted) captures the real
  decoder's gain/shrinkage, lag and cross-talk between DoFs.
* ``r_t`` are real residual sequences ``z_hat - H*z`` replayed by a moving-block bootstrap
  (Kunsch 1989): blocks of ``block_len`` consecutive bins are drawn at random from the
  residual pool, which keeps their temporal autocorrelation and cross-DoF covariance.
* With ``conditioned=True`` the residuals are first standardised by a per-DoF
  velocity-dependent scale ``s_d(|z|) = a_d + c_d |z_d|`` fitted by least squares to
  ``|r|`` (heteroscedastic, speed-dependent error), and re-scaled by the simulated user's
  current intended speed when replayed.

Only bins with ``eval_mask`` are used, and blocks never cross a masked gap.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .data import lag_features
from .metrics import r2_variance_weighted


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Contiguous [start, stop) runs where ``mask`` is True."""
    m = np.concatenate([[False], mask.astype(bool), [False]])
    d = np.diff(m.astype(int))
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0]))


def integrated_autocorr_time(e: np.ndarray, max_lag: int = 200) -> np.ndarray:
    """Per-column integrated autocorrelation time (ticks), Sokal-style windowed estimate."""
    taus = []
    for d in range(e.shape[1]):
        x = e[:, d] - e[:, d].mean()
        ac = np.correlate(x, x, "full")[len(x) - 1:][:max_lag]
        ac = ac / ac[0]
        cut = np.argmax(ac < 0.05) if (ac < 0.05).any() else max_lag
        taus.append(1 + 2 * ac[1:cut].sum())
    return np.array(taus)


@dataclass
class ResidualModel:
    H: np.ndarray  # (K, 7, 7): z_hat_t += H[k] @ z_{t-k}
    b: np.ndarray  # (7,)
    scale_a: np.ndarray  # (7,)
    scale_c: np.ndarray  # (7,)
    segments: list  # list of (T_i, 7) standardised (or raw) residual arrays
    block_len: int
    conditioned: bool
    source: str = ""
    stats: dict = field(default_factory=dict)

    @property
    def n_lags(self) -> int:
        return self.H.shape[0]

    def scale(self, z: np.ndarray) -> np.ndarray:
        if not self.conditioned:
            return np.ones(z.shape[-1]) if z.ndim == 1 else np.ones_like(z)
        return np.maximum(self.scale_a + self.scale_c * np.abs(z), 1e-3)

    def signal(self, z_hist: np.ndarray) -> np.ndarray:
        """Deterministic part for a whole trajectory ``z_hist`` (T, 7)."""
        X = lag_features(z_hist, self.n_lags)
        return X @ self.H.reshape(-1, 7) + self.b

    def sample_residuals(self, T: int, rng: np.random.Generator) -> np.ndarray:
        """Moving-block bootstrap of ``T`` residual bins."""
        out, n = [], 0
        lens = np.array([len(s) for s in self.segments], float)
        p = np.maximum(lens - self.block_len + 1, 0)
        p = p / p.sum()
        while n < T:
            i = rng.choice(len(self.segments), p=p)
            s = rng.integers(0, len(self.segments[i]) - self.block_len + 1)
            out.append(self.segments[i][s:s + self.block_len])
            n += self.block_len
        return np.concatenate(out)[:T]

    def simulate(self, z: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        return self.signal(z) + self.scale(z) * self.sample_residuals(len(z), rng)


def fit_residual_model(truth: list[np.ndarray], pred: list[np.ndarray], masks: list[np.ndarray],
                       n_lags: int = 15, ridge: float = 1.0, block_len: int = 100,
                       conditioned: bool = True, source: str = "") -> ResidualModel:
    """Fit the FIR + residual pool to real (truth, prediction) pairs in z units."""
    Xs, Ys = [], []
    for y, p, m in zip(truth, pred, masks):
        X = lag_features(np.asarray(y, float), n_lags)
        Xs.append(X[m])
        Ys.append(np.asarray(p, float)[m])
    X, Y = np.concatenate(Xs), np.concatenate(Ys)
    xm, ym = X.mean(0), Y.mean(0)
    Xc = X - xm
    W = np.linalg.solve(Xc.T @ Xc + ridge * np.eye(X.shape[1]), Xc.T @ (Y - ym))
    b = ym - xm @ W
    H = W.reshape(n_lags, 7, 7)
    # residuals per contiguous masked run, and the velocity-dependent scale
    segs_raw, absz, absr = [], [], []
    for y, p, m in zip(truth, pred, masks):
        y, p = np.asarray(y, float), np.asarray(p, float)
        r = p - (lag_features(y, n_lags) @ W + b)
        for s, e in _runs(m):
            if e - s >= block_len:
                segs_raw.append((y[s:e], r[s:e]))
    if not segs_raw:
        raise ValueError("no masked run is longer than block_len")
    Z = np.concatenate([z for z, _ in segs_raw])
    R = np.concatenate([r for _, r in segs_raw])
    a, c = np.ones(7), np.zeros(7)
    for d in range(7):
        A = np.stack([np.ones(len(Z)), np.abs(Z[:, d])], 1)
        coef = np.linalg.lstsq(A, np.abs(R[:, d]), rcond=None)[0] * np.sqrt(np.pi / 2)  # E|N(0,s)| = s sqrt(2/pi)
        a[d], c[d] = max(coef[0], 1e-3), max(coef[1], 0.0)
    model = ResidualModel(H, b, a, c, [], block_len, conditioned, source)
    model.segments = [r / model.scale(z) for z, r in segs_raw] if conditioned else [r for _, r in segs_raw]
    Yall = np.concatenate([np.asarray(p, float)[m] for p, m in zip(pred, masks)])
    Tall = np.concatenate([np.asarray(y, float)[m] for y, m in zip(truth, masks)])
    model.stats = {
        "source": source, "n_bins": int(len(R)), "n_segments": len(segs_raw),
        "real_r2_vw": r2_variance_weighted(Tall, Yall),
        "fir_r2_vw": r2_variance_weighted(Tall, np.concatenate([lag_features(np.asarray(y, float), n_lags)[m] @ W + b
                                                                for y, m in zip(truth, masks)])),
        "resid_std": R.std(0).tolist(),
        "resid_tau_ticks": integrated_autocorr_time(R).tolist(),
        "scale_a": a.tolist(), "scale_c": c.tolist(),
        "gain_diag": np.diag(H.sum(0)).tolist(),
    }
    return model


class OnlineResidualDecoder:
    """Streaming decoder driven by real residuals: ``step(z_intended) -> z_hat``.

    Applies the same output gain correction as the synthetic ``OnlineDecoder``.
    """

    takes_intent = True

    def __init__(self, model: ResidualModel, gain: np.ndarray | None = None, horizon: int = 2000):
        self.m = model
        self.gain = np.ones(7) if gain is None else gain
        self.horizon = horizon

    def reset(self, rng: np.random.Generator) -> None:
        self._hist = np.zeros((self.m.n_lags, 7))
        self._res = self.m.sample_residuals(self.horizon, rng)
        self._t = 0

    def step(self, z: np.ndarray) -> np.ndarray:
        self._hist = np.roll(self._hist, 1, axis=0)
        self._hist[0] = z
        sig = np.einsum("kij,kj->i", self.m.H.transpose(0, 2, 1), self._hist) + self.m.b
        r = self._res[self._t % len(self._res)]
        self._t += 1
        return (sig + self.m.scale(z) * r) * self.gain

    def run(self, z: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        self.horizon = max(self.horizon, len(z))
        self.reset(rng)
        return np.array([self.step(x) for x in z])
