"""Classical decoders: ridge Wiener filter, velocity Kalman filter and the position-velocity(-acceleration)
Kalman filter used as the strong Kalman / ReFIT-KF baseline.

Both share the :class:`Decoder` interface used everywhere else:

* ``predict(X)``  - offline, causal decoding of a whole (T, C) count stream -> (T, 2) velocity
* ``reset()`` / ``step(x_t)`` - streaming decoding, one bin at a time (closed-loop simulator)
* ``reset_batch(B)`` / ``step_batch(X_t)`` - B independent streams decoded in lock-step, (B, C) -> (B, 2)
  (the vectorised simulator runs many paired sessions at once)
* ``mu`` / ``sd``  - per-channel normalisation statistics. Replacing them with statistics of a
  new day's *unlabelled* data is the "re-centering" adaptation (Jarosiewicz et al. 2015).
"""
from __future__ import annotations

import copy

import numpy as np

from .features import lagged


class Decoder:
    """Base class: holds normalisation statistics and the velocity scale."""

    mu: np.ndarray
    sd: np.ndarray
    vscale: float = 1.0

    def normalize(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mu) / self.sd

    def predict(self, X: np.ndarray) -> np.ndarray:  # pragma: no cover - interface
        raise NotImplementedError

    def reset(self) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    def step(self, x: np.ndarray) -> np.ndarray:  # pragma: no cover - interface
        raise NotImplementedError

    # Batched streaming. Fallback: B independent copies (subclasses override with vectorised code).
    def reset_batch(self, B: int) -> None:
        self._copies = [copy.deepcopy(self) for _ in range(B)]
        for c in self._copies:
            c.reset()

    def step_batch(self, X: np.ndarray) -> np.ndarray:
        return np.stack([c.step(x) for c, x in zip(self._copies, X)]).astype(np.float32)

    def with_stats(self, mu: np.ndarray, sd: np.ndarray) -> "Decoder":
        """Copy of this decoder using new normalisation statistics (re-centering)."""
        d = copy.deepcopy(self)
        d.mu, d.sd = mu.astype(np.float32), sd.astype(np.float32)
        return d


class WienerFilter(Decoder):
    """Ridge regression from ``n_lags`` bins of spike history to velocity (a Wiener filter).

    ``fit`` optionally shrinks toward prior weights ``W0`` instead of zero, which is the
    natural few-shot recalibration of a linear decoder:
    ``W = argmin ||Y - [Z 1] W||^2 + alpha ||W - W0||^2``. Without a prior the bias is
    unpenalised; with a prior it is shrunk toward the prior bias too (otherwise one short
    reach would set the velocity offset on its own).
    """

    def __init__(self, n_lags: int = 10, alpha: float = 1e3):
        self.n_lags, self.alpha = n_lags, alpha
        self.W: np.ndarray | None = None

    def _design(self, X: np.ndarray) -> np.ndarray:
        Z = lagged(self.normalize(X), self.n_lags)
        return np.concatenate([Z, np.ones((len(Z), 1), np.float32)], 1)

    def fit(self, segments: list[tuple[np.ndarray, np.ndarray]], mu: np.ndarray, sd: np.ndarray,
            W0: np.ndarray | None = None) -> "WienerFilter":
        """Fit on a list of contiguous (X, Y) segments (lags never straddle segment edges)."""
        self.mu, self.sd = mu, sd
        Z = np.concatenate([self._design(X) for X, _ in segments])
        Y = np.concatenate([Y for _, Y in segments]).astype(np.float64)
        D = Z.shape[1]
        reg = self.alpha * np.eye(D)
        if W0 is None:
            reg[-1, -1] = 0.0
        prior = np.zeros((D, Y.shape[1])) if W0 is None else W0
        A = Z.T.astype(np.float64) @ Z + reg
        B = Z.T.astype(np.float64) @ Y + reg @ prior
        self.W = np.linalg.solve(A, B).astype(np.float32)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._design(X) @ self.W

    def reset(self) -> None:
        self._buf = np.zeros((self.n_lags, len(self.mu)), np.float32)

    def step(self, x: np.ndarray) -> np.ndarray:
        self._buf = np.roll(self._buf, 1, axis=0)
        self._buf[0] = self.normalize(x)
        return np.concatenate([self._buf.reshape(-1), [1.0]]).astype(np.float32) @ self.W

    def reset_batch(self, B: int) -> None:
        self._bbuf = np.zeros((B, self.n_lags, len(self.mu)), np.float32)

    def step_batch(self, X: np.ndarray) -> np.ndarray:
        self._bbuf = np.roll(self._bbuf, 1, axis=1)
        self._bbuf[:, 0] = self.normalize(X)
        return (self._bbuf.reshape(len(X), -1) @ self.W[:-1] + self.W[-1]).astype(np.float32)


class KalmanFilter(Decoder):
    """Velocity Kalman filter (Wu et al. 2006) on causally smoothed, normalised counts.

    State ``v_t`` (2-D velocity, in units of ``vscale``) with ``v_t = A v_{t-1} + w``,
    observation ``z_t = H v_t + d + q``. All matrices are fit by least squares on training
    data; decoding uses the steady-state Kalman gain, as in clinical ReFIT-KF systems.
    ``q_scale`` multiplies the fitted observation-noise covariance; it trades smoothing for
    lag and is tuned on validation data (the usual practical KF smoothing knob).
    """

    def __init__(self, smooth_bins: int = 3, ridge: float = 1.0, q_scale: float = 1.0):
        self.smooth_bins, self.ridge, self.q_scale = smooth_bins, ridge, q_scale

    def _obs(self, X: np.ndarray) -> np.ndarray:
        Z = self.normalize(X)
        c = np.cumsum(np.concatenate([np.zeros((self.smooth_bins, Z.shape[1]), np.float32), Z]), 0)
        return (c[self.smooth_bins :] - c[: -self.smooth_bins]) / self.smooth_bins

    def fit(self, segments: list[tuple[np.ndarray, np.ndarray]], mu: np.ndarray, sd: np.ndarray) -> "KalmanFilter":
        self.mu, self.sd = mu, sd
        Ys = [Y.astype(np.float64) for _, Y in segments]
        self.vscale = float(np.concatenate(Ys).std())
        Ys = [Y / self.vscale for Y in Ys]
        Zs = [self._obs(X).astype(np.float64) for X, _ in segments]
        # State transition.
        Yp = np.concatenate([Y[:-1] for Y in Ys])
        Yn = np.concatenate([Y[1:] for Y in Ys])
        A = np.linalg.lstsq(Yp, Yn, rcond=None)[0].T
        Wn = np.cov((Yn - Yp @ A.T).T)
        # Observation model with bias.
        Y = np.concatenate(Ys)
        Z = np.concatenate(Zs)
        Y1 = np.concatenate([Y, np.ones((len(Y), 1))], 1)
        Hd = np.linalg.solve(Y1.T @ Y1 + self.ridge * np.eye(3), Y1.T @ Z).T
        H, d = Hd[:, :2], Hd[:, 2]
        R = Z - Y1 @ Hd.T
        Q = self.q_scale * (np.cov(R.T) + 1e-3 * np.eye(Z.shape[1]))
        # Steady-state gain by iterating the Riccati recursion.
        P = np.eye(2)
        Qi = np.linalg.inv(Q)
        for _ in range(500):
            Pp = A @ P @ A.T + Wn
            P = np.linalg.inv(np.linalg.inv(Pp) + H.T @ Qi @ H)
        K = P @ H.T @ Qi
        self.A, self.H, self.d, self.K = A, H, d, K
        self.M1 = (np.eye(2) - K @ H) @ A  # v_t = M1 v_{t-1} + K (z_t - d)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        Z = self._obs(X).astype(np.float64) - self.d
        KZ = Z @ self.K.T
        out = np.zeros((len(Z), 2))
        v = np.zeros(2)
        for t in range(len(Z)):
            v = self.M1 @ v + KZ[t]
            out[t] = v
        return (out * self.vscale).astype(np.float32)

    def reset(self) -> None:
        self._v = np.zeros(2)
        self._buf = np.zeros((self.smooth_bins, len(self.mu)), np.float32)

    def step(self, x: np.ndarray) -> np.ndarray:
        self._buf = np.roll(self._buf, 1, axis=0)
        self._buf[0] = self.normalize(x)
        z = self._buf.mean(0) - self.d
        self._v = self.M1 @ self._v + self.K @ z
        return (self._v * self.vscale).astype(np.float32)

    def reset_batch(self, B: int) -> None:
        self._bv = np.zeros((B, 2))
        self._bbuf = np.zeros((B, self.smooth_bins, len(self.mu)), np.float32)

    def step_batch(self, X: np.ndarray) -> np.ndarray:
        self._bbuf = np.roll(self._bbuf, 1, axis=1)
        self._bbuf[:, 0] = self.normalize(X)
        z = self._bbuf.mean(1) - self.d
        self._bv = self._bv @ self.M1.T + z @ self.K.T
        return (self._bv * self.vscale).astype(np.float32)


class PVKalmanFilter(Decoder):
    """Kalman filter with a position + velocity (+ acceleration) state: the strong Kalman baseline.

    This is the kinematic state of Wu et al. (2006) and of ReFIT-KF (Gilja et al. 2012):
    ``x_t = [p_t, v_t(, a_t)]`` (2-D each), ``x_t = A x_{t-1} + w``, observation
    ``z_t = H x_t + d + q`` with ``z_t`` the causal ``smooth_bins`` boxcar of z-scored counts
    (equivalent to decoding with a bin width of ``smooth_bins`` x 20 ms, updated every 20 ms),
    taken ``obs_delay`` bins in the past (an extra neural lead on top of the global lag).
    ``sqrt=True`` applies the variance-stabilising square root to counts first (normalised by the
    delta-method approximation of the square-root statistics, so re-centring with raw-count
    statistics still works). ``A``, ``W``, ``H``, ``d`` and the observation noise ``Q`` (scaled by
    ``q_scale``) are fit by least squares; decoding uses the steady-state gain. Offline the position
    state is the filter's own integral of velocity. Only the velocity part is output.

    Diagnosis of the previous, weak Kalman filter (velocity-only state, R^2 about 0.19): with a
    velocity-only state the only dynamics is a first-order low-pass, so every observation-noise
    setting either lags the movement or passes noise. Adding position and acceleration makes the
    prior second/third order (it can represent the bell-shaped velocity profile of a reach), which
    is what lets the filter follow reaches without passing noise; see ``docs/DESIGN.md``.
    """

    def __init__(self, smooth_bins: int = 3, obs_delay: int = 0, order: int = 2, q_scale: float = 1.0,
                 sqrt: bool = False, ridge: float = 1.0):
        self.smooth_bins, self.obs_delay, self.order = smooth_bins, obs_delay, order
        self.q_scale, self.sqrt, self.ridge = q_scale, sqrt, ridge

    @property
    def n_state(self) -> int:
        return 2 if self.order == 0 else 2 * (self.order + 1)

    def normalize(self, X: np.ndarray) -> np.ndarray:
        if not self.sqrt:
            return (X - self.mu) / self.sd
        m = np.sqrt(np.maximum(self.mu, 1e-3))
        return (np.sqrt(np.maximum(X, 0)) - m) / (self.sd / (2 * m))

    def _obs(self, X: np.ndarray) -> np.ndarray:
        Z = self.normalize(X)
        k = self.smooth_bins
        c = np.cumsum(np.concatenate([np.zeros((k, Z.shape[1]), np.float32), Z]), 0)
        Z = (c[k:] - c[:-k]) / k
        if self.obs_delay:
            Z = np.concatenate([np.zeros((self.obs_delay, Z.shape[1]), Z.dtype), Z[: -self.obs_delay]])
        return Z

    def _states(self, V: np.ndarray) -> np.ndarray:
        """Kinematic state trajectory from (scaled) velocity: centred position integral, v, a
        (``order`` 0: velocity only, 1: + position, 2: + position and acceleration)."""
        if self.order == 0:
            return V
        P = np.cumsum(V, 0) * 0.02 * 10  # position in units of vscale x 0.1 s
        cols = [P - P.mean(0), V]
        if self.order >= 2:
            cols.append(np.gradient(V, axis=0) * 10)
        return np.concatenate(cols, 1)

    def fit(self, segments: list[tuple[np.ndarray, np.ndarray]], mu: np.ndarray, sd: np.ndarray,
            vscale: float | None = None, seg_stats: list | None = None) -> "PVKalmanFilter":
        """Fit on (counts, velocity) segments. ``seg_stats`` (one (mu, sd) per segment) normalises
        each segment with its own day's statistics (pooled multi-day fit); the decoder keeps
        ``mu``/``sd``."""
        self.vscale = float(vscale or np.concatenate([Y for _, Y in segments]).std())
        Xs = [self._states(Y.astype(np.float64) / self.vscale) for _, Y in segments]
        Zs = []
        for i, (X, _) in enumerate(segments):
            self.mu, self.sd = seg_stats[i] if seg_stats is not None else (mu, sd)
            Zs.append(self._obs(X).astype(np.float64))
        self.mu, self.sd = mu, sd
        n = self.n_state
        Xp = np.concatenate([X[:-1] for X in Xs])
        Xn = np.concatenate([X[1:] for X in Xs])
        A = np.linalg.lstsq(Xp, Xn, rcond=None)[0].T
        Wn = np.cov((Xn - Xp @ A.T).T) + 1e-6 * np.eye(n)
        X, Z = np.concatenate(Xs), np.concatenate(Zs)
        X1 = np.concatenate([X, np.ones((len(X), 1))], 1)
        Hd = np.linalg.solve(X1.T @ X1 + self.ridge * np.eye(n + 1), X1.T @ Z).T
        H, d = Hd[:, :n], Hd[:, n]
        R = Z - X1 @ Hd.T
        Q = self.q_scale * (np.cov(R.T) + 1e-3 * np.eye(Z.shape[1]))
        Qi = np.linalg.inv(Q)
        HQH = H.T @ Qi @ H
        P = np.eye(n)
        for _ in range(3000):
            Pp = A @ P @ A.T + Wn
            Pn = np.linalg.inv(np.linalg.inv(Pp) + HQH)
            if np.max(np.abs(Pn - P)) < 1e-12:
                P = Pn
                break
            P = Pn
        K = P @ H.T @ Qi
        self.A, self.H, self.d, self.K, self.W_noise = A, H, d, K, Wn
        self.M1 = (np.eye(n) - K @ H) @ A  # x_t = M1 x_{t-1} + K (z_t - d)
        self.vi = slice(0, 2) if self.order == 0 else slice(2, 4)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        KZ = (self._obs(X).astype(np.float64) - self.d) @ self.K.T
        out = np.zeros((len(KZ), self.n_state))
        x = np.zeros(self.n_state)
        M1 = self.M1
        for t in range(len(KZ)):
            x = M1 @ x + KZ[t]
            out[t] = x
        return (out[:, self.vi] * self.vscale).astype(np.float32)

    def reset(self) -> None:
        self.reset_batch(1)

    def step(self, x: np.ndarray) -> np.ndarray:
        return self.step_batch(x[None])[0]

    def reset_batch(self, B: int) -> None:
        self._bx = np.zeros((B, self.n_state))
        self._bbuf = np.zeros((B, self.smooth_bins + self.obs_delay, len(self.mu)), np.float32)

    def step_batch(self, X: np.ndarray) -> np.ndarray:
        self._bbuf = np.roll(self._bbuf, 1, axis=1)
        self._bbuf[:, 0] = self.normalize(X)
        z = self._bbuf[:, self.obs_delay : self.obs_delay + self.smooth_bins].mean(1) - self.d
        self._bx = self._bx @ self.M1.T + z @ self.K.T
        return (self._bx[:, self.vi] * self.vscale).astype(np.float32)
