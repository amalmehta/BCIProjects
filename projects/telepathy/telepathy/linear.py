"""Classical decoders: ridge Wiener filter and velocity Kalman filter.

Both share the :class:`Decoder` interface used everywhere else:

* ``predict(X)``  - offline, causal decoding of a whole (T, C) count stream -> (T, 2) velocity
* ``reset()`` / ``step(x_t)`` - streaming decoding, one bin at a time (closed-loop simulator)
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
