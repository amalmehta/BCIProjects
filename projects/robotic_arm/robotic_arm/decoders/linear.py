"""Linear decoders: Wiener filter (ridge on lagged spike counts) and velocity Kalman filter.

Both operate on z-scored spike counts and z-scored velocities; callers handle the
normalisation so that the same decoder can be evaluated with different (global vs
per-day) input normalisation strategies.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..data import causal_smooth, lag_features


def ridge_solve(x: np.ndarray, y: np.ndarray, alpha: float) -> tuple[np.ndarray, np.ndarray]:
    """Closed-form ridge regression with an unpenalised intercept. Returns (W, b)."""
    xm, ym = x.mean(0), y.mean(0)
    xc, yc = x - xm, y - ym
    gram = xc.T @ xc
    gram[np.diag_indices_from(gram)] += alpha * len(x)
    w = np.linalg.solve(gram, xc.T @ yc)
    return w, ym - xm @ w


def ridge_stats(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    """Sufficient statistics ``([x 1]^T [x 1], [x 1]^T y, n)`` for incremental ridge fits."""
    xa = np.hstack([x, np.ones((len(x), 1), dtype=x.dtype)])
    return xa.T @ xa, xa.T @ y, len(x)


def ridge_from_stats(xtx: np.ndarray, xty: np.ndarray, n: int, alpha: float) -> tuple[np.ndarray, np.ndarray]:
    """Ridge solution from sufficient statistics; the intercept (last row) is unpenalised.

    Mathematically identical to :func:`ridge_solve` on the pooled data.
    """
    g = xtx.copy()
    d = g.shape[0] - 1
    g[np.arange(d), np.arange(d)] += alpha * n
    wb = np.linalg.solve(g, xty)
    return wb[:-1], wb[-1]


@dataclass
class WienerFilter:
    """Wiener filter: ``v_t = sum_k W_k r_{t-k*stride} + b`` over ``n_lags`` causal taps.

    This is the classic BCI linear decoder and FALCON's reference baseline.
    """

    n_lags: int = 10
    alpha: float = 1e-2
    smooth_tau: float = 0.0
    stride: int = 1
    w: np.ndarray | None = None
    b: np.ndarray | None = None

    def features(self, z: np.ndarray) -> np.ndarray:
        return lag_features(causal_smooth(z, self.smooth_tau), self.n_lags, self.stride)

    def fit(self, zs: list[np.ndarray], ys: list[np.ndarray], masks: list[np.ndarray]) -> "WienerFilter":
        x = np.concatenate([self.features(z)[m] for z, m in zip(zs, masks)])
        y = np.concatenate([v[m] for v, m in zip(ys, masks)])
        self.w, self.b = ridge_solve(x, y, self.alpha)
        return self

    def stats(self, zs: list[np.ndarray], ys: list[np.ndarray], masks: list[np.ndarray]):
        """Sufficient statistics of this filter's features on the given blocks."""
        x = np.concatenate([self.features(z)[m] for z, m in zip(zs, masks)])
        y = np.concatenate([v[m] for v, m in zip(ys, masks)])
        return ridge_stats(x, y)

    def fit_stats(self, *stats) -> "WienerFilter":
        """Fit from one or more summed sufficient-statistic tuples (pooled data)."""
        xtx = sum(s[0] for s in stats)
        xty = sum(s[1] for s in stats)
        n = sum(s[2] for s in stats)
        self.w, self.b = ridge_from_stats(xtx, xty, n, self.alpha)
        return self

    def predict(self, z: np.ndarray) -> np.ndarray:
        return self.features(z) @ self.w + self.b


def wiener_grid(zs, ys, masks, val_zs, val_ys, val_masks, max_lags: int, stride: int,
                smooth_tau: float, alphas: list[float], lag_options: list[int], score) -> tuple[WienerFilter, list]:
    """Select (n_lags, alpha) on validation data, building the Gram matrix only once.

    Lag-tap blocks are nested (taps 0..n-1), so every option is a sub-block of the
    full ``max_lags`` Gram matrix. ``score(pred_list) -> float`` (higher is better).
    """
    proto = WienerFilter(max_lags, 0.0, smooth_tau, stride)
    x = np.concatenate([proto.features(z)[m] for z, m in zip(zs, masks)])
    y = np.concatenate([v[m] for v, m in zip(ys, masks)])
    xm, ym = x.mean(0), y.mean(0)
    x -= xm
    gram, xty = x.T @ x, x.T @ (y - ym)
    n, C = len(x), zs[0].shape[1]
    del x
    vfeat = [proto.features(z) for z in val_zs]
    results, best = [], None
    for L in lag_options:
        k = L * C
        for a in alphas:
            g = gram[:k, :k].copy()
            g[np.diag_indices_from(g)] += a * n
            w = np.linalg.solve(g, xty[:k])
            b = ym - xm[:k] @ w
            s = score([f[:, :k] @ w + b for f in vfeat])
            results.append({"n_lags": L, "alpha": a, "val": s})
            if best is None or s > best[0]:
                best = (s, WienerFilter(L, a, smooth_tau, stride, w, b))
    return best[1], results


@dataclass
class KalmanFilter:
    """Velocity Kalman filter (Wu et al. 2006; Kim et al. 2008).

    State ``x_t`` = 7-D velocity (+ constant 1), dynamics ``x_t = A x_{t-1} + w``,
    observation ``y_t = H x_t + q`` where ``y_t`` are causally smoothed z-scored rates
    (optionally lagged by ``obs_lag`` bins so neural activity leads kinematics).
    Uses the steady-state Kalman gain, as is standard in real-time BCIs. With ``decay``
    the state transition is a leaky ``decay * I`` (chosen on validation data).
    """

    smooth_tau: float = 3.0
    obs_lag: int = 0
    ridge: float = 1e-3
    decay: float | None = None  # if set, A = decay * I instead of the least-squares fit
    A: np.ndarray | None = None
    W: np.ndarray | None = None
    H: np.ndarray | None = None
    Q: np.ndarray | None = None
    K: np.ndarray | None = None

    def _obs(self, z: np.ndarray) -> np.ndarray:
        y = causal_smooth(z, self.smooth_tau)
        if self.obs_lag > 0:  # pair kinematics at t with neural activity at t - lag
            y = np.concatenate([np.zeros((self.obs_lag, y.shape[1])), y[: -self.obs_lag]])
        return y

    def fit(self, zs: list[np.ndarray], ys: list[np.ndarray], masks: list[np.ndarray]) -> "KalmanFilter":
        x_prev, x_next, xs, obs = [], [], [], []
        for z, v, m in zip(zs, ys, masks):
            y = self._obs(z)
            pair = m[1:] & m[:-1]
            x_prev.append(v[:-1][pair])
            x_next.append(v[1:][pair])
            xs.append(v[m])
            obs.append(y[m])
        xp, xn = np.concatenate(x_prev), np.concatenate(x_next)
        x, y = np.concatenate(xs), np.concatenate(obs)
        d = x.shape[1]
        if self.decay is None:
            self.A = np.linalg.solve(xp.T @ xp + self.ridge * len(xp) * np.eye(d), xp.T @ xn).T
        else:  # a leakier prior is more robust to observation-model bias across days
            self.A = self.decay * np.eye(d)
        self.W = np.cov((xn - xp @ self.A.T).T)
        xa = np.hstack([x, np.ones((len(x), 1))])
        self.H = np.linalg.solve(xa.T @ xa + self.ridge * len(xa) * np.eye(d + 1), xa.T @ y).T
        resid = y - xa @ self.H.T
        self.Q = np.cov(resid.T) + 1e-3 * np.eye(y.shape[1])
        self.K = self._steady_state_gain()
        return self

    def _steady_state_gain(self, n_iter: int = 500) -> np.ndarray:
        d = self.A.shape[0]
        Hx = self.H[:, :d]
        P = np.eye(d)
        Qinv = np.linalg.inv(self.Q)
        K = np.zeros((d, Hx.shape[0]))
        for _ in range(n_iter):
            Pp = self.A @ P @ self.A.T + self.W
            # information form: K = (Pp^-1 + H' Q^-1 H)^-1 H' Q^-1 (avoids a CxC inverse)
            Pn = np.linalg.inv(np.linalg.inv(Pp) + Hx.T @ Qinv @ Hx)
            K_new = Pn @ Hx.T @ Qinv
            if np.max(np.abs(K_new - K)) < 1e-10:
                K = K_new
                break
            K, P = K_new, Pn
        return K

    def predict(self, z: np.ndarray) -> np.ndarray:
        y = self._obs(z)
        d = self.A.shape[0]
        Hx, h0 = self.H[:, :d], self.H[:, d]
        M = (np.eye(d) - self.K @ Hx) @ self.A
        innov = (y - h0) @ self.K.T
        out = np.zeros((len(y), d))
        x = np.zeros(d)
        for t in range(len(y)):
            x = M @ x + innov[t]
            out[t] = x
        return out

    def step(self, x: np.ndarray, y_t: np.ndarray) -> np.ndarray:
        """One online filter update given the previous state and a smoothed observation."""
        d = self.A.shape[0]
        xp = self.A @ x
        return xp + self.K @ (y_t - self.H[:, :d] @ xp - self.H[:, d])
