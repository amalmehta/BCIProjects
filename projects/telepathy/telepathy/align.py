"""Classical unsupervised stabiliser after Degenhart et al. (2020).

A factor-analysis (FA) model is fit to calibration data and a linear decoder is trained on the
FA latents. On a new day, FA is refit on *unlabelled* data, the subset of channels whose
loadings still agree with the reference ("stable channels") is found iteratively, and an
orthogonal Procrustes rotation maps the new latent space onto the reference one. The latent
decoder is never retrained.
"""
from __future__ import annotations

import copy

import numpy as np
from sklearn.decomposition import FactorAnalysis

from .features import DayData, channel_stats, lagged
from .linear import Decoder


def _smooth(Z: np.ndarray, k: int) -> np.ndarray:
    c = np.cumsum(np.concatenate([np.zeros((k, Z.shape[1]), Z.dtype), Z]), 0)
    return (c[k:] - c[:-k]) / k


def _fa(Z: np.ndarray, q: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    fa = FactorAnalysis(q, random_state=seed).fit(Z)
    return fa.components_.T, fa.noise_variance_, fa.mean_


def procrustes(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Orthogonal O minimising ||A O - B||_F."""
    U, _, Vt = np.linalg.svd(A.T @ B)
    return U @ Vt


class FAStabilizedDecoder(Decoder):
    """FA latents -> ridge Wiener filter; supports Degenhart-style alignment to a new day."""

    def __init__(self, q: int = 10, smooth_bins: int = 5, n_lags: int = 5, alpha: float = 10.0,
                 stable_frac: float = 0.6):
        self.q, self.k, self.n_lags, self.alpha, self.stable_frac = q, smooth_bins, n_lags, alpha, stable_frac

    def _latents(self, X: np.ndarray) -> np.ndarray:
        Z = _smooth(self.normalize(X), self.k) - self.fa_mean
        L, psi, S = self.L, self.psi, self.S
        Ls, pinv = L[S], 1.0 / psi[S]
        M = np.linalg.solve(np.eye(self.q) + (Ls.T * pinv) @ Ls, (Ls.T * pinv))
        return (Z[:, S] @ M.T) @ self.O

    def fit(self, days: list[DayData]) -> "FAStabilizedDecoder":
        """Fit reference FA and latent decoder on the train blocks of the calibration days."""
        Zs, self.day_stats = [], []
        for d in days:
            mu, sd = channel_stats(d.X[d.train])
            self.day_stats.append((mu, sd))
            Zs.append(_smooth((d.X[d.train] - mu) / sd, self.k))
        self.L, self.psi, self.fa_mean = _fa(np.concatenate(Zs), self.q)
        self.L_ref = self.L.copy()
        self.S = np.arange(self.L.shape[0])
        self.O = np.eye(self.q)
        F, Y = [], []
        for d, (mu, sd) in zip(days, self.day_stats):
            self.mu, self.sd = mu, sd
            F.append(lagged(self._latents(d.X[d.train]), self.n_lags))
            Y.append(d.Y[d.train])
        F = np.concatenate([np.concatenate(F), np.ones((sum(len(f) for f in F), 1), np.float32)], 1)
        Y = np.concatenate(Y)
        reg = self.alpha * np.eye(F.shape[1])
        reg[-1, -1] = 0
        self.W = np.linalg.solve(F.T @ F + reg, F.T @ Y)
        self.mu, self.sd = self.day_stats[-1]
        return self

    def aligned_to(self, X_unlab: np.ndarray, n_iter: int = 10) -> "FAStabilizedDecoder":
        """Copy of the decoder aligned to a new day using only unlabelled counts ``X_unlab``."""
        d = copy.deepcopy(self)
        d.mu, d.sd = channel_stats(X_unlab)
        Z = _smooth((X_unlab - d.mu) / d.sd, self.k)
        L, psi, mean = _fa(Z, self.q)
        C = L.shape[0]
        S = np.arange(C)
        for _ in range(n_iter):
            O = procrustes(L[S], self.L_ref[S])
            err = np.linalg.norm(L @ O - self.L_ref, axis=1)
            S = np.sort(np.argsort(err)[: int(self.stable_frac * C)])
        d.L, d.psi, d.fa_mean, d.S = L, psi, mean, S
        d.O = procrustes(L[S], self.L_ref[S])
        return d

    def predict(self, X: np.ndarray) -> np.ndarray:
        F = lagged(self._latents(X), self.n_lags)
        return (np.concatenate([F, np.ones((len(F), 1), np.float32)], 1) @ self.W).astype(np.float32)

    def reset(self) -> None:
        self._hist = np.zeros((self.k + self.n_lags + 1, len(self.mu)), np.float32)

    def step(self, x: np.ndarray) -> np.ndarray:
        self._hist = np.roll(self._hist, -1, axis=0)
        self._hist[-1] = x
        return self.predict(self._hist)[-1]
