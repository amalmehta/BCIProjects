"""Neural encoding model fitted to FALCON H1, used to close the loop in simulation.

Model (per 20 ms bin, channel c):

    counts_c,t ~ Poisson(lambda_c,t)
    log lambda_t = b + C v_t + sigma * L f_t
    f_t = rho f_{t-1} + sqrt(1 - rho^2) eps_t,  eps_t ~ N(0, I_k)

``v_t`` is the 7-D intended velocity in z-scored units. ``b, C`` come from a Poisson GLM
fitted to real H1 spike counts and cued velocities. ``L`` holds the top-k principal axes
of the slowly varying residual activity that velocity does not explain, so the simulated
"nuisance" variability is shaped like the real one. Its scale ``sigma`` is calibrated so
that a decoder trained and tested on simulated data reaches the same offline R² as the
same decoder on real held-out H1 data. Changing ``sigma`` then gives a dial for decoder
quality.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .data import causal_smooth


@dataclass
class EncodingModel:
    b: np.ndarray  # (C,)
    C: np.ndarray  # (C, 7)
    L: np.ndarray  # (C, k) unit-norm columns
    rho: float
    sigma: float = 1.0
    max_rate: float = 10.0  # counts per bin cap (500 Hz)

    @property
    def n_channels(self) -> int:
        return self.b.shape[0]

    def log_rate(self, v: np.ndarray, f: np.ndarray) -> np.ndarray:
        return self.b + v @ self.C.T + self.sigma * f @ self.L.T

    def sample_latents(self, T: int, rng: np.random.Generator, f0: np.ndarray | None = None) -> np.ndarray:
        k = self.L.shape[1]
        f = np.zeros((T, k))
        prev = rng.standard_normal(k) if f0 is None else f0
        s = np.sqrt(1 - self.rho ** 2)
        for t in range(T):
            prev = self.rho * prev + s * rng.standard_normal(k)
            f[t] = prev
        return f

    def sample(self, v: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Simulate spike counts for a whole velocity trajectory ``v`` (T, 7)."""
        f = self.sample_latents(len(v), rng)
        lam = np.minimum(np.exp(self.log_rate(v, f)), self.max_rate)
        return rng.poisson(lam).astype(np.float32)


class OnlineEncoder:
    """Streaming version of :class:`EncodingModel` for the closed-loop simulator."""

    def __init__(self, model: EncodingModel, rng: np.random.Generator):
        self.m, self.rng = model, rng
        self.f = rng.standard_normal(model.L.shape[1])
        self.s = np.sqrt(1 - model.rho ** 2)

    def step(self, v: np.ndarray) -> np.ndarray:
        m = self.m
        self.f = m.rho * self.f + self.s * self.rng.standard_normal(self.f.shape[0])
        lam = np.minimum(np.exp(m.b + m.C @ v + m.sigma * (m.L @ self.f)), m.max_rate)
        return self.rng.poisson(lam).astype(np.float64)


def fit_poisson_glm(counts: np.ndarray, v: np.ndarray, l2: float = 1e-3, iters: int = 200) -> tuple[np.ndarray, np.ndarray]:
    """Fit ``log lambda = b + C v`` for all channels jointly by full-batch L-BFGS."""
    y = torch.as_tensor(counts, dtype=torch.float64)
    x = torch.as_tensor(v, dtype=torch.float64)
    mean = y.mean(0).clamp_min(1e-3)
    b = torch.log(mean).clone().requires_grad_(True)
    C = torch.zeros(y.shape[1], x.shape[1], dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([b, C], lr=1, max_iter=iters, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        eta = b + x @ C.T
        loss = (torch.exp(eta) - y * eta).mean() + l2 * (C ** 2).sum() / y.shape[1]
        loss.backward()
        return loss

    opt.step(closure)
    return b.detach().numpy(), C.detach().numpy()


def fit_encoding_model(counts: np.ndarray, v: np.ndarray, k: int = 8, smooth_tau: float = 10.0) -> EncodingModel:
    """Fit the GLM, then estimate nuisance-latent directions and timescale from residuals."""
    b, C = fit_poisson_glm(counts, v)
    lam = np.exp(b + v @ C.T)
    resid = causal_smooth((counts - lam) / np.sqrt(lam), smooth_tau)
    resid -= resid.mean(0)
    _, _, vt = np.linalg.svd(resid[::5], full_matrices=False)
    L = vt[:k].T  # (C, k), orthonormal columns
    f = resid @ L
    # AR(1) coefficient of the latent trajectories, averaged across factors, then
    # de-biased for the smoothing filter (smoothing alone induces exp(-1/tau)).
    ac = np.mean([np.corrcoef(f[1:, i], f[:-1, i])[0, 1] for i in range(k)])
    rho = float(np.clip(ac, 0.5, 0.995))
    return EncodingModel(b=b, C=C, L=L, rho=rho)
