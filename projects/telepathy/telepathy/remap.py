"""Label-free channel re-matching for electrode re-mapping days.

On two held-out sessions the per-channel firing-rate profile no longer matches the calibration
days (``export.channel_stability``): the same neurons are still recorded, but under different
channel numbers. This module recovers the correspondence **without labels**, from per-channel
statistics of the new day's unlabelled spikes.

Two matchers:

* **Structured hypothesis search** (default). Re-mappings caused by hardware or acquisition
  configuration (connector or bank re-ordering) are not arbitrary permutations: they move
  contiguous blocks of channels. We enumerate a family of block permutations (block size b in
  {1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48}; blocks cyclically shifted and/or reversed; channels
  inside each block optionally reversed; ~1,000 candidates including the identity) and score each
  by how well the permuted day's label-free channel signature matches the calibration days: the
  mean Pearson correlation of per-channel statistics (log rate, log sd, log Fano factor at 20 and
  100 ms, autocorrelation at 1, 2, 5, 10 bins) plus the correlation between the two days'
  channel-by-channel correlation matrices (100 ms boxcar). The identity wins unless another
  candidate beats it by ``margin``.
* **Hungarian / graph-matching refinement** (optional, ``refine=True``): starting from the
  structured solution, alternate the Hungarian algorithm (Kuhn 1955) on a cost combining
  per-channel statistics and each channel's correlation profile to the other (already matched)
  channels, with a bonus for keeping the current assignment. This handles arbitrary
  permutations in principle; on these data per-channel statistics are not distinctive enough
  for it to work (documented negative result, ``results/remap.json``).
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment


def _boxcar(X: np.ndarray, k: int) -> np.ndarray:
    c = np.cumsum(np.concatenate([np.zeros((k, X.shape[1])), X]), 0)
    return (c[k:] - c[:-k]) / k


def channel_features(X: np.ndarray) -> np.ndarray:
    """(C, 8) label-free per-channel statistics of an unlabelled count stream (T, C)."""
    X = X.astype(np.float64)
    m, s = X.mean(0), X.std(0) + 1e-6
    X5 = _boxcar(X, 5)[::5] * 5
    fano = X.var(0) / (m + 1e-3)
    fano5 = X5.var(0) / (X5.mean(0) + 1e-3)
    Z = (X - m) / s
    ac = [(Z[k:] * Z[:-k]).mean(0) for k in (1, 2, 5, 10)]
    return np.stack([np.log(m + 1e-2), np.log(s), np.log(fano + 1e-3), np.log(fano5 + 1e-3), *ac], 1)


def corr_matrix(X: np.ndarray, k: int = 5) -> np.ndarray:
    """Channel-by-channel correlation of ``k``-bin boxcar-smoothed counts (zero diagonal)."""
    Z = _boxcar(X.astype(np.float64), k)
    sd = Z.std(0)
    Z = (Z - Z.mean(0)) / np.where(sd > 0, sd, 1)
    C = Z.T @ Z / len(Z)
    np.fill_diagonal(C, 0)
    return C


def structured_candidates(C: int = 96) -> dict[str, np.ndarray]:
    """Block-permutation hypotheses ``name -> perm`` (new channel ``perm[j]`` plays reference channel j)."""
    out = {"identity": np.arange(C)}
    for b in (1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48):
        if C % b:
            continue
        nb = C // b
        for rev_blocks in (0, 1):
            for rev_in in (0, 1):
                for sh in range(nb):
                    g = np.arange(C).reshape(nb, b)
                    if rev_in:
                        g = g[:, ::-1]
                    order = np.roll(np.arange(nb), sh)
                    if rev_blocks:
                        order = order[::-1]
                    out[f"b{b}_rb{rev_blocks}_ri{rev_in}_s{sh}"] = g[order].reshape(-1)
    uniq, seen = {}, set()
    for k, p in out.items():
        t = p.tobytes()
        if t not in seen:
            seen.add(t)
            uniq[k] = p
    return uniq


class ChannelMatcher:
    """Reference channel signature from calibration days; matches a new day's channels to it."""

    def __init__(self, ref_X: list[np.ndarray], corr_weight: float = 1.0, margin: float = 0.1):
        self.F = np.mean([channel_features(X) for X in ref_X], 0)
        self.Cr = np.mean([corr_matrix(X) for X in ref_X], 0)
        self.corr_weight, self.margin = corr_weight, margin
        self._iu = np.triu_indices(len(self.F), 1)
        self.cands = structured_candidates(len(self.F))

    def score(self, F: np.ndarray, Cn: np.ndarray, perm: np.ndarray) -> float:
        Fp = F[perm]
        fs = np.mean([np.corrcoef(self.F[:, j], Fp[:, j])[0, 1] for j in range(F.shape[1])])
        cs = np.corrcoef(self.Cr[self._iu], Cn[np.ix_(perm, perm)][self._iu])[0, 1]
        return float(fs + self.corr_weight * cs)

    def match(self, X_unlab: np.ndarray, refine: bool = False) -> dict:
        """Return {perm, name, score, identity_score, changed} for unlabelled counts of a new day."""
        F, Cn = channel_features(X_unlab), corr_matrix(X_unlab)
        scores = {k: self.score(F, Cn, p) for k, p in self.cands.items()}
        best = max(scores, key=scores.get)
        s_id = scores["identity"]
        if scores[best] - s_id < self.margin:
            best = "identity"
        perm = self.cands[best].copy()
        out = {"name": best, "score": scores[best], "identity_score": s_id,
               "runner_up": sorted(scores.values())[-2]}
        if refine:
            perm = self.refine(F, Cn, perm)
            out["name"] += "+hungarian"
        out["perm"] = perm
        out["changed"] = bool(not np.array_equal(perm, np.arange(len(perm))))
        return out

    def refine(self, F: np.ndarray, Cn: np.ndarray, perm: np.ndarray, keep_bonus: float = 1.0,
               beta: float = 1.0, iters: int = 10) -> np.ndarray:
        """Hungarian refinement on per-channel statistics + correlation profiles (see module docstring)."""
        mu, sd = self.F.mean(0), self.F.std(0) + 1e-6
        a, b = (F - mu) / sd, (self.F - mu) / sd
        fc = ((a[:, None] - b[None]) ** 2).sum(-1)  # new channel i -> reference slot j
        fc /= np.median(fc)
        C = len(perm)
        for _ in range(iters):
            rows = Cn[:, perm]
            cc = ((rows[:, None, :] - self.Cr[None]) ** 2).sum(-1)
            cc /= np.median(cc)
            bonus = np.zeros((C, C))
            bonus[perm, np.arange(C)] = keep_bonus
            r, c = linear_sum_assignment(fc + beta * cc - bonus)
            p = np.empty(C, int)
            p[c] = r
            if np.array_equal(p, perm):
                break
            perm = p
        return perm


def partial_shuffle(C: int, frac: float, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """The stress-test re-wiring: a random ``frac`` of channels cyclically exchange identities.
    Returns (perm such that X_new = X[:, perm], indices of the moved channels)."""
    n = int(round(frac * C))
    perm = np.arange(C)
    if n < 2:
        return perm, np.array([], int)
    idx = rng.choice(C, n, replace=False)
    perm[idx] = perm[np.roll(idx, 1)]
    return perm, idx
