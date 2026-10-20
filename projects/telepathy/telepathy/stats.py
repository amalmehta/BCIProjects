"""Paired statistics for closed-loop comparisons.

All simulated decoders are evaluated on the same seeds (common random numbers, see
:class:`telepathy.simulator.SessionNoise`), so comparisons are paired by session: we report the
mean paired difference with a percentile bootstrap 95 % CI (Efron & Tibshirani 1993), a
two-sided Wilcoxon signed-rank test (Wilcoxon 1945; zero differences split between ranks,
Pratt-style ``zsplit``) and Holm's step-down correction (Holm 1979) over the family of comparisons.
"""
from __future__ import annotations

import itertools

import numpy as np
from scipy.stats import wilcoxon


def mean_ci(x, n_boot: int = 10000, seed: int = 0, level: float = 0.95) -> dict:
    """Mean with a percentile-bootstrap CI."""
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    bs = x[rng.integers(0, len(x), (n_boot, len(x)))].mean(1)
    a = (1 - level) / 2
    return {"mean": float(x.mean()), "ci95": [float(np.quantile(bs, a)), float(np.quantile(bs, 1 - a))],
            "sd": float(x.std()), "n": int(len(x))}


def paired_test(a, b, n_boot: int = 10000, seed: int = 0) -> dict:
    """Paired comparison of per-session values ``a`` vs ``b`` (same seeds)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a - b
    r = mean_ci(d, n_boot, seed)
    if np.allclose(d, 0):
        p = 1.0
    else:
        p = float(wilcoxon(a, b, zero_method="zsplit").pvalue)
    return {"diff_mean": r["mean"], "diff_ci95": r["ci95"], "p_wilcoxon": p, "n": r["n"]}


def holm(pvals: dict[str, float]) -> dict[str, float]:
    """Holm-Bonferroni adjusted p-values (monotone, capped at 1)."""
    keys = sorted(pvals, key=pvals.get)
    m = len(keys)
    out, running = {}, 0.0
    for i, k in enumerate(keys):
        running = max(running, min(1.0, (m - i) * pvals[k]))
        out[k] = running
    return out


def pairwise(per_session: dict[str, list], alpha: float = 0.05, n_boot: int = 10000) -> dict:
    """All pairwise paired tests between methods, Holm-corrected; ``significant`` uses ``alpha``."""
    res = {}
    for x, y in itertools.combinations(per_session, 2):
        res[f"{x} vs {y}"] = paired_test(per_session[x], per_session[y], n_boot)
    adj = holm({k: v["p_wilcoxon"] for k, v in res.items()})
    for k, v in res.items():
        v["p_holm"] = adj[k]
        v["significant"] = bool(adj[k] < alpha)
    return res
