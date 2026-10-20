"""Uncertainty and paired tests for the decoding and closed-loop results.

* ``bootstrap_ci``: percentile bootstrap CI of a mean over independent units (test blocks
  or held-out days; Efron & Tibshirani 1993).
* ``paired_signflip``: exact two-sided sign-flip permutation test of a paired mean
  difference (all 2^n sign patterns for n <= 16, Monte Carlo otherwise).
* ``add_decoding_stats``: adds ``r2_vw_ci95`` to every decoding entry (resampling blocks,
  per-block scores averaged over seeds first) and a ``paired_tests`` table.
"""

from __future__ import annotations

import itertools

import numpy as np


def bootstrap_ci(values, n_boot: int = 10_000, level: float = 0.95, seed: int = 0) -> list[float]:
    v = np.asarray(values, dtype=np.float64)
    if v.size == 0:
        return [float("nan"), float("nan")]
    if v.size == 1:
        return [float(v[0]), float(v[0])]
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, v.size, size=(n_boot, v.size))].mean(1)
    lo, hi = np.quantile(means, [(1 - level) / 2, 1 - (1 - level) / 2])
    return [float(lo), float(hi)]


def paired_signflip(a, b, n_mc: int = 20_000, seed: int = 0) -> float:
    """Two-sided p-value for H0: E[a - b] = 0 under exchangeable signs of the differences."""
    d = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    n = d.size
    obs = abs(d.mean())
    if n == 0:
        return 1.0
    if n <= 16:
        signs = np.array(list(itertools.product([-1.0, 1.0], repeat=n)))
    else:
        signs = np.random.default_rng(seed).choice([-1.0, 1.0], size=(n_mc, n))
    null = np.abs((signs * d).mean(1))
    return float(np.mean(null >= obs - 1e-12))


def paired_comparison(a, b, seed: int = 0) -> dict:
    d = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    return {"mean_diff": float(d.mean()), "diff_ci95": bootstrap_ci(d, seed=seed),
            "n_units": int(d.size), "n_positive": int((d > 0).sum()), "p_signflip": paired_signflip(a, b)}


WITHIN_TESTS = [("gru", "wiener_daynorm"), ("transformer", "gru"), ("ms_gru", "gru"), ("gru", "kalman"),
                ("transformer", "wiener_daynorm")]
CROSS_TESTS = [
    ("ms_gru_unsup_kl", "ms_gru_zeroshot"), ("ms_gru_unsup_coral", "ms_gru_zeroshot"),
    ("ms_gru_unsup_center", "ms_gru_zeroshot"), ("ms_gru_zeroshot", "ms_gru_zs_fixednorm"),
    ("ms_gru_unsup_kl", "transformer_zeroshot"), ("ms_gru_unsup_coral", "transformer_zeroshot"),
    ("ms_gru_unsup_kl", "ms_gru_45s"), ("ms_gru_unsup_kl", "ms_gru_10s"),
    ("transformer_unsup_coral", "transformer_zeroshot"), ("gru_unsup_coral", "gru_zeroshot"),
    ("ms_gru_45s", "transformer_zeroshot"), ("ms_gru_45s", "gru_45s"), ("ms_gru_45s", "wiener_pooled_45s"),
    ("ms_gru_45s", "ms_gru_ftall_45s"), ("ms_gru_45s", "ms_gru_noaug_45s"), ("gru_zeroshot", "wiener_daynorm_zeroshot"),
]


def add_decoding_stats(out: dict) -> None:
    for split in ("within", "cross"):
        for e in out.get(split, {}).values():
            if isinstance(e, dict) and "r2_vw_blocks" in e:
                e["r2_vw_ci95"] = bootstrap_ci(e["r2_vw_blocks"])
    tests = {}
    for split, pairs in (("within", WITHIN_TESTS), ("cross", CROSS_TESTS)):
        d = out.get(split, {})
        for a, b in pairs:
            if a in d and b in d:
                tests[f"{split}:{a}_vs_{b}"] = paired_comparison(d[a]["r2_vw_blocks"], d[b]["r2_vw_blocks"])
    out["paired_tests"] = tests
