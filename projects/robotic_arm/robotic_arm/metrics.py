"""Evaluation metrics: decoding R² (FALCON protocol) and manipulation-task metrics."""

from __future__ import annotations

import numpy as np


def r2_per_dim(y: np.ndarray, yhat: np.ndarray) -> np.ndarray:
    """Coefficient of determination for each output column (sklearn convention)."""
    y = np.asarray(y, dtype=np.float64)
    yhat = np.asarray(yhat, dtype=np.float64)
    ss_res = ((y - yhat) ** 2).sum(0)
    ss_tot = ((y - y.mean(0)) ** 2).sum(0)
    return 1.0 - ss_res / np.maximum(ss_tot, 1e-300)


def r2_variance_weighted(y: np.ndarray, yhat: np.ndarray) -> float:
    """FALCON's official H1 metric: ``r2_score(..., multioutput='variance_weighted')``.

    Equivalent to ``1 - sum_d SSres_d / sum_d SStot_d``, so high-variance dimensions
    (the grasp dims in H1) dominate.
    """
    y = np.asarray(y, dtype=np.float64)
    yhat = np.asarray(yhat, dtype=np.float64)
    ss_res = ((y - yhat) ** 2).sum()
    ss_tot = ((y - y.mean(0)) ** 2).sum()
    return float(1.0 - ss_res / ss_tot)


def block_scores(y: np.ndarray, yhat: np.ndarray, mask: np.ndarray) -> dict:
    """Score one block using only bins where ``mask`` is True."""
    y, yhat = y[mask], yhat[mask]
    per = r2_per_dim(y, yhat)
    return {"r2_vw": r2_variance_weighted(y, yhat), "r2_mean_dof": float(per.mean()),
            "r2_per_dof": per.tolist()}


def path_efficiency(path: np.ndarray) -> float:
    """Straight-line start-to-end distance divided by travelled path length (1 = straight)."""
    steps = np.linalg.norm(np.diff(path, axis=0), axis=1).sum()
    if steps < 1e-9:
        return 0.0
    return float(np.linalg.norm(path[-1] - path[0]) / steps)


def mean_std(values) -> tuple[float, float]:
    a = np.asarray(values, dtype=np.float64)
    return float(a.mean()), float(a.std(ddof=1) if a.size > 1 else 0.0)
