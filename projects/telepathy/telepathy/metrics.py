"""Evaluation metrics: velocity R^2 and the grid-task achieved bitrate."""
from __future__ import annotations

import math

import numpy as np


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Coefficient of determination averaged over output dimensions.

    ``R^2_d = 1 - sum_t (y_td - yhat_td)^2 / sum_t (y_td - mean_d)^2``, then mean over d.
    This is the "variance-weighted per-dimension" R^2 used by FALCON and NLB.
    """
    y_true = np.asarray(y_true, dtype=np.float64).reshape(len(y_true), -1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(len(y_pred), -1)
    ss_res = ((y_true - y_pred) ** 2).sum(0)
    ss_tot = ((y_true - y_true.mean(0)) ** 2).sum(0)
    return float(np.mean(1.0 - ss_res / np.maximum(ss_tot, 1e-12)))


def webgrid_bitrate(n_targets: int, n_correct: int, n_incorrect: int, seconds: float) -> float:
    """Achieved bitrate on a grid selection task (Nuyujukian 2015; Neuralink Webgrid).

    ``bps = log2(N - 1) * max(S_c - S_i, 0) / t``.
    """
    if seconds <= 0:
        raise ValueError("seconds must be positive")
    return math.log2(n_targets - 1) * max(n_correct - n_incorrect, 0) / seconds


def fitts_index_of_difficulty(distance: float, width: float) -> float:
    """Shannon formulation of Fitts' index of difficulty, ``log2(D / W + 1)``."""
    return math.log2(distance / width + 1.0)
