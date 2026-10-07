"""Turning binned sessions into decoder-ready arrays and train/val/test splits.

Conventions
-----------
* ``X[t]`` are the spike counts in bin ``t`` (shape (T, C)).
* ``Y[t]`` is the cursor velocity ``lag`` bins *later*, i.e. the movement the neural activity
  at time ``t`` is driving (neural activity leads movement by ~100 ms). We call it the
  *intended velocity*; the closed-loop simulator uses the same definition.
* Each day is split in time: first 60 % train, next 10 % validation, last 30 % test.
  Few-shot recalibration draws its labelled reaches from the start of the train block, as a
  short calibration block at the start of a session would.
"""
from __future__ import annotations

import dataclasses

import numpy as np

from .data import Session

TRAIN_FRAC, VAL_FRAC = 0.6, 0.1


@dataclasses.dataclass
class DayData:
    """Lag-aligned arrays for one day plus its split boundaries."""

    name: str
    day_index: int  # days since the first session in the dataset
    X: np.ndarray  # (T, C) float32 counts
    Y: np.ndarray  # (T, 2) float32 intended velocity (mm/s)
    pos: np.ndarray  # (T, 2) cursor position at t + lag
    target: np.ndarray  # (T, 2) target position at t + lag
    trial_starts: np.ndarray  # bin indices of reach starts (within [0, T))

    @property
    def T(self) -> int:
        return len(self.X)

    @property
    def train(self) -> slice:
        return slice(0, int(TRAIN_FRAC * self.T))

    @property
    def val(self) -> slice:
        return slice(int(TRAIN_FRAC * self.T), int((TRAIN_FRAC + VAL_FRAC) * self.T))

    @property
    def test(self) -> slice:
        return slice(int((TRAIN_FRAC + VAL_FRAC) * self.T), self.T)

    def first_trials(self, n: int) -> slice:
        """Bins covering the first ``n`` reaches of the train block (n=0 -> empty slice)."""
        if n <= 0:
            return slice(0, 0)
        starts = self.trial_starts[self.trial_starts < self.train.stop]
        end = int(starts[n]) if n < len(starts) else self.train.stop
        return slice(0, end)


def make_day(session: Session, lag: int, day_index: int) -> DayData:
    """Lag-align a session: ``X[t] = counts[t]``, ``Y[t] = vel[t + lag]``."""
    T = len(session.counts) - lag
    starts = session.target_onsets() - lag
    starts = starts[(starts >= 0) & (starts < T)]
    return DayData(
        name=session.name,
        day_index=day_index,
        X=session.counts[:T].astype(np.float32),
        Y=session.vel[lag : lag + T].astype(np.float32),
        pos=session.pos[lag : lag + T].astype(np.float32),
        target=session.target[lag : lag + T].astype(np.float32),
        trial_starts=np.unique(np.concatenate([[0], starts])).astype(int),
    )


def channel_stats(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-channel mean and standard deviation (std floored to avoid blow-ups)."""
    return X.mean(0).astype(np.float32), np.maximum(X.std(0), 0.1).astype(np.float32)


def lagged(X: np.ndarray, n_lags: int) -> np.ndarray:
    """Causal history features: row t = [X[t], X[t-1], ..., X[t-n_lags+1]] (zero padded)."""
    T, C = X.shape
    out = np.zeros((T, C * n_lags), dtype=np.float32)
    for k in range(n_lags):
        out[k:, k * C : (k + 1) * C] = X[: T - k]
    return out
