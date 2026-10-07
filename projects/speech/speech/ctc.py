"""CTC utilities: greedy (best-path) decoding and phoneme timing."""

from __future__ import annotations

import numpy as np

from .phonemes import BLANK_ID


def greedy_decode(log_probs: np.ndarray) -> list[int]:
    """Best-path CTC decoding: argmax per frame, merge repeats, drop blanks.

    ``log_probs`` has shape (T, C).
    """
    best = log_probs.argmax(-1)
    out, prev = [], -1
    for c in best:
        if c != prev and c != BLANK_ID:
            out.append(int(c))
        prev = c
    return out


def greedy_with_times(log_probs: np.ndarray) -> list[tuple[int, int]]:
    """Like :func:`greedy_decode` but also returns the frame where each token is first emitted."""
    best = log_probs.argmax(-1)
    out, prev = [], -1
    for t, c in enumerate(best):
        if c != prev and c != BLANK_ID:
            out.append((int(c), t))
        prev = c
    return out
