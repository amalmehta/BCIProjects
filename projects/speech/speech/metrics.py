"""Error-rate metrics: phoneme error rate (PER) and word error rate (WER).

Both are Levenshtein distance between hypothesis and reference token
sequences, summed over a set and divided by the summed reference length
(the corpus-level convention used by Willett et al., 2023 and Card et al.,
2024), not an average of per-sentence rates.
"""

from __future__ import annotations

import re
from typing import Sequence

import numpy as np


def edit_distance(ref: Sequence, hyp: Sequence) -> int:
    """Levenshtein distance (substitutions, insertions, deletions all cost 1)."""
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        cur = [i] + [0] * len(hyp)
        for j, h in enumerate(hyp, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r != h))
        prev = cur
    return prev[-1]


def error_rate(refs: Sequence[Sequence], hyps: Sequence[Sequence]) -> float:
    """Corpus-level error rate: total edits / total reference tokens."""
    edits = sum(edit_distance(r, h) for r, h in zip(refs, hyps))
    total = sum(len(r) for r in refs)
    return edits / max(total, 1)


def normalize_text(s: str) -> str:
    """Lower-case, keep letters/apostrophes/spaces, collapse whitespace (as in the B2T benchmarks)."""
    s = s.lower().replace("-", " ")
    s = re.sub(r"[^a-z' ]", "", s)
    return " ".join(s.split())


def words(s: str) -> list[str]:
    """Normalised word tokens of a sentence."""
    return normalize_text(s).split()


def wer(refs: Sequence[str], hyps: Sequence[str]) -> float:
    """Corpus-level word error rate between sentence strings."""
    return error_rate([words(r) for r in refs], [words(h) for h in hyps])


def bootstrap_ci(refs: Sequence[Sequence], hyps: Sequence[Sequence], n: int = 1000, seed: int = 0):
    """95% bootstrap confidence interval of the corpus error rate (resampling sentences)."""
    rng = np.random.default_rng(seed)
    e = np.array([edit_distance(r, h) for r, h in zip(refs, hyps)], float)
    t = np.array([len(r) for r in refs], float)
    idx = rng.integers(0, len(e), size=(n, len(e)))
    rates = e[idx].sum(1) / t[idx].sum(1)
    return float(np.percentile(rates, 2.5)), float(np.percentile(rates, 97.5))
