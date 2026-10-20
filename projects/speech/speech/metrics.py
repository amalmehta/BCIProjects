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


def sentence_edits(refs: Sequence[Sequence], hyps: Sequence[Sequence]) -> tuple[np.ndarray, np.ndarray]:
    """Per-sentence edit counts and reference lengths (the sufficient statistics of a corpus rate)."""
    e = np.array([edit_distance(r, h) for r, h in zip(refs, hyps)], float)
    t = np.array([len(r) for r in refs], float)
    return e, t


def bootstrap_ci_edits(e, t, n: int = 2000, seed: int = 0) -> tuple[float, float]:
    """95% sentence-level bootstrap CI of ``sum(e) / sum(t)`` from per-sentence statistics."""
    e, t = np.asarray(e, float), np.asarray(t, float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(e), size=(n, len(e)))
    rates = e[idx].sum(1) / t[idx].sum(1)
    return float(np.percentile(rates, 2.5)), float(np.percentile(rates, 97.5))


def paired_bootstrap(e_a, e_b, t, n: int = 10000, seed: int = 0) -> dict:
    """Paired sentence-level bootstrap test of two systems scored on the same sentences.

    ``e_a``/``e_b`` are per-sentence edit counts (possibly averaged over seeds) and ``t`` the
    reference lengths. Sentences are resampled jointly for both systems (Koehn, 2004;
    Bisani & Ney, 2004). Returns the observed difference ``rate_a - rate_b``, its 95% CI,
    the fraction of resamples in which A is better (``p_a_better``) and a two-sided p-value.
    """
    e_a, e_b, t = (np.asarray(v, float) for v in (e_a, e_b, t))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(t), size=(n, len(t)))
    tt = t[idx].sum(1)
    d = (e_a[idx].sum(1) - e_b[idx].sum(1)) / tt
    obs = float((e_a.sum() - e_b.sum()) / t.sum())
    p_one = float(np.mean(d >= 0)) if obs < 0 else float(np.mean(d <= 0))
    return {"diff": obs, "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
            "p_a_better": float(np.mean(d < 0)), "p_two_sided": min(1.0, 2 * p_one), "n_resamples": n}
