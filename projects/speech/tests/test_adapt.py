"""Unsupervised day-layer adaptation: statistics matching and pseudo-label construction."""

import numpy as np
import torch

from speech.adapt import confidence, feature_stats, fold_into_day_layer, matching_transform, words_to_phones
from speech.data import Trial
from speech.models import GRUDecoder
from speech.phonemes import SIL_ID


def _trials(X, day=0):
    return [Trial(X[i : i + 50].astype(np.float16), np.array([1]), "", day, "s", 0) for i in range(0, len(X), 50)]


def test_coral_maps_new_day_covariance_onto_reference():
    rng = np.random.default_rng(0)
    F = 6
    L_ref, L_new = rng.normal(size=(F, F)), rng.normal(size=(F, F))
    X_ref = rng.normal(size=(20000, F)) @ L_ref + 1.0
    X_new = rng.normal(size=(20000, F)) @ L_new - 2.0
    m_new, C_new = feature_stats(_trials(X_new))
    m_ref, C_ref = feature_stats(_trials(X_ref))
    A, c = matching_transform(m_new, C_new, m_ref, C_ref, "coral", shrink=0.0)
    Y = X_new.astype(np.float16).astype(np.float64) @ A + c
    assert np.allclose(Y.mean(0), m_ref, atol=0.05)
    assert np.allclose(np.cov(Y.T), C_ref, atol=0.05 * np.abs(C_ref).max())
    A2, _ = matching_transform(m_new, C_new, m_ref, C_ref, "moments")
    assert np.allclose(A2, np.diag(np.diag(A2)))  # diagonal: per-feature scale only


def test_folding_the_map_into_the_day_layer_is_exact():
    torch.manual_seed(0)
    model = GRUDecoder(3, n_features=8, hidden=8, layers=1, patch_size=2, patch_stride=1, smooth_std=0.0, input_dropout=0.0)
    with torch.no_grad():
        model.day.weight[1] += 0.3 * torch.randn(8, 8)
        model.day.bias[1] += 0.1 * torch.randn(1, 8)
    A, c = np.random.default_rng(1).normal(size=(8, 8)), np.random.default_rng(2).normal(size=8)
    fold_into_day_layer(model, new_day=2, src_day=1, A=A, c=c)
    model.eval()
    x = torch.randn(2, 5, 8)
    mapped = x @ torch.tensor(A, dtype=torch.float32) + torch.tensor(c, dtype=torch.float32)
    out_new = model.day(x, torch.tensor([2, 2]))
    out_src = model.day(mapped, torch.tensor([1, 1]))
    assert torch.allclose(out_new, out_src, atol=1e-5)


def test_pseudo_labels_follow_dataset_format_and_confidence():
    lex = {"hi": [(15, 5)], "you": [(36, 33)]}
    ph = words_to_phones("Hi you.", lex)
    assert ph.tolist() == [15, 5, SIL_ID, 36, 33, SIL_ID]
    assert words_to_phones("hi zzz", lex) is None
    lp = np.full((6, 41), -10.0)
    for t, c in enumerate([15, 5, SIL_ID, 36, 33, SIL_ID]):
        lp[t, c] = 0.0
    assert confidence(lp, ph) == 1.0
    assert confidence(lp, np.array([1, 2, 3])) < 0.5
