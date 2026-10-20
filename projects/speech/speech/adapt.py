"""Unsupervised test-time adaptation of the day-specific input layer to a new recording day.

Only the new day's affine input layer (``model.day.weight[new_day]``, ``model.day.bias[new_day]``)
is changed; the shared encoder is frozen. No labels from the new day are used. Two families:

* **Feature-statistics matching.** The new day's unlabelled neural features are mapped onto the
  statistics of the most recent training day, and the map is folded into the copied input layer.
  ``"moments"`` matches per-feature mean and standard deviation; ``"coral"`` matches the full
  covariance (CORAL; Sun, Feng & Saenko, 2016): ``A = C_new^{-1/2} C_ref^{1/2}`` with shrinkage
  towards the diagonal.
* **Self-training with confidence filtering.** The unlabelled sentences are decoded with the
  lexicon + trigram beam search; the decoded words are turned back into phoneme targets with the
  pronunciation lexicon (pseudo-labels; Lee, 2013; Kahn, Lee & Hannun, 2020). Sentences whose
  greedy phoneme string disagrees most with the LM-constrained pseudo-label are discarded, and the
  day layer is fine-tuned with CTC on the rest. This is repeated for a few rounds, re-labelling
  with the adapted layer each time.

Both reuse the training loop of :mod:`speech.train`.
"""

from __future__ import annotations

import copy

import numpy as np
import torch

from .ctc import greedy_decode
from .data import Trial
from .metrics import edit_distance, normalize_text
from .phonemes import SIL_ID
from .train import posteriors, train_loop


# --------------------------------------------------------------------------- statistics matching
def feature_stats(trials: list[Trial], every: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Mean (F,) and covariance (F, F) of the frames of ``trials`` (every ``every``-th frame)."""
    X = np.concatenate([t.features[::every].astype(np.float64) for t in trials])
    m = X.mean(0)
    Xc = X - m
    return m, Xc.T @ Xc / max(1, len(X) - 1)


def _sqrtm_psd(C: np.ndarray, inv: bool = False, eps: float = 1e-6) -> np.ndarray:
    w, V = np.linalg.eigh((C + C.T) / 2)
    w = np.maximum(w, eps)
    return (V * (w ** (-0.5 if inv else 0.5))) @ V.T


def matching_transform(m_new, C_new, m_ref, C_ref, mode: str = "coral", shrink: float = 0.1):
    """Affine map ``x -> x @ A + c`` taking the new day's statistics to the reference statistics."""
    if mode == "moments":
        s_new, s_ref = np.sqrt(np.diag(C_new)), np.sqrt(np.diag(C_ref))
        A = np.diag(s_ref / np.maximum(s_new, 1e-6))
    elif mode == "coral":
        def reg(C):
            return (1 - shrink) * C + shrink * np.diag(np.diag(C))

        A = _sqrtm_psd(reg(C_new), inv=True) @ _sqrtm_psd(reg(C_ref))
    elif mode == "mean":
        A = np.eye(len(m_new))
    else:
        raise ValueError(mode)
    c = m_ref - m_new @ A
    return A, c


def fold_into_day_layer(model, new_day: int, src_day: int, A: np.ndarray, c: np.ndarray) -> None:
    """Set the new day's layer so that layer_new(x) = layer_src(x @ A + c)."""
    W, b = model.day.weight[src_day].detach().double(), model.day.bias[src_day].detach().double()
    A_t, c_t = torch.as_tensor(A, dtype=torch.float64), torch.as_tensor(c, dtype=torch.float64)
    with torch.no_grad():
        model.day.weight[new_day] = (A_t @ W).to(model.day.weight.dtype)
        model.day.bias[new_day] = (c_t[None, :] @ W + b).to(model.day.bias.dtype)


def stats_adapt(base, new_trials, ref_trials, new_day, src_day, mode="coral", shrink=0.1, every=2):
    """Copy ``base`` and initialise the new day's layer by statistics matching (no labels)."""
    model = copy.deepcopy(base)
    m_new, C_new = feature_stats(new_trials, every)
    m_ref, C_ref = feature_stats(ref_trials, every)
    A, c = matching_transform(m_new, C_new, m_ref, C_ref, mode, shrink)
    fold_into_day_layer(model, new_day, src_day, A, c)
    return model


# --------------------------------------------------------------------------- self-training
def words_to_phones(sentence: str, lexicon: dict[str, list[tuple[int, ...]]]) -> np.ndarray | None:
    """Phoneme targets in the dataset's format (``w1 | w2 | ... |``) from a word string; None on OOV."""
    out: list[int] = []
    for w in normalize_text(sentence).split():
        prons = lexicon.get(w)
        if not prons:
            return None
        out += list(prons[0]) + [SIL_ID]
    return np.array(out, np.int64) if out else None


def confidence(lp: np.ndarray, pseudo: np.ndarray) -> float:
    """Agreement between the unconstrained greedy phonemes and the LM-constrained pseudo-label (1 - PER)."""
    g = greedy_decode(lp)
    return 1.0 - edit_distance(list(pseudo), g) / max(1, len(pseudo))


def pseudo_label(model, trials, decode_fn, lexicon, device, keep_frac: float, min_conf: float | None = None):
    """Decode ``trials`` and return (pseudo-labelled trials sorted by confidence, kept, stats)."""
    lps = posteriors(model, trials, device)
    hyps = decode_fn(lps)
    cands = []
    for t, lp, h in zip(trials, lps, hyps):
        ph = words_to_phones(h, lexicon)
        if ph is None or len(ph) == 0:
            continue
        n_out = (t.features.shape[0] - model.patch_size) // model.patch_stride + 1
        if n_out < len(ph):  # CTC-infeasible target
            continue
        cands.append((confidence(lp, ph), Trial(t.features, ph, h, t.day, t.session, t.block)))
    cands.sort(key=lambda c: -c[0])
    k = max(1, int(round(keep_frac * len(cands))))
    kept = [c for c in cands[:k] if min_conf is None or c[0] >= min_conf]
    confs = [c[0] for c in cands]
    return [c[1] for c in kept], {"n_candidates": len(cands), "n_kept": len(kept),
                                  "conf_mean": float(np.mean(confs)) if confs else None,
                                  "conf_threshold": float(kept[-1][0]) if kept else None}


def finetune_day_layer(model, trials, rc, device, seed, steps, lr, batch_size, new_day=None):
    """CTC fine-tuning of the day-specific input layers only (in place); returns the model."""
    for n, p in model.named_parameters():
        p.requires_grad = n.startswith("day.")
    rc = copy.deepcopy(rc)
    rc["train"].update(max_steps=steps, max_minutes=1e9, schedule="steps", warmup_steps=5, lr=lr,
                       batch_size=min(batch_size, len(trials)), eval_every=10**9, lr_min_ratio=0.1,
                       curriculum_steps=0, stop_step=None)
    # No labelled dev data on the new day: keep the last iterate (dev = the training pseudo-labels).
    _, state = train_loop(model, trials, trials[: min(len(trials), 10)], rc, device, seed)
    model.load_state_dict(state)
    for p in model.parameters():
        p.requires_grad = True
    return model


def self_train(model, pool, decode_fn, lexicon, rc, device, *, rounds=2, keep_frac=0.5, steps=40, lr=0.005,
               batch_size=16, seed=0, log=None):
    """Iterative pseudo-label self-training of the day layer on unlabelled ``pool`` trials."""
    model = copy.deepcopy(model)
    for r in range(rounds):
        kept, st = pseudo_label(model, pool, decode_fn, lexicon, device, keep_frac)
        if log is not None:
            log.append({"round": r, **st})
        if not kept:
            break
        torch.manual_seed(seed + r)
        model = finetune_day_layer(model, kept, rc, device, seed + r, steps, lr, batch_size)
    return model
