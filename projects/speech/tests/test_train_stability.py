"""Training-loop stabilisers: curriculum, per-step trace, input rescaling, early stop without schedule change."""

import json

import numpy as np
import torch

from speech.data import Trial
from speech.models import GRUDecoder
from speech.train import lr_lambda, scale_trial, train_loop


def _trials(n=8, F=16):
    rng = np.random.default_rng(0)
    out = []
    for i in range(n):
        T = 20 + 6 * i
        out.append(Trial(rng.normal(size=(T, F)).astype(np.float16), np.array([1, 2, 40]), "x", 0, "s", 0))
    return out


def _rc(**train):
    t = dict(batch_size=2, max_steps=6, max_minutes=10, schedule="steps", lr=1e-3, lr_min_ratio=0.1, warmup_steps=2,
             weight_decay=0.0, grad_clip=1.0, eval_every=3, inter_ctc_weight=0.0, curriculum_steps=3, curriculum_frac=0.5)
    t.update(train)
    return {"train": t, "aug": {}, "model": {"patch_size": 3, "patch_stride": 2}}


def test_trace_curriculum_and_stop_step(tmp_path):
    torch.manual_seed(0)
    trials = _trials()
    model = GRUDecoder(1, n_features=16, hidden=8, layers=1, patch_size=3, patch_stride=2, smooth_std=0.0, blank_bias=4.0)
    hist, state = train_loop(model, trials, trials[:2], _rc(stop_step=5), torch.device("cpu"), 0, tmp_path / "log.jsonl")
    tr = [json.loads(l) for l in (tmp_path / "trace.jsonl").read_text().splitlines()]
    assert [r["step"] for r in tr] == [1, 2, 3, 4, 5]  # stopped at 5 of a 6-step schedule
    assert all(r["grad_norm"] >= 0 and 0 <= r["blank_argmax_frac"] <= 1 for r in tr)
    assert tr[0]["mean_p_blank"] > 0.4  # blank-bias initialisation
    # the LR schedule is that of the full 6 steps (stop_step does not compress it)
    assert np.isclose(tr[4]["lr"], 1e-3 * lr_lambda(4, 2, 6, 0.1))
    assert state is not None and hist[-1]["step"] == 5


def test_scale_trial_rescales_features():
    t = _trials(1)[0]
    s = scale_trial(t, 2.0)
    assert np.allclose(s.features.astype(np.float32), 2 * t.features.astype(np.float32), atol=1e-2)
    assert s.phones is t.phones
