"""Tests for the second-round additions: recalibrated thresholds, frame dynamics, threshold-aware and
temporal encoders, known-dead masking and dropout augmentation."""
import math

import numpy as np
import pytest
import torch

from blindsight.encoders import (BakedThresholdAwareEncoder, Condition, MaskedEncoder, ThresholdAwareEncoder)
from blindsight.simulator import PhospheneSimulator, SimParams, TemporalSimulator, frame_update
from blindsight.temporal import make_sequences, rollout, temporal_encoder_from
from blindsight.train import end_to_end_loss, failure_mask, patient_condition


def test_fernandez_threshold_model_matches_reported_moments():
    p = SimParams(threshold_model="fernandez2021")
    thr = p.threshold_current_ua(p.sample_thresholds(np.random.default_rng(0), 200_000))
    assert abs(thr.mean() - 66.8) < 0.5 and abs(thr.std() - 36.5) < 0.5
    assert thr.min() >= p.rheobase_a * 1e6 - 1e-6  # truncated at the rheobase
    # threshold_mean stays the activation-space mean, so the nominal patient sits at 66.8 uA
    assert math.isclose(float(p.threshold_current_ua(p.threshold_mean)), 66.8, rel_tol=1e-6)


def test_dynaphos_threshold_model_unchanged():
    p = SimParams()
    a = p.sample_thresholds(np.random.default_rng(0), 50_000)
    assert abs(a.mean() - p.threshold_mean) / p.threshold_mean < 0.05  # clipping at 0 raises the mean ~3 %


def test_frame_update_from_rest_equals_static_mode():
    p = SimParams()
    amp = torch.tensor([[30.0, 60.0, 100.0]])
    act, _ = frame_update(p, amp, torch.zeros(1, 3), torch.zeros(1, 3))
    assert torch.allclose(act, PhospheneSimulator(p).activation(amp), rtol=1e-6)


def test_frame_update_matches_euler_over_on_off_frames():
    p = SimParams()
    ts = TemporalSimulator(p, 1, dt_s=p.train_s / 50, substeps=20)
    act, tr = torch.zeros(1), torch.zeros(1)
    for f in range(6):
        amp = torch.tensor([90.0 if f % 2 == 0 else 0.0])
        for _ in range(50):
            ts.step(amp)
        act, tr = frame_update(p, amp, act, tr)
        assert abs(float(act) - float(ts.act)) <= 0.01 * float(ts.act) + 1e-12
        assert abs(float(tr) - float(ts.trace)) <= 0.01 * float(ts.trace) + 1e-12


def test_habituation_eventually_silences_a_constant_drive():
    p = SimParams()
    act, tr = torch.zeros(1), torch.zeros(1)
    acts = []
    for _ in range(30):
        act, tr = frame_update(p, torch.tensor([100.0]), act, tr)
        acts.append(float(act))
    assert acts[-1] < 0.3 * max(acts)


def test_brightness_gate_equals_gating_the_current(setup):
    pt = setup.patient(3)
    g = torch.Generator().manual_seed(1)
    level = 70 + 30 * torch.rand(2, setup.n, generator=g)
    gate = (torch.rand(2, setup.n, generator=g) > 0.5).float()
    a = setup.sim(level, pt, gate=gate)
    b = setup.sim(gate * level, pt)
    assert torch.allclose(a, b, atol=1e-6)


def _ta(setup, temporal=False):
    torch.manual_seed(0)
    return ThresholdAwareEncoder(setup.context(calibrated=False), width=8, temporal=temporal).eval()


def test_threshold_aware_currents_are_supra_threshold_or_zero(setup):
    enc = _ta(setup)
    with torch.no_grad():
        enc.head[2].bias[0] = 10.0  # force gates on so the supra-threshold mapping is exercised
    pt = setup.patient(4)
    cond = patient_condition(setup, pt)
    x = torch.rand(3, 1, setup.params.resolution, setup.params.resolution)
    amp = enc(x, cond)
    on = amp > 0
    assert on.any()
    thr = cond.threshold_ua.expand_as(amp)
    assert torch.all(amp[on] >= thr[on] + enc.margin - 1e-4)
    assert torch.all(amp <= setup.params.max_current_ua + 1e-4)
    unusable = (cond.alive == 0) | (cond.threshold_ua + enc.margin >= setup.params.max_current_ua)
    assert torch.all(amp[:, unusable] == 0)


def test_threshold_aware_respects_known_dead_mask(setup):
    enc = _ta(setup)
    cond = patient_condition(setup, setup.patient(4))
    keep = torch.ones(setup.n)
    keep[: setup.n // 2] = 0
    amp = enc(torch.rand(2, 1, setup.params.resolution, setup.params.resolution), cond.masked(keep))
    assert torch.all(amp[:, : setup.n // 2] == 0)


def test_threshold_aware_training_step_has_gradients_for_gates(setup, cfg):
    enc = _ta(setup).train()
    pt = setup.patient(100)
    cond = patient_condition(setup, pt)
    x = torch.rand(4, 1, setup.params.resolution, setup.params.resolution)
    loss, _ = end_to_end_loss(enc, None, setup, pt, x, {**cfg["train"], "w_recon": 0.0}, cond=cond)
    loss.backward()
    assert enc.head[2].weight.grad[0].abs().sum() > 0  # gate logit row


def test_baked_onnx_export_keeps_io(setup64, tmp_path):
    onnx = pytest.importorskip("onnx")
    ort = pytest.importorskip("onnxruntime")
    enc = _ta(setup64)
    baked = BakedThresholdAwareEncoder(enc, patient_condition(setup64, setup64.patient(1000))).eval()
    x = torch.rand(2, 1, 64, 64)
    path = tmp_path / "ta.onnx"
    torch.onnx.export(baked, x, str(path), input_names=["image"], output_names=["amplitude_ua"],
                      dynamic_axes={"image": {0: "batch"}, "amplitude_ua": {0: "batch"}}, opset_version=17)
    onnx.checker.check_model(onnx.load(str(path)))
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    assert [i.name for i in sess.get_inputs()] == ["image"]
    assert [o.name for o in sess.get_outputs()] == ["amplitude_ua"]
    got = sess.run(None, {"image": x.numpy()})[0]
    assert got.shape == (2, setup64.n)
    assert np.abs(got - baked(x).detach().numpy()).max() < 1e-3


def test_masked_encoder_frees_top_k_budget(setup):
    class Const(torch.nn.Module):
        def forward(self, x):
            return (50.0 + 40.0 * torch.arange(setup.n) / setup.n).expand(x.shape[0], -1)
    keep = torch.ones(setup.n)
    keep[::2] = 0
    amp = setup.safety(MaskedEncoder(Const(), keep)(torch.zeros(1, 1, 4, 4)))
    assert torch.all(amp[0, ::2] == 0)
    assert int((amp > 0).sum()) == min(setup.safety.max_active or setup.n, int(keep.sum()))


def test_failure_mask_fraction():
    g = torch.Generator().manual_seed(0)
    m = failure_mask(4000, 200, 0.5, 1.0, g, torch.device("cpu"))
    assert abs(float(1 - m.mean()) - 0.25) < 0.01  # E[U(0, 0.5)] = 0.25
    assert float(failure_mask(10, 50, 0.5, 0.0, g, torch.device("cpu")).min()) == 1.0


def test_make_sequences_static_and_moving():
    g = torch.Generator().manual_seed(0)
    x = torch.zeros(2, 1, 32, 32)
    x[:, :, 14:18, 14:18] = 1
    st = make_sequences(x, 5, "static", 1.0, g)
    assert st.shape == (2, 5, 1, 32, 32) and torch.equal(st[:, 2], x)
    mv = make_sequences(x, 5, "moving", 2.0, g)
    assert torch.allclose(mv[:, 2], x, atol=1e-5)  # centred at mid-sequence
    assert not torch.allclose(mv[:, 0], mv[:, 4])


def test_temporal_encoder_starts_as_threshold_aware(setup):
    ta = _ta(setup)
    tmp = temporal_encoder_from(ta, setup, width=8).eval()
    cond = patient_condition(setup, setup.patient(4))
    x = torch.rand(2, 1, setup.params.resolution, setup.params.resolution)
    t_cond = Condition(cond.threshold_ua, cond.alive, trace_ua=torch.zeros(2, setup.n), prev_amp_ua=torch.rand(2, setup.n) * 50)
    assert torch.allclose(ta(x, cond), tmp(x, t_cond), atol=1e-5)


def test_rollout_first_frame_equals_static_percept(setup):
    ta = _ta(setup)
    pt = setup.patient(4)
    cond = patient_condition(setup, pt)
    x = torch.rand(2, 1, setup.params.resolution, setup.params.resolution)
    seq = make_sequences(x, 3, "static", 1.0, torch.Generator().manual_seed(0))
    p, a, wasted, _ = rollout("threshold_aware", ta, setup, pt, seq, cond)
    static = setup.sim(setup.safety(ta(x, cond)) * pt.alive, pt)
    assert torch.allclose(p[:, 0], static, atol=1e-5)
    assert not wasted[:, 0].any()  # threshold-aware currents from rest always evoke a phosphene
