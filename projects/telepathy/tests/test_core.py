"""Fast unit tests: metrics, features, decoders, models, alignment and the simulator."""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from telepathy.align import FAStabilizedDecoder, procrustes
from telepathy.data import Session
from telepathy.features import DayData, channel_stats, lagged, make_day
from telepathy.clda import CLDAConfig, finetune_closed_loop, refit_labels
from telepathy.linear import KalmanFilter, PVKalmanFilter, WienerFilter
from telepathy.remap import ChannelMatcher, partial_shuffle, structured_candidates
from telepathy.stats import holm, mean_ci, paired_test, pairwise
from telepathy.metrics import r2_score, webgrid_bitrate
from telepathy.models import GRUDecoder, StableLatentDecoder, alibi_band_bias
from telepathy.simulator import (EncodingModel, UserModel, WebGridConfig, _EncNet, _fast_mlp, run_webgrid,
                                 run_webgrid_batch)
from telepathy.train import (AdaptConfig, TorchDecoder, TrainConfig, adapt_sld, gaussian_kl, latent_stats,
                             train_neural)

torch.set_num_threads(1)  # tiny ops: one thread is fastest on a shared CPU


def synthetic_day(T: int = 3000, C: int = 24, seed: int = 0, lag: int = 2) -> DayData:
    """Cosine-tuned Poisson neurons driven by a smooth random 2-D velocity."""
    rng = np.random.default_rng(seed)
    v = np.cumsum(rng.standard_normal((T + 50, 2)), 0)
    v = np.convolve(v[:, 0], np.ones(25) / 25, "same"), np.convolve(v[:, 1], np.ones(25) / 25, "same")
    v = np.stack(v, 1)[25:-25]
    v = 50 * (v - v.mean(0)) / v.std(0)
    pd = rng.uniform(0, 2 * np.pi, C)
    W = np.stack([np.cos(pd), np.sin(pd)], 1)
    rates = 0.6 * np.exp(0.02 * v @ W.T)
    counts = rng.poisson(rates).astype(np.float32)
    # neural activity leads velocity by `lag` bins
    vel = np.concatenate([np.zeros((lag, 2)), v[:-lag]]).astype(np.float32)
    pos = np.cumsum(vel, 0) * 0.02
    target = np.repeat(pos[::60], 60, axis=0)[:T]
    s = Session("synth_20200101_01", counts, pos.astype(np.float32), vel, target.astype(np.float32), 20.0)
    return make_day(s, lag, 0)


# --------------------------------------------------------------------------- metrics
def test_r2_hand_computed():
    y = np.array([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]])
    yhat = np.array([[1.0, 2.0], [2.0, 5.0], [4.0, 6.0]])
    # dim0: ss_res=1, ss_tot=2 -> 0.5 ; dim1: ss_res=1, ss_tot=8 -> 0.875
    assert r2_score(y, yhat) == pytest.approx((0.5 + 0.875) / 2)
    assert r2_score(y, y) == pytest.approx(1.0)


def test_webgrid_bitrate_hand_computed():
    # 35x35 grid, 30 correct, 2 incorrect in 60 s: log2(1224) * 28 / 60
    assert webgrid_bitrate(35 * 35, 30, 2, 60.0) == pytest.approx(math.log2(1224) * 28 / 60)
    assert webgrid_bitrate(100, 3, 5, 60.0) == 0.0


# --------------------------------------------------------------------------- features
def test_lagged_and_splits():
    X = np.arange(12, dtype=np.float32).reshape(6, 2)
    L = lagged(X, 3)
    assert L.shape == (6, 6)
    np.testing.assert_array_equal(L[3], [6, 7, 4, 5, 2, 3])
    np.testing.assert_array_equal(L[0], [0, 1, 0, 0, 0, 0])
    d = synthetic_day()
    assert d.train.stop <= d.val.start and d.val.stop <= d.test.start and d.test.stop == d.T
    assert d.first_trials(0) == slice(0, 0)
    assert d.first_trials(2).stop > d.first_trials(1).stop > 0


# --------------------------------------------------------------------------- decoders
def test_linear_decoders_learn_and_stream():
    d = synthetic_day()
    mu, sd = channel_stats(d.X[d.train])
    seg = [(d.X[d.train], d.Y[d.train])]
    w = WienerFilter(5, 10.0).fit(seg, mu, sd)
    k = KalmanFilter(3).fit(seg, mu, sd)
    assert r2_score(d.Y[d.test], w.predict(d.X)[d.test]) > 0.4
    assert r2_score(d.Y[d.test], k.predict(d.X)[d.test]) > 0.3
    for dec in (w, k):  # streaming == offline
        dec.reset()
        stream = np.array([dec.step(x) for x in d.X[:100]])
        np.testing.assert_allclose(stream, dec.predict(d.X[:100]), atol=1e-3)


def test_wiener_prior_ridge_shrinks_to_prior():
    d = synthetic_day()
    mu, sd = channel_stats(d.X)
    seg = [(d.X[:50], d.Y[:50])]
    W0 = np.ones((24 * 2 + 1, 2), np.float32)
    w = WienerFilter(2, 1e9).fit(seg, mu, sd, W0=W0)
    np.testing.assert_allclose(w.W[:-1], W0[:-1], atol=1e-3)


def test_fa_stabilizer_recovers_permuted_day():
    d = synthetic_day(T=4000)
    fa = FAStabilizedDecoder(q=4, stable_frac=0.5).fit([d])
    base = r2_score(d.Y[d.test], fa.predict(d.X)[d.test])
    rng = np.random.default_rng(1)
    moved = rng.choice(d.X.shape[1], 10, replace=False)
    perm = np.arange(d.X.shape[1])
    perm[moved] = np.roll(moved, 1)
    Xp = d.X[:, perm]  # 10 of 24 electrodes now record different neurons
    broken = r2_score(d.Y[d.test], fa.with_stats(*channel_stats(Xp[d.train])).predict(Xp)[d.test])
    fixed = r2_score(d.Y[d.test], fa.aligned_to(Xp[d.train]).predict(Xp)[d.test])
    assert broken < base - 0.2
    assert fixed > broken + 0.2


def test_procrustes_recovers_rotation():
    rng = np.random.default_rng(0)
    A = rng.standard_normal((30, 3))
    Q, _ = np.linalg.qr(rng.standard_normal((3, 3)))
    np.testing.assert_allclose(procrustes(A, A @ Q), Q, atol=1e-8)


# --------------------------------------------------------------------------- models
def test_alibi_mask_is_causal_and_banded():
    b = alibi_band_bias(6, 2, 3)
    assert torch.isinf(b[0, 0, 1])  # future masked
    assert torch.isinf(b[0, 5, 1])  # beyond window masked
    assert b[0, 5, 3].item() < b[0, 5, 5].item() == 0.0


@pytest.mark.parametrize("backbone", ["transformer", "gru"])
def test_sld_shapes_causality_and_new_day(backbone):
    m = StableLatentDecoder(10, 3, d=16, window=4, backbone=backbone).eval()
    x = torch.randn(2, 20, 10)
    vel, logr, h = m(x, torch.tensor([0, 2]))
    assert vel.shape == (2, 20, 2) and logr.shape == (2, 20, 10) and h.shape == (2, 20, 16)
    x2 = x.clone()
    x2[:, 15:] += 5.0  # changing the future must not change the past
    v2, _, _ = m(x2, torch.tensor([0, 2]))
    torch.testing.assert_close(vel[:, :15], v2[:, :15])
    k = m.add_day()
    assert k == 3 and m.adapt_w.shape[0] == 4
    torch.testing.assert_close(m.adapt_w[3], torch.eye(10))
    assert m.adapter_penalty(torch.tensor([3])).item() == 0.0


def test_torch_decoder_streaming_matches_offline():
    for model in (StableLatentDecoder(8, 1, d=16, window=4), StableLatentDecoder(8, 1, d=16, backbone="gru"),
                  GRUDecoder(8, hidden=16)):
        dec = TorchDecoder(model, 0, np.zeros(8, np.float32), np.ones(8, np.float32), 2.0)
        X = np.random.default_rng(0).poisson(1.0, (40, 8)).astype(np.float32)
        off = dec.predict(X)
        dec.reset()
        on = np.array([dec.step(x) for x in X])
        np.testing.assert_allclose(on, off, atol=1e-4)


def test_gaussian_kl_zero_for_identical_and_positive_otherwise():
    mu, cov = torch.zeros(3), torch.eye(3)
    assert gaussian_kl(mu, cov, mu, cov).item() == pytest.approx(0.0, abs=1e-5)
    assert gaussian_kl(mu + 1, cov, mu, cov).item() > 0.4


def test_tiny_overfit_and_adaptation():
    """SLD fits a small synthetic day; adaptation to a channel-permuted copy helps."""
    d = synthetic_day(T=2500)
    cfg = TrainConfig(steps=120, batch=16, window=48, warmup_bins=16, eval_every=60, augment=False, ssl_weight=0.2,
                      lr=5e-3)
    model, vscale, stats, hist = train_neural("sld", [d], cfg, {"d": 32}, log=lambda *_: None)
    assert hist["val_r2"][-1] > 0.4
    perm = np.random.default_rng(3).permutation(d.X.shape[1])
    Xp = d.X[:, perm]
    mu, sd = channel_stats(Xp[d.train])
    fixed = r2_score(d.Y[d.test], TorchDecoder(model, 0, mu, sd, vscale).predict(Xp)[d.test])
    ref = latent_stats(model, [d], stats, [0])
    m2, k = adapt_sld(model, ref, Xp[d.train], mu, sd, AdaptConfig(steps=10, fewshot_steps=80),
                      labelled=(Xp[d.first_trials(10)], d.Y[d.first_trials(10)]), vscale=vscale)
    adapted = r2_score(d.Y[d.test], TorchDecoder(m2, k, mu, sd, vscale).predict(Xp)[d.test])
    assert adapted > fixed


# --------------------------------------------------------------------------- simulator
def _toy_encoder(C: int = 12) -> EncodingModel:
    torch.manual_seed(0)
    enc = EncodingModel(None, 4, 1, 50.0, np.zeros(2, np.float32), 65.0, np.zeros((C, 2), np.float32), use_pos=True)
    enc.net = _EncNet(enc.n_features, C, hidden=16)
    return enc


def test_encoder_features_and_numpy_forward():
    enc = _toy_encoder()
    V = np.arange(20, dtype=np.float32).reshape(10, 2)
    F = enc.features(V, np.zeros((10, 2), np.float32))
    assert F.shape == (10, 3 * 5 + 2)
    np.testing.assert_allclose(F[3, :2], V[4] / 50.0)  # newest column = one bin in the future
    np.testing.assert_allclose(F[3, 2:4], V[3] / 50.0)
    np.testing.assert_allclose(F[3, 10], np.linalg.norm(V[4] / 50.0), rtol=1e-6)
    buf = np.stack([V[4 - k] / 50.0 for k in range(5)])  # newest-first buffer at t=3
    np.testing.assert_allclose(enc.step_features(buf, np.zeros(2)), F[3], rtol=1e-6)
    np.testing.assert_allclose(_fast_mlp(enc, F[3]), enc.log_rates(F[3:4])[0], atol=1e-5)


class _OracleDecoder:
    """Returns a fixed velocity toward a known point: sanity check of the task logic."""

    def __init__(self):
        self.mu = np.zeros(1)

    def reset(self):
        pass

    def step(self, x):
        return np.zeros(2)


def test_webgrid_runs_and_bitrate_consistent():
    enc = _toy_encoder()
    res = run_webgrid(_OracleDecoder(), enc, UserModel(), WebGridConfig(grid=5, seconds=4, false_click_rate_hz=5.0),
                      seed=0)
    assert res.cursor.shape == (200, 2)
    assert res.bps == pytest.approx(webgrid_bitrate(25, res.n_correct, res.n_incorrect, 4))
    assert res.n_incorrect + res.n_correct == len(res.selections)


# --------------------------------------------------------------------------- improvement round 2
def test_pv_kalman_learns_streams_and_batches():
    d = synthetic_day()
    mu, sd = channel_stats(d.X[d.train])
    seg = [(d.X[d.train], d.Y[d.train])]
    for order, delay, sq in ((0, 0, False), (1, 1, False), (2, 0, True)):
        k = PVKalmanFilter(3, obs_delay=delay, order=order, sqrt=sq).fit(seg, mu, sd)
        assert r2_score(d.Y[d.test], k.predict(d.X)[d.test]) > 0.3
        k.reset()
        stream = np.array([k.step(x) for x in d.X[:80]])
        np.testing.assert_allclose(stream, k.predict(d.X[:80]), atol=1e-3)
    # ReFIT-KF position feedback: known cursor position is used, output stays finite
    k.reset_batch(2)
    k.set_cursor(np.zeros((2, 2)))
    assert np.all(np.isfinite(k.step_batch(d.X[:2])))
    assert np.allclose(k.K_fb[:2], 0)  # neural data never update the (known) position
    # pooled fit with per-segment statistics equals a fit on pre-normalised data
    k1 = PVKalmanFilter(3).fit(seg, mu, sd, seg_stats=[(mu, sd)])
    k2 = PVKalmanFilter(3).fit(seg, mu, sd)
    np.testing.assert_allclose(k1.K, k2.K, atol=1e-8)


def test_batched_streaming_equals_single_stream():
    d = synthetic_day()
    mu, sd = channel_stats(d.X[d.train])
    seg = [(d.X[d.train], d.Y[d.train])]
    X = np.stack([d.X[:60], d.X[100:160], d.X[300:360]], 1)  # (T, B, C)
    decs = [WienerFilter(4, 10.0).fit(seg, mu, sd), KalmanFilter(3).fit(seg, mu, sd),
            PVKalmanFilter(3, obs_delay=1).fit(seg, mu, sd),
            TorchDecoder(GRUDecoder(24, hidden=8), 0, mu, sd, 2.0),
            TorchDecoder(StableLatentDecoder(24, 1, d=8, backbone="gru"), 0, mu, sd, 2.0),
            TorchDecoder(StableLatentDecoder(24, 1, d=8, window=4), 0, mu, sd, 2.0)]
    for dec in decs:
        dec.reset_batch(3)
        batch = np.stack([dec.step_batch(x) for x in X], 1)
        for b in range(3):
            np.testing.assert_allclose(batch[b], dec.predict(X[:, b]), atol=1e-3)


def test_batched_webgrid_sessions_independent_of_batch():
    """Common random numbers: a session's result depends only on its seed, not on the batch."""
    enc = _toy_encoder()
    w = WienerFilter(2, 10.0)
    w.mu, w.sd = np.ones(12, np.float32), np.ones(12, np.float32)
    w.W = np.random.default_rng(0).standard_normal((25, 2)).astype(np.float32) * 5
    cfg = WebGridConfig(grid=4, seconds=3, false_click_rate_hz=3.0)
    batch = run_webgrid_batch(w, enc, UserModel(), cfg, [3, 4, 5], gain=[1.0, 1.5, 0.75], smooth=0.3)
    for r, s, g in zip(batch, (3, 4, 5), (1.0, 1.5, 0.75)):
        alone = run_webgrid(w, enc, UserModel(), cfg, gain=g, smooth=0.3, seed=s)
        np.testing.assert_allclose(r.cursor, alone.cursor, atol=1e-4)  # BLAS rounding only
        assert r.selections == alone.selections and r.bps == alone.bps
    # oracle sessions with the same seed share targets regardless of decoder
    o = run_webgrid_batch(None, enc, UserModel(), cfg, [3])[0]
    assert o.targets[0] == batch[0].targets[0]


def test_paired_stats():
    rng = np.random.default_rng(0)
    a = rng.normal(1.0, 0.3, 40)
    b = a - 0.5 + rng.normal(0, 0.05, 40)
    r = paired_test(a, b)
    assert r["diff_ci95"][0] > 0.4 and r["p_wilcoxon"] < 1e-6
    assert paired_test(a, a)["p_wilcoxon"] == 1.0
    m = mean_ci(a)
    assert m["ci95"][0] < m["mean"] < m["ci95"][1]
    adj = holm({"x": 0.01, "y": 0.04, "z": 0.03})
    assert adj == pytest.approx({"x": 0.03, "z": 0.06, "y": 0.06})
    pw = pairwise({"a": a, "b": b, "c": a + 0.001 * rng.normal(size=40)})
    assert pw["a vs b"]["significant"] and not pw["a vs c"]["significant"]


def test_channel_matcher_recovers_block_remap():
    """A connector-style block re-ordering is found from unlabelled spikes; normal days keep identity."""
    rng = np.random.default_rng(0)
    C, T = 24, 6000
    rates = rng.gamma(2.0, 0.3, C)
    L = rng.standard_normal((C, 3)) * 0.4
    def day(seed):
        r = np.random.default_rng(seed)
        z = np.cumsum(r.standard_normal((T, 3)), 0) * 0.05
        z = z - np.convolve(z[:, 0], np.ones(50) / 50, "same")[:, None]
        return r.poisson(rates * np.exp(np.clip(z @ L.T, -3, 3))).astype(np.float32)
    m = ChannelMatcher([day(1), day(2)], margin=0.05)
    assert len(structured_candidates(C)) > 50
    assert not m.match(day(3))["changed"]
    true = structured_candidates(C)["b2_rb1_ri0_s0"]  # pair reversal
    X = day(4)[:, true]
    res = m.match(X)
    assert np.array_equal(true[res["perm"]], np.arange(C))
    sh, moved = partial_shuffle(C, 0.25, rng)
    assert len(moved) == 6 and np.all(sh[moved] != moved)


def test_refit_labels_and_closed_loop_finetune():
    class R:  # minimal closed-loop record
        grid = 2
        cursor = np.array([[0.0, 0.0], [10.0, 10.0]])
        target_pos = np.array([[10.0, 0.0], [10.0, 10.0]])
        decoded = np.array([[0.0, 3.0], [5.0, 5.0]])
        counts = np.zeros((2, 4), np.float32)
    (_, y), = refit_labels([R()], side_mm=40.0)
    np.testing.assert_allclose(y[0], [3.0, 0.0], atol=1e-6)  # rotated toward target, speed kept
    np.testing.assert_allclose(y[1], [0.0, 0.0])  # on the target: zero
    d = synthetic_day(T=1500)
    mu, sd = channel_stats(d.X[d.train])
    m = StableLatentDecoder(24, 1, d=16, backbone="gru")
    dec = TorchDecoder(m, 0, mu, sd, 50.0)
    seg = [(d.X[d.train], d.Y[d.train])]
    before = r2_score(d.Y[d.test], dec.predict(d.X)[d.test])
    new = finetune_closed_loop(dec, seg, CLDAConfig(steps=60, lr_adapter=3e-3, window=48, warmup_bins=8))
    assert r2_score(d.Y[d.test], new.predict(d.X)[d.test]) > before
    # adapter-only: shared weights unchanged
    torch.testing.assert_close(new.model.readin.weight, m.readin.weight)
    g = TorchDecoder(GRUDecoder(24, hidden=16), 0, mu, sd, 50.0)
    g2 = finetune_closed_loop(g, seg, CLDAConfig(steps=5, window=48, warmup_bins=8), offline=seg)
    assert not torch.equal(g2.model.out.weight, g.model.out.weight)
