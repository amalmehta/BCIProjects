"""Fast unit tests: metrics, features, decoders, models, alignment and the simulator."""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from telepathy.align import FAStabilizedDecoder, procrustes
from telepathy.data import Session
from telepathy.features import DayData, channel_stats, lagged, make_day
from telepathy.linear import KalmanFilter, WienerFilter
from telepathy.metrics import r2_score, webgrid_bitrate
from telepathy.models import GRUDecoder, StableLatentDecoder, alibi_band_bias
from telepathy.simulator import (EncodingModel, UserModel, WebGridConfig, _EncNet, _fast_mlp, run_webgrid)
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
