import numpy as np
import torch

from robotic_arm.decoders.linear import KalmanFilter, WienerFilter, ridge_from_stats, ridge_solve, ridge_stats
from robotic_arm.decoders.neural import SeqDecoder, TrainCfg, fit_new_readin, train_model
from robotic_arm.metrics import r2_variance_weighted


def _synthetic(T=4000, C=30, D=3, seed=0):
    """Smooth velocities, neural = linear tuning + noise."""
    rng = np.random.default_rng(seed)
    v = np.zeros((T, D))
    for t in range(1, T):
        v[t] = 0.97 * v[t - 1] + 0.25 * rng.standard_normal(D)
    H = rng.standard_normal((C, D))
    z = v @ H.T + 0.5 * rng.standard_normal((T, C))
    return z.astype(np.float32), v.astype(np.float32)


def test_ridge_from_stats_equals_direct_solve():
    rng = np.random.default_rng(0)
    x, y = rng.standard_normal((300, 8)), rng.standard_normal((300, 2))
    w1, b1 = ridge_solve(x, y, 0.1)
    w2, b2 = ridge_from_stats(*ridge_stats(x, y), 0.1)
    assert np.allclose(w1, w2) and np.allclose(b1, b2)
    # pooling statistics == fitting on concatenated data
    s1, s2 = ridge_stats(x[:100], y[:100]), ridge_stats(x[100:], y[100:])
    w3, b3 = ridge_from_stats(s1[0] + s2[0], s1[1] + s2[1], s1[2] + s2[2], 0.1)
    assert np.allclose(w1, w3) and np.allclose(b1, b3)


def test_wiener_and_kalman_decode_synthetic_data():
    z, v = _synthetic()
    m = np.ones(len(z), bool)
    tr, te = slice(0, 3000), slice(3000, None)
    wf = WienerFilter(n_lags=3, alpha=1e-3).fit([z[tr]], [v[tr]], [m[tr]])
    assert r2_variance_weighted(v[te], wf.predict(z[te])) > 0.8
    kf = KalmanFilter(smooth_tau=0.0).fit([z[tr]], [v[tr]], [m[tr]])
    assert r2_variance_weighted(v[te], kf.predict(z[te])) > 0.8


def test_gru_shapes_and_tiny_overfit():
    torch.manual_seed(0)
    z, v = _synthetic(T=600, C=20)
    model = SeqDecoder(20, 3, n_readins=2, d_in=16, hidden=32, layers=1, dropout=0.0)
    out = model(torch.as_tensor(z[None, :50]), 1)
    assert out.shape == (1, 50, 3)
    m = np.ones(len(z), bool)
    data = ([z], [v], [m], [0])
    first = ((model.predict(z, 0) - v) ** 2).mean()
    train_model(model, data, data, TrainCfg(window=50, batch=8, steps=300, lr=5e-3, eval_every=100, patience=10),
                log=lambda *_: None)
    last = ((model.predict(z, 0) - v) ** 2).mean()
    assert last < 0.3 * first


def test_new_readin_leaves_core_frozen():
    torch.manual_seed(0)
    z, v = _synthetic(T=300, C=20)
    model = SeqDecoder(20, 3, n_readins=2, d_in=16, hidden=32, layers=1, dropout=0.0)
    core_before = [p.detach().clone() for p in model.core.parameters()]
    ri = fit_new_readin(model, [z], [v], [np.ones(len(z), bool)], steps=20, window=50, batch=4)
    assert all(torch.equal(a, b) for a, b in zip(core_before, model.core.parameters()))
    assert ri.weight.shape == model.readins[0].weight.shape
    assert all(p.requires_grad for p in model.parameters())


def test_batched_per_day_readins_match_individual_calls():
    torch.manual_seed(0)
    model = SeqDecoder(20, 3, n_readins=3, d_in=8, hidden=16, layers=1, dropout=0.0).eval()
    x = torch.randn(4, 10, 20)
    days = torch.tensor([0, 2, 1, 2])
    batched = model(x, days)
    single = torch.cat([model(x[i:i + 1], int(days[i])) for i in range(4)])
    assert torch.allclose(batched, single, atol=1e-6)
