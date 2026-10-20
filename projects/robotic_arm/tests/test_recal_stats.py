import numpy as np
import torch

from robotic_arm.decoders.neural import (SeqDecoder, align_readin_kl, align_readin_moments, gaussian_kl,
                                         latent_stats, mean_readin)
from robotic_arm.stats import add_decoding_stats, bootstrap_ci, paired_signflip


def _model(C=12, seed=0):
    torch.manual_seed(seed)
    return SeqDecoder(C, 3, n_readins=2, d_in=6, hidden=16, layers=1, dropout=0.0).eval()


def _data(T=1500, C=12, seed=0):
    rng = np.random.default_rng(seed)
    lat = np.zeros((T, 3))
    for t in range(1, T):
        lat[t] = 0.95 * lat[t - 1] + 0.3 * rng.standard_normal(3)
    M = rng.standard_normal((C, 3))
    return (lat @ M.T + 0.3 * rng.standard_normal((T, C))).astype(np.float32), M


def test_moment_alignment_matches_reference_latent_moments():
    model = _model()
    z, _ = _data()
    ref = latent_stats(model, [z[:700], z[700:]], [0, 1], window=40, burn=8)
    # new day: channels re-scaled and shifted (a drifted recording)
    rng = np.random.default_rng(1)
    z_new = (z * rng.uniform(0.5, 2.0, z.shape[1]) + rng.normal(0, 1, z.shape[1])).astype(np.float32)
    ri = align_readin_moments(model, [z_new], ref, shrink=0.0)
    with torch.no_grad():
        u = ri(torch.as_tensor(z_new)).double()
    assert torch.allclose(u.mean(0), ref.mu_u.double(), atol=1e-3)
    cov = torch.cov(u.T)
    assert torch.allclose(cov, ref.cov_u.double(), atol=0.05 * float(ref.cov_u.abs().max()))
    # mean-only re-centring keeps the weights of the initial read-in
    ri_c = align_readin_moments(model, [z_new], ref, mean_only=True)
    assert torch.allclose(ri_c.weight, mean_readin(model).weight)


def test_gaussian_kl_zero_for_identical_and_positive_otherwise():
    c = torch.eye(3, dtype=torch.float64) * 2
    mu = torch.zeros(3, dtype=torch.float64)
    assert abs(float(gaussian_kl(mu, c, mu, c))) < 1e-8
    assert float(gaussian_kl(mu + 1, c, mu, c)) > 0.1


def test_kl_alignment_reduces_divergence_and_freezes_core():
    model = _model()
    z, _ = _data()
    ref = latent_stats(model, [z], [0], window=40, burn=8)
    z_new = (2.0 * z + 1.0).astype(np.float32)
    core = [p.detach().clone() for p in model.core.parameters()]

    def kl_h(ri):
        with torch.no_grad():
            st = latent_stats(model, [z_new], [ri], window=40, burn=8)
        return float(gaussian_kl(st.mu_h.double(), st.cov_h.double(), ref.mu_h.double(), ref.cov_h.double()))

    base = mean_readin(model)
    ri = align_readin_kl(model, [z_new], ref, steps=60, lr=5e-3, window=40, burn=8, batch=8, anchor=0.0,
                         init=base)
    assert kl_h(ri) < kl_h(base)
    assert all(torch.equal(a, b) for a, b in zip(core, model.core.parameters()))
    assert all(p.requires_grad for p in model.parameters())


def test_bootstrap_and_signflip():
    lo, hi = bootstrap_ci([1.0, 2.0, 3.0, 4.0])
    assert 1.0 <= lo < 2.5 < hi <= 4.0
    # 7 strictly positive differences: exact two-sided p = 2/128
    assert abs(paired_signflip(np.arange(1, 8) + 1.0, np.zeros(7)) - 2 / 128) < 1e-12
    assert paired_signflip([1, -1], [0, 0]) == 1.0


def test_add_decoding_stats_adds_ci_and_paired_tests():
    out = {"within": {"gru": {"r2_vw_blocks": [0.6, 0.7, 0.5]}, "wiener_daynorm": {"r2_vw_blocks": [0.3, 0.4, 0.2]}},
           "cross": {}}
    add_decoding_stats(out)
    assert len(out["within"]["gru"]["r2_vw_ci95"]) == 2
    t = out["paired_tests"]["within:gru_vs_wiener_daynorm"]
    assert abs(t["mean_diff"] - 0.3) < 1e-9 and t["n_positive"] == 3
