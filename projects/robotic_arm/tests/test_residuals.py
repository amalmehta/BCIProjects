import numpy as np

from robotic_arm.closed_loop import calibrate_residual
from robotic_arm.residuals import OnlineResidualDecoder, fit_residual_model, integrated_autocorr_time


def _real_like(T=3000, seed=0, lag=3, gain=0.6):
    rng = np.random.default_rng(seed)
    z = np.zeros((T, 7))
    for t in range(1, T):
        z[t] = 0.98 * z[t - 1] + 0.2 * rng.standard_normal(7)
    e = np.zeros((T, 7))
    for t in range(1, T):  # AR(1) errors, heteroscedastic in |z|
        e[t] = 0.9 * e[t - 1] + 0.3 * rng.standard_normal(7)
    pred = np.zeros_like(z)
    pred[lag:] = gain * z[:-lag]
    pred += e * (0.5 + 0.5 * np.abs(z))
    return z, pred


def test_fir_recovers_gain_and_lag():
    z, p = _real_like()
    m = fit_residual_model([z], [p], [np.ones(len(z), bool)], n_lags=6, ridge=1e-3, block_len=50)
    assert abs(np.diag(m.H.sum(0)).mean() - 0.6) < 0.08  # total gain
    assert np.diag(m.H[3]).mean() > 0.4 and abs(np.diag(m.H[0]).mean()) < 0.1  # peak at the true lag
    assert all(c > 0.1 for c in m.scale_c)  # picked up the speed dependence


def test_block_bootstrap_preserves_autocorrelation_and_scale():
    z, p = _real_like()
    m = fit_residual_model([z], [p], [np.ones(len(z), bool)], n_lags=6, block_len=100, conditioned=False)
    r = m.sample_residuals(5000, np.random.default_rng(1))
    pool = np.concatenate(m.segments)
    assert np.allclose(r.std(0), pool.std(0), rtol=0.25)
    tau_pool, tau_r = integrated_autocorr_time(pool).mean(), integrated_autocorr_time(r).mean()
    assert tau_r > 5 and abs(tau_r - tau_pool) / tau_pool < 0.35


def test_blocks_do_not_cross_masked_gaps():
    z, p = _real_like(T=1000)
    mask = np.ones(1000, bool)
    mask[480:520] = False
    m = fit_residual_model([z], [p], [mask], n_lags=4, block_len=200, conditioned=False)
    assert [len(s) for s in m.segments] == [480, 480]


def test_online_decoder_matches_batch_simulation_and_calibrates():
    z, p = _real_like()
    m = fit_residual_model([z], [p], [np.ones(len(z), bool)], n_lags=6, block_len=100)
    dec = OnlineResidualDecoder(m, horizon=len(z))
    online = dec.run(z[:500], np.random.default_rng(3))
    rng = np.random.default_rng(3)
    batch = m.signal(z[:500]) + m.scale(z[:500]) * m.sample_residuals(500, rng)
    assert np.allclose(online, batch)
    cal = calibrate_residual(m, [z[:1500]], [z[1500:]], np.random.default_rng(0))
    assert 0.0 < cal.r2_test < 1.0 and cal.err.tau > 1
    assert cal.decoder.takes_intent
