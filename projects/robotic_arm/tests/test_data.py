import numpy as np

from robotic_arm.data import Normalizer, bin_spike_times, causal_smooth, lag_features


def test_binning_matches_histogram_convention():
    # bins end at 0.02, 0.04, 0.06; bin i covers (end-0.02, end]
    spikes = [np.array([0.001, 0.019, 0.021, 0.059]), np.array([])]
    out = bin_spike_times(spikes, np.array([0.02, 0.04, 0.06]))
    assert out.shape == (3, 2)
    assert out[:, 0].tolist() == [2, 1, 1]
    assert out[:, 1].sum() == 0


def test_lag_features_shapes_and_causality():
    x = np.arange(10, dtype=float)[:, None] * np.ones((1, 2))
    f = lag_features(x, 3, stride=2)
    assert f.shape == (10, 6)
    assert np.allclose(f[5], [5, 5, 3, 3, 1, 1])
    assert np.allclose(f[1, 2:], 0)  # no data from before t=0


def test_causal_smooth_has_no_future_leakage():
    x = np.zeros((50, 1))
    x[30] = 1.0
    y = causal_smooth(x, 3.0)
    assert np.allclose(y[:30], 0) and y[30] > 0


def test_normalizer_roundtrip():
    rng = np.random.default_rng(1)
    x = rng.normal(3, 2, (100, 4))
    n = Normalizer.fit(x)
    assert np.allclose(n(x).mean(0), 0, atol=1e-9)
    assert np.allclose(n.inverse(n(x)), x)


def test_minival_is_stripped_from_calibration():
    from datetime import date

    from robotic_arm.data import Block, _strip_minival

    rng = np.random.default_rng(0)
    T = 50
    full = Block("held_in_calib", date(2000, 1, 1), "000000", rng.poisson(1, (T, 3)).astype(float),
                 rng.normal(size=(T, 7)), np.zeros((T, 7)), np.ones(T, bool), np.zeros(T))
    mini = full.slice(0, 20)
    (stripped,) = _strip_minival([full], [mini])
    assert stripped.n_bins == 30
    assert np.array_equal(stripped.spikes, full.spikes[20:])
