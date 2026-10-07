import numpy as np
from sklearn.metrics import r2_score

from robotic_arm.metrics import path_efficiency, r2_per_dim, r2_variance_weighted


def test_r2_hand_computed():
    y = np.array([[1.0], [2.0], [3.0]])
    yhat = np.array([[1.0], [2.0], [4.0]])
    # SSres = 1, SStot = 2 -> R2 = 0.5
    assert np.isclose(r2_per_dim(y, yhat)[0], 0.5)


def test_variance_weighted_matches_sklearn():
    rng = np.random.default_rng(0)
    y = rng.standard_normal((200, 3)) * np.array([1.0, 5.0, 0.1])
    yhat = y + rng.standard_normal((200, 3))
    assert np.isclose(r2_variance_weighted(y, yhat), r2_score(y, yhat, multioutput="variance_weighted"))
    assert np.allclose(r2_per_dim(y, yhat), r2_score(y, yhat, multioutput="raw_values"))


def test_path_efficiency():
    straight = np.linspace([0, 0, 0], [1, 0, 0], 11)
    assert np.isclose(path_efficiency(straight), 1.0)
    detour = np.array([[0, 0, 0], [0.5, 0.5, 0], [1, 0, 0]], float)
    assert np.isclose(path_efficiency(detour), 1 / np.sqrt(2))
