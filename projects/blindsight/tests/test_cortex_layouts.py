import numpy as np

from blindsight.cortex import WedgeDipole, hemifield_to_cartesian, horton_hoyt_magnification
from blindsight.layouts import make_layout


def test_round_trip_visual_field_cortex():
    m = WedgeDipole()
    ecc = np.array([0.3, 1.0, 2.5, 6.0, 15.0])
    theta = np.array([-1.2, -0.3, 0.0, 0.7, 1.4])
    x, y = m.to_cortex(ecc, theta)
    e2, t2 = m.to_visual_field(x, y)
    np.testing.assert_allclose(e2, ecc, rtol=1e-9)
    np.testing.assert_allclose(t2, theta, atol=1e-9)


def test_fovea_maps_to_origin_and_horizontal_meridian_to_x_axis():
    m = WedgeDipole()
    x, y = m.to_cortex(np.array([1e-9, 3.0]), np.array([0.0, 0.0]))
    assert abs(x[0]) < 1e-6 and abs(y[0]) < 1e-9 and abs(y[1]) < 1e-12 and x[1] > 0


def test_magnification_matches_horton_hoyt_and_finite_difference():
    m = WedgeDipole()
    r = np.linspace(0.5, 10, 20)
    rel = np.abs(m.magnification(r) - horton_hoyt_magnification(r)) / horton_hoyt_magnification(r)
    assert rel.max() < 0.09  # dipole term only matters as r approaches b = 120 deg
    h = 1e-5
    fd = (m.to_cortex(r + h, 0 * r)[0] - m.to_cortex(r - h, 0 * r)[0]) / (2 * h)
    np.testing.assert_allclose(fd, m.magnification(r), rtol=1e-5)


def test_hemifield_sign_convention():
    x, _ = hemifield_to_cartesian(np.array([2.0, 2.0]), np.array([0.0, 0.0]), np.array([0, 1]))
    assert x[0] == 2.0 and x[1] == -2.0


def test_layouts_counts_and_extent():
    m = WedgeDipole()
    for name, n in [("utah_96", 96), ("utah_1024", 1024), ("threads_256", 256)]:
        lay = make_layout(name, 0.2, 6.0, 0)
        assert lay.n == n
        vf = lay.visual_field(m)
        assert vf["ecc"].min() >= 0.2 - 1e-6 and vf["ecc"].max() <= 6.0 + 1e-6
    utah = make_layout("utah_96", 0.2, 6.0, 0)
    d = np.hypot(np.diff(np.sort(np.unique(utah.cortex_x_mm.round(6)))), 0)
    np.testing.assert_allclose(d, 0.4, atol=1e-6)
    threads = make_layout("threads_256", 0.2, 6.0, 0)
    assert set(np.unique(threads.hemisphere)) == {0, 1}
