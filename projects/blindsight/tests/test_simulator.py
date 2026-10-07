import math

import numpy as np
import torch

from blindsight.simulator import PhospheneSimulator, SimParams, TemporalSimulator


def test_zero_and_subthreshold_give_black(setup):
    pt = setup.nominal()
    assert setup.sim(torch.zeros(2, setup.n), pt).abs().max() == 0
    sub = float(setup.params.threshold_current_ua(setup.params.threshold_mean)) - 1.0
    assert setup.sim(torch.full((1, setup.n), sub), pt).max() == 0


def test_threshold_current_inverts_activation(setup):
    p = setup.params
    i_thr = float(p.threshold_current_ua(p.threshold_mean))
    act = setup.sim.activation(torch.tensor([i_thr]))
    assert math.isclose(float(act), p.threshold_mean, rel_tol=1e-5)
    # hand-computed: duty d = 170e-6 * 300, kd = 13.955 d, gain = (exp(-kd T) - exp(-9 T)) / (9 - kd)
    d = 170e-6 * 300
    kd = 13.95528162 * d
    gain = (math.exp(-kd * 0.166) - math.exp(-9.0 * 0.166)) / (9.0 - kd)
    assert math.isclose(i_thr, 23.9 + p.threshold_mean / (170e-6 * 300 * gain) * 1e6, rel_tol=1e-9)


def test_brightness_monotone_and_bounded(setup):
    pt = setup.nominal()
    amps = torch.linspace(0, 100, 21)
    b = setup.sim.brightness(setup.sim.activation(amps), pt.threshold[:1].expand(21))
    assert torch.all(b[1:] >= b[:-1]) and b.max() <= 1 and b.min() >= 0


def test_sigma_grows_with_eccentricity_and_current(setup):
    s = setup.sim.sigma_deg(torch.tensor([80.0, 80.0, 40.0]), torch.tensor([1.0, 5.0, 5.0]))
    assert s[1] > s[0] and s[1] > s[2]


def test_closed_form_activation_matches_euler_integration():
    p = SimParams()
    sim = PhospheneSimulator(p)
    ts = TemporalSimulator(p, 1, dt_s=p.train_s / 50, substeps=20)
    amp = torch.tensor([70.0])
    for _ in range(50):
        a = ts.step(amp)
    closed = sim.activation(amp)
    # Same equations (incl. the habituation trace); residual = Euler error + neglected trace decay.
    assert abs(float(a) - float(closed)) / float(closed) < 0.01


def test_fast_renderer_close_to_exact(setup):
    pt = setup.patient(5)
    amp = torch.rand(4, setup.n, generator=torch.Generator().manual_seed(0)) * 100
    exact = setup.sim(amp, pt)
    fast = setup.sim(amp, pt, fast=True)
    assert (exact - fast).abs().max() < 0.03


def test_gradient_flows_through_soft_gate(setup):
    pt = setup.patient(5)
    amp = torch.full((1, setup.n), 60.0, requires_grad=True)
    setup.sim(amp, pt, hard=False, fast=True).sum().backward()
    assert amp.grad.abs().sum() > 0


def test_dead_electrodes_render_nothing(setup):
    pt = setup.patient(5)
    pt.alive = torch.zeros(setup.n)
    assert setup.sim(torch.full((1, setup.n), 100.0), pt).max() == 0


def test_single_phosphene_peak_location(setup):
    p = setup.params
    pt = setup.nominal()
    amp = torch.zeros(1, setup.n)
    k = int(torch.argmax(pt.ecc_deg))
    amp[0, k] = 100.0
    img = setup.sim(amp, pt)[0, 0]
    r, c = np.unravel_index(int(img.argmax()), img.shape)
    R = p.resolution
    x = (c + 0.5) / R * p.fov_deg - p.fov_deg / 2
    y = p.fov_deg / 2 - (r + 0.5) / R * p.fov_deg
    assert math.hypot(x - float(pt.x_deg[k]), y - float(pt.y_deg[k])) <= p.fov_deg / R

