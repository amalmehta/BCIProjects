import numpy as np

from robotic_arm.autonomy import Arbitration, ArmView, DecoderErrorModel, MaxEntInference, NoiseAwareInference
from robotic_arm.closed_loop import calibrate, run_episode
from robotic_arm.encoding import EncodingModel, OnlineEncoder, fit_poisson_glm
from robotic_arm.task import Scene, command_to_decoded, decoded_to_command, reach_policy


def _toy_encoder(C=60, kappa=1.5, seed=0):
    rng = np.random.default_rng(seed)
    return EncodingModel(b=np.log(rng.uniform(0.1, 0.5, C)), C=kappa * rng.standard_normal((C, 7)) / np.sqrt(7),
                         L=np.linalg.qr(rng.standard_normal((C, 3)))[0], rho=0.98, sigma=0.1)


def _smooth_vel(T, rng):
    v = np.zeros((T, 7))
    for t in range(1, T):
        v[t] = 0.98 * v[t - 1] + 0.2 * rng.standard_normal(7)
    return v


def test_poisson_glm_recovers_tuning():
    rng = np.random.default_rng(0)
    enc = _toy_encoder(C=10, kappa=1.0)
    v = _smooth_vel(20000, rng)
    enc.sigma = 0.0
    counts = enc.sample(v, rng)
    b, C = fit_poisson_glm(counts, v)
    assert np.corrcoef(C.ravel(), enc.C.ravel())[0, 1] > 0.95


def test_online_encoder_rates():
    rng = np.random.default_rng(0)
    enc = _toy_encoder()
    oe = OnlineEncoder(enc, rng)
    x = np.array([oe.step(np.zeros(7)) for _ in range(2000)])
    assert x.shape == (2000, 60)
    assert np.allclose(x.mean(0).mean(), np.exp(enc.b).mean(), rtol=0.3)


def test_command_mapping_roundtrip():
    u = np.array([0.05, -0.02, 0.01, 0.3, 0.5])
    assert np.allclose(decoded_to_command(command_to_decoded(u)), u)


def test_goal_inference_concentrates_and_can_switch():
    scene = Scene(np.array([[0.4, -0.2, 0.05], [0.4, 0.2, 0.05], [0.6, 0.0, 0.05]]), np.zeros(3))
    view = ArmView(np.array([0.45, 0.0, 0.3]), 0.0, 1.0)
    err = DecoderErrorModel(np.eye(7), np.eye(7) * 0.3, tau=5.0)

    def update(inf, u):
        if isinstance(inf, NoiseAwareInference):
            return inf.update(u, view, scene, z_hat=command_to_decoded(u))
        return inf.update(u, view, scene)

    for inf in (MaxEntInference(3, beta=10.0, hazard=0.01), NoiseAwareInference(3, err, hazard=0.01)):
        u0 = reach_policy(view.p, 0.0, 1.0, scene.positions[0], 0.0, 0.12)
        for _ in range(100):
            b = update(inf, u0)
        assert np.argmax(b) == 0 and b[0] > 0.8
        u1 = reach_policy(view.p, 0.0, 1.0, scene.positions[1], 0.0, 0.12)
        for _ in range(300):
            b = update(inf, u1)
        assert np.argmax(b) == 1


def test_arbitration_weights():
    scene = Scene(np.array([[0.4, -0.2, 0.05], [0.6, 0.2, 0.05]]), np.zeros(2))
    view = ArmView(np.array([0.45, 0.0, 0.3]), 0.0, 1.0)
    u = np.array([0.05, 0, 0, 0, 0])
    cmd, a, share = Arbitration("direct")(u, np.array([0.5, 0.5]), view, scene)
    assert a == 0 and share == 1 and np.allclose(cmd, u)
    _, a_far, _ = Arbitration("distance")(u, np.array([0.5, 0.5]), view, scene)
    near = ArmView(scene.positions[0] + np.array([0, 0, 0.01]), 0.0, 1.0)
    _, a_near, _ = Arbitration("distance")(u, np.array([0.5, 0.5]), near, scene)
    assert a_far < 0.05 and a_near > 0.9  # Downey et al. 2016 constants
    _, a_unsure, _ = Arbitration("confidence")(u, np.array([0.5, 0.5]), view, scene)
    _, a_sure, _ = Arbitration("confidence")(u, np.array([0.99, 0.01]), view, scene)
    assert a_unsure < a_sure


def test_closed_loop_episode_with_good_decoder_succeeds():
    rng = np.random.default_rng(0)
    enc = _toy_encoder(C=80, kappa=3.0)
    cal = calibrate(enc, [_smooth_vel(6000, rng)], [_smooth_vel(2000, rng)], rng)
    assert cal.r2_test > 0.5
    scene = Scene.random(np.random.default_rng(3), 3)
    res = run_episode(scene, 0, cal, enc, "dasa", np.random.default_rng(4), max_steps=900, record=True)
    assert res.success, res
    assert len(res.trace["tcp"]) == len(res.trace["alpha"]) and len(res.trace["q"][0]) == 7
