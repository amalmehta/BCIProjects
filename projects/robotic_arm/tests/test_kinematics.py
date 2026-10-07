import numpy as np

from robotic_arm.arm import Q_HOME, TOOL_D, Arm, SafetyConfig, fk, fk_frames, jacobian



def test_zero_configuration_points_straight_up():
    """All joints at zero: links stack vertically (sum of |d|) plus the tool offset;
    the two small lateral offsets (d2, d4) shift the TCP sideways."""
    p = fk(np.zeros(7))[:3, 3]
    vertical = 0.2848 + 0.4208 + 0.3143 + 0.1674 + TOOL_D
    assert np.isclose(p[2], vertical, atol=1e-9)
    assert np.isclose(abs(p[1]), 0.0118 + 0.0128, atol=1e-9)
    assert np.isclose(p[0], 0.0, atol=1e-9)


def test_home_test_vector():
    """The test vector exported for the website's JS forward kinematics."""
    T = fk(Q_HOME)
    assert np.allclose(T[:3, 3], [0.45, 0.0, 0.30], atol=2e-3)
    assert np.allclose(T[:3, 2], [0, 0, -1], atol=1e-2)  # tool points down


def test_jacobian_matches_finite_differences():
    rng = np.random.default_rng(0)
    q = Q_HOME + rng.normal(0, 0.2, 7)
    J = jacobian(fk_frames(q))
    eps = 1e-6
    for i in range(7):
        dq = np.zeros(7)
        dq[i] = eps
        dp = (fk(q + dq)[:3, 3] - fk(q - dq)[:3, 3]) / (2 * eps)
        assert np.allclose(J[:3, i], dp, atol=1e-6)


def test_arm_reaches_target_and_respects_table():
    arm = Arm()
    target = np.array([0.55, 0.15, -0.10])  # below the table: must be clipped
    for _ in range(600):
        arm.step(2.0 * (target - arm.tcp), 0.0, 0.0)
    lo = SafetyConfig().workspace_lo
    assert arm.tcp[2] >= lo[2] - 1e-6
    assert np.allclose(arm.tcp[:2], target[:2], atol=0.01)
    assert arm.events["table_clip"] > 0
    assert all(f[2, 3] >= SafetyConfig().link_z_min - 1e-9 for f in arm.frames[2:-1])


def test_speed_cap():
    arm = Arm()
    p0 = arm.tcp
    arm.step(np.array([5.0, 0, 0]), 0.0, 0.0)
    assert np.linalg.norm(arm.tcp - p0) <= SafetyConfig().v_max * arm.dt * 1.05
