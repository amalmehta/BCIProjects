"""7-DoF arm kinematics (Kinova Gen3 classical DH table) with a safety-filtered
resolved-rate controller.

Forward kinematics uses the classical Denavit-Hartenberg convention
``T_i = Rz(theta_i) Tz(d_i) Tx(a_i) Rx(alpha_i)`` with the table from the Kinova Gen3
User Guide (7-DoF, all ``a_i = 0``). A gripper tool-centre-point offset of
``TOOL_D`` metres along the flange z-axis is appended (our assumption, roughly a
Robotiq 2F-85 gripper). The same table is exported to the website so the JS renderer can
reproduce FK exactly.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

PI = np.pi
# rows: (alpha, a, d, theta_offset); row 0 is the fixed base transform, rows 1-7 joints,
# row 8 is the fixed flange (interface) transform folded into joint 7 per Kinova's table.
DH_BASE = (PI, 0.0, 0.0, 0.0)
DH_JOINTS = [
    (PI / 2, 0.0, -0.2848, 0.0),
    (PI / 2, 0.0, -0.0118, PI),
    (PI / 2, 0.0, -0.4208, PI),
    (PI / 2, 0.0, -0.0128, PI),
    (PI / 2, 0.0, -0.3143, PI),
    (PI / 2, 0.0, 0.0, PI),
    (PI, 0.0, -0.1674, PI),
]
TOOL_D = 0.12  # gripper TCP offset along flange z (m)
# Kinova Gen3 joint limits (rad); joints 1,3,5,7 are continuous (we cap at +-2pi).
Q_LIMITS = np.deg2rad(np.array([
    [-360, 360], [-128.9, 128.9], [-360, 360], [-147.8, 147.8],
    [-360, 360], [-120.3, 120.3], [-360, 360]], dtype=np.float64))
QDOT_MAX = np.deg2rad(np.array([50, 50, 50, 50, 70, 70, 70], dtype=np.float64))
# Start posture (deg): TCP at about (0.45, 0, 0.30) m with the tool pointing straight down.
Q_HOME = np.deg2rad(np.array([-2.0, 20.7, -176.9, -86.1, 178.9, 73.2, -178.8]))


def dh(alpha: float, a: float, d: float, theta: float) -> np.ndarray:
    """Classical DH homogeneous transform Rz(theta) Tz(d) Tx(a) Rx(alpha)."""
    ct, st, ca, sa = np.cos(theta), np.sin(theta), np.cos(alpha), np.sin(alpha)
    return np.array([[ct, -st * ca, st * sa, a * ct],
                     [st, ct * ca, -ct * sa, a * st],
                     [0.0, sa, ca, d],
                     [0.0, 0.0, 0.0, 1.0]])


_ALPHA = np.array([r[0] for r in DH_JOINTS])
_A = np.array([r[1] for r in DH_JOINTS])
_D = np.array([r[2] for r in DH_JOINTS])
_OFF = np.array([r[3] for r in DH_JOINTS])
_CA, _SA = np.cos(_ALPHA), np.sin(_ALPHA)
_T_BASE = dh(*DH_BASE)
_T_TOOL = np.eye(4)
_T_TOOL[2, 3] = TOOL_D


def fk_frames(q: np.ndarray) -> list[np.ndarray]:
    """World transforms of the base, each joint frame (1..7) and the TCP (9 frames)."""
    th = np.asarray(q, dtype=np.float64) + _OFF
    ct, st = np.cos(th), np.sin(th)
    Ts = np.zeros((7, 4, 4))
    Ts[:, 0, 0], Ts[:, 0, 1], Ts[:, 0, 2], Ts[:, 0, 3] = ct, -st * _CA, st * _SA, _A * ct
    Ts[:, 1, 0], Ts[:, 1, 1], Ts[:, 1, 2], Ts[:, 1, 3] = st, ct * _CA, -ct * _SA, _A * st
    Ts[:, 2, 1], Ts[:, 2, 2], Ts[:, 2, 3] = _SA, _CA, _D
    Ts[:, 3, 3] = 1.0
    T = _T_BASE
    frames = [T]
    for i in range(7):
        T = T @ Ts[i]
        frames.append(T)
    frames.append(T @ _T_TOOL)
    return frames


def fk(q: np.ndarray) -> np.ndarray:
    """TCP homogeneous transform."""
    return fk_frames(q)[-1]


def jacobian(frames: list[np.ndarray]) -> np.ndarray:
    """Geometric 6x7 Jacobian of the TCP (linear; angular) for revolute joints.

    Joint i rotates about the z-axis of frame i-1 (classical DH).
    """
    p = frames[-1][:3, 3]
    z = np.stack([frames[i][:3, 2] for i in range(7)])
    o = np.stack([frames[i][:3, 3] for i in range(7)])
    return np.vstack([_cross(z, p - o).T, z.T])


def tool_down_rotation(yaw: float) -> np.ndarray:
    """Rotation with the tool z-axis pointing down and the gripper rotated by ``yaw``."""
    c, s = np.cos(yaw), np.sin(yaw)
    rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])
    rx = np.array([[1.0, 0, 0], [0, -1, 0], [0, 0, -1]])
    return rz @ rx


def _cross(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Row-wise 3-D cross product (much faster than np.cross for tiny arrays)."""
    return np.stack([a[..., 1] * b[..., 2] - a[..., 2] * b[..., 1],
                     a[..., 2] * b[..., 0] - a[..., 0] * b[..., 2],
                     a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]], axis=-1)


def orientation_error(R: np.ndarray, Rd: np.ndarray) -> np.ndarray:
    """Small-angle orientation error (axis * angle) taking R to Rd, in the world frame."""
    return 0.5 * _cross(R.T, Rd.T).sum(0)


def tool_yaw(R: np.ndarray) -> float:
    """Yaw of the gripper's x-axis in the horizontal plane."""
    return float(np.arctan2(R[1, 0], R[0, 0]))


@dataclass
class SafetyConfig:
    v_max: float = 0.20  # m/s translational speed cap
    yaw_rate_max: float = 1.2  # rad/s
    workspace_lo: tuple = (0.20, -0.45, 0.02)
    workspace_hi: tuple = (0.80, 0.45, 0.55)
    link_z_min: float = 0.05  # joint origins (not the TCP) must stay this far above the table
    joint_margin: float = np.deg2rad(5.0)


@dataclass
class ArmState:
    q: np.ndarray = field(default_factory=lambda: Q_HOME.copy())
    yaw_des: float = 0.0
    gripper: float = 1.0  # 1 open, 0 closed


class Arm:
    """Resolved-rate arm controller with a safety filter.

    Commands are task-space: translational TCP velocity (m/s), yaw rate (rad/s) and
    gripper velocity (1/s). The controller keeps the tool pointing down, tracks the
    commanded yaw, and uses the null space to stay near the home posture.
    """

    def __init__(self, safety: SafetyConfig | None = None, dt: float = 0.02,
                 damping: float = 0.03, k_orient: float = 4.0, k_null: float = 0.5):
        self.safety = safety or SafetyConfig()
        self.dt, self.damping, self.k_orient, self.k_null = dt, damping, k_orient, k_null
        self.state = ArmState()
        self.frames = fk_frames(self.state.q)
        self.state.yaw_des = tool_yaw(self.frames[-1][:3, :3])
        self.events = {"speed_clip": 0, "workspace_clip": 0, "table_clip": 0,
                       "joint_limit_clip": 0, "link_collision_block": 0}

    def reset(self, q: np.ndarray | None = None) -> None:
        self.state = ArmState(q=(Q_HOME if q is None else q).copy())
        self.frames = fk_frames(self.state.q)
        self.state.yaw_des = tool_yaw(self.frames[-1][:3, :3])
        self.events = {k: 0 for k in self.events}

    @property
    def tcp(self) -> np.ndarray:
        return self.frames[-1][:3, 3].copy()

    def safety_filter(self, p: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Task-space safety at the velocity level: workspace box and table plane first
        (so a blocked component does not eat the speed budget), then the speed cap."""
        s = self.safety
        v = np.asarray(v, dtype=np.float64).copy()
        nxt = p + v * self.dt
        lo, hi = np.array(s.workspace_lo), np.array(s.workspace_hi)
        clipped = np.clip(nxt, lo, hi)
        if not np.allclose(clipped, nxt):
            if nxt[2] < lo[2]:
                self.events["table_clip"] += 1
            else:
                self.events["workspace_clip"] += 1
            v = (clipped - p) / self.dt
        speed = np.linalg.norm(v)
        if speed > s.v_max:
            v *= s.v_max / speed
            self.events["speed_clip"] += 1
        return v

    def step(self, v_cmd: np.ndarray, yaw_rate: float, grip_rate: float) -> dict:
        """Advance one control tick; returns the TCP position and applied commands."""
        st, s = self.state, self.safety
        frames = self.frames
        T = frames[-1]
        p = T[:3, 3]
        v = self.safety_filter(p, v_cmd)
        yaw_rate = float(np.clip(yaw_rate, -s.yaw_rate_max, s.yaw_rate_max))
        st.yaw_des = float(np.clip(st.yaw_des + yaw_rate * self.dt, -PI, PI))
        e_o = orientation_error(T[:3, :3], tool_down_rotation(st.yaw_des))
        twist = np.concatenate([v, self.k_orient * e_o + np.array([0, 0, yaw_rate])])
        J = jacobian(frames)
        JJt = J @ J.T + self.damping ** 2 * np.eye(6)
        J_pinv = J.T @ np.linalg.inv(JJt)
        null = (np.eye(7) - J_pinv @ J) @ (self.k_null * (Q_HOME - st.q))
        null[6] = 0.0  # leave the wrist roll to the yaw task
        qdot = J_pinv @ twist + null
        # joint speed cap (uniform scaling keeps the task-space direction)
        ratio = np.max(np.abs(qdot) / QDOT_MAX)
        if ratio > 1:
            qdot /= ratio
        # joint position limits
        q_next = st.q + qdot * self.dt
        lo, hi = Q_LIMITS[:, 0] + s.joint_margin, Q_LIMITS[:, 1] - s.joint_margin
        viol = (q_next < lo) | (q_next > hi)
        if viol.any():
            qdot[viol] = 0.0
            self.events["joint_limit_clip"] += 1
            q_next = st.q + qdot * self.dt
        # link-table collision: block motions that drive a joint origin (wrist/elbow/flange;
        # the TCP itself is bounded by the workspace box) further below the clearance
        next_frames = fk_frames(q_next)
        z_now = np.array([f[2, 3] for f in frames[2:-1]])
        z_links = np.array([f[2, 3] for f in next_frames[2:-1]])
        if ((z_links < s.link_z_min) & (z_links < z_now)).any():
            self.events["link_collision_block"] += 1
            q_next, next_frames = st.q, frames
            qdot = np.zeros(7)
        st.q = q_next
        self.frames = next_frames
        st.gripper = float(np.clip(st.gripper + grip_rate * self.dt, 0.0, 1.0))
        return {"tcp": self.tcp, "v_applied": v, "qdot": qdot}
