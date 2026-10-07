"""Tabletop reach-and-grasp task, simulated user intent and the shared command space.

Command space (what both the decoded user and the autonomy produce), physical units:
``u = [vx, vy, vz (m/s), yaw_rate (rad/s), grip_rate (1/s, + closes)]``.

Mapping between the decoded 7-D FALCON H1 space (z-scored ``tx, ty, tz, rx, g1, g2, g3``)
and the command space is fixed: translation dims scale by ``V_SCALE``, ``rx`` by
``YAW_SCALE`` and the grip rate is the mean of the three grasp dims times ``GRIP_SCALE``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

V_SCALE = 0.08  # m/s per z-scored unit of decoded translation velocity
YAW_SCALE = 0.6  # rad/s per z-scored unit of rx
GRIP_SCALE = 0.8  # aperture/s per z-scored unit of grasp
TABLE_LO = np.array([0.33, -0.30])
TABLE_HI = np.array([0.67, 0.30])
GRASP_Z = 0.05  # grasp height of every object (m above table)
GRASP_RADIUS = 0.03  # TCP must be within this distance of the grasp point (m)
GRASP_YAW_TOL = np.deg2rad(20.0)
CLOSED = 0.2  # aperture below which the gripper counts as closed
OBJECT_COLORS = ["#d1495b", "#edae49", "#00798c", "#66a182", "#8d6cab"]
OBJECT_NAMES = ["red block", "yellow block", "teal block", "green block", "purple block"]


def wrap_half_pi(a: np.ndarray | float) -> np.ndarray | float:
    """Wrap an angle to [-pi/2, pi/2): a parallel gripper is symmetric under 180 deg."""
    return (np.asarray(a) + np.pi / 2) % np.pi - np.pi / 2


def decoded_to_command(z: np.ndarray) -> np.ndarray:
    """Map a z-scored 7-D decoded velocity to the 5-D command space."""
    return np.array([V_SCALE * z[0], V_SCALE * z[1], V_SCALE * z[2], YAW_SCALE * z[3],
                     GRIP_SCALE * float(np.mean(z[4:7]))])


def command_to_decoded(u: np.ndarray) -> np.ndarray:
    """Inverse map (used to express the user's intent in the decoder's z-scored space)."""
    g = u[4] / GRIP_SCALE
    return np.array([u[0] / V_SCALE, u[1] / V_SCALE, u[2] / V_SCALE, u[3] / YAW_SCALE, g, g, g])


@dataclass
class Scene:
    """Objects on the table: grasp points (K, 3) and grasp yaw angles (K,)."""

    positions: np.ndarray
    yaws: np.ndarray
    sizes: np.ndarray = field(default_factory=lambda: np.zeros(0))

    @property
    def n(self) -> int:
        return len(self.positions)

    @classmethod
    def random(cls, rng: np.random.Generator, n_objects: int = 3, min_sep: float = 0.12) -> "Scene":
        pts: list[np.ndarray] = []
        while len(pts) < n_objects:
            xy = rng.uniform(TABLE_LO, TABLE_HI)
            if all(np.linalg.norm(xy - p[:2]) >= min_sep for p in pts):
                pts.append(np.array([xy[0], xy[1], GRASP_Z]))
        yaws = rng.uniform(-np.pi / 2, np.pi / 2, n_objects)
        sizes = np.tile([0.05, 0.05, 0.10], (n_objects, 1))
        return cls(np.array(pts), yaws, sizes)


def reach_policy(p: np.ndarray, yaw: float, aperture: float, goal_pos: np.ndarray, goal_yaw: float,
                 v_max: float, k_p: float = 2.5, k_yaw: float = 2.0, yaw_rate_max: float = 1.0) -> np.ndarray:
    """Goal-directed command used both by the simulated user and by the autonomy.

    The TCP heads for a point above the object whose height shrinks with horizontal
    error (a top-down approach), aligns yaw, opens the gripper while travelling and
    closes it once the TCP is at the grasp point with the right yaw.
    """
    xy_err = np.linalg.norm(goal_pos[:2] - p[:2])
    yaw_err = float(wrap_half_pi(goal_yaw - yaw))
    target = goal_pos.copy()
    target[2] += min(0.12, 1.5 * xy_err) + (0.04 if abs(yaw_err) > GRASP_YAW_TOL else 0.0)
    d = target - p
    dist = np.linalg.norm(d)
    speed = min(v_max, k_p * dist)
    v = d / dist * speed if dist > 1e-9 else np.zeros(3)
    yaw_rate = float(np.clip(k_yaw * yaw_err, -yaw_rate_max, yaw_rate_max))
    at_goal = np.linalg.norm(goal_pos - p) < 0.8 * GRASP_RADIUS and abs(yaw_err) < 0.7 * GRASP_YAW_TOL
    grip = 1.0 if at_goal else (-1.0 if aperture < 1.0 else 0.0)
    return np.array([v[0], v[1], v[2], yaw_rate, grip])


@dataclass
class UserModel:
    """Simulated BCI user: delayed feedback controller with motor noise and optional
    change of mind. Produces the *intended* command; the neural pipeline corrupts it."""

    goal: int
    v_max: float = 0.12
    delay_steps: int = 10  # 200 ms visuomotor delay
    noise: float = 0.15  # intent noise, in z-scored decoder units
    switch_goal: int | None = None
    switch_frac: float = 0.7  # switch late, once assistance is committed to the first goal
    switched: bool = False
    switch_step: int | None = None

    def maybe_switch(self, step: int, p0: np.ndarray, p: np.ndarray, scene: Scene) -> None:
        if self.switch_goal is None or self.switched:
            return
        total = np.linalg.norm(scene.positions[self.goal] - p0)
        if np.linalg.norm(p - p0) >= self.switch_frac * total:
            self.goal, self.switched, self.switch_step = self.switch_goal, True, step

    def intent(self, p: np.ndarray, yaw: float, aperture: float, scene: Scene,
               rng: np.random.Generator) -> np.ndarray:
        u = reach_policy(p, yaw, aperture, scene.positions[self.goal], scene.yaws[self.goal], self.v_max)
        z = command_to_decoded(u) + self.noise * rng.standard_normal(7)
        return np.clip(z, -3.0, 3.0)
