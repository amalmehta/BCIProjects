"""Shared autonomy: goal inference over candidate objects + arbitration policies.

All arbitration policies see the same inputs every tick: the decoded user command
``u_hat`` (command space, see :mod:`task`), the arm state, the scene, and their own goal
posterior. They return the command sent to the safety-filtered arm plus the autonomy
weight and the user's share of the command.

Implemented methods:

* ``direct``      decoder only (no assistance).
* ``fixed``       linear policy blending with constant alpha toward the MAP goal.
* ``confidence``  Dragan & Srinivasa (2013) policy blending, alpha ramps with the MAP
                  goal's posterior probability (MaxEnt goal prediction).
* ``distance``    Downey et al. (2016)-style blending, autonomy weight is a sigmoid of
                  the distance to the nearest object's grasp pose (no intent inference).
* ``hindsight``   Javdani et al. (2015/2018)-style assistance: the robot adds the action
                  that minimises the posterior-expected cost-to-go over all goals,
                  instead of committing to one goal (QMDP / hindsight optimisation).
* ``dasa``        proposed Decoder-Aware Shared Autonomy: goal likelihood under the
                  decoder's calibrated error model (gain ``G``, covariance ``Sigma``,
                  autocorrelation-corrected), a goal-switching (HMM) prior, and a
                  model-consistency gate that hands control back when the decoded
                  commands stop being explained by the assisted goal (the user is
                  pushing against the assistance).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .task import Scene, command_to_decoded, reach_policy

COMMAND_WEIGHTS = np.array([1 / 0.1, 1 / 0.1, 1 / 0.1, 1 / 0.8, 1 / 1.0])  # normalise dims


@dataclass
class ArmView:
    """What the autonomy knows about the robot (no sensing delay)."""

    p: np.ndarray
    yaw: float
    aperture: float


# ----------------------------------------------------------------------------- inference
class MaxEntInference:
    """MaxEnt-IOC goal prediction (Ziebart; Dragan & Srinivasa 2013; Javdani 2015).

    Per tick, the log-likelihood of command ``u`` for goal ``g`` with a distance cost and
    look-ahead ``h`` is ``-beta * (|u| h + d(p + u h, g) - d(p, g))``. Evidence accumulates
    with an optional goal-switching probability ``hazard`` per tick.
    """

    def __init__(self, n_goals: int, beta: float = 20.0, horizon: float = 0.5, hazard: float = 0.0):
        self.beta, self.h, self.hazard = beta, horizon, hazard
        self.logb = np.full(n_goals, -np.log(n_goals))

    @property
    def posterior(self) -> np.ndarray:
        return np.exp(self.logb)

    def update(self, u_hat: np.ndarray, view: ArmView, scene: Scene) -> np.ndarray:
        v = u_hat[:3]
        d0 = np.linalg.norm(scene.positions - view.p, axis=1)
        d1 = np.linalg.norm(scene.positions - (view.p + v * self.h), axis=1)
        ll = -self.beta * (np.linalg.norm(v) * self.h + d1 - d0)
        self._accumulate(ll)
        return self.posterior

    def _accumulate(self, ll: np.ndarray) -> None:
        b = np.exp(self.logb)
        if self.hazard > 0:
            b = (1 - self.hazard) * b + self.hazard / len(b)
        logb = np.log(b + 1e-300) + ll
        self.logb = logb - np.logaddexp.reduce(logb)


@dataclass
class DecoderErrorModel:
    """Calibrated relation between intended (z) and decoded (z_hat) velocity:
    ``z_hat = G z + e``, ``e ~ N(0, Sigma)`` with integrated autocorrelation time ``tau``."""

    G: np.ndarray  # (7, 7)
    Sigma: np.ndarray  # (7, 7)
    tau: float  # ticks

    @classmethod
    def fit(cls, z_true: np.ndarray, z_dec: np.ndarray, max_lag: int = 200) -> "DecoderErrorModel":
        G = np.linalg.lstsq(z_true, z_dec, rcond=None)[0].T
        e = z_dec - z_true @ G.T
        Sigma = np.cov(e.T) + 1e-3 * np.eye(e.shape[1])
        # integrated autocorrelation time, averaged over dims (Sokal's windowed estimate)
        taus = []
        for d in range(e.shape[1]):
            x = e[:, d] - e[:, d].mean()
            ac = np.correlate(x[:20000], x[:20000], "full")[len(x[:20000]) - 1:][:max_lag]
            ac = ac / ac[0]
            cut = np.argmax(ac < 0.05) if (ac < 0.05).any() else max_lag
            taus.append(1 + 2 * ac[1:cut].sum())
        return cls(G, Sigma, float(np.mean(taus)))


class NoiseAwareInference(MaxEntInference):
    """Goal likelihood under the decoder's own error model (proposed).

    ``log p(z_hat | g) = -0.5 (z_hat - G z*_g)^T Sigma^-1 (z_hat - G z*_g) / tau`` where
    ``z*_g`` is the user model's intended velocity for goal g. Dividing by the
    autocorrelation time ``tau`` counts correlated decoder errors once per effective
    sample rather than once per tick (otherwise the posterior is badly over-confident).
    """

    def __init__(self, n_goals: int, err: DecoderErrorModel, hazard: float = 0.002,
                 user_v_max: float = 0.12, temper: float = 1.0):
        super().__init__(n_goals, hazard=hazard)
        self.err = err
        self.Sinv = np.linalg.inv(err.Sigma)
        self.user_v_max = user_v_max
        self.temper = temper

    def update(self, u_hat: np.ndarray, view: ArmView, scene: Scene, z_hat: np.ndarray | None = None) -> np.ndarray:
        if z_hat is None:
            z_hat = command_to_decoded(u_hat)
        ll = np.empty(scene.n)
        for g in range(scene.n):
            ug = reach_policy(view.p, view.yaw, view.aperture, scene.positions[g], scene.yaws[g], self.user_v_max)
            r = z_hat - self.err.G @ command_to_decoded(ug)
            ll[g] = -0.5 * r @ self.Sinv @ r
        self._accumulate(ll / (self.err.tau * self.temper))
        return self.posterior


# ----------------------------------------------------------------------------- arbitration
@dataclass
class Arbitration:
    """Base class: stateless blending toward the posterior's MAP goal."""

    name: str = "direct"
    v_max: float = 0.12  # autonomy speed (m/s)
    alpha_const: float = 0.5
    conf_lo: float = 0.4
    conf_hi: float = 0.9
    alpha_max: float = 0.8
    dist_a: float = 11.6  # Downey et al. 2016 sigmoid constants
    dist_o: float = 6.9
    envelope: float = 0.25  # m, distance that maps to D = 1
    ho_lambda: float = 1.0  # hindsight: cost on the robot's added action
    ho_horizon: float = 0.5
    gate_tau: float = 0.5  # s, EMA time constant of the consistency statistic
    gate_c0: float = 2.0  # normalised Mahalanobis residual at which the gate is half open
    gate_slope: float = 0.3
    dt: float = 0.02
    err: DecoderErrorModel | None = None  # decoder error model (dasa only)
    user_v_max: float = 0.12  # user-model speed assumed by dasa
    use_gate: bool = True
    _stat: float = field(default=1.0, init=False)

    def reset(self) -> None:
        self._stat = 1.0

    def autonomous_action(self, view: ArmView, scene: Scene, g: int) -> np.ndarray:
        return reach_policy(view.p, view.yaw, view.aperture, scene.positions[g], scene.yaws[g], self.v_max)

    def __call__(self, u_hat: np.ndarray, b: np.ndarray, view: ArmView, scene: Scene,
                 z_hat: np.ndarray | None = None) -> tuple[np.ndarray, float, float]:
        """Return (command, autonomy weight alpha, user share of the command)."""
        if self.name == "direct":
            return u_hat, 0.0, 1.0
        if self.name == "hindsight":
            return self._hindsight(u_hat, b, view, scene)
        if self.name == "distance":
            d = np.linalg.norm(scene.positions - view.p, axis=1)
            g = int(np.argmin(d))
            D = min(d[g] / self.envelope, 1.0)
            alpha = 1.0 - 1.0 / (1.0 + np.exp(-self.dist_a * D + self.dist_o))
        else:
            g = int(np.argmax(b))
            if self.name == "fixed":
                alpha = self.alpha_const
            elif self.name == "confidence":
                alpha = self.alpha_max * float(np.clip((b[g] - self.conf_lo) / (self.conf_hi - self.conf_lo), 0, 1))
            elif self.name == "dasa":
                alpha = self._dasa_alpha(z_hat if z_hat is not None else command_to_decoded(u_hat),
                                         b, view, scene, g)
            else:
                raise ValueError(self.name)
        a = self.autonomous_action(view, scene, g)
        return self._blend(u_hat, a, alpha)

    def _blend(self, u_hat, a, alpha):
        u = (1 - alpha) * u_hat + alpha * a
        wu = np.linalg.norm(COMMAND_WEIGHTS * (1 - alpha) * u_hat)
        wa = np.linalg.norm(COMMAND_WEIGHTS * alpha * a)
        return u, float(alpha), float(wu / (wu + wa + 1e-12))

    def _dasa_alpha(self, z_hat, b, view, scene, g) -> float:
        """Confidence ramp (as in ``confidence``) times a model-consistency gate.

        The gate monitors ``r^T Sigma^-1 r / dim`` for the MAP goal's predicted decoder
        output (``r = z_hat - G z*_g``); it is ~1 when the user is steering toward g and
        grows when the decoded commands are inconsistent with g, e.g. when the user
        resists the assistance. This is innovation monitoring calibrated to the
        decoder's own noise, so no hand-tuned agreement threshold is needed per noise level.
        """
        ramp = float(np.clip((b[g] - self.conf_lo) / (self.conf_hi - self.conf_lo), 0, 1))
        if not self.use_gate or self.err is None:
            return self.alpha_max * ramp
        ug = reach_policy(view.p, view.yaw, view.aperture, scene.positions[g], scene.yaws[g], self.user_v_max)
        r = z_hat - self.err.G @ command_to_decoded(ug)
        m = float(r @ np.linalg.solve(self.err.Sigma, r)) / len(r)
        k = self.dt / self.gate_tau
        self._stat = (1 - k) * self._stat + k * m
        gate = 1.0 / (1.0 + np.exp((self._stat - self.gate_c0) / self.gate_slope))
        return self.alpha_max * ramp * gate

    def _hindsight(self, u_hat, b, view, scene):
        """Expected-cost-minimising added action (posterior-weighted, no MAP commitment)."""
        h = self.ho_horizon
        acts = np.array([self.autonomous_action(view, scene, g) for g in range(scene.n)])
        # translation: minimise sum_g b_g |p + (v_hat + a) h - t_g| + lambda |a|^2, where t_g
        # is where goal g's policy would take the TCP. Weiszfeld-style fixed point.
        targets = view.p + acts[:, :3] * h
        base = view.p + u_hat[:3] * h
        a = np.zeros(3)
        for _ in range(10):
            w = b / (np.linalg.norm(base + a * h - targets, axis=1) + 1e-3)
            a = h * (w[:, None] * (targets - base)).sum(0) / (h * h * w.sum() + 2 * self.ho_lambda)
        # yaw / gripper: posterior-weighted correction toward each goal's policy
        conf = float(b.max())
        rest = (b[:, None] * (acts[:, 3:] - u_hat[3:])).sum(0) * conf
        add = np.concatenate([a, rest])
        u = u_hat + add
        wu = np.linalg.norm(COMMAND_WEIGHTS * u_hat)
        wa = np.linalg.norm(COMMAND_WEIGHTS * add)
        share = float(wu / (wu + wa + 1e-12))
        return u, 1.0 - share, share
