"""Closed-loop simulation: simulated user -> encoding model -> decoder -> shared autonomy
-> safety-filtered 7-DoF arm, at 50 Hz (20 ms, the FALCON bin width).

This is a *simulated* closed loop. No human is in the loop; the user is a delayed
feedback-control model and the neural activity comes from an encoding model fitted to
FALCON H1 (see :mod:`encoding`).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from .arm import Arm, tool_yaw
from .autonomy import Arbitration, ArmView, DecoderErrorModel, MaxEntInference, NoiseAwareInference
from .data import Normalizer
from .decoders.linear import WienerFilter
from .encoding import EncodingModel, OnlineEncoder
from .metrics import path_efficiency, r2_variance_weighted
from .task import (CLOSED, GRASP_RADIUS, GRASP_YAW_TOL, Scene, UserModel, decoded_to_command,
                   wrap_half_pi)

DT = 0.02


@dataclass
class OnlineDecoder:
    """Streaming Wiener filter + exponential output smoothing + per-DoF gain correction."""

    wf: WienerFilter
    norm: Normalizer
    gain: np.ndarray
    smooth_tau: float = 4.0
    _buf: deque = field(default_factory=deque, init=False)
    _y: np.ndarray | None = field(default=None, init=False)

    def reset(self) -> None:
        self._buf = deque([np.zeros(self.wf.w.shape[0] // self.wf.n_lags)] * self.wf.n_lags, maxlen=self.wf.n_lags)
        self._y = np.zeros(self.wf.w.shape[1])

    def step(self, counts: np.ndarray) -> np.ndarray:
        self._buf.appendleft(self.norm(counts))
        raw = np.concatenate(self._buf) @ self.wf.w + self.wf.b
        a = np.exp(-1.0 / self.smooth_tau)
        self._y = a * self._y + (1 - a) * raw
        return self._y * self.gain

    def run(self, counts: np.ndarray) -> np.ndarray:
        self.reset()
        return np.array([self.step(c) for c in counts])


@dataclass
class Calibration:
    """Everything produced by the simulated open-loop calibration block."""

    decoder: OnlineDecoder
    err: DecoderErrorModel
    r2_test: float  # offline variance-weighted R² of the decoder (before speed gain)


def calibrate(enc: EncodingModel, calib_vel: list[np.ndarray], test_vel: list[np.ndarray],
              rng: np.random.Generator, n_lags: int = 10, alpha: float = 0.1,
              smooth_tau: float = 4.0, gain_exponent: float = 0.5, n_test_repeats: int = 4) -> Calibration:
    """Simulate an open-loop calibration block by replaying real H1 cued velocities
    through the encoding model, fit the online decoder, then measure its offline R² and
    error model on a separate simulated test block."""
    spikes = [enc.sample(v, rng) for v in calib_vel]
    norm = Normalizer.fit(np.concatenate(spikes))
    wf = WienerFilter(n_lags=n_lags, alpha=alpha).fit([norm(s) for s in spikes], calib_vel,
                                                      [np.ones(len(v), bool) for v in calib_vel])
    dec = OnlineDecoder(wf, norm, np.ones(7), smooth_tau)
    # per-DoF gain correction from the calibration block: undo the regression shrinkage
    pred = np.concatenate([dec.run(s) for s in spikes])
    truth = np.concatenate(calib_vel)
    slope = (pred * truth).sum(0) / (truth * truth).sum(0)
    test_vel = list(test_vel) * n_test_repeats  # several noise draws -> lower-variance R² estimate
    test_spikes = [enc.sample(v, rng) for v in test_vel]
    truth_t = np.concatenate(test_vel)
    r2 = r2_variance_weighted(truth_t, np.concatenate([dec.run(s) for s in test_spikes]))
    # partial speed-gain correction (clinicians tune output gain; full unbiasing
    # would amplify noise by 1/slope)
    dec.gain = np.clip(slope, 0.1, 1.5) ** (-gain_exponent)
    pred_t = np.concatenate([dec.run(s) for s in test_spikes])
    err = DecoderErrorModel.fit(truth_t, pred_t)
    return Calibration(dec, err, r2)


def calibrate_residual(model, calib_vel: list[np.ndarray], test_vel: list[np.ndarray], rng: np.random.Generator,
                       gain_exponent: float = 0.5, n_test_repeats: int = 4) -> Calibration:
    """Calibration for a decoder driven by real residuals (see :mod:`residuals`).

    Mirrors :func:`calibrate`: the same partial speed-gain correction is estimated on a
    simulated calibration block, and the offline R² and DASA's error model (G, Sigma, tau)
    are measured on a separate simulated test block, built from real cued velocities.
    """
    from .residuals import OnlineResidualDecoder

    dec = OnlineResidualDecoder(model)
    pred = np.concatenate([dec.run(v, rng) for v in calib_vel])
    truth = np.concatenate(calib_vel)
    slope = (pred * truth).sum(0) / (truth * truth).sum(0)
    test_vel = list(test_vel) * n_test_repeats
    preds_t = [dec.run(v, rng) for v in test_vel]
    truth_t = np.concatenate(test_vel)
    r2 = r2_variance_weighted(truth_t, np.concatenate(preds_t))
    dec.gain = np.clip(slope, 0.1, 1.5) ** (-gain_exponent)
    pred_t = np.concatenate([p * dec.gain for p in preds_t])
    err = DecoderErrorModel.fit(truth_t, pred_t)
    dec.horizon = 2000
    return Calibration(dec, err, r2)


@dataclass
class EpisodeResult:
    success: bool
    wrong_grasp: bool
    timeout: bool
    time_to_grasp: float
    path_eff: float
    user_share: float
    mean_alpha: float
    n_missed_grasps: int
    goal_acc_curve: np.ndarray  # MAP==true goal at 10 normalised-time bins
    yield_latency: float  # s from change of mind until MAP settles on the new goal (nan if n/a)
    safety: dict
    trace: dict | None = None


def run_episode(scene: Scene, goal: int, cal: Calibration, enc: EncodingModel, method: str,
                rng: np.random.Generator, arb_kwargs: dict | None = None, infer_kwargs: dict | None = None,
                switch_goal: int | None = None, max_steps: int = 1000, record: bool = False,
                user_kwargs: dict | None = None) -> EpisodeResult:
    """Run one reach-and-grasp episode under a given assistance method."""
    arm = Arm(dt=DT)
    user = UserModel(goal=goal, switch_goal=switch_goal, **(user_kwargs or {}))
    if getattr(cal.decoder, "takes_intent", False):  # real-residual decoder: no spikes
        cal.decoder.reset(rng)
        decode = cal.decoder.step
    else:
        encoder = OnlineEncoder(enc, rng)
        cal.decoder.reset()

        def decode(z):
            return cal.decoder.step(encoder.step(z))
    arb = Arbitration(name=method, dt=DT, err=cal.err, **(arb_kwargs or {}))
    arb.reset()
    ik = dict(infer_kwargs or {})
    if method == "dasa":
        err = cal.err
        if ik.pop("identity_error_model", False):  # ablation: ignore the calibrated decoder errors
            err = DecoderErrorModel(np.eye(7), np.eye(7), 1.0)
        inf = NoiseAwareInference(scene.n, err, **ik)
    else:
        inf = MaxEntInference(scene.n, **ik)

    p0 = arm.tcp
    hist = deque(maxlen=user.delay_steps + 1)
    path, alphas, shares, map_ok = [p0], [], [], []
    trace = {k: [] for k in ("tcp", "q", "gripper", "u_user", "u_cmd", "alpha", "posterior", "goal")} if record else None
    prev_ap = arm.state.gripper
    missed, outcome, t_end = 0, "timeout", max_steps
    for t in range(max_steps):
        R = arm.frames[-1][:3, :3]
        view = ArmView(arm.tcp, tool_yaw(R), arm.state.gripper)
        hist.append(view)
        user.maybe_switch(t, p0, view.p, scene)
        seen = hist[0]  # delayed visual feedback
        z_int = user.intent(seen.p, seen.yaw, seen.aperture, scene, rng)
        z_hat = decode(z_int)
        u_hat = decoded_to_command(z_hat)
        if method == "dasa":
            b = inf.update(u_hat, view, scene, z_hat=z_hat)
        else:
            b = inf.update(u_hat, view, scene)
        u, alpha, share = arb(u_hat, b, view, scene, z_hat=z_hat)
        arm.step(u[:3], u[3], -u[4])
        alphas.append(alpha)
        shares.append(share)
        map_ok.append(int(np.argmax(b)) == user.goal)
        path.append(arm.tcp)
        if record:
            trace["tcp"].append(arm.tcp.tolist())
            trace["q"].append(arm.state.q.tolist())
            trace["gripper"].append(arm.state.gripper)
            trace["u_user"].append(u_hat.tolist())
            trace["u_cmd"].append(u.tolist())
            trace["alpha"].append(alpha)
            trace["posterior"].append(b.tolist())
            trace["goal"].append(user.goal)
        ap = arm.state.gripper
        if prev_ap >= CLOSED > ap:  # gripper just closed: did it close on an object?
            p = arm.tcp
            yaw = tool_yaw(arm.frames[-1][:3, :3])
            d = np.linalg.norm(scene.positions - p, axis=1)
            k = int(np.argmin(d))
            if d[k] < GRASP_RADIUS and abs(wrap_half_pi(yaw - scene.yaws[k])) < GRASP_YAW_TOL:
                outcome = "success" if k == user.goal else "wrong"
                t_end = t + 1
                break
            missed += 1
        prev_ap = ap

    # yield latency: first step after the switch from which MAP stays on the new goal
    yield_latency = float("nan")
    if user.switched:
        ok = np.array(map_ok[user.switch_step:])
        bad = np.where(~ok)[0]
        if len(ok) and (len(bad) == 0 or bad[-1] < len(ok) - 1):
            yield_latency = (0 if len(bad) == 0 else bad[-1] + 1) * DT
    curve = np.array([np.mean(c) if len(c) else np.nan for c in np.array_split(np.array(map_ok, float), 10)])
    return EpisodeResult(
        success=outcome == "success", wrong_grasp=outcome == "wrong", timeout=outcome == "timeout",
        time_to_grasp=t_end * DT, path_eff=path_efficiency(np.array(path)),
        user_share=float(np.mean(shares)), mean_alpha=float(np.mean(alphas)), n_missed_grasps=missed,
        goal_acc_curve=curve, yield_latency=yield_latency, safety=dict(arm.events),
        trace=trace)
