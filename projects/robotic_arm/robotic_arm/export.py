"""Export small JSON/PNG files for the website (results/site_export, < 5 MB total).

Files:
* ``metrics.json``  decoder R² tables + shared-autonomy comparison tables.
* ``episodes.json`` recorded closed-loop episodes + arm kinematics (DH) for three.js.
* ``decoding.json`` true vs decoded velocity on held-out test blocks.
* ``*.png``         key figures.

Usage: ``python -m robotic_arm.export``
"""

from __future__ import annotations

import json
import shutil

import numpy as np

from . import DOF_NAMES, RESULTS_ROOT
from .arm import DH_BASE, DH_JOINTS, Q_HOME, Q_LIMITS, TOOL_D, fk
from .data import Normalizer, held_out_pairs, load_blocks
from .task import (GRASP_RADIUS, GRASP_YAW_TOL, OBJECT_COLORS, OBJECT_NAMES, V_SCALE)

OUT = RESULTS_ROOT / "site_export"
METHOD_LABELS = {"dasa": "Decoder-aware shared autonomy (proposed)", "confidence": "Confidence-based blending",
                 "hindsight": "Hindsight-optimisation assistance", "distance": "Distance-based blending",
                 "fixed": "Fixed blending (alpha constant)", "direct": "Decoder only (no assistance)"}
DECODER_LABELS = {"wiener_daynorm": "Wiener filter", "wiener_globalnorm": "Wiener filter (global normalisation)",
                  "kalman": "Kalman filter", "gru": "GRU", "transformer": "Causal Transformer",
                  "ms_gru": "Multi-day GRU (proposed)", "ms_gru_noaug": "Multi-day GRU, no augmentation"}


def _r(x, n=4):
    if x is None:
        return None
    if isinstance(x, (list, tuple)):
        return [_r(v, n) for v in x]
    if isinstance(x, dict):
        return {k: _r(v, n) for k, v in x.items()}
    if isinstance(x, (float, np.floating)):
        return None if not np.isfinite(x) else round(float(x), n)
    return x


JS_FK = (
    "function dh(al,a,d,th){const ct=Math.cos(th),st=Math.sin(th),ca=Math.cos(al),sa=Math.sin(al);"
    "return [[ct,-st*ca,st*sa,a*ct],[st,ct*ca,-ct*sa,a*st],[0,sa,ca,d],[0,0,0,1]];}"
    "function mul(A,B){return A.map(r=>B[0].map((_,j)=>r.reduce((s,x,k)=>s+x*B[k][j],0)));}"
    "function fk(arm,q){let T=dh(arm.base.alpha,arm.base.a,arm.base.d,arm.base.theta);const frames=[T];"
    "arm.joints.forEach((j,i)=>{T=mul(T,dh(j.alpha,j.a,j.d,q[i]+j.theta_offset));frames.push(T);});"
    "frames.push(mul(T,[[1,0,0,0],[0,1,0,0],[0,0,1,arm.tool_d],[0,0,0,1]]));"
    "return frames; /* frames[k][r][3] = origin; last = TCP */}"
)


def kinematics_spec() -> dict:
    q = Q_HOME.tolist()
    return {
        "convention": "classical DH: T_i = Rz(theta_i + offset_i) * Tz(d_i) * Tx(a_i) * Rx(alpha_i); "
                      "world = T_base * T_1(q1) * ... * T_7(q7) * Tz(tool_d). Units: metres, radians. z up, "
                      "table surface at z = 0, robot base at the origin.",
        "source": "Kinova Gen3 7-DoF User Guide (R07), Table 94; tool_d is our gripper TCP assumption",
        "base": {"alpha": DH_BASE[0], "a": DH_BASE[1], "d": DH_BASE[2], "theta": DH_BASE[3]},
        "joints": [{"alpha": a, "a": aa, "d": d, "theta_offset": off} for a, aa, d, off in DH_JOINTS],
        "tool_d": TOOL_D,
        "joint_limits_rad": Q_LIMITS.tolist(),
        "test_vector": {"q": q, "tcp_xyz": fk(Q_HOME)[:3, 3].tolist(),
                        "q_zero_tcp_xyz": fk(np.zeros(7))[:3, 3].tolist()},
        "js_reference": JS_FK,
    }


def export_episodes(max_points: int = 300) -> dict:
    show = json.loads((RESULTS_ROOT / "closed_loop_showcase.json").read_text())
    cl = json.loads((RESULTS_ROOT / "closed_loop_metrics.json").read_text())
    eps = []
    for e in show:
        tr = e["trace"]
        n = len(tr["tcp"])
        stride = int(np.ceil(n / max_points))
        idx = list(range(0, n, stride))
        if idx[-1] != n - 1:
            idx.append(n - 1)
        P = e["scene"]["positions"]
        eps.append({
            "id": f"{e['method']}_{e['scenario']}_{e['idx']}",
            "method": e["method"], "method_label": METHOD_LABELS[e["method"]], "scenario": e["scenario"],
            "objects": [{"id": k, "name": OBJECT_NAMES[k], "color": OBJECT_COLORS[k], "grasp_xyz": _r(P[k]),
                         "yaw": _r(e["scene"]["yaws"][k]), "size_xyz": e["scene"]["sizes"][k]} for k in range(len(P))],
            "initial_goal": e["initial_goal"], "final_goal": e["final_goal"],
            "outcome": "success" if e["success"] else ("wrong_object" if e["wrong"] else "timeout"),
            "time_to_grasp_s": _r(e["time"], 2), "path_efficiency": _r(e["path_eff"], 3),
            "user_share": _r(e["user_share"], 3),
            "dt_s": 0.02 * stride, "n_steps": len(idx),
            "steps": {
                "t": [round(i * 0.02, 2) for i in idx],
                "tcp_xyz": [_r(tr["tcp"][i]) for i in idx],
                "gripper_open": [_r(tr["gripper"][i], 3) for i in idx],
                "q": [_r(tr["q"][i]) for i in idx],
                "u_user": [_r(tr["u_user"][i]) for i in idx],
                "u_command": [_r(tr["u_cmd"][i]) for i in idx],
                "alpha": [_r(tr["alpha"][i], 3) for i in idx],
                "goal_posterior": [_r(tr["posterior"][i], 3) for i in idx],
                "true_goal": [tr["goal"][i] for i in idx],
            },
        })
    return {
        "description": "Simulated closed-loop reach-and-grasp episodes (no human in the loop): a simulated user "
                       "drives an encoding model fitted to FALCON H1, decoded by a Wiener filter, arbitrated by "
                       "shared autonomy and executed by a safety-filtered 7-DoF Kinova Gen3 kinematic model.",
        "command_vector": "[vx, vy, vz (m/s), yaw_rate (rad/s), grip_rate (1/s, positive closes)]",
        "decoder_offline_r2": _r(cl["levels"][str(cl["nominal_kappa"])]["offline_r2"], 3),
        "grasp_rule": {"radius_m": GRASP_RADIUS, "yaw_tolerance_rad": round(GRASP_YAW_TOL, 4),
                       "closed_below_aperture": 0.2},
        "speed_scale_m_per_s_per_unit": V_SCALE,
        "arm": kinematics_spec(),
        "table": {"z": 0.0, "x_range": [0.0, 0.8], "y_range": [-0.45, 0.45]},
        "episodes": eps,
    }


def export_decoding(n_trials: int = 4, window: int = 400, stride: int = 2) -> dict:
    preds = np.load(RESULTS_ROOT / "decoding_preds.npz")
    B = load_blocks()
    ny = Normalizer.fit(np.concatenate([b.vel[b.eval_mask] for b in B["held_in_calib"]]))
    pairs = held_out_pairs(B["held_out_calib"])
    last = max(b.day for b in B["held_in_calib"])
    trials = []
    for _, b in pairs[:: max(1, len(pairs) // n_trials)][:n_trials]:
        y = ny(b.vel)
        speed = np.abs(y).sum(1) * b.eval_mask
        s = int(np.argmax(np.convolve(speed, np.ones(window), "valid")))  # most active window
        sl = slice(s, s + window, stride)
        tr = {"day": b.day.isoformat(), "days_since_training": (b.day - last).days,
              "t": [round(i * 0.02, 2) for i in range(0, window, stride)], "dof_names": DOF_NAMES,
              "true": _r(y[sl].tolist(), 3)}
        for name, key in [("proposed", "ms_gru"), ("wiener", "wiener"), ("gru", "gru"),
                          ("proposed_label_free", "ms_gru_unsup_kl")]:
            k = f"{key}|{b.key}"
            if k in preds.files:
                tr[name] = _r(preds[k][sl].tolist(), 3)
        trials.append(tr)
    return {"description": "Held-out FALCON H1 test blocks (later days than any training data). Velocities are "
                           "z-scored per DoF. 'proposed' = multi-day GRU with a 45 s supervised read-in "
                           "recalibration; 'wiener' = Wiener filter zero-shot; 'gru' = GRU fine-tuned on 45 s; "
                           "'proposed_label_free' = multi-day GRU with a label-free (KL-aligned) read-in.",
            "bin_s": 0.02 * stride, "trials": trials}


def export_metrics() -> dict:
    d = json.loads((RESULTS_ROOT / "decoding_metrics.json").read_text())
    c = json.loads((RESULTS_ROOT / "closed_loop_metrics.json").read_text())
    within = [{"decoder": k, "label": DECODER_LABELS.get(k, k), "r2_vw": _r(v["r2_vw_mean"]),
               "r2_vw_std": _r(v.get("r2_vw_seed_std", v.get("r2_vw_std"))),
               "r2_per_dof": _r(v["r2_pooled_per_dof"]),
               "r2_vw_ci95": _r(v.get("r2_vw_ci95")), "n_seeds": v.get("n_seeds", 1),
               "r2_vw_per_seed": _r(v.get("r2_vw_per_seed"))} for k, v in d["within"].items()]
    cross = [{"setting": k, "r2_vw": _r(v["r2_vw_mean"]), "r2_vw_seed_std": _r(v.get("r2_vw_seed_std")),
              "r2_per_day": _r(v["r2_vw_blocks"]),
              "r2_vw_ci95": _r(v.get("r2_vw_ci95")), "n_seeds": v.get("n_seeds", 1)} for k, v in d["cross"].items()]
    lf = {}
    for m in ("ms_gru", "gru", "transformer"):
        if f"{m}_unsup_kl" in d["cross"]:
            lf[m] = {k: {"r2_vw": _r(d["cross"][f"{m}_{k}"]["r2_vw_mean"]),
                         "r2_vw_ci95": _r(d["cross"][f"{m}_{k}"].get("r2_vw_ci95")),
                         "r2_vw_seed_std": _r(d["cross"][f"{m}_{k}"].get("r2_vw_seed_std"))}
                     for k in ("zs_fixednorm", "zeroshot", "unsup_center", "unsup_coral", "unsup_kl")
                     if f"{m}_{k}" in d["cross"]}
    label_free = {"note": "No labels from the new day: block A's neural data only. zs_fixednorm = no per-day "
                          "re-normalisation; zeroshot = per-channel z-scoring to block A; unsup_center / unsup_coral "
                          "= latent mean / mean+covariance matching of the read-in output to the held-in days; "
                          "unsup_kl = KL alignment of read-in and core-state Gaussians (NoMAD-style).",
                  "models": lf} if lf else None
    nk = str(c["nominal_kappa"])
    cols = ("success", "success_ci95", "wrong_grasp", "timeout", "time_to_grasp_mean", "path_eff", "user_share",
            "mean_alpha", "yield_latency_median")
    sa = {m: {k: _r(c["main"][nk][m][k]) for k in cols} | {"label": METHOD_LABELS[m]} for m in c["methods"]}
    com = {m: {k: _r(c["change_of_mind"][m][k]) for k in cols} | {"label": METHOD_LABELS[m]} for m in c["methods"]}
    clut = {m: {k: _r(c["clutter"][m][k]) for k in cols} | {"label": METHOD_LABELS[m]} for m in c["methods"]}
    sweep = [{"kappa": lv["kappa"], "offline_r2": _r(lv["offline_r2"]),
              "success": {m: _r(c["main"][str(lv["kappa"])][m]["success"]) for m in c["methods"]},
              "time_to_grasp": {m: _r(c["main"][str(lv["kappa"])][m]["time_to_grasp_mean"]) for m in c["methods"]}}
             for lv in sorted(c["levels"].values(), key=lambda v: v["offline_r2"])]
    return {
        "decoding": {"dataset": "FALCON H1 (DANDI 000954)", "metric": "R² variance-weighted (FALCON), mean over blocks",
                     "dof_names": DOF_NAMES, "held_in_minival": within, "held_out_days": cross,
                     "days_since_train": d["meta"]["days_since_train"],
                     "label_free_recalibration": label_free,
                     "paired_tests": _r(d.get("paired_tests", {})),
                     "uncertainty": "r2_vw_ci95 = percentile bootstrap 95% CI over test blocks (13 minival blocks / "
                                    "7 held-out days) of the seed-averaged per-block R²; paired tests = exact "
                                    "sign-flip permutation test on per-block differences."},
        "shared_autonomy": {"note": "Simulated closed loop, not human-in-the-loop.",
                            "nominal_decoder_offline_r2": _r(c["levels"][nk]["offline_r2"]),
                            "real_session_r2": _r(c["real_session_r2"]),
                            "real_wiener600_r2": _r(c["real_wiener600_r2"]),
                            "standard": sa, "change_of_mind": com, "clutter": clut, "r2_sweep": sweep,
                            "authority_sweep": {m: [{k: _r(v) for k, v in p.items() if not isinstance(v, (dict, list))}
                                                    for p in pts] for m, pts in c.get("authority_sweep", {}).items()},
                            "ablations": {k: {s: {kk: _r(v[s][kk]) for kk in ("success", "wrong_grasp",
                                                                              "time_to_grasp_mean")}
                                              for s in v} for k, v in c["ablations"].items()},
                            "ablations_weakest_decoder": {k: {s: {kk: _r(v[s][kk]) for kk in ("success", "wrong_grasp")}
                                                              for s in v} for k, v in c.get("ablations_hard", {}).items()},
                            "paired_tests": c["tests"],
                            "real_residuals": export_real_residuals(cols)},
    }


def export_real_residuals(cols) -> dict | None:
    """Closed loop driven by real FALCON H1 decoder residuals (new key; the synthetic keys are unchanged)."""
    path = RESULTS_ROOT / "closed_loop_real_metrics.json"
    if not path.exists():
        return None
    r = json.loads(path.read_text())
    out = {"note": "Decoder noise replayed from real held-out FALCON H1 decoder errors: z_hat = FIR(intended "
                   "velocity) + speed-scaled, block-bootstrapped real residuals. Methods re-tuned on separate seeds.",
           "primary": r["primary"], "tuned": r["tuning"]["chosen"], "sources": {}}
    for name, v in r["sources"].items():
        st = v["residual_stats"]
        out["sources"][name] = {
            "decoder_preds": v["preds"], "conditioned": v["conditioned"],
            "real_offline_r2": _r(st["real_r2_vw"]), "fir_r2": _r(st["fir_r2_vw"]),
            "sim_offline_r2": _r(v["sim_offline_r2"]), "err_tau_ticks": _r(v["err_tau"], 2),
            "residual_tau_ticks": _r(st["resid_tau_ticks"], 2), "n_bins": st["n_bins"],
            **{scen: {m: {k: _r(v[scen][m][k]) for k in cols} | {"label": METHOD_LABELS[m]} for m in r["methods"]}
               for scen in ("standard", "change_of_mind", "clutter")},
            "paired_tests": v["tests"], "n_episodes": v["n_episodes"]}
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "decoding.json").write_text(json.dumps(export_decoding()))
    (OUT / "episodes.json").write_text(json.dumps(export_episodes()))
    (OUT / "metrics.json").write_text(json.dumps(export_metrics(), indent=1))
    for png in ("decoding_within.png", "decoding_cross_session.png", "success_vs_r2.png", "change_of_mind.png",
                "goal_inference.png", "example_episodes.png", "arm_3d.png", "ablations.png", "decoding_traces.png",
                "authority_tradeoff.png", "label_free_recal.png", "real_residuals.png"):
        src = RESULTS_ROOT / "figures" / png
        if src.exists():
            shutil.copy(src, OUT / png)
    total = sum(f.stat().st_size for f in OUT.iterdir())
    print(f"site_export: {total / 1e6:.2f} MB in {len(list(OUT.iterdir()))} files")


if __name__ == "__main__":
    main()
