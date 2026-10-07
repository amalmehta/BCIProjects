"""Shared-autonomy experiments in the simulated closed loop.

Pipeline: fit the encoding model to one real FALCON H1 session -> simulate the open-loop
calibration block -> fit the online decoder -> run reach-and-grasp episodes for every
assistance method, decoder-quality level and scenario (standard / change of mind).

Usage: ``python -m robotic_arm.run_closed_loop --config configs/cpu.yaml``
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import copy  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
import hashlib  # noqa: E402
import multiprocessing as mp  # noqa: E402
from multiprocessing import Pool  # noqa: E402

import numpy as np  # noqa: E402
from scipy.stats import binomtest  # noqa: E402

from . import RESULTS_ROOT  # noqa: E402
from .closed_loop import calibrate, run_episode  # noqa: E402
from .config import load_config  # noqa: E402
from .data import Normalizer, load_blocks  # noqa: E402
from .decoders.linear import WienerFilter  # noqa: E402
from .encoding import fit_encoding_model  # noqa: E402
from .metrics import r2_variance_weighted  # noqa: E402
from .task import Scene  # noqa: E402

DEC_LAGS, DEC_ALPHA = 10, 0.1  # closed-loop Wiener filter (same config for real & simulated)
CLUTTER_OBJECTS, CLUTTER_SEP = 5, 0.08  # "clutter" scenario: 5 objects, >= 8 cm apart


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# --------------------------------------------------------------------------- setup
def setup(cfg) -> dict:
    """Fit the encoding model and calibrate decoders for every decoder-quality level."""
    cl = cfg.closed_loop
    B = load_blocks()
    ny = Normalizer.fit(np.concatenate([b.vel[b.eval_mask] for b in B["held_in_calib"]]))
    cal_blocks = [b for b in B["held_in_calib"] if b.day.isoformat() == cl.calib_day]
    test_blocks = [b for b in B["held_in_minival"] if b.day.isoformat() == cl.calib_day]
    counts = np.concatenate([b.spikes for b in cal_blocks])
    v = np.concatenate([ny(b.vel) for b in cal_blocks])
    enc = fit_encoding_model(counts, v)
    enc.sigma = cl.latent_sigma
    # real-data reference: same decoder, same session
    nx = Normalizer.fit(counts)
    wf = WienerFilter(DEC_LAGS, DEC_ALPHA).fit([nx(b.spikes) for b in cal_blocks], [ny(b.vel) for b in cal_blocks],
                                               [b.eval_mask for b in cal_blocks])
    real_r2 = float(np.mean([r2_variance_weighted(ny(b.vel)[b.eval_mask], wf.predict(nx(b.spikes))[b.eval_mask])
                             for b in test_blocks]))
    # second reference: FALCON-style Wiener filter (600 ms history) trained on all held-in days
    days = sorted({b.day for b in B["held_in_calib"]})
    dn = {d: Normalizer.fit(np.concatenate([b.spikes for b in B["held_in_calib"] if b.day == d])) for d in days}
    tr = B["held_in_calib"]
    wf6 = WienerFilter(15, 0.1, 2.0, 2).fit([dn[b.day](b.spikes) for b in tr], [ny(b.vel) for b in tr],
                                           [b.eval_mask for b in tr])
    real_r2_multi = float(np.mean([r2_variance_weighted(b.vel[b.eval_mask],
                                                        ny.inverse(wf6.predict(dn[b.day](b.spikes)))[b.eval_mask])
                                   for b in B["held_in_minival"]]))
    log(f"real reference R2: same-session 10-tap Wiener {real_r2:.3f}; multi-day 600 ms Wiener {real_r2_multi:.3f}")
    cal_vel = [ny(b.vel) for b in cal_blocks]
    test_vel = [ny(b.vel) for b in test_blocks]
    rng = np.random.default_rng(cl.seed)
    levels = {}
    for kappa in cl.kappas:
        e = copy.deepcopy(enc)
        e.C = enc.C * kappa
        cal = calibrate(e, cal_vel, test_vel, rng, DEC_LAGS, DEC_ALPHA, gain_exponent=cl.gain_exponent)
        levels[float(kappa)] = (e, cal)
        log(f"kappa {kappa}: simulated offline R2 {cal.r2_test:.3f} (real {real_r2:.3f}); err tau {cal.err.tau:.1f}")
    return {"enc": enc, "levels": levels, "real_r2": real_r2, "real_r2_multi": real_r2_multi, "ny": ny}


# --------------------------------------------------------------------------- episodes
def _episode_job(args):
    (method, kappa, idx, scenario, enc, cal, arb_kwargs, infer_kwargs, seed, n_objects, max_steps, record) = args
    rs = np.random.default_rng([seed, idx])
    if scenario == "clutter":  # more, closer objects: harder goal inference
        scene = Scene.random(rs, CLUTTER_OBJECTS, CLUTTER_SEP)
    else:
        scene = Scene.random(rs, n_objects)
    goal = int(rs.integers(scene.n))
    others = [g for g in range(scene.n) if g != goal]
    switch = int(rs.choice(others)) if scenario == "change_of_mind" else None
    rng = np.random.default_rng([seed, idx, 7])
    res = run_episode(scene, goal, cal, enc, method, rng, arb_kwargs, infer_kwargs, switch_goal=switch,
                      max_steps=max_steps, record=record)
    out = {"method": method, "kappa": kappa, "idx": idx, "scenario": scenario, "success": res.success,
           "wrong": res.wrong_grasp, "timeout": res.timeout, "time": res.time_to_grasp, "path_eff": res.path_eff,
           "user_share": res.user_share, "alpha": res.mean_alpha, "missed": res.n_missed_grasps,
           "acc_curve": res.goal_acc_curve.tolist(), "yield_latency": res.yield_latency, "safety": res.safety}
    if record:
        out["trace"] = res.trace
        out["scene"] = {"positions": scene.positions.tolist(), "yaws": scene.yaws.tolist(),
                        "sizes": scene.sizes.tolist()}
        out["initial_goal"], out["final_goal"] = goal, switch if switch is not None else goal
    return out


_POOL: Pool | None = None
_CACHE: dict | None = None


def _job_key(job) -> str:
    """Deterministic key of an episode job (episodes are fully determined by it)."""
    method, kappa, idx, scenario, enc, cal, arb, inf, seed, n_obj, max_steps, record = job
    blob = json.dumps([method, kappa, idx, scenario, round(cal.r2_test, 10), arb, inf, seed, n_obj, max_steps],
                      sort_keys=True, default=str)
    return hashlib.sha1(blob.encode()).hexdigest()


def _cache_path():
    return RESULTS_ROOT / "closed_loop_cache.jsonl"


def run_jobs(jobs: list, workers: int, timeout_per_job: float = 60.0) -> list:
    """Run episode jobs on one reusable worker pool, with an on-disk cache.

    Finished episodes are appended to ``closed_loop_cache.jsonl`` so an interrupted run
    resumes where it stopped. If a worker dies (``Pool.map`` would then hang forever), the
    batch times out, the pool is rebuilt and the missing jobs are retried.
    """
    global _POOL, _CACHE
    if _CACHE is None:
        _CACHE = {}
        if _cache_path().exists():
            for line in _cache_path().read_text().splitlines():
                k, v = line.split("\t", 1)
                _CACHE[k] = json.loads(v)
    keys = [None if j[-1] else _job_key(j) for j in jobs]
    todo = [j for j, k in zip(jobs, keys) if k is None or k not in _CACHE]
    results: list = []
    for attempt in range(3):
        if not todo:
            break
        try:
            if workers <= 1:
                results = [_episode_job(j) for j in todo]
            else:
                if _POOL is None:
                    _POOL = Pool(workers)
                results = _POOL.map_async(_episode_job, todo, chunksize=1).get(
                    timeout=120 + timeout_per_job * len(todo) / workers)
            break
        except mp.TimeoutError:
            log(f"worker pool stalled (attempt {attempt + 1}); rebuilding it")
            _POOL.terminate()
            _POOL = None
    else:
        raise RuntimeError("episode jobs failed three times")
    fresh = iter(results)
    out = []
    with open(_cache_path(), "a") as fh:
        for j, k in zip(jobs, keys):
            if k is not None and k in _CACHE:
                out.append(_CACHE[k])
                continue
            r = next(fresh)
            if k is not None:
                _CACHE[k] = r
                fh.write(k + "\t" + json.dumps(r) + "\n")
            out.append(r)
    return out


def summarize(rows: list[dict]) -> dict:
    """Aggregate episode rows into the metrics reported in the paper."""
    n = len(rows)
    succ = np.array([r["success"] for r in rows], float)
    s_rows = [r for r in rows if r["success"]]
    lo, hi = _wilson(succ.sum(), n)
    yl = [r["yield_latency"] for r in rows if r["scenario"] == "change_of_mind"]
    safety_keys = rows[0]["safety"].keys()
    return {
        "n": n, "success": float(succ.mean()), "success_ci95": [lo, hi],
        "wrong_grasp": float(np.mean([r["wrong"] for r in rows])),
        "timeout": float(np.mean([r["timeout"] for r in rows])),
        "time_to_grasp_mean": float(np.mean([r["time"] for r in s_rows])) if s_rows else None,
        "time_to_grasp_median": float(np.median([r["time"] for r in s_rows])) if s_rows else None,
        "path_eff": float(np.mean([r["path_eff"] for r in s_rows])) if s_rows else None,
        "user_share": float(np.mean([r["user_share"] for r in rows])),
        "mean_alpha": float(np.mean([r["alpha"] for r in rows])),
        "missed_grasps": float(np.mean([r["missed"] for r in rows])),
        "goal_acc_curve": np.nanmean([r["acc_curve"] for r in rows], 0).tolist(),
        "yield_latency_median": float(np.nanmedian(yl)) if yl and np.isfinite(yl).any() else None,
        "yield_settled_frac": float(np.mean(np.isfinite(yl))) if yl else None,
        "safety_events_per_episode": {k: float(np.mean([r["safety"][k] for r in rows])) for k in safety_keys},
    }


def _wilson(k: float, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return float(c - h), float(c + h)


def paired_test(rows_a: list[dict], rows_b: list[dict]) -> dict:
    """Exact McNemar test on paired episode successes (same scenes and seeds)."""
    a = {(r["idx"], r["scenario"]): r["success"] for r in rows_a}
    b = {(r["idx"], r["scenario"]): r["success"] for r in rows_b}
    keys = sorted(set(a) & set(b))
    n10 = sum(a[k] and not b[k] for k in keys)
    n01 = sum(b[k] and not a[k] for k in keys)
    p = binomtest(n10, n10 + n01, 0.5).pvalue if n10 + n01 > 0 else 1.0
    return {"a_only": n10, "b_only": n01, "p_mcnemar": float(p)}


# --------------------------------------------------------------------------- main
def method_settings(tuned: dict, method: str) -> tuple[dict, dict]:
    t = tuned.get(method, {})
    return t.get("arb", {}), t.get("infer", {})


AUTHORITY_SWEEP = [("dasa", "alpha_max", [0.4, 0.6, 0.8, 0.95]),
                   ("confidence", "alpha_max", [0.4, 0.6, 0.8, 0.95]),
                   ("fixed", "alpha_const", [0.3, 0.5, 0.7, 0.9]),
                   ("hindsight", "ho_lambda", [5.0, 1.0, 0.2, 0.05])]

TUNING_GRID = {
    "fixed": [{"arb": {"alpha_const": a}, "infer": {"beta": b}} for a in (0.3, 0.5, 0.7) for b in (5.0, 20.0)],
    "confidence": [{"arb": {}, "infer": {"beta": b}} for b in (5.0, 10.0, 20.0, 40.0)],
    "hindsight": [{"arb": {"ho_lambda": l}, "infer": {"beta": b}} for l in (0.2, 1.0) for b in (5.0, 20.0)],
    "distance": [{"arb": {"envelope": e}, "infer": {}} for e in (0.15, 0.25, 0.35)],
    "dasa": [{"arb": {"gate_c0": c}, "infer": {}} for c in (1.5, 2.5, 4.0)],
}


def _retune_beta(jobs_for, arb: dict, kappa: float, cl) -> float:
    """Re-tune the MaxEnt temperature for confidence blending once a switching prior is added."""
    best = None
    for beta in (5.0, 20.0, 40.0, 80.0):
        inf = {"hazard": 0.002, "beta": beta}
        rows = run_jobs(jobs_for("confidence", kappa, range(cl.tune_episodes), "standard", arb, inf, 10_000)
                        + jobs_for("confidence", kappa, range(cl.tune_episodes), "change_of_mind", arb, inf, 10_000),
                        cl.workers)
        s = summarize(rows)
        obj = s["success"] - s["wrong_grasp"] - 0.01 * (s["time_to_grasp_mean"] or 20.0)
        if best is None or obj > best[0]:
            best = (obj, beta)
    log(f"re-tuned beta for confidence+switching: {best[1]}")
    return best[1]


def run(cfg) -> dict:
    cl = cfg.closed_loop
    t_start = time.time()
    S = setup(cfg)
    levels, real_r2, real_multi = S["levels"], S["real_r2"], S["real_r2_multi"]
    kappas = sorted(levels)
    r2s = [levels[k][1].r2_test for k in kappas]
    # nominal level = the simulated decoder whose offline R² is closest to that of a
    # FALCON-style Wiener filter on real held-in data
    nominal = float(kappas[int(np.argmin([abs(r - real_multi) for r in r2s]))])
    out: dict = {"real_session_r2": real_r2, "real_wiener600_r2": real_multi, "levels": {str(k): {"kappa": k, "offline_r2": levels[k][1].r2_test,
                                                                  "err_tau": levels[k][1].err.tau}
                                                              for k in kappas},
                 "nominal_kappa": nominal, "methods": list(cl.methods)}
    log(f"real R2 {real_r2:.3f}; simulated R2 by kappa: {dict(zip(kappas, np.round(r2s, 3)))}")

    def jobs_for(method, kappa, idxs, scenario, arb, inf, seed_offset=0, record=False):
        enc, cal = levels[kappa]
        return [(method, kappa, i, scenario, enc, cal, arb, inf, cl.seed + seed_offset, cl.n_objects,
                 cl.max_steps, record) for i in idxs]

    # 1) tune each method's main hyper-parameters on separate seeds, standard + change of mind
    tuned, tune_log = {}, {}
    tune_idx = range(cl.tune_episodes)
    for method, grid in TUNING_GRID.items():
        if method not in cl.methods:
            continue
        best = None
        for setting in grid:
            rows = run_jobs(jobs_for(method, nominal, tune_idx, "standard", setting["arb"], setting["infer"], 10_000)
                            + jobs_for(method, nominal, tune_idx, "change_of_mind", setting["arb"],
                                       setting["infer"], 10_000), cl.workers)
            s = summarize(rows)
            obj = s["success"] - s["wrong_grasp"] - 0.01 * (s["time_to_grasp_mean"] or 20.0)
            tune_log.setdefault(method, []).append({"setting": setting, "success": s["success"],
                                                    "wrong": s["wrong_grasp"], "objective": obj})
            if best is None or obj > best[0]:
                best = (obj, setting)
        tuned[method] = best[1]
        log(f"tuned {method}: {best[1]} (objective {best[0]:.3f})")
    out["tuning"] = {"chosen": tuned, "log": tune_log}

    # 2) main comparison across decoder-quality levels (standard scenario)
    idxs = range(cl.episodes)
    all_rows: dict = {}
    out["main"] = {}
    for kappa in kappas:
        for method in cl.methods:
            arb, inf = method_settings(tuned, method)
            rows = run_jobs(jobs_for(method, kappa, idxs, "standard", arb, inf), cl.workers)
            all_rows[(method, kappa, "standard")] = rows
            out["main"].setdefault(str(kappa), {})[method] = summarize(rows)
            s = out["main"][str(kappa)][method]
            log(f"kappa {kappa} {method:10s} success {s['success']:.2f} wrong {s['wrong_grasp']:.2f} "
                f"T {s['time_to_grasp_mean']} share {s['user_share']:.2f}  [{time.time()-t_start:.0f}s]")

    # 3) change of mind at the nominal level
    out["change_of_mind"] = {}
    for method in cl.methods:
        arb, inf = method_settings(tuned, method)
        rows = run_jobs(jobs_for(method, nominal, range(cl.change_of_mind_episodes), "change_of_mind", arb, inf),
                        cl.workers)
        all_rows[(method, nominal, "change_of_mind")] = rows
        out["change_of_mind"][method] = summarize(rows)
        s = out["change_of_mind"][method]
        log(f"change-of-mind {method:10s} success {s['success']:.2f} wrong {s['wrong_grasp']:.2f} "
            f"yield {s['yield_latency_median']}")

    # 3b) clutter at the nominal level
    out["clutter"] = {}
    for method in cl.methods:
        arb, inf = method_settings(tuned, method)
        rows = run_jobs(jobs_for(method, nominal, range(cl.episodes), "clutter", arb, inf), cl.workers)
        all_rows[(method, nominal, "clutter")] = rows
        out["clutter"][method] = summarize(rows)
        s = out["clutter"][method]
        log(f"clutter {method:10s} success {s['success']:.2f} wrong {s['wrong_grasp']:.2f} T {s['time_to_grasp_mean']}")

    # 3c) authority-performance trade-off: sweep each method's assistance strength
    out["authority_sweep"] = {}
    for method, key, values in AUTHORITY_SWEEP:
        if method not in cl.methods:
            continue
        arb0, inf = method_settings(tuned, method)
        for val in values:
            rows = []
            for scen in ("standard", "clutter"):
                rows += run_jobs(jobs_for(method, nominal, range(cl.ablation_episodes), scen, {**arb0, key: val}, inf),
                                 cl.workers)
            s = summarize(rows)
            out["authority_sweep"].setdefault(method, []).append({key: val, **s})
            log(f"authority {method:10s} {key}={val}: success {s['success']:.2f} share {s['user_share']:.2f} "
                f"T {s['time_to_grasp_mean']}")

    # 4) paired significance tests vs the proposed method (nominal level)
    out["tests"] = {}
    for scen in ("standard", "change_of_mind", "clutter"):
        for method in cl.methods:
            if method == "dasa":
                continue
            out["tests"][f"{scen}:dasa_vs_{method}"] = paired_test(all_rows[("dasa", nominal, scen)],
                                                                   all_rows[(method, nominal, scen)])

    # 5) ablations of the proposed method (nominal level, both scenarios)
    arb0, inf0 = method_settings(tuned, "dasa")
    tau = levels[nominal][1].err.tau
    conf_arb, conf_inf = method_settings(tuned, "confidence")
    ablations = {
        "dasa": (arb0, inf0, "dasa"),
        "no_gate": ({**arb0, "use_gate": False}, inf0, "dasa"),
        "no_switching_prior": (arb0, {**inf0, "hazard": 0.0}, "dasa"),
        "no_autocorr_correction": (arb0, {**inf0, "temper": 1.0 / tau}, "dasa"),
        "uncalibrated_error_model": (arb0, {**inf0, "identity_error_model": True}, "dasa"),
        "mismatched_user_model": ({**arb0, "user_v_max": 0.06}, {**inf0, "user_v_max": 0.06}, "dasa"),
        "confidence+switching": (conf_arb, {**conf_inf, "hazard": 0.002}, "confidence"),
        "confidence+switching_retuned": (conf_arb, {"hazard": 0.002, "beta": _retune_beta(
            jobs_for, conf_arb, nominal, cl)}, "confidence"),
    }
    # ablations at the nominal level and at the hardest (lowest-R²) level, where the
    # nominal level saturates
    hard = float(kappas[0])
    out["ablations"], out["ablations_hard"] = {}, {}
    out["hard_kappa"] = hard
    n_ab = range(cl.ablation_episodes)
    for name, (arb, inf, method) in ablations.items():
        for level, store in ((nominal, out["ablations"]), (hard, out["ablations_hard"])):
            res = {}
            for scen in ("standard", "change_of_mind", "clutter"):
                rows = run_jobs(jobs_for(method, level, n_ab, scen, arb, inf), cl.workers)
                res[scen] = summarize(rows)
            store[name] = res
            log(f"ablation k={level} {name:28s} std {res['standard']['success']:.2f} "
                f"com {res['change_of_mind']['success']:.2f} (wrong {res['change_of_mind']['wrong_grasp']:.2f}) "
                f"clutter {res['clutter']['success']:.2f}")

    # 6) recorded episodes for the website
    showcase = []
    plan = [("direct", "standard", 0), ("dasa", "standard", 0), ("confidence", "standard", 1),
            ("hindsight", "standard", 2), ("dasa", "standard", 3), ("dasa", "change_of_mind", 4),
            ("confidence", "change_of_mind", 4), ("dasa", "clutter", 6)]
    for method, scen, idx in plan:
        arb, inf = method_settings(tuned, method)
        showcase.append(_episode_job(jobs_for(method, nominal, [idx], scen, arb, inf, 50_000, True)[0]))
    out["_showcase"] = showcase
    out["wall_time_s"] = time.time() - t_start
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/cpu.yaml")
    ap.add_argument("--resume", action="store_true", help="reuse episodes cached by an interrupted run")
    args = ap.parse_args()
    cfg = load_config(args.config)
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    if not args.resume and _cache_path().exists():
        _cache_path().unlink()
    try:
        out = run(cfg)
    finally:
        if _POOL is not None:
            _POOL.close()
    showcase = out.pop("_showcase")
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_ROOT / "closed_loop_metrics.json", "w") as fh:
        json.dump(out, fh, indent=1)
    with open(RESULTS_ROOT / "closed_loop_showcase.json", "w") as fh:
        json.dump(showcase, fh)
    log("wrote closed_loop_metrics.json and closed_loop_showcase.json")


if __name__ == "__main__":
    main()
