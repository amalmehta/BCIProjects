"""Aggregate evaluation outputs into results/metrics.json (the single source for paper and site)."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

METRICS = ["ssim", "ssim_lowpass", "psnr", "acc", "decoder_ssim", "active_electrodes", "charge_nc_per_phase",
           "frac_active_subthreshold", "wasted_charge_frac", "median_active_current_ua"]
V1_DIR = "v1_dynaphos_thresholds"  # first-round results (original dynaphos threshold fit), archived
ENCODER_LABELS = {
    "intensity_uncalibrated": "Intensity (uncalibrated)",
    "intensity": "Intensity (calibrated)",
    "edge": "Edges (Sobel)",
    "salient_edge": "Salient edges",
    "learned": "Learned (generic)",
    "learned_adapted": "Learned + patient adapter",
    "per_image_optimisation": "Per-image optimisation",
    "learned_threshold_aware": "Threshold-aware (+ failure aug.)",
    "learned_threshold_aware_noaug": "Threshold-aware (no aug.)",
    "learned_threshold_aware_noisy_thresholds": "Threshold-aware, 20% threshold error",
    "learned_threshold_aware_noaug_noisy_thresholds": "Threshold-aware (no aug.), 20% threshold error",
    "learned_dropout_aug": "Learned + failure aug.",
}


def _agg(rows: list[dict], keys: list[str]) -> dict:
    out = {"n_seeds": len(rows)}
    for k in keys:
        vals = [r[k] for r in rows if r.get(k) is not None]
        if vals:
            out[k] = float(np.mean(vals))
            out[k + "_std"] = float(np.std(vals)) if len(vals) > 1 else 0.0
    if len(rows) == 1 and "ssim_patient_std" in rows[0]:
        out["ssim_patient_std"] = rows[0]["ssim_patient_std"]
    return out


def summarize(cfg: dict, log=print) -> dict:
    res_dir = Path(cfg["results_dir"])
    ev = res_dir / "eval"
    out: dict = {"note": "All numbers produced by blindsight/run.py with configs/cpu.yaml on CPU. "
                         "Learned rows: mean and std over training seeds; each value itself averages 5 held-out "
                         "simulated patients. Baseline rows are deterministic (std over patients in ssim_patient_std).",
                 "encoder_labels": ENCODER_LABELS, "main": {}, "robustness": {}}
    for d in cfg["datasets"]:
        f = ev / f"{d}.json"
        if not f.exists():
            continue
        data = json.loads(f.read_text())
        groups: dict[str, list] = defaultdict(list)
        for r in data["rows"]:
            groups[r["encoder"]].append(r)
        out["main"][d] = {"layout": data["layout"], "n_electrodes": data["n_electrodes"],
                          "image_ceiling_acc": data["image_ceiling_acc"],
                          "encoders": {k: _agg(v, METRICS) for k, v in groups.items()}}
        rob = {}
        for k, v in groups.items():
            if "dropout" not in v[0]:
                continue
            rob[k] = {"0.0": {m: float(np.mean([r[m] for r in v])) for m in ("ssim", "ssim_lowpass", "acc")}}
            rob[k].update({lvl: {m: float(np.mean([r["dropout"][lvl][m] for r in v]))
                                 for m in ("ssim", "ssim_lowpass", "acc")} for lvl in v[0]["dropout"]})
        out["robustness"][d] = rob
        robk = {}
        for k, v in groups.items():
            if "dropout_known" not in v[0]:
                continue
            robk[k] = {"0.0": rob[k]["0.0"]}
            robk[k].update({lvl: {m: float(np.mean([r["dropout_known"][lvl][m] for r in v]))
                                  for m in ("ssim", "ssim_lowpass", "acc")} for lvl in v[0]["dropout_known"]})
        out.setdefault("robustness_known_dead", {})[d] = robk
        out.setdefault("budget_diagnostics", {})[d] = {
            k: {"frac_active_above_threshold": float(np.mean([r["frac_active_above_threshold"] for r in v])),
                "frac_active_subthreshold": float(np.mean([r["frac_active_subthreshold"] for r in v])),
                "wasted_charge_frac": float(np.mean([r["wasted_charge_frac"] for r in v])),
                "median_active_current_ua": float(np.mean([r["median_active_current_ua"] for r in v]))}
            for k, v in groups.items() if "frac_active_subthreshold" in v[0]}
    v1 = res_dir / V1_DIR
    f = ev / "ablations.json"
    if not f.exists() and (v1 / "eval" / "ablations.json").exists():
        f = v1 / "eval" / "ablations.json"
    if f.exists():
        out["ablations"] = json.loads(f.read_text())
        out["ablations"]["simulator"] = "v1_dynaphos_thresholds" if V1_DIR in str(f) else "current"
    f = ev / "adapt_shots.json"
    shots_sim = "current"
    if not f.exists() and (v1 / "eval" / "adapt_shots.json").exists():
        f, shots_sim = v1 / "eval" / "adapt_shots.json", "v1_dynaphos_thresholds"
    if f.exists():
        rows = json.loads(f.read_text())["rows"]
        groups = defaultdict(list)
        for r in rows:
            groups[(r["mode"], r["shots"])].append(r)
        generic = {r["patient"]: r["ssim"] for r in rows if r["mode"] == "generic"}
        out["adaptation_shots"] = [
            {"mode": m, "shots": s, "ssim": float(np.mean([r["ssim"] for r in v])),
             "ssim_gain": float(np.mean([r["ssim"] - generic[r["patient"]] for r in v])),
             "ssim_gain_std": float(np.std([r["ssim"] - generic[r["patient"]] for r in v])), "n": len(v)}
            for (m, s), v in sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][1]))]
        out["adaptation_shots_simulator"] = shots_sim
    temporal = {}
    for f in sorted(ev.glob("temporal_*.json")):
        t = json.loads(f.read_text())
        temporal[t["dataset"]] = t
    if temporal:
        out["temporal"] = temporal
    if (v1 / "metrics.json").exists():
        old = json.loads((v1 / "metrics.json").read_text())
        out["v1_dynaphos_thresholds"] = {
            "note": "First-round results with the original dynaphos threshold fit (47 +/- 15 uA), same code and "
                    "protocol otherwise; kept for the old-vs-new comparison. Files in results/" + V1_DIR + "/.",
            "main": old["main"], "robustness": old["robustness"],
            "budget_diagnostics": old.get("budget_diagnostics"), "simulator_validation": old.get("simulator_validation")}
    f = res_dir / "validation.json"
    if f.exists():
        out["simulator_validation"] = json.loads(f.read_text())
    runs = sorted((res_dir / "runs").glob("*/train_log.json"))
    out["training"] = {p.parent.name: {k: json.loads(p.read_text()).get(k) for k in ("steps", "train_time_s", "n_params_encoder")}
                       for p in runs}
    out["config"] = {"layout": cfg["layout"]["name"], "fov_deg": cfg["sim"]["fov_deg"],
                     "threshold_model": cfg["sim"].get("threshold_model", "dynaphos"),
                     "resolution": cfg["sim"]["resolution"], "max_current_ua": cfg["sim"]["max_current_ua"],
                     "max_active_frac": cfg["safety"]["max_active_frac"], "patients": cfg["patients"],
                     "train": cfg["train"], "eval": cfg["eval"], "adapt": cfg["adapt"]}
    (res_dir / "metrics.json").write_text(json.dumps(out, indent=1))
    for d, v in out["main"].items():
        log(f"{d}: " + "; ".join(f"{k} acc={e.get('acc', float('nan')):.3f} ssim_lp={e['ssim_lowpass']:.3f}"
                                  for k, e in v["encoders"].items() if e.get("acc") is not None))
    return out
