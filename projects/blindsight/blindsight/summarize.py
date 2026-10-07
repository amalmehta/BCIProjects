"""Aggregate evaluation outputs into results/metrics.json (the single source for paper and site)."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

METRICS = ["ssim", "ssim_lowpass", "psnr", "acc", "decoder_ssim", "active_electrodes", "charge_nc_per_phase"]
ENCODER_LABELS = {
    "intensity_uncalibrated": "Intensity (uncalibrated)",
    "intensity": "Intensity (calibrated)",
    "edge": "Edges (Sobel)",
    "salient_edge": "Salient edges",
    "learned": "Learned (generic)",
    "learned_adapted": "Learned + patient adapter",
    "per_image_optimisation": "Per-image optimisation",
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
    f = ev / "ablations.json"
    if f.exists():
        out["ablations"] = json.loads(f.read_text())
    f = ev / "adapt_shots.json"
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
    f = ev / "diagnostics.json"
    if f.exists():
        out["budget_diagnostics"] = json.loads(f.read_text())
    f = res_dir / "validation.json"
    if f.exists():
        out["simulator_validation"] = json.loads(f.read_text())
    runs = sorted((res_dir / "runs").glob("*/train_log.json"))
    out["training"] = {p.parent.name: {k: json.loads(p.read_text())[k] for k in ("steps", "train_time_s", "n_params_encoder")}
                       for p in runs}
    out["config"] = {"layout": cfg["layout"]["name"], "fov_deg": cfg["sim"]["fov_deg"],
                     "resolution": cfg["sim"]["resolution"], "max_current_ua": cfg["sim"]["max_current_ua"],
                     "max_active_frac": cfg["safety"]["max_active_frac"], "patients": cfg["patients"],
                     "train": cfg["train"], "eval": cfg["eval"], "adapt": cfg["adapt"]}
    (res_dir / "metrics.json").write_text(json.dumps(out, indent=1))
    for d, v in out["main"].items():
        log(f"{d}: " + "; ".join(f"{k} acc={e.get('acc', float('nan')):.3f} ssim_lp={e['ssim_lowpass']:.3f}"
                                  for k, e in v["encoders"].items() if e.get("acc") is not None))
    return out
