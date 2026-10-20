"""Export small artefacts for the website's in-browser demos (results/site_export/).

* ``electrodes.json``      - per-electrode geometry, thresholds and every simulator constant,
                              the exact rendering formula, and a numeric test case.
* ``baseline_encoder.json`` - the intensity-sampling baseline as an algorithm + parameters + test case.
* ``encoder.onnx`` (+ ``encoder_letters.onnx``) - learned encoders (opset 17) with I/O spec in
                              ``encoder_onnx.json``.
* ``metrics.json``          - encoder x metric tables (copied from results/metrics.json).
* ``gallery/``              - input / baseline percept / learned percept triplets (PNG).
* ``figures/``              - main figures (PNG).
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import torch

from .config import build_setup
from .encoders import BakedThresholdAwareEncoder, IntensityEncoder, ThresholdAwareEncoder
from .train import load_encoder, patient_condition

DEMO_PATIENT_SEED = 1000

RENDER_FORMULA = [
    "Inputs: per-electrode amplitude I_n in microamps (after the safety layer), electrode fields from 'electrodes'.",
    "1. If alive_n == 0 then I_n = 0.",
    "2. Effective current (A): I_eff_n = max(0, (I_n * 1e-6 - rheobase_a) * pulse_width_s * frequency_hz).",
    "3. Tissue activation after one static train (closed form incl. habituation trace): A_n = I_eff_n * activation_gain_s, "
    "activation_gain_s = (exp(-kd * train_s) - exp(-train_s / tau_act_s)) / (1 / tau_act_s - kd), "
    "kd = trace_increase_rate * pulse_width_s * frequency_hz (precomputed value given in constants).",
    "4. Brightness: b_n = A_n > threshold_activation_n ? 1 / (1 + exp(-brightness_slope * (A_n - a50))) : 0.",
    "5. Size: r_mm = sqrt(max(I_n,0) * 1e-6 / current_spread_a_per_mm2 + 1e-12); "
    "M_n = k * (1 / (ecc_deg_n + a) - 1 / (ecc_deg_n + b)); sigma_n = sqrt((r_mm / M_n)^2 + sigma_floor_deg^2) (degrees).",
    "6. Pixel (row i, col j) of an R x R render has centre x = (j + 0.5) / R * fov_deg - fov_deg / 2, "
    "y = fov_deg / 2 - (i + 0.5) / R * fov_deg (degrees; +y is up, row 0 is the top).",
    "7. value(x, y) = min(1, max(0, sum_n b_n * exp(-((x - x_deg_n)^2 + (y - y_deg_n)^2) / (2 sigma_n^2)))).",
    "Gaussian tails below 1e-6 per axis are zeroed in Python (affects values by < 1e-5).",
]


def _round(v, nd=5):
    return float(np.round(float(v), nd))


def export_electrodes(setup, out: Path) -> dict:
    p = setup.params
    lay = setup.layout
    nom = setup.nominal()
    pat = setup.patient(DEMO_PATIENT_SEED)
    vf_ecc = pat.ecc_deg.cpu().numpy()
    x, y = pat.x_deg.cpu().numpy(), pat.y_deg.cpu().numpy()
    thr_ua = pat.threshold_current_ua(p).cpu().numpy()
    sim = setup.sim
    sig_thr = sim.sigma_deg(torch.as_tensor(np.minimum(thr_ua, p.max_current_ua), dtype=torch.float32), pat.ecc_deg.cpu())
    sig_max = sim.sigma_deg(torch.full((lay.n,), p.max_current_ua), pat.ecc_deg.cpu())
    electrodes = []
    for i in range(lay.n):
        electrodes.append({
            "id": i, "implant": int(lay.implant[i]), "hemisphere": "left" if lay.hemisphere[i] == 0 else "right",
            "cortex_x_mm": _round(lay.cortex_x_mm[i], 4), "cortex_y_mm": _round(lay.cortex_y_mm[i], 4),
            "nominal_x_deg": _round(nom.x_deg[i]), "nominal_y_deg": _round(nom.y_deg[i]),
            "x_deg": _round(x[i]), "y_deg": _round(y[i]), "ecc_deg": _round(vf_ecc[i]),
            "polar_angle_deg": _round(np.degrees(np.arctan2(y[i], x[i])), 3),
            "threshold_activation": float(f"{float(pat.threshold[i]):.6e}"), "threshold_ua": _round(thr_ua[i], 3),
            "sigma_deg_at_threshold": _round(sig_thr[i]), "sigma_deg_at_max_current": _round(sig_max[i]),
            "alive": int(pat.alive[i]),
        })
    consts = {k: v for k, v in p.to_dict().items() if k != "model"}
    consts.update({f"{k}": v for k, v in p.to_dict()["model"].items()})
    consts["activation_gain_s"] = p.activation_gain
    # Test case: a deterministic amplitude pattern rendered with the exact Python renderer.
    g = torch.Generator().manual_seed(0)
    amp = (torch.rand(1, lay.n, generator=g) * p.max_current_ua)
    amp = setup.safety(amp)
    with torch.no_grad():
        img = sim(amp.to(setup.device), pat)[0, 0].cpu()
    R = p.resolution
    pix = [(int(R * fr), int(R * fc)) for fr, fc in
           [(0.5, 0.5), (0.16, 0.63), (0.5, 0.31), (0.78, 0.78), (0.08, 0.08), (0.44, 0.56), (0.63, 0.47), (0.31, 0.34)]]
    cases = []
    for (r, c) in pix:
        xd = (c + 0.5) / R * p.fov_deg - p.fov_deg / 2
        yd = p.fov_deg / 2 - (r + 0.5) / R * p.fov_deg
        cases.append({"row": r, "col": c, "x_deg": _round(xd), "y_deg": _round(yd), "value": _round(img[r, c], 6)})
    doc = {
        "description": "Blindsight phosphene simulator, demo patient (seed 1000): electrode geometry and constants "
                       "for rendering phosphenes in the browser. Independent project; not affiliated with Neuralink.",
        "layout": lay.name, "n_electrodes": lay.n, "patient_seed": DEMO_PATIENT_SEED,
        "field_of_view_deg": p.fov_deg, "reference_resolution": R,
        "safety": {"max_current_ua": p.max_current_ua, "max_active_electrodes": setup.safety.max_active},
        "constants": consts,
        "rendering_formula": RENDER_FORMULA,
        "coordinates": "cortex_*: flattened wedge-dipole V1 coordinates (mm), x along the eccentricity axis; "
                       "x_deg/y_deg: the demo patient's true phosphene centres (nominal + 0.4 mm cortical jitter); "
                       "nominal_*: planned centres the encoders use. +x right, +y up, fixation at (0, 0).",
        "electrodes": electrodes,
        "test_case": {"amplitude_rule": "amp_n = safety(U_n * max_current_ua) with U ~ torch.rand(seed 0); "
                                        "the post-safety amplitudes are listed in 'amplitudes_ua'",
                      "amplitudes_ua": [_round(a, 4) for a in amp[0].tolist()],
                      "resolution": R, "pixels": cases, "tolerance": 1e-4},
    }
    (out / "electrodes.json").write_text(json.dumps(doc, separators=(",", ":")))
    return doc


def export_baseline(setup, out: Path) -> None:
    p = setup.params
    pat = setup.patient(DEMO_PATIENT_SEED)
    enc = IntensityEncoder(setup.context(pat))
    R = p.resolution
    yy, xx = torch.meshgrid(torch.arange(R, dtype=torch.float32), torch.arange(R, dtype=torch.float32), indexing="ij")
    test_img = ((xx / (R - 1)) * 0.6 + 0.4 * (((xx - 40) ** 2 + (yy - 24) ** 2) < 100).float()).clamp(0, 1)
    with torch.no_grad():
        amp = setup.safety(enc(test_img.view(1, 1, R, R).to(setup.device)))[0].cpu()
    nz = torch.nonzero(amp).flatten()[:6].tolist() + [int(torch.argmin(amp))]
    doc = {
        "name": "Intensity sampling (threshold-calibrated)",
        "description": "Standard pixelised encoding: blur, sample brightness at each phosphene centre, "
                       "map linearly from the electrode's threshold current to the safety ceiling.",
        "input": {"resolution": R, "field_of_view_deg": p.fov_deg,
                  "preprocess": "Convert to grayscale with luma = 0.299 R + 0.587 G + 0.114 B (values in [0,1]); "
                                "center-crop to square; resize to resolution x resolution with bilinear interpolation. "
                                "The full image spans the field of view."},
        "steps": [
            "1. Gaussian blur, sigma_px = 1.0, kernel radius = ceil(3 * sigma_px) = 3, weights exp(-t^2 / (2 sigma^2)) "
            "normalised to sum 1, applied separably (horizontal then vertical) with reflect padding "
            "(index -1 -> 1, R -> R-2).",
            "2. For electrode n use its nominal centre (nominal_x_deg, nominal_y_deg) from electrodes.json: "
            "col = (x + fov/2) / fov * R - 0.5, row = (fov/2 - y) / fov * R - 0.5, each clamped to [0, R - 1 - 1e-4]; "
            "bilinear interpolation between floor/floor+1 neighbours (clamped to R-1).",
            "3. v_n = clamp(sample, 0, 1). If v_n <= cutoff then amp_n = 0, else "
            "amp_n = t_n + v_n * (max_current_ua - t_n) with t_n = min(threshold_ua_n, max_current_ua).",
            "4. amp_n = 0 for electrodes with alive_n == 0 (known dead after calibration).",
            "5. Safety: clamp to [0, max_current_ua]; keep only the max_active_electrodes largest amplitudes "
            "(ties at the k-th value are kept), zero the rest.",
            "6. Render with the formula in electrodes.json.",
        ],
        "params": {"blur_sigma_px": 1.0, "cutoff": 0.1, "max_current_ua": p.max_current_ua,
                   "max_active_electrodes": setup.safety.max_active, "threshold_source": "electrodes[n].threshold_ua"},
        "test_case": {"image": "img[r][c] = clamp(0.6 * c / (R - 1) + 0.4 * ((c - 40)^2 + (r - 24)^2 < 100 ? 1 : 0), 0, 1), "
                               "R = 64 (already at model resolution, skip the resize)",
                      "expected_amplitudes_ua": {str(i): _round(amp[i], 4) for i in nz},
                      "n_nonzero": int((amp > 0).sum()), "tolerance": 1e-3},
    }
    (out / "baseline_encoder.json").write_text(json.dumps(doc, indent=1))


def export_onnx(cfg: dict, dataset: str, path: Path, log, variant: str = "main") -> dict:
    """Export one seed-0 encoder. A threshold-aware encoder gets the demo patient's calibration
    (thresholds, alive mask, coverage map) baked in as constants, so the graph keeps the generic
    I/O: ``image`` (batch, 1, 64, 64) -> ``amplitude_ua`` (batch, N)."""
    import onnx
    import onnxruntime as ort

    run = Path(cfg["results_dir"]) / "runs" / f"{dataset}_{variant}_s0"
    enc, _, setup = load_encoder(cfg, variant, run)
    conditioned = isinstance(enc, ThresholdAwareEncoder)
    if conditioned:
        enc = BakedThresholdAwareEncoder(enc, patient_condition(setup, setup.patient(DEMO_PATIENT_SEED)))
    enc = enc.cpu().eval()
    R = setup.params.resolution
    dummy = torch.rand(2, 1, R, R)
    torch.onnx.export(enc, dummy, str(path), input_names=["image"], output_names=["amplitude_ua"],
                      dynamic_axes={"image": {0: "batch"}, "amplitude_ua": {0: "batch"}}, opset_version=17)
    onnx.checker.check_model(onnx.load(str(path)))
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    with torch.no_grad():
        ref = enc(dummy).numpy()
    got = sess.run(None, {"image": dummy.numpy()})[0]
    err = float(np.abs(ref - got).max())
    log(f"onnx {path.name}: {path.stat().st_size / 1e3:.0f} kB, max |torch - onnxruntime| = {err:.2e} uA")
    ops = sorted({n.op_type for n in onnx.load(str(path)).graph.node})
    return {"file": path.name, "trained_on": dataset, "seed": 0, "variant": variant, "size_bytes": path.stat().st_size,
            "max_abs_diff_vs_torch_ua": err, "onnx_ops": ops, "opset": 17,
            "conditioning": (f"threshold-aware encoder; the demo patient's (seed {DEMO_PATIENT_SEED}) calibrated "
                             "thresholds and alive mask from electrodes.json are baked into the graph as constants. "
                             "Electrodes failed later in the browser are therefore unannounced to the encoder.")
            if conditioned else "generic encoder: image only, no patient calibration"}


def export_gallery(cfg: dict, out: Path) -> None:
    from PIL import Image

    gal = out / "gallery"
    gal.mkdir(parents=True, exist_ok=True)
    index = []
    ev = Path(cfg["results_dir"]) / "eval"
    for d in cfg["datasets"]:
        f = ev / f"{d}_examples.pt"
        if not f.exists():
            continue
        ex = torch.load(f)
        for i in range(4):
            tiles = [ex["input"][i, 0], ex["intensity"][i, 0], ex["learned"][i, 0]]
            strip = torch.cat([torch.nn.functional.pad(t, (2, 2, 2, 2), value=0.25) for t in tiles], 1)
            arr = (strip.clamp(0, 1).numpy() * 255).astype(np.uint8)
            img = Image.fromarray(arr).resize((arr.shape[1] * 2, arr.shape[0] * 2), Image.NEAREST)
            name = f"{d}_{i}.png"
            img.save(gal / name, optimize=True)
            index.append({"file": name, "dataset": d, "panels": ["input", "intensity baseline percept",
                                                                 "learned encoder percept"]})
    (gal / "index.json").write_text(json.dumps(index, indent=1))


def export_site(cfg: dict, log=print) -> None:
    out = Path(cfg["results_dir"]) / "site_export"
    out.mkdir(parents=True, exist_ok=True)
    setup = build_setup(cfg)
    export_electrodes(setup, out)
    export_baseline(setup, out)
    onnx_info = []
    variant = cfg.get("export", {}).get("onnx_variant", "main")
    for d, name in (("cifar", "encoder.onnx"), ("letters", "encoder_letters.onnx")):
        if (Path(cfg["results_dir"]) / "runs" / f"{d}_{variant}_s0" / "encoder.pt").exists():
            onnx_info.append(export_onnx(cfg, d, out / name, log, variant))
    spec = {
        "models": onnx_info,
        "input": {"name": "image", "shape": ["batch", 1, setup.params.resolution, setup.params.resolution],
                  "dtype": "float32", "range": "[0, 1] grayscale, same preprocessing as baseline_encoder.json "
                                               "(luma, square crop, bilinear resize); full image spans the field of view"},
        "output": {"name": "amplitude_ua", "shape": ["batch", setup.n], "dtype": "float32",
                   "meaning": "stimulation amplitude per electrode (microamps), electrode order as in electrodes.json"},
        "postprocess": "Apply the safety layer (clamp to [0, max_current_ua], keep top max_active_electrodes), "
                       "zero electrodes with alive == 0, then render with electrodes.json's formula.",
        "runtime": "onnxruntime-web (wasm backend); operator list per model under models[].onnx_ops",
    }
    (out / "encoder_onnx.json").write_text(json.dumps(spec, indent=1))
    metrics = Path(cfg["results_dir"]) / "metrics.json"
    if metrics.exists():
        shutil.copy(metrics, out / "metrics.json")
    export_gallery(cfg, out)
    figs = out / "figures"
    figs.mkdir(exist_ok=True)
    for f in sorted((Path(cfg["results_dir"]) / "figures").glob("*.png")):
        shutil.copy(f, figs / f.name)
    total = sum(f.stat().st_size for f in out.rglob("*") if f.is_file())
    log(f"site export: {total / 1e6:.2f} MB in {out}")
