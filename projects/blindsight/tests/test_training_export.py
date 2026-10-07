import json
import math

import numpy as np
import torch

from blindsight.encoders import Decoder, LearnedEncoder
from blindsight.export import export_baseline, export_electrodes
from blindsight.train import end_to_end_loss


def test_tiny_overfit(setup, cfg):
    """A small encoder/decoder should fit 8 fixed images through the simulator."""
    torch.manual_seed(0)
    R = setup.params.resolution
    x = torch.zeros(8, 1, R, R)
    for i in range(8):  # bars at different positions/orientations
        if i % 2:
            x[i, 0, :, 4 + 3 * i // 2: 8 + 3 * i // 2] = 1
        else:
            x[i, 0, 4 + 3 * i // 2: 8 + 3 * i // 2, :] = 1
    enc = LearnedEncoder(setup.context(calibrated=False), width=8)
    dec = Decoder(width=8)
    opt = torch.optim.Adam(list(enc.parameters()) + list(dec.parameters()), lr=5e-3)
    pt = setup.patient(100)
    w = dict(cfg["train"])
    first = None
    for _ in range(80):
        loss, _ = end_to_end_loss(enc, dec, setup, pt, x, w, train=True)
        first = first if first is not None else float(loss)
        opt.zero_grad(); loss.backward(); opt.step()
    assert float(loss) < 0.6 * first


def _render_from_json(doc, amps, row, col):
    c = doc["constants"]
    R = doc["test_case"]["resolution"]
    fov = doc["field_of_view_deg"]
    x = (col + 0.5) / R * fov - fov / 2
    y = fov / 2 - (row + 0.5) / R * fov
    total = 0.0
    for e, amp in zip(doc["electrodes"], amps):
        if not e["alive"]:
            continue
        i_eff = max(0.0, (amp * 1e-6 - c["rheobase_a"]) * c["pulse_width_s"] * c["frequency_hz"])
        act = i_eff * c["activation_gain_s"]
        b = 1 / (1 + math.exp(-c["brightness_slope"] * (act - c["a50"]))) if act > e["threshold_activation"] else 0.0
        r_mm = math.sqrt(max(amp, 0) * 1e-6 / c["current_spread_a_per_mm2"] + 1e-12)
        mag = c["k"] * (1 / (e["ecc_deg"] + c["a"]) - 1 / (e["ecc_deg"] + c["b"]))
        sig = math.sqrt((r_mm / mag) ** 2 + c["sigma_floor_deg"] ** 2)
        total += b * math.exp(-((x - e["x_deg"]) ** 2 + (y - e["y_deg"]) ** 2) / (2 * sig ** 2))
    return min(1.0, max(0.0, total))


def test_exported_render_formula_reproduces_test_case(setup64, tmp_path):
    export_electrodes(setup64, tmp_path)
    doc = json.loads((tmp_path / "electrodes.json").read_text())
    amps = doc["test_case"]["amplitudes_ua"]
    for px in doc["test_case"]["pixels"]:
        got = _render_from_json(doc, amps, px["row"], px["col"])
        assert abs(got - px["value"]) < 2e-3, (px, got)


def test_exported_baseline_spec_reproduces_test_case(setup64, tmp_path):
    export_electrodes(setup64, tmp_path)
    export_baseline(setup64, tmp_path)
    el = json.loads((tmp_path / "electrodes.json").read_text())
    bl = json.loads((tmp_path / "baseline_encoder.json").read_text())
    R, fov, p = el["reference_resolution"], el["field_of_view_deg"], bl["params"]
    rr, cc = np.meshgrid(np.arange(R), np.arange(R), indexing="ij")
    img = np.clip(0.6 * cc / (R - 1) + 0.4 * (((cc - 40) ** 2 + (rr - 24) ** 2) < 100), 0, 1)
    t = np.arange(-3, 4)
    k = np.exp(-t ** 2 / 2.0); k /= k.sum()
    pad = np.pad(img, ((0, 0), (3, 3)), mode="reflect")
    img = sum(k[i] * pad[:, i:i + R] for i in range(7))
    pad = np.pad(img, ((3, 3), (0, 0)), mode="reflect")
    img = sum(k[i] * pad[i:i + R, :] for i in range(7))
    amps = []
    for e in el["electrodes"]:
        col = np.clip((e["nominal_x_deg"] + fov / 2) / fov * R - 0.5, 0, R - 1 - 1e-4)
        row = np.clip((fov / 2 - e["nominal_y_deg"]) / fov * R - 0.5, 0, R - 1 - 1e-4)
        c0, r0 = int(col), int(row); fc, fr = col - c0, row - r0
        c1, r1 = min(c0 + 1, R - 1), min(r0 + 1, R - 1)
        v = (img[r0, c0] * (1 - fr) * (1 - fc) + img[r0, c1] * (1 - fr) * fc
             + img[r1, c0] * fr * (1 - fc) + img[r1, c1] * fr * fc)
        v = min(max(v, 0), 1)
        tn = min(e["threshold_ua"], p["max_current_ua"])
        a = tn + v * (p["max_current_ua"] - tn) if v > p["cutoff"] else 0.0
        amps.append(a * e["alive"])
    amps = np.array(amps)
    kth = np.sort(amps)[::-1][p["max_active_electrodes"] - 1]
    amps = np.where(amps >= kth, amps, 0)
    for i, val in bl["test_case"]["expected_amplitudes_ua"].items():
        assert abs(amps[int(i)] - val) < 0.05, (i, amps[int(i)], val)


def test_javascript_reimplementation_matches(setup64, tmp_path):
    """The website's JavaScript renderer/baseline (tests/js/render_check.mjs) reproduces the exports."""
    import shutil
    import subprocess
    from pathlib import Path

    import pytest

    node = shutil.which("node")
    if node is None:
        pytest.skip("node not installed")
    export_electrodes(setup64, tmp_path)
    export_baseline(setup64, tmp_path)
    script = Path(__file__).parent / "js" / "render_check.mjs"
    out = subprocess.run([node, str(script), str(tmp_path)], capture_output=True, text=True)
    assert out.returncode == 0, out.stdout + out.stderr
