"""Simulator validation against published models and psychophysics.

Writes ``results/validation.json`` and figures ``fig_sim_*.png``:

* cortical magnification of the wedge-dipole map vs. Horton & Hoyt (1991);
* phosphene size vs. eccentricity and current (Tehovnik current spread; constant
  cortical extent, Winawer & Parvizi 2016), with the Bosking et al. (2017) saturating
  size model for comparison;
* brightness vs. current against Fernandez et al. (2021, Fig. 6A, values digitised by
  the dynaphos authors and distributed with their code);
* distribution of single-electrode threshold currents vs. Fernandez et al. (2021);
* habituation under repeated stimulation (protocol of Schmidt et al. 1996 as used in
  van der Grinten et al. 2024, Fig. 4);
* electrode layouts on cortex and in the visual field.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from .cortex import horton_hoyt_magnification
from .layouts import make_layout
from .plotting import COLORS, FIGSIZE, savefig, setup_style
from .simulator import PhospheneSimulator, SimParams, TemporalSimulator, sample_patient

# Fernandez et al. (2021) Fig. 6A, relative brightness vs. amplitude, as digitised (visual
# inspection) in the dynaphos repository: test/data/Fernandez_2021_fig6A.csv.
FERNANDEZ_6A_UA = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100], dtype=float)
FERNANDEZ_6A_BRIGHTNESS = np.array([0.04, 0.08, 0.2, 0.36, 0.48, 0.8, 0.8, 0.8, 1.0, 0.92])
# Fernandez et al. (2021) abstract: mean single-electrode threshold 66.8 +/- 36.5 uA.
FERNANDEZ_THRESHOLD_UA = (66.8, 36.5)


def bosking_size_mm(i_ua: np.ndarray, md: float = 0.7, i_half: float = 40.0, slope: float = 0.08) -> np.ndarray:
    """Saturating activated-cortex diameter (mm), the Bosking et al. (2017)-based option in dynaphos."""
    return md / (1 + np.exp(-slope * (np.asarray(i_ua) - i_half)))


def run_validation(cfg: dict, log=print) -> dict:
    import matplotlib.pyplot as plt

    setup_style()
    out_dir = Path(cfg["results_dir"])
    fig_dir = out_dir / "figures"
    p = SimParams(fov_deg=cfg["sim"]["fov_deg"], resolution=cfg["sim"]["resolution"])
    sim = PhospheneSimulator(p)
    model = p.model
    res: dict = {}

    # 1. Cortical magnification ------------------------------------------------------------
    ecc = np.linspace(0.1, 40, 400)
    m_ours, m_hh = model.magnification(ecc), horton_hoyt_magnification(ecc)
    res["magnification_max_rel_diff_0_to_10deg"] = float(np.max(np.abs(m_ours - m_hh)[ecc <= 10] / m_hh[ecc <= 10]))
    fig, axes = plt.subplots(1, 2, figsize=(FIGSIZE[0], FIGSIZE[1] * 0.9))
    ax = axes[0]
    ax.loglog(ecc, m_hh, color=COLORS[1], lw=2.5, label="Horton & Hoyt 1991")
    ax.loglog(ecc, m_ours, color=COLORS[0], lw=1.5, ls="--", label="Wedge-dipole (ours)")
    ax.set_xlabel("Eccentricity (deg)"); ax.set_ylabel("Magnification (mm/deg)")
    ax.set_title("Cortical magnification"); ax.legend(frameon=False)
    ax = axes[1]
    for e in [0.5, 1, 2, 4, 8, 16]:
        t = np.linspace(-np.pi / 2, np.pi / 2, 100)
        x, y = model.to_cortex(np.full_like(t, e), t)
        ax.plot(x, y, color=COLORS[0], lw=1)
        ax.text(x[50], y[-1] + 1.5, f"{e:g}°", ha="center", fontsize=8, color=COLORS[0])
    for a in np.radians([-90, -45, 0, 45, 90]):
        r = np.linspace(0.05, 16, 200)
        x, y = model.to_cortex(r, np.full_like(r, a * 0.999))
        ax.plot(x, y, color=COLORS[1], lw=1)
    ax.set_aspect("equal"); ax.set_xlabel("Cortical x (mm)"); ax.set_ylabel("Cortical y (mm)")
    ax.set_title("Iso-eccentricity / iso-angle lines on V1")
    savefig(fig, fig_dir / "fig_sim_magnification.png")

    # 2. Phosphene size --------------------------------------------------------------------
    ecc = np.linspace(0.2, 10, 200)
    fig, axes = plt.subplots(1, 2, figsize=(FIGSIZE[0], FIGSIZE[1] * 0.9))
    ax = axes[0]
    res["phosphene_diameter_deg"] = {}
    for c, amp in zip(COLORS, [50, 80, 100]):
        d_mm = 2 * np.sqrt(amp * 1e-6 / p.current_spread_a_per_mm2)
        size = d_mm / model.magnification(ecc)
        ax.plot(ecc, size, color=c, lw=2, label=f"{amp} µA (cortex {d_mm:.2f} mm)")
        res["phosphene_diameter_deg"][f"{amp}uA"] = {f"{e}deg": float(d_mm / model.magnification(e)) for e in (1, 2, 5)}
    ax.set_xlabel("Eccentricity (deg)"); ax.set_ylabel("Phosphene diameter (deg)")
    ax.set_title("Size grows with eccentricity"); ax.legend(frameon=False, fontsize=8)
    ax = axes[1]
    amps = np.linspace(0, 150, 300)
    ax.plot(amps, 2 * np.sqrt(amps * 1e-6 / p.current_spread_a_per_mm2), color=COLORS[0], lw=2,
            label="sqrt(I/K) spread (Tehovnik)")
    ax.plot(amps, bosking_size_mm(amps), color=COLORS[1], lw=2, ls="--", label="Saturating (Bosking-based)")
    ax.set_xlabel("Current (µA)"); ax.set_ylabel("Activated cortex diameter (mm)")
    ax.set_title("Size vs. current"); ax.legend(frameon=False, fontsize=8)
    savefig(fig, fig_dir / "fig_sim_size.png")

    # 3. Brightness vs. current --------------------------------------------------------------
    amps_t = torch.linspace(0, 120, 241)
    act = sim.activation(amps_t)
    bright = torch.sigmoid(p.brightness_slope * (act - p.a50)) * (act > p.threshold_mean).float()
    model_at = np.interp(FERNANDEZ_6A_UA, amps_t.numpy(), bright.numpy())
    ss_res = np.sum((FERNANDEZ_6A_BRIGHTNESS - model_at) ** 2)
    ss_tot = np.sum((FERNANDEZ_6A_BRIGHTNESS - FERNANDEZ_6A_BRIGHTNESS.mean()) ** 2)
    res["brightness_r2_vs_fernandez_fig6a"] = float(1 - ss_res / ss_tot)
    # Threshold distribution (current needed to reach each electrode's activation threshold).
    rng = np.random.default_rng(0)
    thr_a = np.clip(rng.normal(p.threshold_mean, p.threshold_sd, 20000), 0, None)
    thr_ua = p.threshold_current_ua(thr_a)
    res["threshold_current_ua"] = {"mean": float(thr_ua.mean()), "sd": float(thr_ua.std()),
                                   "p5": float(np.percentile(thr_ua, 5)), "p95": float(np.percentile(thr_ua, 95)),
                                   "fernandez2021_mean_sd": list(FERNANDEZ_THRESHOLD_UA)}
    fig, axes = plt.subplots(1, 2, figsize=(FIGSIZE[0], FIGSIZE[1] * 0.9))
    ax = axes[0]
    ax.plot(amps_t, bright, color=COLORS[0], lw=2, label="Model (mean threshold)")
    ax.scatter(FERNANDEZ_6A_UA, FERNANDEZ_6A_BRIGHTNESS, color=COLORS[1], zorder=3, s=22,
               label="Fernández 2021 Fig. 6A")
    ax.set_xlabel("Current (µA), 166 ms train"); ax.set_ylabel("Relative brightness")
    ax.set_title(f"Brightness (R² = {res['brightness_r2_vs_fernandez_fig6a']:.2f})"); ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax = axes[1]
    ax.hist(thr_ua, bins=60, color=COLORS[0], alpha=0.8, density=True, label="Model electrodes")
    m, s = FERNANDEZ_THRESHOLD_UA
    ax.axvline(m, color=COLORS[1], lw=2, label=f"Fernández 2021 mean {m} µA")
    ax.axvspan(m - s, m + s, color=COLORS[1], alpha=0.12, label="± 1 sd")
    ax.set_xlabel("Threshold current (µA)"); ax.set_ylabel("Density"); ax.set_title("Detection thresholds")
    ax.legend(frameon=False, fontsize=8)
    savefig(fig, fig_dir / "fig_sim_brightness_threshold.png")

    # 4. Habituation -------------------------------------------------------------------------
    hp = SimParams(pulse_width_s=100e-6, frequency_hz=200.0)
    tsim = TemporalSimulator(hp, 1, dt_s=0.125 / 5, substeps=4)
    peaks, times, t_now = [], [], 0.0
    schedule = [4.0] * 50 + [200.0] * 5
    for gap in schedule:
        a_peak = 0.0
        for _ in range(5):  # 125 ms train
            a_peak = max(a_peak, float(tsim.step(torch.tensor([90.0]))[0]))
        peaks.append(float(torch.sigmoid(torch.tensor(hp.brightness_slope * (a_peak - hp.a50)))))
        times.append(t_now)
        # Rest until the next train (no stimulation): integrate in coarse steps.
        rest = TemporalSimulator(hp, 1, dt_s=gap, substeps=200)
        rest.act, rest.trace = tsim.act, tsim.trace
        rest.step(torch.tensor([0.0]))
        tsim.act, tsim.trace = rest.act, rest.trace
        t_now += 0.125 + gap
    peaks = np.array(peaks)
    res["habituation"] = {"first": float(peaks[0]), "after_50_trains": float(peaks[49]),
                          "after_recovery": float(peaks[-1])}
    fig, ax = plt.subplots(figsize=(FIGSIZE[0] * 0.6, FIGSIZE[1] * 0.85))
    ax.plot(np.arange(len(peaks)), peaks / peaks[0], "o-", color=COLORS[0], ms=3)
    ax.axvline(49.5, color="0.6", ls=":")
    ax.text(51, 0.95 * peaks.max() / peaks[0], "200 s gaps", fontsize=8, color="0.4")
    ax.set_xlabel("Pulse train #  (4 s apart, then 200 s)"); ax.set_ylabel("Relative peak brightness")
    ax.set_title("Habituation to repeated stimulation")
    savefig(fig, fig_dir / "fig_sim_habituation.png")

    # 5. Layouts -----------------------------------------------------------------------------
    names = ["utah_96", "utah_1024", "threads_256", "threads_1024"]
    labels = {"utah_96": "Utah array, 96 ch", "utah_1024": "16 Utah arrays, 1024 ch",
              "threads_256": "Threads, 256 sites", "threads_1024": "Threads, 1024 sites"}
    fig, axes = plt.subplots(2, 4, figsize=(FIGSIZE[0] * 1.35, FIGSIZE[1] * 1.5))
    res["layouts"] = {}
    for j, name in enumerate(names):
        lay = make_layout(name, cfg["layout"]["ecc_min"], cfg["layout"]["ecc_max"], cfg["layout"]["seed"])
        vf = lay.visual_field(model)
        ax = axes[0, j]
        for h, c in ((0, COLORS[0]), (1, COLORS[1])):
            sel = lay.hemisphere == h
            ax.scatter(lay.cortex_x_mm[sel], lay.cortex_y_mm[sel], s=2, color=c,
                       label="left V1 (right field)" if h == 0 else "right V1 (left field)")
        if j == len(names) - 1:
            ax.legend(frameon=True, fontsize=7, loc="lower right", markerscale=3)
        ax.set_title(labels[name], fontsize=9); ax.set_aspect("equal")
        ax.set_xlabel("Cortical x (mm)", fontsize=8)
        if j == 0:
            ax.set_ylabel("Cortical y (mm)", fontsize=8)
        ax = axes[1, j]
        pat = sample_patient(lay, p, 1000)
        amp = torch.full((1, lay.n), 80.0)
        img = sim(amp, pat)[0, 0]
        half = p.fov_deg / 2
        ax.imshow(img, cmap="gray", extent=[-half, half, -half, half], vmin=0, vmax=1)
        ax.set_xlabel("Visual field x (deg)", fontsize=8)
        if j == 0:
            ax.set_ylabel("y (deg)", fontsize=8)
        res["layouts"][name] = {"n": lay.n, "ecc_min": float(vf["ecc"].min()), "ecc_max": float(vf["ecc"].max()),
                                "frac_field_within_0.5deg": _coverage(vf["x"], vf["y"], p.fov_deg)}
    savefig(fig, fig_dir / "fig_sim_layouts.png")

    (out_dir / "validation.json").write_text(json.dumps(res, indent=1))
    log(f"validation: {json.dumps(res)[:600]}")
    return res


def _coverage(x: np.ndarray, y: np.ndarray, fov: float, radius: float = 0.5, n: int = 120) -> float:
    """Fraction of the circular field of view within ``radius`` deg of some phosphene centre."""
    c = (np.arange(n) + 0.5) / n * fov - fov / 2
    gx, gy = np.meshgrid(c, c)
    inside = gx ** 2 + gy ** 2 <= (fov / 2) ** 2
    pts = np.stack([gx[inside], gy[inside]], 1)
    d2 = ((pts[:, None, :] - np.stack([x, y], 1)[None]) ** 2).sum(-1).min(1)
    return float((d2 <= radius ** 2).mean())
