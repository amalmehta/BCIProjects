"""Result figures (PNG, 200 dpi) from results/metrics.json and the saved example percepts."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from .plotting import COLORS, FIGSIZE, TEXT_2, savefig, setup_style
from .summarize import ENCODER_LABELS

ORDER = ["intensity_uncalibrated", "intensity", "edge", "salient_edge", "learned", "learned_adapted",
         "per_image_optimisation", "learned_dropout_aug", "learned_threshold_aware_noaug", "learned_threshold_aware"]
_PALETTE = COLORS + ["#7a5c2e", "#5f6b7a"]
ENC_COLOR = {k: _PALETTE[i] for i, k in enumerate(ORDER)}
ENC_COLOR["learned_threshold_aware"] = "#111111"
DATASET_LABELS = {"mnist": "MNIST digits", "letters": "EMNIST letters", "cifar": "CIFAR-10 (grayscale)"}


def _main_bars(m: dict, fig_dir: Path) -> None:
    import matplotlib.pyplot as plt

    ds = list(m["main"])
    metrics = [("acc", "Recognition accuracy"), ("ssim_lowpass", "Low-pass SSIM"), ("ssim", "SSIM")]
    fig, axes = plt.subplots(len(metrics), len(ds), figsize=(FIGSIZE[0] * 1.3, 2.1 * len(metrics)), squeeze=False)
    for j, d in enumerate(ds):
        enc = m["main"][d]["encoders"]
        for i, (key, label) in enumerate(metrics):
            ax = axes[i, j]
            names = [k for k in ORDER if k in enc and enc[k].get(key) is not None]
            vals = [enc[k][key] for k in names]
            errs = [enc[k].get(key + "_std", 0.0) for k in names]
            ypos = np.arange(len(names))[::-1]
            ax.barh(ypos, vals, xerr=errs, color=[ENC_COLOR[k] for k in names], height=0.7,
                    error_kw={"ecolor": TEXT_2, "lw": 1, "capsize": 2})
            if key == "acc":
                ax.axvline(m["main"][d]["image_ceiling_acc"], color=TEXT_2, ls=":", lw=1)
            for y, v in zip(ypos, vals):
                ax.text(v, y, f" {v:.2f}", va="center", fontsize=7, color=TEXT_2)
            ax.set_yticks(ypos)
            ax.set_yticklabels([ENCODER_LABELS[k] for k in names] if j == 0 else [], fontsize=7)
            ax.set_xlim(0, 1.12 if key != "ssim" else max(vals) * 1.25 + 0.02)
            ax.grid(axis="y", visible=False)
            if i == 0:
                ax.set_title(DATASET_LABELS.get(d, d))
            ax.set_xlabel(label, fontsize=8)
    savefig(fig, fig_dir / "fig_main_results.png")


def _dropout(m: dict, fig_dir: Path) -> None:
    """Failure curves: rows = unannounced / known-dead; columns = datasets; metric = accuracy and low-pass SSIM."""
    import matplotlib.pyplot as plt

    ds = list(m["robustness"])
    show = ["intensity", "edge", "learned", "learned_adapted", "learned_dropout_aug", "learned_threshold_aware_noaug",
            "learned_threshold_aware"]
    blocks = [("robustness", "acc", "Unannounced: accuracy"), ("robustness_known_dead", "acc", "Known-dead: accuracy"),
              ("robustness", "ssim_lowpass", "Unannounced: low-pass SSIM"),
              ("robustness_known_dead", "ssim_lowpass", "Known-dead: low-pass SSIM")]
    blocks = [b for b in blocks if b[0] in m]
    fig, axes = plt.subplots(len(blocks), len(ds), figsize=(FIGSIZE[0] * 1.3, 1.9 * len(blocks)), squeeze=False)
    for i, (src, key, lab) in enumerate(blocks):
        for j, d in enumerate(ds):
            ax = axes[i, j]
            for k in show:
                if k not in m[src].get(d, {}):
                    continue
                lv = sorted(m[src][d][k], key=float)
                ax.plot([100 * float(v) for v in lv], [m[src][d][k][v][key] for v in lv], "o-",
                        color=ENC_COLOR[k], ms=3, lw=1.4, label=ENCODER_LABELS[k])
            if i == 0:
                ax.set_title(DATASET_LABELS.get(d, d))
            if i == len(blocks) - 1:
                ax.set_xlabel("Extra electrodes failed (%)")
            if j == 0:
                ax.set_ylabel(lab, fontsize=7)
    axes[0, -1].legend(frameon=False, fontsize=6, loc="lower left")
    savefig(fig, fig_dir / "fig_dropout_robustness.png")


def _budget(m: dict, fig_dir: Path) -> None:
    """Sub-threshold share of active currents and charge per frame, by encoder and dataset."""
    import matplotlib.pyplot as plt

    show = ["intensity", "learned", "learned_adapted", "learned_threshold_aware_noaug", "learned_threshold_aware"]
    ds = list(m["main"])
    fig, axes = plt.subplots(1, 3, figsize=(FIGSIZE[0] * 1.2, FIGSIZE[1] * 0.85))
    w = 0.8 / len(show)
    for ax, key, lab in ((axes[0], "frac_active_subthreshold", "Active currents below threshold"),
                         (axes[1], "charge_nc_per_phase", "Charge per frame (nC/phase)"),
                         (axes[2], "acc", "Recognition accuracy")):
        for i, k in enumerate(show):
            xs = [j + (i - (len(show) - 1) / 2) * w for j in range(len(ds))]
            ys = [m["main"][d]["encoders"].get(k, {}).get(key, np.nan) for d in ds]
            ax.bar(xs, ys, width=w, color=ENC_COLOR[k], label=ENCODER_LABELS[k])
        ax.set_xticks(range(len(ds)))
        ax.set_xticklabels([DATASET_LABELS.get(d, d).split(" ")[0] for d in ds], fontsize=8)
        ax.set_title(lab, fontsize=9)
        ax.grid(axis="x", visible=False)
    axes[0].legend(frameon=False, fontsize=6, loc="upper right")
    savefig(fig, fig_dir / "fig_threshold_budget.png")


def _temporal(m: dict, fig_dir: Path) -> None:
    """Per-frame low-pass SSIM over a 2.7 s sequence under habituation (static and moving images)."""
    import matplotlib.pyplot as plt

    t = m.get("temporal")
    if not t:
        return
    names = {"intensity": ("Intensity (frame by frame)", ENC_COLOR["intensity"], "-"),
             "learned": ("Learned generic (frame by frame)", ENC_COLOR["learned"], "-"),
             "learned_threshold_aware": ("Threshold-aware (frame by frame)", "#777777", "-"),
             "threshold_aware_compensated": ("Threshold-aware + trace compensation", COLORS[3], "--"),
             "learned_temporal": ("Temporal encoder (BPTT)", "#111111", "-")}
    ds = list(t)
    fig, axes = plt.subplots(len(ds), 2, figsize=(FIGSIZE[0], 2.0 * len(ds)), squeeze=False)
    for i, d in enumerate(ds):
        dt = t[d]["frame_s"]
        for j, mode in enumerate(("static", "moving")):
            ax = axes[i, j]
            for k, (lab, c, ls) in names.items():
                r = t[d]["modes"][mode].get(k)
                if r:
                    ax.plot(dt * np.arange(1, len(r["ssim_lowpass"]) + 1), r["ssim_lowpass"], ls, color=c, lw=1.5, label=lab)
            ax.set_title(f"{DATASET_LABELS.get(d, d)}, {mode}", fontsize=9)
            if i == len(ds) - 1:
                ax.set_xlabel("Time (s), 166 ms frames")
            if j == 0:
                ax.set_ylabel("Low-pass SSIM")
    axes[0, 1].legend(frameon=False, fontsize=6)
    savefig(fig, fig_dir / "fig_temporal_habituation.png")


def _adapt(m: dict, fig_dir: Path) -> None:
    import matplotlib.pyplot as plt

    rows = [r for r in m.get("adaptation_shots", []) if r["mode"] != "generic"]
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(FIGSIZE[0] * 0.6, FIGSIZE[1] * 0.9))
    for c, mode, lab in ((COLORS[5], "adapter", "Per-electrode adapter (2N params)"), (COLORS[1], "full", "Full fine-tune")):
        rr = sorted([r for r in rows if r["mode"] == mode], key=lambda r: r["shots"])
        ax.errorbar([r["shots"] for r in rr], [r["ssim_gain"] for r in rr], yerr=[r["ssim_gain_std"] for r in rr],
                    fmt="o-", color=c, ms=4, capsize=2, lw=1.6, label=lab)
    ax.axhline(0, color=TEXT_2, lw=0.8)
    ax.set_xscale("log")
    ax.set_xlabel("Unlabeled images used for adaptation")
    ax.set_ylabel("SSIM gain over generic encoder")
    ax.set_title("Few-shot patient adaptation")
    ax.legend(frameon=False, fontsize=7)
    savefig(fig, fig_dir / "fig_adaptation_shots.png")


def _ablations(m: dict, fig_dir: Path) -> None:
    import matplotlib.pyplot as plt

    ab = m.get("ablations")
    if not ab:
        return
    rows = ab["rows"]
    lay_order = ["utah_96", "threads_64", "threads_256", "utah_1024", "main"]
    lay_label = {"utah_96": "Utah\n96", "threads_64": "Threads\n64", "threads_256": "Threads\n256",
                 "utah_1024": "16×Utah\n1024", "main": "Threads\n1024"}
    fig, axes = plt.subplots(1, 2, figsize=(FIGSIZE[0], FIGSIZE[1] * 0.9))
    for ax, key, lab in ((axes[0], "acc", "Recognition accuracy"), (axes[1], "ssim_lowpass", "Low-pass SSIM")):
        for enc, c, off in (("intensity", ENC_COLOR["intensity"], -0.18), ("learned", ENC_COLOR["learned"], 0.18)):
            xs, ys = [], []
            for i, v in enumerate(lay_order):
                r = [q for q in rows if q["variant"] == v and q["encoder"] == enc]
                if r:
                    xs.append(i + off); ys.append(np.mean([q[key] for q in r]))
            ax.bar(xs, ys, width=0.34, color=c, label=ENCODER_LABELS[enc])
        ax.set_xticks(range(len(lay_order)))
        ax.set_xticklabels([lay_label[v] for v in lay_order], fontsize=7)
        ax.set_ylabel(lab); ax.set_ylim(0, 1.05); ax.grid(axis="x", visible=False)
    axes[0].legend(frameon=False, fontsize=7, loc="upper left")
    axes[0].set_title("Electrode count and layout (MNIST)")
    savefig(fig, fig_dir / "fig_ablation_layouts.png")


def _examples(cfg: dict, fig_dir: Path) -> None:
    import matplotlib.pyplot as plt

    ev = Path(cfg["results_dir"]) / "eval"
    cols = ["input", "intensity", "edge", "salient_edge", "learned", "learned_adapted", "learned_threshold_aware"]
    titles = {"input": "Input", **ENCODER_LABELS, "learned_adapted": "Learned +\nadapter",
              "learned_threshold_aware": "Threshold-\naware"}
    ds = [d for d in cfg["datasets"] if (ev / f"{d}_examples.pt").exists()]
    if not ds:
        return
    n_per = 3
    fig, axes = plt.subplots(len(ds) * n_per, len(cols), figsize=(FIGSIZE[0], 1.25 * n_per * len(ds)))
    for di, d in enumerate(ds):
        ex = torch.load(ev / f"{d}_examples.pt")
        for i in range(n_per):
            for j, c in enumerate(cols):
                ax = axes[di * n_per + i, j]
                if c not in ex:
                    ax.axis("off")
                    continue
                ax.imshow(ex[c][i, 0].numpy(), cmap="gray", vmin=0, vmax=1)
                ax.set_xticks([]); ax.set_yticks([])
                for s in ax.spines.values():
                    s.set_visible(False)
                if di == 0 and i == 0:
                    ax.set_title(titles[c].replace(" (", "\n("), fontsize=7)
    savefig(fig, fig_dir / "fig_examples.png")


def _training(cfg: dict, fig_dir: Path) -> None:
    import matplotlib.pyplot as plt

    runs = Path(cfg["results_dir"]) / "runs"
    fig, axes = plt.subplots(1, len(cfg["datasets"]), figsize=(FIGSIZE[0] * 1.2, FIGSIZE[1] * 0.8), squeeze=False)
    for j, d in enumerate(cfg["datasets"]):
        ax = axes[0, j]
        for s, c in zip(cfg["seeds"], COLORS):
            f = runs / f"{d}_main_s{s}" / "train_log.json"
            if not f.exists():
                continue
            h = json.loads(f.read_text())["history"]
            ax.plot([r["step"] for r in h], [r["loss"] for r in h], color=c, lw=1.4, label=f"seed {s}")
        ax.set_title(DATASET_LABELS.get(d, d)); ax.set_xlabel("Step")
        if j == 0:
            ax.set_ylabel("Training loss")
    axes[0, 0].legend(frameon=False, fontsize=7)
    savefig(fig, fig_dir / "fig_training_curves.png")


def make_figures(cfg: dict, log=print) -> None:
    setup_style()
    fig_dir = Path(cfg["results_dir"]) / "figures"
    m = json.loads((Path(cfg["results_dir"]) / "metrics.json").read_text())
    _main_bars(m, fig_dir)
    _dropout(m, fig_dir)
    _budget(m, fig_dir)
    _temporal(m, fig_dir)
    _adapt(m, fig_dir)
    _ablations(m, fig_dir)
    _examples(cfg, fig_dir)
    _training(cfg, fig_dir)
    log(f"figures written to {fig_dir}")
