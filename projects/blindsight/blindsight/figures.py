"""Result figures (PNG, 200 dpi) from results/metrics.json and the saved example percepts."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from .plotting import COLORS, FIGSIZE, TEXT_2, savefig, setup_style
from .summarize import ENCODER_LABELS

ORDER = ["intensity_uncalibrated", "intensity", "edge", "salient_edge", "learned", "learned_adapted",
         "per_image_optimisation"]
ENC_COLOR = {k: COLORS[i] for i, k in enumerate(ORDER)}
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
    import matplotlib.pyplot as plt

    ds = list(m["robustness"])
    fig, axes = plt.subplots(1, len(ds), figsize=(FIGSIZE[0] * 1.3, FIGSIZE[1] * 0.85), squeeze=False)
    for j, d in enumerate(ds):
        ax = axes[0, j]
        for k in ORDER:
            if k not in m["robustness"][d]:
                continue
            lv = sorted(m["robustness"][d][k], key=float)
            ax.plot([100 * float(v) for v in lv], [m["robustness"][d][k][v]["acc"] for v in lv], "o-",
                    color=ENC_COLOR[k], ms=4, lw=1.6, label=ENCODER_LABELS[k])
        ax.set_title(DATASET_LABELS.get(d, d))
        ax.set_xlabel("Extra electrodes failed (%)")
        if j == 0:
            ax.set_ylabel("Recognition accuracy")
    axes[0, -1].legend(frameon=False, fontsize=7, loc="lower left")
    savefig(fig, fig_dir / "fig_dropout_robustness.png")


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
    cols = ["input", "intensity", "edge", "salient_edge", "learned", "learned_adapted"]
    titles = {"input": "Input", **ENCODER_LABELS, "learned_adapted": "Learned +\nadapter"}
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
    _adapt(m, fig_dir)
    _ablations(m, fig_dir)
    _examples(cfg, fig_dir)
    _training(cfg, fig_dir)
    log(f"figures written to {fig_dir}")
