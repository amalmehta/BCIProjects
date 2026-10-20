"""Publication figures (PNG, 200 dpi) from the results directory."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .phonemes import ALPHABET  # noqa: E402
from .train import load_posteriors  # noqa: E402

# validated categorical palette (fixed order) + neutral inks
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
RUN_COLOR = {"gru_baseline": C[0], "gru_aug": C[1], "conformer_aug": C[2], "gru_noaug": C[3],
             "gru_noday": C[4], "gru_bi_aug": C[6]}
SHORT = {"gru_baseline": "GRU baseline", "gru_aug": "GRU + new aug.", "conformer_aug": "Causal Conformer",
         "gru_noaug": "GRU, no aug.", "gru_noday": "GRU, no day layers", "gru_bi_aug": "Bidir. GRU + new aug."}


def _style():
    plt.rcParams.update({
        "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9, "axes.edgecolor": INK2,
        "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False,
        "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.axisbelow": True, "legend.frameon": False, "figure.dpi": 200, "savefig.dpi": 200,
        "savefig.bbox": "tight", "lines.linewidth": 2,
    })


def _read_jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []


def fig_training_curves(cfg, res, out):
    """(a) dev PER, last round's recipe (round1) vs. this round's; (b) per-step training loss of every seed."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.3))
    ax = axes[0]
    for run in ("gru_baseline", "gru_aug", "conformer_aug"):
        for seed in range(3):
            for root, style, lw, tag in ((res / "round1" / "runs", "--", 1.0, "previous recipe"), (res / "runs", "-", 1.6, "stabilised recipe")):
                h = _read_jsonl(root / f"{run}_s{seed}" / "log.jsonl")
                if h:
                    ax.plot([r["step"] for r in h], [100 * r["dev_per"] for r in h], style, color=RUN_COLOR[run], lw=lw,
                            label=f"{SHORT[run]}, {tag}" if seed == 0 or (tag == "previous recipe" and run == "gru_aug" and seed == 0) else None)
    ax.set_xlabel("optimizer step (batch 32)")
    ax.set_ylabel("dev phoneme error rate (%)")
    ax.set_title("(a) Dev PER per seed: dashed = previous recipe, solid = stabilised", fontsize=9)
    ax.legend(fontsize=6.5)
    ax = axes[1]
    k = 10
    for run in ("gru_baseline", "conformer_aug"):
        for seed in range(3):
            t = _read_jsonl(res / "runs" / f"{run}_s{seed}" / "trace.jsonl")
            if len(t) > k:
                y = np.convolve([r["loss"] for r in t], np.ones(k) / k, mode="valid")
                ax.plot(np.arange(len(y)) + k, y, color=RUN_COLOR[run], lw=1.2, alpha=0.9,
                        label=f"{SHORT[run]} (3 seeds)" if seed == 0 else None)
    t = _read_jsonl(res / "diag" / "runs" / "gru_old_s1" / "trace.jsonl")
    if len(t) > k:
        y = np.convolve([r["loss"] for r in t], np.ones(k) / k, mode="valid")
        ax.plot(np.arange(len(y)) + k, y, "--", color=INK, lw=1.2, label="GRU seed 1, previous recipe (probe)")
    ax.set_ylim(0.8, 6)
    ax.set_xlabel("optimizer step")
    ax.set_ylabel("training CTC loss (10-step mean)")
    ax.set_title("(b) Training loss, every seed", fontsize=9)
    ax.legend(fontsize=7)
    fig.savefig(out / "training_curves.png")
    plt.close(fig)


def fig_main_results(metrics, out):
    """Per-seed error rates for each model x decoding stage (bar = seed 0, circles = other seeds, diamond = ensemble)."""
    runs = [r for r in RUN_COLOR if r in metrics["runs"]]
    stages = [("per_greedy", "greedy (PER)"), ("wer_lexicon", "lexicon, no LM"),
              ("wer_ngram", "lexicon + 3-gram"), ("wer_llm", "3-gram + GPT-2 rescoring")]
    fig, axes = plt.subplots(1, len(stages), figsize=(10, 3.2))
    for ax, (key, title) in zip(axes, stages):
        vmax = 0.0
        for i, run in enumerate(runs):
            seeds = metrics["runs"][run]
            s0 = seeds.get("0", {}).get(key)
            if s0 is not None:
                ax.barh(i, 100 * s0, color=RUN_COLOR[run], height=0.62)
                ax.text(100 * s0 + 1.5, i - 0.18, f"{100 * s0:.1f}", va="center", fontsize=7, color=INK)
                vmax = max(vmax, 100 * s0)
            for sd, v in seeds.items():
                if sd != "0" and v.get(key) is not None:
                    ax.plot(100 * v[key], i, "o", mfc="white", mec=INK, ms=5, mew=1.2)
                    vmax = max(vmax, 100 * v[key])
            ens = metrics["ensembles"].get(f"{run}_ens", {}).get(key)
            if ens is not None:
                ax.plot(100 * ens, i + 0.2, "D", color=INK, ms=4)
                vmax = max(vmax, 100 * ens)
        ax.set_xlim(0, max(vmax * 1.15, 10))
        ax.set_ylim(len(runs) - 0.5, -0.5)
        ax.set_yticks(range(len(runs)))
        ax.set_yticklabels([SHORT[r] for r in runs] if ax is axes[0] else [])
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("PER (%)" if key == "per_greedy" else "WER (%)")
        ax.grid(axis="y", visible=False)
    fig.suptitle("Held-out test sentences (n = 426). Bar = seed 0, ○ = seeds 1-2, ◆ = posterior-averaged seed ensemble",
                 fontsize=9, y=1.02)
    fig.savefig(out / "main_results.png")
    plt.close(fig)


def fig_calibration(metrics, out):
    cal = metrics.get("calibration")
    if not cal:
        return
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.1))
    reuse = [r for r in cal["rows"] if r["mode"] == "reuse"]
    for ax, key, lab in zip(axes, ("wer", "per"), ("word error rate (%) with 3-gram LM", "phoneme error rate (%), greedy")):
        for mode, color, name in (("day_layer", C[0], "fine-tune new day's input layer only"), ("full", C[1], "fine-tune whole network")):
            rs = [r for r in cal["rows"] if r["mode"] == mode]
            if not rs:
                continue
            ns = sorted({r["n_sentences"] for r in rs})
            m = [100 * np.mean([r[key] for r in rs if r["n_sentences"] == n]) for n in ns]
            lo = [100 * np.min([r[key] for r in rs if r["n_sentences"] == n]) for n in ns]
            hi = [100 * np.max([r[key] for r in rs if r["n_sentences"] == n]) for n in ns]
            ax.fill_between(ns, lo, hi, color=color, alpha=0.15, lw=0)
            ax.plot(ns, m, "-o", color=color, ms=4, label=name)
        if reuse:
            ax.axhline(100 * reuse[0][key], color=INK2, ls="--", lw=1.2, label="no new data (reuse latest day's layer)")
        for mode, color, ls, name in (("tta_coral", C[3], ":", "unsupervised: CORAL statistics matching"),
                                      ("tta_self_train", C[2], "-.", "unsupervised: pseudo-label self-training"),
                                      ("tta_coral+self_train", C[6], "-", "unsupervised: CORAL + self-training")):
            rs = [r for r in cal["rows"] if r["mode"] == mode]
            if rs:
                ax.axhline(100 * rs[0][key], color=color, ls=ls, lw=1.4, label=name)
        ax.set_xscale("log")
        ticks = sorted({r["n_sentences"] for r in cal["rows"] if r["n_sentences"] > 0})
        ax.set_xticks(ticks)
        ax.set_xticklabels([str(t) for t in ticks])
        ax.minorticks_off()
        ax.set_xlabel("calibration sentences from the new day")
        ax.set_ylabel(lab)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, fontsize=7, bbox_to_anchor=(0.5, -0.13))
    fig.suptitle(f"Calibrating to a new day ({cal['heldout_session']}, {cal['n_test_sentences']} test sentences); "
                 f"band = range over {1 + max(r['repeat'] for r in cal['rows'])} random subsets", fontsize=9, y=1.02)
    fig.savefig(out / "calibration.png")
    plt.close(fig)


def fig_robustness(metrics, out):
    rob = metrics.get("robustness")
    if not rob:
        return
    fig, ax = plt.subplots(figsize=(4.6, 3.1))
    for run, pers in rob["per"].items():
        base = run.rsplit("_s", 1)[0]
        ax.plot(100 * np.array(rob["drop_fractions"]), 100 * np.array(pers), "-o", ms=4,
                color=RUN_COLOR.get(base, INK2), label=SHORT.get(base, base))
    ax.set_xlabel("electrodes silenced at test time (%)")
    ax.set_ylabel("test phoneme error rate (%)")
    ax.set_title("Robustness to electrode loss (no retraining)")
    ax.legend(fontsize=7)
    fig.savefig(out / "robustness.png")
    plt.close(fig)


def fig_day_swap(metrics, out):
    swap = (metrics.get("robustness") or {}).get("day_swap")
    if not swap:
        return
    conds = list(next(iter(swap.values())).keys())
    fig, ax = plt.subplots(figsize=(5.2, 2.6))
    w = 0.8 / len(swap)
    for j, (run, vals) in enumerate(swap.items()):
        base = run.rsplit("_s", 1)[0]
        y = np.arange(len(conds)) + (j - (len(swap) - 1) / 2) * w
        v = [100 * vals[c] for c in conds]
        ax.barh(y, v, height=w * 0.9, color=RUN_COLOR.get(base, INK2), label=SHORT.get(base, base))
        for yi, vi in zip(y, v):
            ax.text(vi + 1, yi, f"{vi:.1f}", va="center", fontsize=7, color=INK)
    ax.set_yticks(range(len(conds)))
    ax.set_yticklabels(conds)
    ax.invert_yaxis()
    ax.set_xlabel("test phoneme error rate (%)")
    ax.set_title("Which day's input layer is used at test time?", fontsize=9)
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, 105)
    ax.legend(fontsize=7, loc="lower right")
    fig.savefig(out / "day_layer_swap.png")
    plt.close(fig)


def fig_latency(metrics, out):
    lat = metrics.get("latency")
    if not lat:
        return
    fig, ax = plt.subplots(figsize=(5.2, 2.6))
    names, vals, notes = [], [], []
    for run, r in lat["models"].items():
        base = run.rsplit("_s", 1)[0]
        names.append(SHORT.get(base, base))
        if r.get("per_frame_ms") is not None:
            vals.append(r["per_frame_ms"])
            notes.append(f"{r['per_frame_ms']:.1f} ms / {r['hop_ms']:.0f} ms frame")
        else:
            vals.append(r["full_forward_ms"])
            notes.append(f"{r['full_forward_ms']:.0f} ms after sentence end")
    y = np.arange(len(names))
    ax.barh(y, vals, color=[RUN_COLOR.get(r.rsplit('_s', 1)[0], INK2) for r in lat["models"]], height=0.6)
    for i, t in enumerate(notes):
        ax.text(vals[i] * 1.05, i, t, va="center", fontsize=7, color=INK)
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.set_xlabel("CPU compute (ms, log scale)")
    ax.set_title("Streaming cost per frame vs. whole-sentence latency")
    ax.grid(axis="y", visible=False)
    ax.set_xlim(right=max(vals) * 8)
    fig.savefig(out / "latency.png")
    plt.close(fig)


def fig_posteriors(cfg, res, out):
    run = cfg["report"]["posterior_run"]
    post = load_posteriors(res / "runs" / run / "posteriors_test.npz")
    cands = [i for i, s in enumerate(post["sentences"]) if 5 <= len(s.split()) <= 7]
    i = cands[3] if len(cands) > 3 else 0
    lp = post["log_probs"][i]
    hop = 20 * cfg["data"]["bin_factor"] * cfg["models"]["gru"]["patch_stride"] / 1000
    p = np.exp(lp).T
    fig, ax = plt.subplots(figsize=(9, 3.6))
    ax.imshow(p[1:], aspect="auto", cmap="Blues", origin="lower", interpolation="nearest",
              extent=[0, p.shape[1] * hop, 0.5, len(ALPHABET) - 0.5])
    ax.plot(np.arange(p.shape[1]) * hop + hop / 2, 1 + 5 * p[0], color=C[1], lw=1, label="blank probability (scaled)")
    ax.set_yticks(range(1, len(ALPHABET)))
    ax.set_yticklabels(ALPHABET[1:], fontsize=4.5)
    ax.set_xlabel("time (s)")
    ax.grid(False)
    ax.set_title(f'CTC phoneme posteriors — "{post["sentences"][i]}"', fontsize=9)
    ax.legend(loc="upper right", fontsize=7)
    fig.savefig(out / "posteriors_example.png")
    plt.close(fig)


def fig_corpus(metrics, out):
    run = next((r for r in ("conformer_aug", "gru_baseline", "gru_aug") if r in metrics["runs"]), None)
    if run is None or "wer_ngram_by_corpus" not in metrics["runs"][run].get("0", {}):
        return
    byc = metrics["runs"][run]["0"]["wer_ngram_by_corpus"]
    key = run
    names = sorted(byc, key=lambda c: -byc[c]["n"])
    fig, ax = plt.subplots(figsize=(4.6, 2.4))
    ax.barh(range(len(names)), [100 * byc[c]["wer"] for c in names], color=C[0], height=0.6)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels([f"{c} (n={byc[c]['n']})" for c in names])
    ax.invert_yaxis()
    ax.set_xlabel("WER (%), lexicon + 3-gram")
    ax.set_title(f"WER by cue-sentence corpus ({SHORT.get(key, key)}, seed 0)", fontsize=9)
    ax.grid(axis="y", visible=False)
    fig.savefig(out / "wer_by_corpus.png")
    plt.close(fig)


def make_all(cfg: dict, res: Path, metrics: dict) -> None:
    """Render every figure into ``results/figures`` (and copy the site-relevant ones to site_export)."""
    _style()
    out = res / "figures"
    out.mkdir(parents=True, exist_ok=True)
    fig_training_curves(cfg, res, out)
    fig_main_results(metrics, out)
    fig_calibration(metrics, out)
    fig_robustness(metrics, out)
    fig_latency(metrics, out)
    fig_day_swap(metrics, out)
    fig_posteriors(cfg, res, out)
    fig_corpus(metrics, out)
    site = res / "site_export"
    for name in ("main_results.png", "calibration.png", "posteriors_example.png", "robustness.png"):
        if (out / name).exists():
            (site / name).write_bytes((out / name).read_bytes())
