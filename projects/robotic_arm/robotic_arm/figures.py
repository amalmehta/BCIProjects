"""Figures for the paper and website (PNG, 200 dpi) from the results JSON files.

Usage: ``python -m robotic_arm.figures``
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from . import DOF_NAMES, RESULTS_ROOT  # noqa: E402
from .arm import Q_HOME, fk_frames  # noqa: E402
from .task import OBJECT_COLORS  # noqa: E402

FIG = RESULTS_ROOT / "figures"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948")
GRAY = "#8a8984"

METHOD_STYLE = {  # fixed colour per method (never by rank); proposed = slot 1
    "dasa": ("DASA (proposed)", BLUE),
    "confidence": ("Confidence blend", ORANGE),
    "hindsight": ("Hindsight assist", AQUA),
    "distance": ("Distance blend", YELLOW),
    "fixed": ("Fixed blend", MAGENTA),
    "direct": ("Decoder only", GRAY),
}
DECODER_STYLE = {
    "wiener_daynorm": ("Wiener filter", GRAY),
    "kalman": ("Kalman filter", YELLOW),
    "gru": ("GRU", ORANGE),
    "transformer": ("Causal Transformer", AQUA),
    "ms_gru_noaug": ("Multi-day GRU, no aug.", MAGENTA),
    "ms_gru": ("Multi-day GRU (proposed)", BLUE),
}


def style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
        "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
        "grid.linewidth": 0.6, "axes.axisbelow": True, "legend.frameon": False, "figure.dpi": 100,
        "savefig.dpi": 200, "savefig.bbox": "tight", "lines.linewidth": 2.0})


def _save(fig, name: str) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / name)
    plt.close(fig)


# ------------------------------------------------------------------ decoding
def fig_decoding_within(d: dict) -> None:
    names = [k for k in DECODER_STYLE if k in d["within"]]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.5, 3.2), gridspec_kw={"width_ratios": [1, 1.5]})
    vals = [d["within"][k]["r2_vw_mean"] for k in names]
    errs = [d["within"][k].get("r2_vw_seed_std", 0.0) for k in names]
    y = np.arange(len(names))
    a1.barh(y, vals, xerr=errs, color=[DECODER_STYLE[k][1] for k in names], height=0.6,
            error_kw={"ecolor": INK, "elinewidth": 1, "capsize": 2})
    for i, v in enumerate(vals):
        a1.text(v + 0.01, i, f"{v:.2f}", va="center", fontsize=8, color=INK)
    a1.set_yticks(y, [DECODER_STYLE[k][0] for k in names])
    a1.invert_yaxis()
    a1.set_xlabel("R² (FALCON variance-weighted)")
    a1.set_title("Held-in sessions (minival)")
    a1.grid(axis="y", visible=False)
    M = np.array([[d["within"][k]["r2_pooled_per_dof"][dof] for dof in DOF_NAMES] for k in names])
    im = a2.imshow(np.clip(M, -0.2, 0.8), cmap="Blues", vmin=-0.2, vmax=0.8, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            a2.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=7.5,
                    color="white" if M[i, j] > 0.45 else INK)
    a2.set_xticks(range(7), DOF_NAMES)
    a2.set_yticks(range(len(names)), [DECODER_STYLE[k][0] for k in names])
    a2.grid(False)
    a2.set_title("Per-DoF R² (pooled over held-in test blocks)")
    fig.colorbar(im, ax=a2, fraction=0.04, pad=0.02)
    fig.tight_layout()
    _save(fig, "decoding_within.png")


def fig_cross_session(d: dict) -> None:
    c = d["cross"]
    ks = d["meta"]["recal_seconds"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.5, 3.3))
    series = [
        ("Multi-day GRU: new read-in only (proposed)", BLUE, "ms_gru_zeroshot", "ms_gru_{}s"),
        ("Multi-day GRU: fine-tune all", VIOLET, "ms_gru_zeroshot", "ms_gru_ftall_{}s"),
        ("GRU: fine-tune all", ORANGE, "gru_zeroshot", "gru_{}s"),
        ("Wiener: held-in + new data", GRAY, "wiener_daynorm_zeroshot", "wiener_pooled_{}s"),
        ("Wiener: new data only", YELLOW, None, "wiener_scratch_{}s"),
    ]
    for label, col, z0, pat in series:
        xs, ys = [], []
        if z0 and z0 in c:
            xs.append(0)
            ys.append(c[z0]["r2_vw_mean"])
        for k in ks:
            key = pat.format(k)
            if key in c:
                xs.append(k)
                ys.append(c[key]["r2_vw_mean"])
        if xs:
            a1.plot(xs, ys, "-o", color=col, label=label, ms=4)
    a1.axhline(0, color=MUTED, lw=0.8)
    a1.set_xlabel("Labelled seconds from the new day")
    a1.set_ylabel("R² on held-out test block")
    a1.set_title("Cross-session recalibration (7 later days)")
    a1.legend(fontsize=7, loc="lower right")
    days = d["meta"]["days_since_train"]
    kmax = max(ks)
    for label, col, key in [("Wiener zero-shot", GRAY, "wiener_daynorm_zeroshot"),
                            ("GRU zero-shot", ORANGE, "gru_zeroshot"),
                            (f"Multi-day GRU, {kmax} s read-in", BLUE, f"ms_gru_{kmax}s")]:
        if key in c:
            a2.plot(days, c[key]["r2_vw_blocks"], "-o", color=col, label=label, ms=4)
    a2.axhline(0, color=MUTED, lw=0.8)
    a2.set_xlabel("Days since last training day")
    a2.set_ylabel("R² (per held-out day)")
    a2.set_title("Decay with time")
    a2.legend(fontsize=7)
    fig.tight_layout()
    _save(fig, "decoding_cross_session.png")


def fig_traces(traces: dict) -> None:
    if not traces:
        return
    t = traces[0]
    dofs = ["tx", "tz", "rx", "g1"]
    fig, axes = plt.subplots(len(dofs), 1, figsize=(8, 5.2), sharex=True)
    time = np.array(t["t"])
    for ax, dof in zip(axes, dofs):
        i = DOF_NAMES.index(dof)
        ax.plot(time, np.array(t["true"])[:, i], color=INK, lw=1.5, label="Cued velocity")
        for key, col, lab, lw in [("wiener", GRAY, "Wiener (zero-shot)", 1.2), ("gru", ORANGE, "GRU (45 s fine-tune)", 1.2),
                                  ("proposed", BLUE, "Multi-day GRU (45 s read-in)", 1.5)]:
            if key in t:
                ax.plot(time, np.array(t[key])[:, i], color=col, lw=lw, label=lab)
        ax.set_ylabel(dof)
    axes[0].set_title(f"Held-out day {t['day']}: cued vs decoded velocity (z-scored)")
    axes[-1].set_xlabel("Time (s)")
    h, lab = axes[0].get_legend_handles_labels()
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.legend(h, lab, ncol=4, fontsize=7.5, loc="lower center")
    _save(fig, "decoding_traces.png")


# ------------------------------------------------------------------ closed loop
def fig_success_vs_r2(c: dict) -> None:
    levels = sorted(c["levels"].values(), key=lambda v: v["offline_r2"])
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.5, 3.4))
    for m in c["methods"]:
        lab, col = METHOD_STYLE[m]
        x = [lv["offline_r2"] for lv in levels]
        s = [c["main"][str(lv["kappa"])][m] for lv in levels]
        y = [v["success"] for v in s]
        lo = [max(0.0, v["success"] - v["success_ci95"][0]) for v in s]
        hi = [max(0.0, v["success_ci95"][1] - v["success"]) for v in s]
        a1.errorbar(x, y, yerr=[lo, hi], color=col, marker="o", ms=4, capsize=2, elinewidth=0.8, label=lab)
        tt = [v["time_to_grasp_mean"] if v["time_to_grasp_mean"] is not None else np.nan for v in s]
        a2.plot(x, tt, "-o", color=col, ms=4, label=lab)
    refs = [(c.get("real_wiener600_r2"), "real H1 Wiener\n(600 ms, held-in)"),
            (c.get("real_session_r2"), "real H1 Wiener\n(10 taps, 1 session)")]
    for ax in (a1, a2):
        for x, _ in refs:
            if x is not None:
                ax.axvline(x, color=MUTED, ls="--", lw=0.8)
        ax.set_xlabel("Offline decoder R² (simulated)")
    for x, lab in refs:
        if x is not None:
            a1.text(x + 0.008, 0.30 if "600" in lab else 0.36, lab, fontsize=6.5, color=MUTED)
    a1.set_ylabel("Grasp success rate")
    a1.set_ylim(-0.02, 1.05)
    a1.set_title("Task success vs decoder quality")
    a2.set_ylabel("Time to grasp, successes (s)")
    a2.set_title("Speed")
    a1.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    _save(fig, "success_vs_r2.png")


def fig_goal_inference(c: dict) -> None:
    nk = str(c["nominal_kappa"])
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.5, 3.2), sharey=True)
    x = (np.arange(10) + 0.5) / 10
    for m in c["methods"]:
        if m in ("direct", "distance"):
            continue
        lab, col = METHOD_STYLE[m]
        a1.plot(x, c["main"][nk][m]["goal_acc_curve"], "-o", color=col, ms=3, label=lab)
        a2.plot(x, c["change_of_mind"][m]["goal_acc_curve"], "-o", color=col, ms=3, label=lab)
    a1.set_title("Standard reach-and-grasp")
    a2.set_title("Change of mind (switch at 70% of the path)")
    for ax in (a1, a2):
        ax.set_xlabel("Normalised episode time")
        ax.axhline(1 / 3, color=MUTED, ls=":", lw=0.8)
    a1.set_ylabel("P(MAP goal = current true goal)")
    a1.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    _save(fig, "goal_inference.png")


def fig_change_of_mind(c: dict) -> None:
    ms = c["methods"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.5, 3.0))
    y = np.arange(len(ms))
    for i, m in enumerate(ms):
        lab, col = METHOD_STYLE[m]
        s = c["change_of_mind"][m]
        a1.barh(i, s["success"], color=col, height=0.6)
        a1.barh(i, s["wrong_grasp"], left=s["success"], color=RED, height=0.6, alpha=0.35)
        a1.text(s["success"] + s["wrong_grasp"] + 0.02, i, f"{s['success']:.2f}", va="center", fontsize=8)
        yl = s["yield_latency_median"]
        if yl is not None:
            a2.barh(i, yl, color=col, height=0.6)
            a2.text(yl + 0.03, i, f"{yl:.2f} s", va="center", fontsize=8)
    for ax in (a1, a2):
        ax.set_yticks(y, [METHOD_STYLE[m][0] for m in ms])
        ax.invert_yaxis()
        ax.grid(axis="y", visible=False)
    a2.set_yticklabels([])
    a1.set_xlim(0, 1.15)
    a1.set_xlabel("Success on the new goal (light red: grasped the abandoned object)")
    a2.set_xlabel("Median time for goal estimate to settle on the new goal (s)")
    a1.set_title("User changes their mind")
    a2.set_title("How fast assistance yields")
    fig.tight_layout()
    _save(fig, "change_of_mind.png")


def fig_authority(c: dict) -> None:
    """Success and speed against the user's share of the command (assistance-strength sweep)."""
    sw = c.get("authority_sweep", {})
    if not sw:
        return
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.5, 3.2))
    for m, pts in sw.items():
        lab, col = METHOD_STYLE[m]
        x = [p["user_share"] for p in pts]
        a1.plot(x, [p["success"] for p in pts], "-o", color=col, ms=4, label=lab)
        a2.plot(x, [p["time_to_grasp_mean"] if p["time_to_grasp_mean"] else np.nan for p in pts], "-o", color=col,
                ms=4, label=lab)
    nk = str(c["nominal_kappa"])
    d = c["main"][nk]["direct"]
    a1.plot([1.0], [d["success"]], "s", color=GRAY, ms=6, label=METHOD_STYLE["direct"][0])
    a1.set_ylabel("Success (standard + clutter)")
    a2.set_ylabel("Time to grasp (s)")
    for ax in (a1, a2):
        ax.set_xlabel("User share of the command (1 = full user control)")
    a1.set_title("Authority vs success")
    a2.set_title("Authority vs speed")
    a1.legend(fontsize=7, loc="lower left")
    fig.tight_layout()
    _save(fig, "authority_tradeoff.png")


def fig_ablations(c: dict) -> None:
    labels = {"dasa": "DASA (full)", "no_gate": "- consistency gate", "no_switching_prior": "- switching prior",
              "no_autocorr_correction": "- autocorrelation correction",
              "uncalibrated_error_model": "- calibrated error model (G=I, Σ=I)",
              "mismatched_user_model": "user model speed off by 2x",
              "confidence+switching": "Confidence + switching prior",
              "confidence+switching_retuned": "Confidence + switching, β re-tuned"}
    panels = [("ablations", f"nominal decoder (R² {c['levels'][str(c['nominal_kappa'])]['offline_r2']:.2f})")]
    if "ablations_hard" in c:
        panels.append(("ablations_hard", f"weakest decoder (R² {c['levels'][str(c['hard_kappa'])]['offline_r2']:.2f})"))
    fig, axes = plt.subplots(1, len(panels), figsize=(5.2 * len(panels), 3.9), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, (key, title) in zip(axes, panels):
        ab = c[key]
        names = list(ab)
        y = np.arange(len(names))
        w = 0.26
        for j, (scen, col, lab) in enumerate([("standard", BLUE, "Standard"), ("change_of_mind", ORANGE, "Change of mind"),
                                              ("clutter", AQUA, "Clutter (5 objects)")]):
            vals = [ab[n][scen]["success"] for n in names]
            off = (j - 1) * (w + 0.02)
            ax.barh(y + off, vals, height=w, color=col, label=lab)
            for i, v in enumerate(vals):
                ax.text(v + 0.01, i + off, f"{v:.2f}", va="center", fontsize=6)
        ax.set_yticks(y, [labels.get(n, n) for n in names])
        ax.set_xlim(0, 1.15)
        ax.set_xlabel("Success rate")
        ax.grid(axis="y", visible=False)
        ax.set_title(f"Ablations, {title}", fontsize=9)
    axes[0].invert_yaxis()
    axes[-1].legend(fontsize=7, loc="lower right", frameon=True, framealpha=0.9, edgecolor="none")
    fig.tight_layout()
    _save(fig, "ablations.png")


def fig_episodes(show: list) -> None:
    pick = [e for e in show if e["idx"] == 0 and e["scenario"] == "standard"][:2]
    com = [e for e in show if e["scenario"] == "change_of_mind" and e["method"] == "dasa"][:1]
    eps = pick + com
    if not eps:
        return
    fig, axes = plt.subplots(2, len(eps), figsize=(3.4 * len(eps), 5.6), gridspec_kw={"height_ratios": [1.25, 1]})
    for j, e in enumerate(eps):
        ax, bx = axes[0, j], axes[1, j]
        P = np.array(e["scene"]["positions"])
        tcp = np.array(e["trace"]["tcp"])
        for k, p in enumerate(P):
            ax.add_patch(plt.Rectangle((p[1] - 0.025, p[0] - 0.025), 0.05, 0.05, color=OBJECT_COLORS[k], alpha=0.9))
        ax.plot(tcp[:, 1], tcp[:, 0], color=INK, lw=1.2)
        ax.plot(tcp[0, 1], tcp[0, 0], "o", color=INK, ms=5)
        g0, g1 = e["initial_goal"], e["final_goal"]
        ttl = f"{METHOD_STYLE[e['method']][0]}: {'success' if e['success'] else ('wrong object' if e['wrong'] else 'timeout')}"
        ax.set_title(ttl + (f"\n(goal switches {g0}->{g1})" if g0 != g1 else ""), fontsize=8.5)
        ax.set_xlim(0.35, -0.35)
        ax.set_ylim(0.28, 0.72)
        ax.set_aspect("equal")
        ax.set_xlabel("y (m)")
        ax.set_ylabel("x (m)")
        post = np.array(e["trace"]["posterior"])
        t = np.arange(len(post)) * 0.02
        for k in range(post.shape[1]):
            bx.plot(t, post[:, k], color=OBJECT_COLORS[k], lw=1.5)
        bx.plot(t, e["trace"]["alpha"], color=INK, lw=1.0, ls="--", label="autonomy weight α")
        bx.set_ylim(-0.02, 1.02)
        bx.set_xlabel("Time (s)")
        bx.set_ylabel("Goal posterior / α")
        bx.legend(fontsize=7, loc="upper right", frameon=True, framealpha=0.9, edgecolor="none")
    fig.tight_layout()
    _save(fig, "example_episodes.png")


def fig_arm(show: list) -> None:
    """3-D render of the Kinova Gen3 kinematic chain over the tabletop scene."""
    fig = plt.figure(figsize=(5.2, 4.2))
    ax = fig.add_subplot(projection="3d")
    e = next((s for s in show if s["method"] == "dasa"), None)
    qs = [Q_HOME] + ([np.array(e["trace"]["q"][len(e["trace"]["q"]) // 2]), np.array(e["trace"]["q"][-1])] if e else [])
    for i, q in enumerate(qs):
        pts = np.array([f[:3, 3] for f in fk_frames(q)])
        ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], "-o", color=BLUE if i == len(qs) - 1 else GRAY, lw=3 - i * 0.5,
                ms=3, alpha=1.0 if i == len(qs) - 1 else 0.6)
    if e:
        for k, p in enumerate(np.array(e["scene"]["positions"])):
            ax.bar3d(p[0] - 0.025, p[1] - 0.025, 0, 0.05, 0.05, 0.1, color=OBJECT_COLORS[k], alpha=0.8)
        tcp = np.array(e["trace"]["tcp"])
        ax.plot(tcp[:, 0], tcp[:, 1], tcp[:, 2], color=INK, lw=0.8)
    xx, yy = np.meshgrid([0.0, 0.8], [-0.45, 0.45])
    ax.plot_surface(xx, yy, np.zeros_like(xx), color="#d9d6cc", alpha=0.3)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")
    ax.set_box_aspect((0.8, 0.9, 0.8))
    ax.view_init(22, -130)
    ax.set_title("Kinova Gen3 kinematics (DH), home and grasp postures")
    _save(fig, "arm_3d.png")


def main() -> None:
    style()
    dec_path = RESULTS_ROOT / "decoding_metrics.json"
    cl_path = RESULTS_ROOT / "closed_loop_metrics.json"
    if dec_path.exists():
        d = json.loads(dec_path.read_text())
        fig_decoding_within(d)
        fig_cross_session(d)
        if (RESULTS_ROOT / "decoding_preds.npz").exists():
            from .export import export_decoding

            fig_traces(export_decoding()["trials"])
    if cl_path.exists():
        c = json.loads(cl_path.read_text())
        fig_success_vs_r2(c)
        fig_goal_inference(c)
        fig_change_of_mind(c)
        fig_ablations(c)
        fig_authority(c)
        show = json.loads((RESULTS_ROOT / "closed_loop_showcase.json").read_text())
        fig_episodes(show)
        fig_arm(show)
    print(f"figures written to {FIG}")


if __name__ == "__main__":
    main()
