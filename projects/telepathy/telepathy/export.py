"""Figures, headline metrics and the website export, built only from result files on disk."""
from __future__ import annotations

import json
import pickle
import shutil

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# Fixed entity -> colour mapping (validated categorical palette, light mode), never cycled.
COLORS = {"sld": "#2a78d6", "gru": "#eb6834", "wiener": "#1baf7a", "kalman": "#eda100", "fa": "#e87ba4",
          "oracle": "#8a8985", "gru_aug": "#4a3aa7"}
NAMES = {"sld": "SLD (proposed)", "gru": "GRU", "wiener": "Wiener filter", "kalman": "Kalman filter",
         "fa": "FA stabiliser", "oracle": "Oracle (perfect decoder)", "gru_aug": "GRU + augmentation",
         "sld_no_aug": "SLD w/o augmentation", "sld_no_ssl": "SLD w/o masked-spike loss",
         "sld_no_adapters": "SLD w/o day adapters", "sld_transformer": "SLD, transformer backbone"}
VARIANT_STYLE = {"fixed": ":", "rc": "--", "aligned": "-", "naive": ":"}
VARIANT_NAME = {"fixed": "fixed", "rc": "re-centred", "aligned": "aligned (unsupervised)", "naive": "no re-centring"}


def _style():
    plt.rcParams.update({
        "figure.dpi": 100, "savefig.dpi": 200, "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": "#e6e5e0",
        "grid.linewidth": 0.6, "axes.edgecolor": "#8a8985", "axes.labelcolor": "#2b2b2a", "xtick.color": "#52514e",
        "ytick.color": "#52514e", "legend.frameon": False, "lines.linewidth": 2.0, "font.family": "DejaVu Sans",
    })


def _base(method: str) -> tuple[str, str]:
    for v in ("aligned", "fixed", "naive", "rc"):
        if method.endswith("_" + v):
            return method[: -len(v) - 1], v
    return method, ""


def _save(fig, ctx, name):
    fig.tight_layout()
    fig.savefig(ctx.res / "figures" / name, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ----------------------------------------------------------------------------------------
def fig_within(ctx, within):
    keys = [k for k in ("wiener_per_day", "kalman_per_day", "fa", "wiener", "kalman", "gru", "gru_aug", "sld")
            if k in within]
    labels = {"wiener_per_day": "Wiener\n(per day)", "kalman_per_day": "Kalman\n(per day)", "fa": "FA\nstabiliser",
              "wiener": "Wiener\n(5 days)", "kalman": "Kalman\n(5 days)", "gru": "GRU", "gru_aug": "GRU\n+aug",
              "sld": "SLD\n(proposed)"}
    fig, ax = plt.subplots(figsize=(6.4, 2.8))
    for i, k in enumerate(keys):
        base = k.replace("_per_day", "")
        m, s = within[k]["mean"], within[k]["std"]
        ax.bar(i, m, 0.7, color=COLORS.get(base, "#8a8985"), yerr=s, capsize=3, edgecolor="white", linewidth=2)
        ax.text(i, m + s + 0.015, f"{m:.3f}", ha="center", va="bottom", fontsize=8, color="#2b2b2a")
    ax.set_xticks(range(len(keys)), [labels[k] for k in keys])
    ax.set_ylabel("Velocity $R^2$ (held-out test blocks)")
    ax.set_ylim(0, 1)
    ax.set_title("Within-day decoding, 5 calibration days")
    ax.grid(axis="x", visible=False)
    _save(fig, ctx, "fig_within_day.png")


def fig_cross(ctx, cross):
    x = np.array(cross["day_index"])
    M = cross["methods"]
    show = ["wiener_fixed", "wiener_rc", "kalman_rc", "fa_aligned", "gru_fixed", "gru_rc", "sld_rc", "sld_aligned"]
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    for m in show:
        if m not in M:
            continue
        b, v = _base(m)
        y = np.array([np.mean(r) for r in M[m]])
        ax.plot(x, y, VARIANT_STYLE.get(v, "-"), color=COLORS[b], marker="o", ms=3.5,
                label=f"{NAMES[b].replace(' (proposed)', '')}, {VARIANT_NAME.get(v, v)}")
    ax.set_xlabel("Days since first recording (calibration days 0-15)")
    ax.set_ylabel("Velocity $R^2$ on held-out day")
    ax.set_ylim(-0.5, 1)
    ax.set_xlim(0, max(x) + 4)
    ax.axvspan(0, ctx.anchors[-1].day_index, color="#efeee9", lw=0)
    ax.text(1, 0.93, "calibration", fontsize=7, color="#52514e")
    ax.text(max(x) + 3, -0.48, "values below -0.5 clipped", fontsize=6.5, color="#52514e", ha="right")
    ax.legend(fontsize=7, ncol=2, loc="lower left")
    ax.set_title("Across days without labels: 19 later sessions over 4 months")
    _save(fig, ctx, "fig_cross_day.png")


def fig_fewshot(ctx, fs):
    Ns = fs["trials"]
    xs = [max(n, 0.5) for n in Ns]
    fig, ax = plt.subplots(figsize=(4.8, 3.2))
    spec = {"wiener_scratch": ("wiener", ":", "Wiener, from scratch"),
            "wiener_prior": ("wiener", "--", "Wiener, prior-centred ridge"),
            "gru_finetune": ("gru", "--", "GRU, fine-tune all"),
            "sld_fewshot": ("sld", "--", "SLD, adapter only"),
            "sld_align_fewshot": ("sld", "-", "SLD, align + adapter")}
    for k, (b, ls, lab) in spec.items():
        if k not in fs["methods"]:
            continue
        d = fs["methods"][k]
        pts = [(x, np.mean(d[str(n)]), np.std(d[str(n)])) for x, n in zip(xs, Ns) if str(n) in d]
        px, py, pe = map(np.array, zip(*pts))
        ax.plot(px, py, ls, color=COLORS[b], marker="o", ms=4, label=lab)
        ax.fill_between(px, py - pe, py + pe, color=COLORS[b], alpha=0.08, lw=0)
    ax.set_xscale("log")
    ax.set_xticks(xs, [str(n) for n in Ns])
    ax.set_xlabel("Labelled reaches on the new day")
    ax.set_ylabel("Velocity $R^2$ (mean ± sd over days)")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=7, loc="lower right")
    ax.set_title("Few-shot recalibration")
    _save(fig, ctx, "fig_fewshot.png")


def fig_stress(ctx, st):
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.6), sharey=True)
    titles = {"drop": "Electrodes silenced", "shift": "Electrodes re-wired (identity scrambled)"}
    for ax, mode in zip(axes, ("drop", "shift")):
        R = st["results"][mode]
        f = np.array(R["fracs"]) * 100
        for m, vals in R["methods"].items():
            b, v = _base(m)
            if b not in COLORS:
                continue
            y = np.array([x["mean"] for x in vals])
            e = np.array([x["std"] for x in vals])
            ax.errorbar(f, y, e, fmt=VARIANT_STYLE.get(v, "-"), color=COLORS[b], marker="o", ms=3, capsize=2,
                        label=f"{NAMES[b].replace(' (proposed)', '')}, {VARIANT_NAME.get(v, v)}")
        ax.set_title(titles[mode])
        ax.set_xlabel("% of 96 channels affected")
    axes[0].set_ylabel("Velocity $R^2$")
    axes[0].set_ylim(-0.3, 1)
    h, lab = axes[1].get_legend_handles_labels()
    fig.tight_layout(rect=(0, 0.2, 1, 1))
    fig.legend(h, lab, loc="lower center", ncol=3, fontsize=6.5, bbox_to_anchor=(0.5, 0.0))
    fig.savefig(ctx.res / "figures" / "fig_stress.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def fig_bps(ctx, cl):
    order = [k for k in ("oracle", "wiener", "kalman", "gru", "sld") if k in cl["main"]]
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.0), gridspec_kw={"width_ratios": [1, 1.3]})
    ax = axes[0]
    for i, k in enumerate(order):
        m, s = cl["main"][k]["mean"], cl["main"][k]["std"]
        ax.bar(i, m, 0.7, color=COLORS[k], yerr=s, capsize=3, edgecolor="white", linewidth=2)
        ax.text(i, m + s + 0.1, f"{m:.2f}", ha="center", fontsize=8)
    for k in ("wiener", "kalman"):
        if k in cl.get("refit", {}):
            i = order.index(k)
            a = cl["refit"][k]["after"]["mean"]
            ax.plot([i - 0.35, i + 0.35], [a, a], color="#2b2b2a", lw=1.5, ls="--")
    ax.set_xticks(range(len(order)), ["Oracle", "Wiener", "Kalman", "GRU", "SLD"][: len(order)] if order[0] == "oracle"
                  else order)
    ax.set_ylabel("Simulated bits per second")
    ax.set_title(f"WebGrid {cl['grid']}×{cl['grid']} (simulated, click)")
    ax.grid(axis="x", visible=False)
    ax = axes[1]
    g = cl["grid_sweep"]["grids"]
    for k in order:
        y = [x["mean"] for x in cl["grid_sweep"]["methods"][k]]
        ax.plot(g, y, "-", color=COLORS[k], marker="o", ms=3.5, label=NAMES[k])
    ax.set_xlabel("Grid size n (n×n targets)")
    ax.set_ylabel("Simulated bits per second")
    ax.set_title("Grid-size sweep")
    ax.legend(fontsize=7)
    _save(fig, ctx, "fig_webgrid_bps.png")


def fig_crossday_bps(ctx, cl):
    cd = cl.get("crossday")
    if not cd:
        return
    x = np.array(cd["day_index"])
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    for m, vals in cd["methods"].items():
        b, v = _base(m)
        y = np.array([q["mean"] for q in vals])
        e = np.array([q["std"] for q in vals])
        ax.errorbar(x, y, e, fmt=VARIANT_STYLE.get(v, "-"), color=COLORS[b], marker="o", ms=3.5, capsize=2,
                    label=f"{NAMES[b].replace(' (proposed)', '').replace(' (perfect decoder)', '')}"
                          + (f", {VARIANT_NAME.get(v, v)}" if v else ""))
    ax.set_xlabel("Day of simulated session (decoder calibrated days 0-15)")
    ax.set_ylabel("Simulated bits per second")
    ax.set_title(f"Closed-loop bitrate on later days (WebGrid {cl['grid']}×{cl['grid']})")
    ax.legend(fontsize=6.5, ncol=2)
    _save(fig, ctx, "fig_crossday_bps.png")


def fig_fidelity(ctx, cl):
    fid = cl["fidelity"]
    fig, ax = plt.subplots(figsize=(3.2, 3.0))
    for k, v in fid.items():
        ax.scatter(v["real"], v["sim"], s=40, color=COLORS[k], label=NAMES[k], zorder=3, edgecolor="white")
    ax.plot([0, 1], [0, 1], color="#8a8985", lw=1)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("$R^2$ on real spikes")
    ax.set_ylabel("$R^2$ on simulated spikes")
    ax.set_title("Simulator fidelity (open loop)")
    ax.legend(fontsize=6.5, loc="upper left")
    _save(fig, ctx, "fig_sim_fidelity.png")


def fig_webgrid_trace(ctx, traces, grid, side, center):
    keys = [k for k in ("oracle", "wiener", "gru", "sld") if k in traces]
    fig, axes = plt.subplots(1, len(keys), figsize=(2.2 * len(keys), 2.4))
    origin = np.asarray(center) - side / 2
    for ax, k in zip(axes, keys):
        r = traces[k]
        c = r.cursor[:500]
        for t in np.linspace(0, side, grid + 1):
            ax.axvline(origin[0] + t, color="#efeee9", lw=0.4, zorder=0)
            ax.axhline(origin[1] + t, color="#efeee9", lw=0.4, zorder=0)
        ax.plot(c[:, 0], c[:, 1], color=COLORS[k], lw=1)
        tp = np.unique(r.target_pos[:500], axis=0)
        ax.scatter(tp[:, 0], tp[:, 1], marker="s", s=30, facecolor="none", edgecolor="#2b2b2a", lw=1)
        ax.set_xlim(origin[0], origin[0] + side)
        ax.set_ylim(origin[1], origin[1] + side)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        ax.set_title(f"{NAMES[k].replace(' (perfect decoder)', '').replace(' (proposed)', '')}\n"
                     f"{r.bps:.2f} bps, {r.n_correct} ok / {r.n_incorrect} wrong", fontsize=8)
    _save(fig, ctx, "fig_webgrid_traces.png")


def fig_traj(ctx, traj):
    trials = traj["trials"][:4]
    fig, axes = plt.subplots(1, len(trials), figsize=(2.3 * len(trials), 2.5))
    for ax, tr in zip(np.atleast_1d(axes), trials):
        ax.plot(*np.array(tr["true_pos"]).T, color="#2b2b2a", lw=2.2, label="true")
        for k, p in tr["decoded_pos"].items():
            ax.plot(*np.array(p).T, color=COLORS[k], lw=1.3, label=NAMES[k].replace(" (proposed)", ""))
        ax.scatter(*tr["target"], marker="s", s=40, facecolor="none", edgecolor="#2b2b2a")
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_title(f"{tr['session']} (day {tr['day_index']})", fontsize=7.5)
        ax.set_xticks([])
        ax.set_yticks([])
    np.atleast_1d(axes)[0].legend(fontsize=6, loc="best")
    _save(fig, ctx, "fig_trajectories.png")


# ----------------------------------------------------------------------------------------
def _downsample(a: np.ndarray, n: int = 200) -> list:
    idx = np.linspace(0, len(a) - 1, min(n, len(a))).astype(int)
    return np.round(a[idx], 2).tolist()


def build_trajectories(ctx) -> dict:
    """True vs decoded velocity/position for a few held-out reaches (decoded position is the
    integral of decoded velocity from the true start point, as a cursor would move)."""
    from .experiments import available_models, neural_aligned, neural_rc, unlabelled
    from .features import channel_stats

    lin = ctx.linear()
    models = available_models(ctx)
    picks = [(ctx.anchors[-1], "within-day")] + [(ctx.tests[i], "cross-day") for i in (0, 7, 12, -1)]
    out = {"bin_ms": 20, "note": "decoded position = true start + cumulative sum of decoded velocity", "trials": []}
    for d, kind in picks:
        Xu = unlabelled(d)
        mu, sd = channel_stats(Xu)
        if kind == "within-day":
            decs = {"wiener": lin["wiener"], "kalman": lin["kalman"]}
            if "gru" in models:
                decs["gru"] = neural_rc(models["gru"][0], Xu)
            if "sld" in models:
                from .experiments import neural_fixed
                decs["sld"] = neural_fixed(models["sld"][0])
        else:
            decs = {"wiener": lin["wiener"].with_stats(mu, sd), "kalman": lin["kalman"].with_stats(mu, sd)}
            if "gru" in models:
                decs["gru"] = neural_rc(models["gru"][0], Xu)
            if "sld" in models:
                ck = models["sld"][0]
                decs["sld"] = neural_aligned(ctx, ck, Xu, seed=ck["seed"], cache_key=f"{ck['tag']}|{d.name}")
        preds = {k: v.predict(d.X) for k, v in decs.items()}
        starts = d.trial_starts[(d.trial_starts > d.test.start + 50) & (d.trial_starts < d.T - 200)]
        for s in starts[[3, 10]] if kind == "within-day" else starts[[5]]:
            e = d.trial_starts[d.trial_starts > s][0] + 10
            sl = slice(s, e)
            p0 = d.pos[s]
            rec = {"session": d.name, "day_index": d.day_index, "split": kind, "t_ms": _downsample(np.arange(e - s) * 20.0),
                   "target": np.round(d.target[s + 1], 2).tolist(), "true_vel": _downsample(d.Y[sl]),
                   "true_pos": _downsample(d.pos[sl]), "decoded_vel": {}, "decoded_pos": {}}
            for k, p in preds.items():
                rec["decoded_vel"][k] = _downsample(p[sl])
                rec["decoded_pos"][k] = _downsample(p0 + np.cumsum(p[sl], 0) * 0.02)
            out["trials"].append(rec)
    out["decoder_variants"] = {"within-day": "decoders as calibrated", "cross-day":
                               "Wiener/Kalman/GRU re-centred, SLD unsupervised-aligned (no labels)"}
    return out


def build_webgrid(cl, traces, side, center) -> dict:
    out = {"simulated": True, "note": "Closed-loop simulator, not a human or animal result.",
           "workspace_mm": {"side": side, "center": list(center)}, "sessions": {}}
    for k, r in traces.items():
        T = len(r.cursor)
        step = max(1, int(np.ceil(T / 3000)))
        out["sessions"][k] = {
            "grid": r.grid, "seconds": r.seconds, "bps": round(r.bps, 3), "n_correct": r.n_correct,
            "n_incorrect": r.n_incorrect, "tuning": cl["tuned"][k],
            "targets": [{"t": t, "row": rc[0], "col": rc[1]} for t, rc in r.targets],
            "selections": [{"t": t, "row": rc[0], "col": rc[1], "correct": ok} for t, rc, ok in r.selections],
            "cursor": {"t": np.round(np.arange(0, T, step) * 0.02, 2).tolist(),
                       "x": np.round(r.cursor[::step, 0], 1).tolist(), "y": np.round(r.cursor[::step, 1], 1).tolist()},
        }
    return out


def channel_stability(ctx) -> dict:
    """Label-free check for electrode re-mapping: correlation of per-channel mean firing rate
    (unlabelled train block) between each held-out day and the last calibration day."""
    ref = ctx.anchors[-1].X[ctx.anchors[-1].train].mean(0)
    corr = [float(np.corrcoef(ref, d.X[d.train].mean(0))[0, 1]) for d in ctx.tests]
    return {"days": [d.name for d in ctx.tests], "day_index": [d.day_index for d in ctx.tests],
            "rate_corr_vs_last_calibration_day": corr, "threshold": 0.2,
            "flagged": [d.name for d, c in zip(ctx.tests, corr) if c < 0.2]}


def export_all(ctx) -> None:
    _style()
    (ctx.res / "channel_stability.json").write_text(json.dumps(channel_stability(ctx), indent=1))
    (ctx.res / "figures").mkdir(exist_ok=True)
    site = ctx.res / "site_export"
    site.mkdir(exist_ok=True)
    R = {}
    for name in ("prep", "within_day", "cross_day", "fewshot", "stress", "closed_loop", "latency",
                 "channel_stability", "closed_loop_ablation"):
        p = ctx.res / f"{name}.json"
        if p.exists():
            R[name] = json.loads(p.read_text())
    if "within_day" in R:
        fig_within(ctx, R["within_day"])
    if "cross_day" in R:
        fig_cross(ctx, R["cross_day"])
    if "fewshot" in R:
        fig_fewshot(ctx, R["fewshot"])
    if "stress" in R:
        fig_stress(ctx, R["stress"])
    from .simulator import WebGridConfig

    wc = WebGridConfig()
    traces = None
    if "closed_loop" in R:
        cl = R["closed_loop"]
        fig_bps(ctx, cl)
        fig_crossday_bps(ctx, cl)
        fig_fidelity(ctx, cl)
        traces = pickle.load(open(ctx.ckpt / "webgrid_traces.pkl", "rb"))
        fig_webgrid_trace(ctx, traces, cl["grid"], wc.side_mm, wc.center)
        (site / "webgrid.json").write_text(json.dumps(build_webgrid(cl, traces, wc.side_mm, wc.center)))
    traj = build_trajectories(ctx)
    fig_traj(ctx, traj)
    (site / "trajectories.json").write_text(json.dumps(traj))
    # --- headline metrics
    metrics = headline(R)
    (ctx.res / "metrics.json").write_text(json.dumps(metrics, indent=1))
    (site / "metrics.json").write_text(json.dumps(metrics, indent=1))
    if "fewshot" in R:
        fs = R["fewshot"]
        rec = {"x_label": "labelled reaches on the new day", "trials": fs["trials"], "n_days": len(fs["days"]),
               "days": fs["days"], "methods": {}}
        for k, d in fs["methods"].items():
            rec["methods"][k] = [{"x": int(n), "mean": float(np.mean(v)), "std": float(np.std(v))}
                                 for n, v in sorted(d.items(), key=lambda kv: int(kv[0]))]
        (site / "recalibration.json").write_text(json.dumps(rec, indent=1))
    for p in (ctx.res / "figures").glob("*.png"):
        shutil.copy(p, site / p.name)


def headline(R: dict) -> dict:
    """Headline numbers per decoder. Every value is read from a result file of this run."""
    out = {"note": "Offline R^2 on real data (O'Doherty et al. 2017, monkey Indy). Bits per second come from a "
                   "closed-loop simulator, not from a human or animal.", "decoders": {}}
    W = R.get("within_day", {})
    C = R.get("cross_day", {}).get("methods", {})
    flagged = set(R.get("channel_stability", {}).get("flagged", []))
    keep = [i for i, n in enumerate(R.get("cross_day", {}).get("days", [])) if n not in flagged]
    CL = R.get("closed_loop", {})
    rows = {"wiener": ("wiener", "wiener_rc", "wiener_fixed", "wiener"),
            "kalman": ("kalman", "kalman_rc", "kalman_fixed", "kalman"),
            "fa_stabiliser": ("fa", "fa_aligned", "fa_fixed", None),
            "gru": ("gru", "gru_rc", "gru_fixed", "gru"),
            "sld_proposed": ("sld", "sld_aligned", "sld_fixed", "sld")}
    for name, (wk, ck, fk, bk) in rows.items():
        r = {}
        if wk in W:
            r["within_day_r2"] = {"mean": W[wk]["mean"], "std": W[wk]["std"]}
        variants = [("cross_day_r2", ck), ("cross_day_r2_fixed", fk)]
        if name == "sld_proposed":
            variants.append(("cross_day_r2_recentred", "sld_rc"))
        for label, key in variants:
            if key in C:
                per_day = np.array([float(np.mean(v)) for v in C[key]])
                r[label] = {"mean": float(np.mean(per_day)), "std": float(np.std(per_day)), "variant": key,
                            "mean_excluding_remapped_days": float(np.mean(per_day[keep])),
                            "n_days": len(per_day)}
        if bk and bk in CL.get("main", {}):
            r["webgrid_bps_sim"] = {"mean": CL["main"][bk]["mean"], "std": CL["main"][bk]["std"],
                                    "grid": CL["grid"]}
        if bk and bk in R.get("latency", {}):
            r["latency_ms_per_bin_cpu"] = R["latency"][bk]
        out["decoders"][name] = r
    if "oracle" in CL.get("main", {}):
        out["oracle_webgrid_bps_sim"] = {"mean": CL["main"]["oracle"]["mean"], "std": CL["main"]["oracle"]["std"]}
    if flagged:
        out["remapped_days"] = sorted(flagged)
    if "stress" in R:
        st = {}
        for mode in ("drop", "shift"):
            res = R["stress"]["results"][mode]
            st[mode] = {"fracs": res["fracs"], **{k: [round(v["mean"], 4) for v in vals]
                                                  for k, vals in res["methods"].items()}}
        out["stress_r2"] = st
    if "fewshot" in R:
        out["fewshot_r2_mean"] = {k: {n: float(np.mean(v)) for n, v in d.items()}
                                  for k, d in R["fewshot"]["methods"].items()}
    if CL.get("refit"):
        out["refit_bps_sim"] = {k: {"before": v["before"]["mean"], "after": v["after"]["mean"]}
                                for k, v in CL["refit"].items()}
    if CL.get("crossday"):
        out["crossday_bps_sim_mean"] = {k: float(np.mean([x["mean"] for x in v]))
                                        for k, v in CL["crossday"]["methods"].items()}
    if "closed_loop_ablation" in R:
        out["ablation_bps_sim"] = {k: {"mean": v["mean"], "std": v["std"]} for k, v in R["closed_loop_ablation"].items()}
    if "prep" in R:
        out["lag_ms"] = R["prep"]["lag_ms"]
        out["anchor_days"] = R["prep"]["anchor_days"]
        out["n_test_days"] = len(R["prep"]["test_days"])
    return out
