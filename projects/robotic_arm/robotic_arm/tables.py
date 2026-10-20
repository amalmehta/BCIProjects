"""Generate the paper's LaTeX tables (and a macro file of in-text numbers) from results JSON,
so every number in the paper comes from a run.

Usage: ``python -m robotic_arm.tables``  -> paper/tables/*.tex, paper/tables/numbers.tex
"""

from __future__ import annotations

import json

import numpy as np

from . import DOF_NAMES, PROJECT_ROOT, RESULTS_ROOT

# paper tables live next to the paper unless results were redirected (RA_RESULTS)
OUT = PROJECT_ROOT / "paper" / "tables" if RESULTS_ROOT == PROJECT_ROOT / "results" else RESULTS_ROOT / "tables"
DEC = [("wiener_globalnorm", "Wiener filter, global norm."), ("wiener_daynorm", "Wiener filter"),
       ("kalman", "Kalman filter"), ("gru", "GRU"), ("transformer", "Causal Transformer"),
       ("ms_gru_noaug", "Multi-day GRU, no aug."), ("ms_gru", "\\textbf{Multi-day GRU (ours)}")]
METH = [("direct", "Decoder only"), ("fixed", "Fixed blend"), ("distance", "Distance blend [Downey'16]"),
        ("confidence", "Confidence blend [Dragan'13]"), ("hindsight", "Hindsight assist [Javdani'15]"),
        ("dasa", "\\textbf{DASA (ours)}")]


def f(x, nd=2, pm=None):
    if x is None:
        return "--"
    s = f"{x:.{nd}f}"
    if pm is not None:
        s += f"{{\\scriptsize$\\pm${pm:.{nd}f}}}"
    return s


def decoding_tables(d: dict) -> tuple[str, str]:
    w = d["within"]
    rows = []
    for k, lab in DEC:
        if k not in w:
            continue
        e = w[k]
        std = e.get("r2_vw_seed_std")
        n = e.get("n_seeds", 1)
        per = " & ".join(f(e["r2_pooled_per_dof"][dof]) for dof in DOF_NAMES)
        ci = e.get("r2_vw_ci95")
        cis = f"[{ci[0]:.2f}, {ci[1]:.2f}]" if ci else "--"
        rows.append(f"{lab} & {f(e['r2_vw_mean'], 3, std if n and n > 1 else None)} & {cis} & {n or 1} & {per} \\\\")
    within = ("\\begin{tabular}{l c c c ccccccc}\n\\toprule\n"
              "Decoder & R$^2_{\\mathrm{vw}}$ & 95\\% CI & seeds & " + " & ".join(f"\\texttt{{{x}}}" for x in DOF_NAMES)
              + " \\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
    c = d["cross"]
    ks = d["meta"]["recal_seconds"]
    spec = [("Wiener filter, global norm.", "wiener_globalnorm_zeroshot", None),
            ("Wiener filter, held-in + new", "wiener_daynorm_zeroshot", "wiener_pooled_{}s"),
            ("Wiener filter, new data only", None, "wiener_scratch_{}s"),
            ("Kalman filter", "kalman_zeroshot", None),
            ("GRU, fine-tune all", "gru_zeroshot", "gru_{}s"),
            ("Causal Transformer, fine-tune all", "transformer_zeroshot", "transformer_{}s"),
            ("Multi-day GRU, fine-tune all", "ms_gru_zeroshot", "ms_gru_ftall_{}s"),
            ("Multi-day GRU no aug., new read-in", "ms_gru_noaug_zeroshot", "ms_gru_noaug_{}s"),
            ("\\textbf{Multi-day GRU, new read-in (ours)}", "ms_gru_zeroshot", "ms_gru_{}s")]
    rows = []
    for lab, z0, pat in spec:
        cells = []
        for key in [z0] + [pat.format(k) if pat else None for k in ks]:
            if key and key in c:
                e = c[key]
                std = e.get("r2_vw_seed_std") if e.get("n_seeds", 1) > 1 else None
                cells.append(f(e["r2_vw_mean"], 3, std))
            else:
                cells.append("--")
        rows.append(f"{lab} & " + " & ".join(cells) + " \\\\")
    cross = ("\\begin{tabular}{l c " + "c" * len(ks) + "}\n\\toprule\n"
             "& zero-shot & " + " & ".join(f"{k}\\,s" for k in ks) + " \\\\\n\\midrule\n"
             + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
    return within, cross


def sa_tables(c: dict) -> tuple[str, str, str]:
    nk = str(c["nominal_kappa"])
    rows = []
    for k, lab in METH:
        s, m, cl = c["main"][nk][k], c["change_of_mind"][k], c["clutter"][k]
        rows.append(
            f"{lab} & {f(s['success'])} & {f(s['wrong_grasp'])} & "
            f"{f(s['time_to_grasp_mean'], 1)} & {f(s['path_eff'])} & {f(s['user_share'])} & "
            f"{f(m['success'])} & {f(m['wrong_grasp'])} & {f(m['yield_latency_median'])} & "
            f"{f(cl['success'])} & {f(cl['wrong_grasp'])} \\\\")
    main = ("\\begin{tabular}{l ccccc ccc cc}\n\\toprule\n"
            "& \\multicolumn{5}{c}{Standard (3 objects)} & \\multicolumn{3}{c}{Change of mind} "
            "& \\multicolumn{2}{c}{Clutter (5 obj.)} \\\\\n"
            "\\cmidrule(lr){2-6}\\cmidrule(lr){7-9}\\cmidrule(lr){10-11}\n"
            "Method & Succ. & Wrong & Time (s) & Path eff. & User share & Succ. & Wrong & Yield (s) & Succ. & Wrong \\\\\n"
            "\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
    levels = sorted(c["levels"].values(), key=lambda v: v["offline_r2"])
    rows = []
    for k, lab in METH:
        rows.append(f"{lab} & " + " & ".join(f(c["main"][str(lv["kappa"])][k]["success"]) for lv in levels) + " \\\\")
    sweep = ("\\begin{tabular}{l " + "c" * len(levels) + "}\n\\toprule\n"
             "Offline decoder R$^2$ & " + " & ".join(f"{lv['offline_r2']:.2f}" for lv in levels)
             + " \\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
    labels = {"dasa": "DASA (full)", "no_gate": "$-$ consistency gate", "no_switching_prior": "$-$ switching prior",
              "no_autocorr_correction": "$-$ autocorrelation correction ($\\tau{=}1$)",
              "mismatched_user_model": "mismatched user model (speed $\\times0.5$)",
              "confidence+switching": "Confidence blend + switching prior",
              "confidence+switching_retuned": "Confidence + switching, $\\beta$ re-tuned",
              "uncalibrated_error_model": "$-$ calibrated error model ($G{=}I,\\Sigma{=}I,\\tau{=}1$)"}
    rows = []
    for k, v in c["ablations"].items():
        h = c.get("ablations_hard", {}).get(k)
        cells = [f(v[sc]["success"]) for sc in ("standard", "change_of_mind", "clutter")]
        cells += [f(h[sc]["success"]) for sc in ("standard", "change_of_mind", "clutter")] if h else ["--"] * 3
        rows.append(f"{labels.get(k, k)} & " + " & ".join(cells) + " \\\\")
    nr = c["levels"][str(c["nominal_kappa"])]["offline_r2"]
    hr = c["levels"][str(c.get("hard_kappa", c["nominal_kappa"]))]["offline_r2"]
    abl = ("\\begin{tabular}{l ccc ccc}\n\\toprule\n"
           f"& \\multicolumn{{3}}{{c}}{{Nominal (R$^2$ {nr:.2f})}} & \\multicolumn{{3}}{{c}}{{Weakest (R$^2$ {hr:.2f})}} \\\\\n"
           "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\n"
           "Variant & Std. & Change & Clutter & Std. & Change & Clutter \\\\\n\\midrule\n"
           + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
    return main, sweep, abl


LF_COLS = [("zs_fixednorm", "no re-norm."), ("zeroshot", "z-score"), ("unsup_center", "+ centre"),
           ("unsup_coral", "+ CORAL"), ("unsup_kl", "+ KL")]


def fci(e, nd=3):
    if not e:
        return "--"
    ci = e.get("r2_vw_ci95")
    s = f"{e['r2_vw_mean']:.{nd}f}"
    return s + (f"{{\\scriptsize\\,[{ci[0]:.2f}, {ci[1]:.2f}]}}" if ci else "")


def label_free_table(d: dict) -> str:
    c = d["cross"]
    kmax = max(d["meta"]["recal_seconds"])
    rows = []
    for m, lab in (("gru", "GRU"), ("transformer", "Causal Transformer"), ("ms_gru", "\\textbf{Multi-day GRU}")):
        if f"{m}_zeroshot" not in c:
            continue
        cells = [fci(c.get(f"{m}_{k}")) for k, _ in LF_COLS]
        cells.append(fci(c.get(f"{m}_{kmax}s")))
        rows.append(f"{lab} & " + " & ".join(cells) + " \\\\")
    return ("\\begin{tabular}{l ccccc c}\n\\toprule\n"
            "& \\multicolumn{5}{c}{No labels from the new day} & Supervised \\\\\n\\cmidrule(lr){2-6}\n"
            "Model & " + " & ".join(t for _, t in LF_COLS) + f" & {kmax}\\,s read-in / FT \\\\\n\\midrule\n"
            + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")


REAL_LABELS = {"wiener_within": "Real Wiener residuals (speed-scaled)", "wiener_within_raw": "Real Wiener, raw bootstrap",
               "proposed_within": "Real multi-day GRU residuals", "wiener_crossday": "Real zero-shot Wiener, later days"}


def real_tables(r: dict, c: dict) -> tuple[str, str]:
    """(a) full comparison under the primary real-residual source; (b) success per source."""
    v = r["sources"][r["primary"]]
    rows = []
    for k, lab in METH:
        s, m, cl = v["standard"][k], v["change_of_mind"][k], v["clutter"][k]
        rows.append(f"{lab} & {f(s['success'])} & {f(s['wrong_grasp'])} & {f(s['time_to_grasp_mean'], 1)} & "
                    f"{f(s['user_share'])} & {f(m['success'])} & {f(m['wrong_grasp'])} & "
                    f"{f(m['yield_latency_median'])} & {f(cl['success'])} & {f(cl['wrong_grasp'])} \\\\")
    main = ("\\begin{tabular}{l cccc ccc cc}\n\\toprule\n"
            "& \\multicolumn{4}{c}{Standard} & \\multicolumn{3}{c}{Change of mind} & \\multicolumn{2}{c}{Clutter} \\\\\n"
            "\\cmidrule(lr){2-5}\\cmidrule(lr){6-8}\\cmidrule(lr){9-10}\n"
            "Method & Succ. & Wrong & Time (s) & User share & Succ. & Wrong & Yield (s) & Succ. & Wrong \\\\\n\\midrule\n"
            + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
    nk = str(c["nominal_kappa"])
    srcs = [("Synthetic noise model", c["levels"][nk]["offline_r2"],
             {"standard": c["main"][nk], "change_of_mind": c["change_of_mind"], "clutter": c["clutter"]}, c["tests"])]
    srcs += [(REAL_LABELS.get(n, n), x["sim_offline_r2"], x, x["tests"]) for n, x in r["sources"].items()]
    rows = []
    for lab, r2, x, tests in srcs:
        cells = [f"{r2:.2f}"]
        for sc in ("standard", "change_of_mind", "clutter"):
            best = max((m for m, _ in METH if m != "dasa"), key=lambda m: x[sc][m]["success"])
            p = tests[f"{sc}:dasa_vs_{best}"]["p_mcnemar"]
            cells += [f(x[sc]["dasa"]["success"]), f"{f(x[sc][best]['success'])} ({best})", f"{p:.2g}"]
        rows.append(f"{lab} & " + " & ".join(cells) + " \\\\")
    per = ("\\begin{tabular}{l c ccc ccc ccc}\n\\toprule\n"
           "& & \\multicolumn{3}{c}{Standard} & \\multicolumn{3}{c}{Change of mind} & \\multicolumn{3}{c}{Clutter} \\\\\n"
           "\\cmidrule(lr){3-5}\\cmidrule(lr){6-8}\\cmidrule(lr){9-11}\n"
           "Decoder noise & R$^2$ & DASA & best other & $p$ & DASA & best other & $p$ & DASA & best other & $p$ \\\\\n"
           "\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
    return main, per


DIGITS = ["Zero", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine"]


def _macro_safe(name: str) -> str:
    """LaTeX macro names may only contain letters."""
    return "".join(DIGITS[int(ch)] if ch.isdigit() else ch for ch in name if ch.isalnum())


def numbers(d: dict, c: dict) -> str:
    """LaTeX macros for in-text numbers."""
    nk = str(c["nominal_kappa"])
    w, x = d["within"], d["cross"]
    kmax = max(d["meta"]["recal_seconds"])
    m = {
        "WienerWithin": f(w["wiener_daynorm"]["r2_vw_mean"], 3),
        "KalmanWithin": f(w["kalman"]["r2_vw_mean"], 3),
        "GRUWithin": f(w["gru"]["r2_vw_mean"], 3) if "gru" in w else "--",
        "MSGRUWithin": f(w["ms_gru"]["r2_vw_mean"], 3) if "ms_gru" in w else "--",
        "WienerZS": f(x["wiener_daynorm_zeroshot"]["r2_vw_mean"], 3),
        "WienerGlobalZS": f(x["wiener_globalnorm_zeroshot"]["r2_vw_mean"], 3),
        "MSGRUZS": f(x["ms_gru_zeroshot"]["r2_vw_mean"], 3) if "ms_gru_zeroshot" in x else "--",
        "MSGRUFew": f(x[f"ms_gru_{kmax}s"]["r2_vw_mean"], 3) if f"ms_gru_{kmax}s" in x else "--",
        "GRUFew": f(x[f"gru_{kmax}s"]["r2_vw_mean"], 3) if f"gru_{kmax}s" in x else "--",
        "WienerPooledFew": f(x[f"wiener_pooled_{kmax}s"]["r2_vw_mean"], 3),
        "RecalMax": str(kmax),
        "RealRsq": f(c["real_session_r2"], 3),
        "RealRsqMulti": f(c["real_wiener600_r2"], 3),
        "NominalRsq": f(c["levels"][nk]["offline_r2"], 3),
        "NominalKappa": f(c["nominal_kappa"], 2),
        "Episodes": str(c["main"][nk]["dasa"]["n"]),
    }
    for k, _ in METH:
        key = k.capitalize().replace("Dasa", "DASA")
        m[f"Succ{key}"] = f(c["main"][nk][k]["success"])
        m[f"Time{key}"] = f(c["main"][nk][k]["time_to_grasp_mean"], 1)
        m[f"Share{key}"] = f(c["main"][nk][k]["user_share"])
        m[f"ComSucc{key}"] = f(c["change_of_mind"][k]["success"])
        m[f"ComWrong{key}"] = f(c["change_of_mind"][k]["wrong_grasp"])
        m[f"Yield{key}"] = f(c["change_of_mind"][k]["yield_latency_median"])
        m[f"ClutSucc{key}"] = f(c["clutter"][k]["success"])
        m[f"ClutWrong{key}"] = f(c["clutter"][k]["wrong_grasp"])
    # safety events across every simulated episode of the run (from the episode cache)
    cache = RESULTS_ROOT / "closed_loop_cache.jsonl"
    if cache.exists():
        n, hit = 0, {}
        for line in cache.read_text().splitlines():
            r = json.loads(line.split("\t", 1)[1])
            n += 1
            for k, v in r["safety"].items():
                hit[k] = hit.get(k, 0) + (v > 0)
        m["SafetyEpisodes"] = f"{n:,}".replace(",", "{,}")
        m["SafetyJointLimit"] = str(hit.get("joint_limit_clip", 0))
        m["SafetyWorkspace"] = str(hit.get("workspace_clip", 0))
        m["SafetyLinkTable"] = str(hit.get("link_collision_block", 0))
        m["SafetyTablePlane"] = str(hit.get("table_clip", 0))
    for name, t in c["tests"].items():
        scen, pair = name.split(":")
        key = {"standard": "Std", "change_of_mind": "Com", "clutter": "Clut"}[scen] + pair.replace(
            "dasa_vs_", "").capitalize()
        m[f"P{key}"] = f"{t['p_mcnemar']:.2g}"
    # uncertainty and label-free recalibration (improvements 9, 11)
    def mac(name, key, split="cross", nd=3):
        e = d[split].get(key)
        if e:
            m[name] = f(e["r2_vw_mean"], nd)
            if "r2_vw_ci95" in e:
                m[name + "CI"] = f"[{e['r2_vw_ci95'][0]:.2f}, {e['r2_vw_ci95'][1]:.2f}]"
            m[name + "SD"] = f(e.get("r2_vw_seed_std") or 0.0, 3)
    for name, key, split in [("WienerWithinX", "wiener_daynorm", "within"), ("GRUWithinX", "gru", "within"),
                             ("TFWithin", "transformer", "within"), ("MSGRUWithinX", "ms_gru", "within"),
                             ("KalmanWithinX", "kalman", "within"),
                             ("GRUZS", "gru_zeroshot", "cross"), ("TFZS", "transformer_zeroshot", "cross"),
                             ("MSGRUZSX", "ms_gru_zeroshot", "cross"), ("MSGRUFixedNorm", "ms_gru_zs_fixednorm", "cross"),
                             ("MSGRUCenter", "ms_gru_unsup_center", "cross"), ("MSGRUCoral", "ms_gru_unsup_coral", "cross"),
                             ("MSGRUKL", "ms_gru_unsup_kl", "cross"), ("TFCoral", "transformer_unsup_coral", "cross"),
                             ("TFKL", "transformer_unsup_kl", "cross"), ("GRUKL", "gru_unsup_kl", "cross"),
                             ("GRUCoral", "gru_unsup_coral", "cross"),
                             ("MSGRUTen", "ms_gru_10s", "cross"), ("MSGRUFewX", f"ms_gru_{kmax}s", "cross"),
                             ("GRUTen", "gru_10s", "cross"), ("TFTen", "transformer_10s", "cross"),
                             ("TFFew", f"transformer_{kmax}s", "cross"), ("GRUFewX", f"gru_{kmax}s", "cross"),
                             ("MSGRUFtAllFew", f"ms_gru_ftall_{kmax}s", "cross"),
                             ("MSGRUNoAugFew", f"ms_gru_noaug_{kmax}s", "cross")]:
        mac(name, key, split)
    for name, t in d.get("paired_tests", {}).items():
        key = _macro_safe("PT" + "".join(w.capitalize() for w in name.replace(":", "_").replace("_vs_", "_Vs_").split("_")))
        m[key] = f"{t['p_signflip']:.2g}"
        m[key + "Diff"] = f(t["mean_diff"], 3)
    m["Seeds"] = str(len(d["meta"]["seeds"]))
    real_path = RESULTS_ROOT / "closed_loop_real_metrics.json"
    if real_path.exists():
        r = json.loads(real_path.read_text())
        for src, tag in (("wiener_within", "RW"), ("wiener_within_raw", "RWRaw"), ("proposed_within", "RP"),
                         ("wiener_crossday", "RX")):
            if src not in r["sources"]:
                continue
            v = r["sources"][src]
            m[f"{tag}RealRsq"] = f(v["residual_stats"]["real_r2_vw"], 3)
            m[f"{tag}SimRsq"] = f(v["sim_offline_r2"], 3)
            m[f"{tag}Tau"] = f(np.mean(v["residual_stats"]["resid_tau_ticks"]) * 0.02, 2)
            for k, _ in METH:
                key = k.capitalize().replace("Dasa", "DASA")
                m[f"{tag}Succ{key}"] = f(v["standard"][k]["success"])
                m[f"{tag}Time{key}"] = f(v["standard"][k]["time_to_grasp_mean"], 1)
                m[f"{tag}Share{key}"] = f(v["standard"][k]["user_share"])
                m[f"{tag}ComSucc{key}"] = f(v["change_of_mind"][k]["success"])
                m[f"{tag}ComWrong{key}"] = f(v["change_of_mind"][k]["wrong_grasp"])
                m[f"{tag}Yield{key}"] = f(v["change_of_mind"][k]["yield_latency_median"])
                m[f"{tag}ClutSucc{key}"] = f(v["clutter"][k]["success"])
            for name, t in v["tests"].items():
                scen, pair = name.split(":")
                key = {"standard": "Std", "change_of_mind": "Com", "clutter": "Clut"}[scen] + pair.replace(
                    "dasa_vs_", "").capitalize()
                m[f"{tag}P{key}"] = f"{t['p_mcnemar']:.2g}"
        m["RealEpisodes"] = str(r["sources"][r["primary"]]["n_episodes"])
    return "\n".join(f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in m.items()) + "\n"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    d = json.loads((RESULTS_ROOT / "decoding_metrics.json").read_text())
    c = json.loads((RESULTS_ROOT / "closed_loop_metrics.json").read_text())
    within, cross = decoding_tables(d)
    main_t, sweep, abl = sa_tables(c)
    extra = [("decoding_labelfree", label_free_table(d))]
    real_path = RESULTS_ROOT / "closed_loop_real_metrics.json"
    if real_path.exists():
        rm, rs = real_tables(json.loads(real_path.read_text()), c)
        extra += [("sa_real", rm), ("sa_real_sources", rs)]
    for name, s in [("decoding_within", within), ("decoding_cross", cross), ("sa_main", main_t),
                    ("sa_sweep", sweep), ("sa_ablations", abl), ("numbers", numbers(d, c))] + extra:
        (OUT / f"{name}.tex").write_text(s)
    if OUT == PROJECT_ROOT / "paper" / "tables":  # keep the paper's figures in sync with the results
        import shutil

        fig_dir = PROJECT_ROOT / "paper" / "figures"
        fig_dir.mkdir(exist_ok=True)
        for png in (RESULTS_ROOT / "figures").glob("*.png"):
            shutil.copy(png, fig_dir / png.name)
    print(f"tables written to {OUT}")


if __name__ == "__main__":
    main()
