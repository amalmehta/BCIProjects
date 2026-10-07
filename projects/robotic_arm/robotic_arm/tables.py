"""Generate the paper's LaTeX tables (and a macro file of in-text numbers) from results JSON,
so every number in the paper comes from a run.

Usage: ``python -m robotic_arm.tables``  -> paper/tables/*.tex, paper/tables/numbers.tex
"""

from __future__ import annotations

import json

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
        rows.append(f"{lab} & {f(e['r2_vw_mean'], 3, std if n and n > 1 else None)} & {per} \\\\")
    within = ("\\begin{tabular}{l c ccccccc}\n\\toprule\n"
              "Decoder & R$^2_{\\mathrm{vw}}$ & " + " & ".join(f"\\texttt{{{x}}}" for x in DOF_NAMES)
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
    return "\n".join(f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in m.items()) + "\n"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    d = json.loads((RESULTS_ROOT / "decoding_metrics.json").read_text())
    c = json.loads((RESULTS_ROOT / "closed_loop_metrics.json").read_text())
    within, cross = decoding_tables(d)
    main_t, sweep, abl = sa_tables(c)
    for name, s in [("decoding_within", within), ("decoding_cross", cross), ("sa_main", main_t),
                    ("sa_sweep", sweep), ("sa_ablations", abl), ("numbers", numbers(d, c))]:
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
