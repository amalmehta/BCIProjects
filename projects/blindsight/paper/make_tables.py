"""Generate LaTeX tables and number macros for the paper from results/metrics.json.

Run from anywhere:  python paper/make_tables.py
Writes paper/generated.tex. Every number in the paper's tables and every \\Num... macro comes from here.
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
M = json.loads((HERE.parent / "results" / "metrics.json").read_text())

ORDER = ["intensity_uncalibrated", "intensity", "edge", "salient_edge", "learned", "learned_adapted",
         "per_image_optimisation"]
LABEL = M["encoder_labels"]
DS_LABEL = {"mnist": "MNIST digits", "letters": "EMNIST letters", "cifar": "CIFAR-10 (grayscale)"}
VARIANT_LABEL = {"main": "Full objective (threads, 1024)", "no_percept_loss": "No percept loss",
                 "no_recon_loss": "No decoder (reconstruction) loss", "no_charge_cost": "No charge cost",
                 "unconstrained": "No safety limits (200\\,\\textmu A, no top-$k$, no charge cost)",
                 "threads_64": "Threads, 64 sites", "threads_256": "Threads, 256 sites",
                 "utah_96": "Utah array, 96 ch.", "utah_1024": "16 Utah arrays, 1024 ch."}


def fmt(v, nd=3, std=None):
    if v is None:
        return "--"
    s = f"{v:.{nd}f}"
    if std:
        s += f"{{\\scriptsize$\\pm${std:.{nd}f}}}"
    return s


def macro(name: str, value: str) -> str:
    return f"\\newcommand{{\\{name}}}{{{value}}}\n"


def main_table() -> str:
    out = []
    for d, block in M["main"].items():
        enc = block["encoders"]
        best = {k: max(enc[e][k] for e in enc if enc[e].get(k) is not None and e != "per_image_optimisation")
                for k in ("ssim", "ssim_lowpass", "acc")}
        rows = []
        for e in ORDER:
            if e not in enc:
                continue
            r = enc[e]
            cells = []
            for k, nd in (("ssim", 3), ("ssim_lowpass", 3), ("psnr", 2), ("acc", 3)):
                c = fmt(r.get(k), nd, r.get(k + "_std") if r.get("n_seeds", 1) > 1 else None)
                if k in best and r.get(k) is not None and abs(r[k] - best[k]) < 1e-12:
                    c = f"\\textbf{{{c}}}"
                cells.append(c)
            cells.append(f"{r['active_electrodes']:.0f}")
            cells.append(f"{r['charge_nc_per_phase'] / 1000:.2f}")
            rows.append(f"{LABEL[e]} & " + " & ".join(cells) + r" \\")
        out.append(f"\\midrule\n\\multicolumn{{7}}{{l}}{{\\textit{{{DS_LABEL[d]}}} "
                   f"(classifier on original images: {block['image_ceiling_acc']:.3f})}}\\\\\n" + "\n".join(rows))
    return ("\\begin{table*}[t]\n\\centering\\small\n"
            "\\caption{Main results on five held-out simulated patients (threads layout, 1024 sites). "
            "Learned rows: mean $\\pm$ std over three training seeds. SSIM/PSNR compare the percept with the input; "
            "low-pass SSIM blurs both by $\\sigma=0.375^\\circ$ first. Accuracy is that of a CNN trained on each "
            "encoder's percepts. Charge is the summed charge per phase over all active electrodes (\\textmu C). "
            "Per-image optimisation is a slow, non-causal reference (128 images per patient, no classifier). "
            "Best causal encoder in bold.}\n\\label{tab:main}\n"
            "\\begin{tabular}{lcccccc}\n\\toprule\n"
            "Encoder & SSIM & Low-pass SSIM & PSNR (dB) & Accuracy & Active el. & Charge (\\textmu C) \\\\\n"
            + "\n".join(out) + "\n\\bottomrule\n\\end{tabular}\n\\end{table*}\n")


def robustness_table() -> str:
    ds = list(M["robustness"])
    levels = sorted(next(iter(M["robustness"][ds[0]].values())), key=float)
    head1 = " & ".join(f"\\multicolumn{{{len(levels)}}}{{c}}{{{DS_LABEL[d]}}}" for d in ds)
    rules = " ".join(f"\\cmidrule(lr){{{2 + i * len(levels)}-{1 + (i + 1) * len(levels)}}}" for i in range(len(ds)))
    head2 = " & ".join(f"{100 * float(lv):.0f}\\%" for _ in ds for lv in levels)
    rows = []
    for e in ORDER:
        if not all(e in M["robustness"][d] for d in ds):
            continue
        cells = " & ".join(f"{M['robustness'][d][e][lv]['acc']:.3f}" for d in ds for lv in levels)
        rows.append(f"{LABEL[e]} & {cells} \\\\")
    return ("\\begin{table*}[t]\n\\centering\\small\\setlength{\\tabcolsep}{4pt}\n\\caption{Recognition accuracy after additional, unannounced "
            "electrode failures (0\\% = intact implant; encoders and classifiers are not recalibrated or retrained).}\n"
            "\\label{tab:dropout}\n\\begin{tabular}{l" + "c" * (len(levels) * len(ds)) + "}\n\\toprule\n"
            f"& {head1} \\\\\n{rules}\nEncoder & {head2} \\\\\n\\midrule\n" + "\n".join(rows)
            + "\n\\bottomrule\n\\end{tabular}\n\\end{table*}\n")


def ablation_table() -> str:
    rows = M.get("ablations", {}).get("rows", [])
    if not rows:
        return ""
    order = ["main", "no_percept_loss", "no_recon_loss", "no_charge_cost", "unconstrained",
             "utah_96", "threads_64", "threads_256", "utah_1024"]
    lines = []
    for v in order:
        for enc in ("learned", "intensity"):
            rr = [r for r in rows if r["variant"] == v and r["encoder"] == enc]
            if not rr:
                continue
            r = rr[0]
            name = VARIANT_LABEL[v] if enc == "learned" else "\\quad intensity baseline, same layout"
            lines.append(f"{name} & {r['n_electrodes']} & {fmt(r['ssim'])} & {fmt(r['ssim_lowpass'])} & "
                         f"{fmt(r['acc'])} & {r['active_electrodes']:.0f} & {r['charge_nc_per_phase'] / 1000:.2f} \\\\")
        if v in ("main", "unconstrained"):
            lines.append("\\midrule")
    return ("\\begin{table*}[t]\n\\centering\\small\\setlength{\\tabcolsep}{5pt}\n\\caption{Ablations on MNIST (training seed 0, five held-out "
            "patients). Upper block: loss terms and safety limits (1024 thread sites). Lower block: electrode "
            "count and layout, each with the intensity baseline on the same layout. Active el.\\ counts electrodes with nonzero "
            "current; in the two collapsed runs (no percept loss, Utah 96) these currents are far below the rheobase.}\n\\label{tab:ablation}\n"
            "\\begin{tabular}{lcccccc}\n\\toprule\n"
            "Variant & $N$ & SSIM & Low-pass SSIM & Accuracy & Active el. & Charge (\\textmu C) \\\\\n\\midrule\n"
            + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n\\end{table*}\n")


def macros() -> str:
    s = ""
    v = M.get("simulator_validation", {})
    if v:
        s += macro("NumMagErr", f"{100 * v['magnification_max_rel_diff_0_to_10deg']:.1f}")
        s += macro("NumBrightRsq", f"{v['brightness_r2_vs_fernandez_fig6a']:.2f}")
        t = v["threshold_current_ua"]
        s += macro("NumThrMean", f"{t['mean']:.1f}")
        s += macro("NumThrSd", f"{t['sd']:.1f}")
        h = v["habituation"]
        s += macro("NumHabDrop", f"{100 * (1 - h['after_50_trains'] / h['first']):.0f}")
        lay = v["layouts"]
        for k, name in (("utah_96", "UtahNine"), ("utah_1024", "UtahThou"), ("threads_256", "ThrTwo"),
                        ("threads_1024", "ThrThou")):
            s += macro(f"NumCov{name}", f"{100 * lay[k]['frac_field_within_0.5deg']:.0f}")
        d = v["phosphene_diameter_deg"]
        s += macro("NumSizeOne", f"{d['80uA']['1deg']:.2f}")
        s += macro("NumSizeFive", f"{d['80uA']['5deg']:.2f}")
    names = {"mnist": "Mnist", "letters": "Letters", "cifar": "Cifar"}
    keys = {"intensity": "Int", "intensity_uncalibrated": "IntUn", "edge": "Edge", "salient_edge": "Sal",
            "learned": "Learn", "learned_adapted": "Adapt", "per_image_optimisation": "Oracle"}
    for d, block in M["main"].items():
        s += macro(f"Num{names[d]}Ceil", f"{block['image_ceiling_acc']:.3f}")
        for e, k in keys.items():
            r = block["encoders"].get(e)
            if not r:
                continue
            for met, short in (("acc", "Acc"), ("ssim_lowpass", "Lp"), ("ssim", "Ssim"), ("psnr", "Psnr")):
                if r.get(met) is not None:
                    s += macro(f"Num{names[d]}{k}{short}", f"{r[met]:.3f}" if met != "psnr" else f"{r[met]:.2f}")
            s += macro(f"Num{names[d]}{k}Charge", f"{r['charge_nc_per_phase'] / 1000:.2f}")
            s += macro(f"Num{names[d]}{k}Active", f"{r['active_electrodes']:.0f}")
        rob = M["robustness"].get(d, {})
        for e, k in keys.items():
            if e in rob:
                s += macro(f"Num{names[d]}{k}AccDropHalf", f"{rob[e]['0.5']['acc']:.3f}")
    saves = [1 - b["encoders"]["learned"]["charge_nc_per_phase"] / b["encoders"]["intensity"]["charge_nc_per_phase"]
             for b in M["main"].values()]
    s += macro("NumChargeSaveMin", f"{100 * min(saves):.0f}")
    s += macro("NumChargeSaveMax", f"{100 * max(saves):.0f}")
    for r in M.get("adaptation_shots", []):
        if r["mode"] == "generic":
            continue
        tag = ("Adpt" if r["mode"] == "adapter" else "Full") + {16: "A", 64: "B", 256: "C", 1024: "D"}[r["shots"]]
        s += macro(f"NumShot{tag}", f"{r['ssim_gain']:+.4f}")
    tags = {"main": "Main", "no_percept_loss": "NoPer", "no_recon_loss": "NoRec", "no_charge_cost": "NoChg",
            "unconstrained": "Uncon", "threads_64": "ThrA", "threads_256": "ThrB", "utah_96": "UtahA", "utah_1024": "UtahB"}
    for r in M.get("ablations", {}).get("rows", []):
        t = "Ab" + tags[r["variant"]] + ("Int" if r["encoder"] == "intensity" else "")
        s += macro(t + "Ssim", f"{r['ssim']:.3f}") + macro(t + "Lp", f"{r['ssim_lowpass']:.3f}")
        s += macro(t + "Acc", f"{r['acc']:.3f}") + macro(t + "Active", f"{r['active_electrodes']:.0f}")
        s += macro(t + "Charge", f"{r['charge_nc_per_phase'] / 1000:.2f}")
    diag = M.get("budget_diagnostics", {})
    for d, block in diag.items():
        for e, k in (("intensity", "Int"), ("learned", "Learn"), ("learned_adapted", "Adapt")):
            s += macro(f"Num{names[d]}{k}SupraPct", f"{100 * block[e]['frac_active_above_threshold']:.0f}")
            s += macro(f"Num{names[d]}{k}MedAmp", f"{block[e]['median_active_current_ua']:.0f}")
    tr = M.get("training", {})
    if tr:
        t = [v["train_time_s"] for k, v in tr.items() if "_main_" in k]
        s += macro("NumTrainMin", f"{sum(t) / len(t) / 60:.0f}")
        s += macro("NumEncParams", f"{next(iter(tr.values()))['n_params_encoder']:,}".replace(",", "{,}"))
    return s


if __name__ == "__main__":
    text = "% Auto-generated by paper/make_tables.py from results/metrics.json. Do not edit.\n"
    text += macros()
    text += "\\newcommand{\\GenMainTable}{%\n" + main_table() + "}\n"
    text += "\\newcommand{\\GenDropoutTable}{%\n" + robustness_table() + "}\n"
    text += "\\newcommand{\\GenAblationTable}{%\n" + ablation_table() + "}\n"
    (HERE / "generated.tex").write_text(text)
    print(f"wrote {HERE / 'generated.tex'} ({len(text)} chars)")
