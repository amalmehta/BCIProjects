"""Generate the paper's numbers (``paper/numbers.tex``) and tables (``paper/tables/*.tex``)
from result files, so that every number in the paper traces back to a run.

    python -m telepathy.report [--results results] [--paper paper]
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
LAT_NAMES = {"wiener": "Wiener filter", "kalman": "Kalman filter (PVA)", "kalman_v1": "Kalman filter (v1, velocity)", "fa": "FA stabiliser", "gru": "GRU",
             "sld": "SLD (GRU backbone)", "sld_transformer": "SLD (transformer backbone)"}
DISPLAY = {"oracle": "Oracle", "wiener": "Wiener", "kalman": "Kalman (PVA)", "kalman_v1": "Kalman v1", "gru": "GRU",
           "sld": "SLD", "sld_head": "SLD (adapter+head)"}


def _f(x: float, nd: int = 3) -> str:
    return f"{x:.{nd}f}"


def _pm(m: float, s: float, nd: int = 3) -> str:
    return f"{m:.{nd}f} $\\pm$ {s:.{nd}f}"


_WORDS = {0: "Zero", 1: "One", 2: "Two", 3: "Three", 4: "Four", 5: "Five", 6: "Six", 8: "Eight", 10: "Ten",
          15: "Fifteen", 20: "Twenty", 30: "Thirty", 35: "ThirtyFive", 40: "Forty", 50: "Fifty"}


def _macro_name(k: str) -> str:
    """LaTeX macro names cannot contain digits: 'sweepGruG10' -> 'sweepGruGTen'."""
    k = re.sub(r"_(\w)", lambda m: m.group(1).upper(), k)
    return re.sub(r"\d+", lambda m: _WORDS[int(m.group())], k)


def _per_day(vals) -> np.ndarray:
    return np.array([np.nanmean(v) for v in vals], dtype=float)


def build(res: Path, paper: Path) -> None:
    J = {p.stem: json.loads(p.read_text()) for p in res.glob("*.json")}
    W, C, F, S, CL, P, L = (J.get(k, {}) for k in
                            ("within_day", "cross_day", "fewshot", "stress", "closed_loop", "prep", "latency"))
    M = C.get("methods", {})
    macros: dict[str, str] = {}

    def cross_mean(key):
        return float(np.nanmean(_per_day(M[key]))) if key in M else float("nan")

    macros["sldWithin"] = _f(W["sld"]["mean"])
    macros["gruWithin"] = _f(W["gru"]["mean"])
    macros["wienerWithin"] = _f(W["wiener"]["mean"])
    macros["sldRcCross"] = _f(cross_mean("sld_rc"))
    macros["sldAlignedCross"] = _f(cross_mean("sld_aligned"))
    macros["gruRcCross"] = _f(cross_mean("gru_rc"))
    macros["wienerFixedCross"] = _f(cross_mean("wiener_fixed"))
    macros["maxDay"] = str(max(C["day_index"]))
    # Days whose electrode identities changed (label-free detector, see export.channel_stability)
    CS = J.get("channel_stability", {})
    flagged = set(CS.get("flagged", []))
    keep = [i for i, n in enumerate(C["days"]) if n not in flagged]
    macros["nFlagged"] = str(len(flagged))
    macros["nKeep"] = str(len(keep))
    macros["flaggedDays"] = ", ".join(str(C["day_index"][i]) for i, n in enumerate(C["days"]) if n in flagged)
    macros["flagCorr"] = ", ".join(f"{c:.2f}" for c, n in zip(CS.get("rate_corr_vs_last_calibration_day", []),
                                                             CS.get("days", [])) if n in flagged)
    others = [c for c, n in zip(CS.get("rate_corr_vs_last_calibration_day", []), CS.get("days", [])) if n not in flagged]
    if others:
        macros["otherCorrMin"] = f"{min(others):.2f}"
        macros["otherCorrMax"] = f"{max(others):.2f}"

    def keep_mean(key):
        return float(np.nanmean(_per_day(M[key])[keep])) if key in M else float("nan")

    for key in ("sld_rc", "sld_aligned", "gru_rc", "gru_aug_rc", "wiener_rc", "wiener_fixed", "fa_aligned",
                "kalman_rc", "sld_fixed", "gru_fixed", "sld_no_adapters_rc", "sld_no_aug_rc", "sld_no_ssl_rc",
                "sld_transformer_rc", "kalman_v1_rc", "kalman_v1_fixed", "kalman_fixed", "wiener_rc_remap",
                "kalman_rc_remap", "gru_rc_remap", "sld_rc_remap", "sld_aligned_remap"):
        name = "".join(w.capitalize() for w in key.split("_"))
        macros[f"keep{name}"] = _f(keep_mean(key))
        macros[f"all{name}"] = _f(cross_mean(key))
    di = np.array(C["day_index"])
    gain = _per_day(M["sld_aligned"]) - _per_day(M["sld_rc"])
    kk = np.array([i in keep for i in range(len(di))])
    macros["alignGainEarly"] = f"{np.mean(gain[kk & (di <= 42)]):+.3f}"
    macros["alignGainLate"] = f"{np.mean(gain[kk & (di >= 82)]):+.3f}"
    for n in flagged:
        i = C["days"].index(n)
        tag = "A" if C["day_index"][i] < 100 else "B"
        macros[f"flag{tag}Day"] = str(C["day_index"][i])
        macros[f"flag{tag}Rc"] = _f(float(np.mean(M["sld_rc"][i])))
        macros[f"flag{tag}Aligned"] = ", ".join(_f(v) for v in M["sld_aligned"][i])
    macros["lagMs"] = str(P["lag_ms"])
    macros["vmax"] = f"{P['user_vmax']:.0f}"
    macros["dzero"] = f"{P['user_d0']:.0f}"
    if S:
        for mode in ("drop", "shift"):
            R = S["results"][mode]
            fr = R["fracs"]
            for k, vals in R["methods"].items():
                for f_, v in zip(fr, vals):
                    name = "".join(w.capitalize() for w in k.split("_"))
                    macros[f"st{mode.capitalize()}{name}{int(round(f_ * 100))}"] = _f(v["mean"])
    if F:
        for k, d in F["methods"].items():
            name = "".join(w.capitalize() for w in k.split("_"))
            for n, v in d.items():
                macros[f"fs{name}N{n}"] = _f(float(np.mean(v)))
        macros["fsDays"] = str(len(F["days"]))
    if CL:
        macros["gridSize"] = str(CL["grid"])
        for k in ("wiener", "kalman"):
            if k in CL.get("refit", {}):
                macros[f"{k}RefitBps"] = _f(CL["refit"][k]["after"]["mean"], 2)
        for k, v in CL["fidelity"].items():
            macros[f"fid{k.capitalize()}Real"] = _f(v["real"])
            macros[f"fid{k.capitalize()}Sim"] = _f(v["sim"])
        g = CL["grid_sweep"]["grids"]
        for k, vals in CL["grid_sweep"]["methods"].items():
            for gg, v in zip(g, vals):
                macros[f"sweep{k.capitalize()}G{gg}"] = _f(v["mean"], 2)
        for k, v in CL["dwell"].items():
            macros[f"dwell{k.capitalize()}"] = _f(v["mean"], 2)
        if "crossday" in CL:
            cd = CL["crossday"]
            for k, vals in cd["methods"].items():
                name = "".join(w.capitalize() for w in k.split("_"))
                macros[f"cdBps{name}"] = _f(float(np.mean([v["mean"] for v in vals])), 2)
    SA = J.get("closed_loop_ablation", {})
    for k, v in SA.items():
        name = "".join(w.capitalize() for w in k.split("_"))
        macros[f"simAbl{name}"] = _f(v["mean"], 2)
        macros[f"simAbl{name}Sd"] = _f(v["std"], 2)
    if CL:
        for k in ("sld", "gru", "wiener", "kalman", "oracle", "kalman_v1"):
            if k in CL["main"]:
                macros[_macro_name(k.replace("_v1", "VOne")) + "Bps"] = _f(CL["main"][k]["mean"], 2)
        macros["noiseScale"] = f"{CL['encoder']['noise_scale']:g}"

    tables = paper / "tables"
    tables.mkdir(exist_ok=True)
    # ---- improvement round 2: strong Kalman, channel re-matching, CLDA, paired statistics
    K = J.get("kalman", {})
    if K:
        b = K["best"]
        macros["kfSmooth"] = str(b["smooth_bins"] * 20)
        macros["kfDelay"] = str(b["obs_delay"] * 20)
        macros["kfOrder"] = {0: "velocity", 1: "position--velocity", 2: "position--velocity--acceleration"}[b["order"]]
        macros["kfQ"] = f"{b['q_scale']:g}"
        macros["kfSqrt"] = "with" if b["sqrt"] else "without"
        for o, v in K["best_val_by_order"].items():
            macros[f"kfValOrder{o}"] = _f(v)
        for o, v in K["best_val_by_sqrt"].items():
            macros[f"kfValSqrt{o.capitalize()}"] = _f(v)
        macros["kfWithin"] = _f(W["kalman"]["mean"])
        macros["kfPerDayWithin"] = _f(W["kalman_per_day"]["mean"])
        macros["kfVOneWithin"] = _f(W["kalman_v1"]["mean"])
    RM = J.get("remap", {})
    if RM:
        macros["remapCands"] = str(RM["n_candidates"])
        ch = [(n, r) for n, r in RM["days"].items() if r.get("changed")]
        macros["remapNChanged"] = str(len(ch))
        macros["remapNKept"] = str(len(RM["days"]) - len(ch))
        for n, r in ch:
            i = C["days"].index(n)
            tag = "A" if C["day_index"][i] < 100 else "B"
            macros[f"rm{tag}Found"] = r["name"].replace("_", "\\_")
            macros[f"rm{tag}Score"] = _f(r["score"], 2)
            macros[f"rm{tag}IdScore"] = _f(r["identity_score"], 2)
            for k, v in r["after"].items():
                nm = "".join(w.capitalize() for w in k.split("_"))
                macros[f"rm{tag}{nm}"] = _f(float(np.mean(v)))
            macros[f"rm{tag}SldAlignedSeeds"] = ", ".join(_f(v) for v in r["after"]["sld_aligned"])
            macros[f"rm{tag}SldRcSeeds"] = ", ".join(_f(v) for v in r["after"]["sld_rc"])
        syn = RM.get("synthetic", {})
        if "structured" in syn:
            st = syn["structured"]
            macros["rmSynNDays"] = str(len(syn["days"]))
            macros["rmSynN"] = str(len(st["rows"]))
            for k in ("recovery", "wiener_rc_shuffled", "wiener_rc_matched", "sld_rc_shuffled", "sld_rc_matched"):
                macros["rmSyn" + "".join(w.capitalize() for w in k.split("_"))] = _f(st[k], 2 if k == "recovery" else 3)
        if "random" in syn:
            rows = ["\\begin{tabular}{lcccccc}", "\\toprule",
                    "Channels shuffled & \\multicolumn{2}{c}{recovered (moved ch.)} & \\multicolumn{3}{c}{SLD re-centred $R^2$} & Wiener $R^2$ \\\\",
                    " & structured & +Hungarian & none & structured & +Hungarian & +Hungarian \\\\", "\\midrule"]
            for f_ in syn["random"]["fracs"]:
                m = syn["random"]["methods"][str(f_)]
                rows.append(f"{int(round(100 * f_))}\\,\\% & {m['structured_recovery_moved']:.2f} & {m['hungarian_recovery_moved']:.2f} & "
                            f"{m['sld_rc_shuffled']:.3f} & {m['structured_sld_rc']:.3f} & {m['hungarian_sld_rc']:.3f} & "
                            f"{m['hungarian_wiener_rc']:.3f} \\\\")
                macros[f"rmRand{int(round(100 * f_))}HungRec"] = _f(m["hungarian_recovery_moved"], 2)
                macros[f"rmRand{int(round(100 * f_))}HungSld"] = _f(m["hungarian_sld_rc"])
                macros[f"rmRand{int(round(100 * f_))}NoneSld"] = _f(m["sld_rc_shuffled"])
            rows += ["\\bottomrule", "\\end{tabular}"]
            (tables / "remap_random.tex").write_text("\n".join(rows) + "\n")
    if CL and "paired_tests" in CL:
        macros["nSimSessions"] = str(CL["n_sessions"])
        for k, v in CL["main"].items():
            if "ci95" in v:
                nm = "".join(w.capitalize() for w in k.split("_"))
                macros[f"ci{nm}"] = f"[{v['ci95'][0]:.2f}, {v['ci95'][1]:.2f}]"
        rows = ["\\begin{tabular}{lccc}", "\\toprule", "Comparison & $\\Delta$ bits/s [95\\,\\% CI] & $p_{\\mathrm{Holm}}$ & sig. \\\\",
                "\\midrule"]
        for k, v in sorted(CL["paired_tests"].items(), key=lambda kv: kv[1]["p_holm"]):
            a, b_ = k.split(" vs ")
            rows.append(f"{DISPLAY.get(a, a)} $-$ {DISPLAY.get(b_, b_)} & {v['diff_mean']:+.2f} "
                        f"[{v['diff_ci95'][0]:+.2f}, {v['diff_ci95'][1]:+.2f}] & {v['p_holm']:.2g} & "
                        f"{'yes' if v['significant'] else 'no'} \\\\")
        rows += ["\\bottomrule", "\\end{tabular}"]
        (tables / "paired.tex").write_text("\n".join(rows) + "\n")
        macros["nSigPairs"] = str(sum(v["significant"] for v in CL["paired_tests"].values()))
        macros["nPairs"] = str(len(CL["paired_tests"]))
        for k in ("wiener", "kalman"):
            if "paired" in CL.get("refit", {}).get(k, {}):
                macros[f"{k}RefitP"] = f"{CL['refit'][k]['paired']['p_wilcoxon']:.2g}"
                macros[f"{k}RefitCi"] = "[{:.2f}, {:.2f}]".format(*CL["refit"][k]["after"]["ci95"])
    CD = J.get("clda", {})
    if CD:
        rows = ["\\begin{tabular}{lcccccc}", "\\toprule",
                "Decoder & \\multicolumn{2}{c}{bits/s} & $\\Delta$ [95\\,\\% CI] & $p$ & \\multicolumn{2}{c}{offline $R^2$} \\\\",
                " & before & after & & & before & after \\\\", "\\midrule"]
        for k, rec in CD["decoders"].items():
            r0, rN = rec["rounds"][0], rec["rounds"][-1]
            pv = rN["paired_vs_before"]
            rows.append(f"{DISPLAY.get(k, k)} & {r0['bps']['mean']:.2f} & {rN['bps']['mean']:.2f} & "
                        f"{pv['diff_mean']:+.2f} [{pv['diff_ci95'][0]:+.2f}, {pv['diff_ci95'][1]:+.2f}] & {pv['p_wilcoxon']:.2g} & "
                        f"{r0['offline_r2']:.3f} & {rN['offline_r2']:.3f} \\\\")
            nm = "".join(w.capitalize() for w in k.split("_"))
            macros[f"clda{nm}Before"] = _f(r0["bps"]["mean"], 2)
            macros[f"clda{nm}After"] = _f(rN["bps"]["mean"], 2)
            macros[f"clda{nm}Best"] = _f(max(x["bps"]["mean"] for x in rec["rounds"]), 2)
            macros[f"clda{nm}P"] = f"{pv['p_wilcoxon']:.2g}"
            macros[f"clda{nm}RBefore"] = _f(r0["offline_r2"])
            macros[f"clda{nm}RAfter"] = _f(rN["offline_r2"])
        rows += ["\\bottomrule", "\\end{tabular}"]
        (tables / "clda.tex").write_text("\n".join(rows) + "\n")
        macros["cldaRounds"] = str(CD["config"]["rounds"])
        macros["cldaSessions"] = str(CD["config"]["sessions_per_round"])
    # ---- Table 1: main results
    rows = [
        ("Wiener filter (per day)", "wiener_per_day", None, None, None),
        ("Kalman PVA (per day)", "kalman_per_day", None, None, None),
        ("Wiener filter (5 days)", "wiener", "wiener_fixed", "wiener_rc", "wiener"),
        ("Kalman v1, velocity state (5 days)", "kalman_v1", "kalman_v1_fixed", "kalman_v1_rc", "kalman_v1"),
        ("Kalman PVA (5 days)", "kalman", "kalman_fixed", "kalman_rc", "kalman"),
        ("FA stabiliser~\\cite{degenhart2020stabilization}", "fa", "fa_fixed", "fa_aligned", None),
        ("GRU", "gru", "gru_fixed", "gru_rc", "gru"),
        ("SLD (proposed), re-centred", "sld", "sld_fixed", "sld_rc", "sld"),
        ("SLD (proposed), aligned", None, None, "sld_aligned", None),
    ]
    out = ["\\begin{tabular}{lcccc}", "\\toprule",
           "Decoder & Within-day $R^2$ & \\multicolumn{2}{c}{Cross-day $R^2$ (19 days, no labels)} & Sim.\\ bits/s \\\\",
           " & & fixed & adapted & " + (f"({CL['grid']}$\\times${CL['grid']})" if CL else "") + " \\\\", "\\midrule"]
    for name, wk, fk, ak, bk in rows:
        w = _pm(W[wk]["mean"], W[wk]["std"]) if wk and wk in W else "--"
        f = _pm(np.nanmean(_per_day(M[fk])), np.nanstd(_per_day(M[fk]))) if fk and fk in M else "--"
        a = _pm(np.nanmean(_per_day(M[ak])), np.nanstd(_per_day(M[ak]))) if ak and ak in M else "--"
        b = _pm(CL["main"][bk]["mean"], CL["main"][bk]["std"], 2) if bk and CL and bk in CL["main"] else "--"
        out.append(f"{name} & {w} & {f} & {a} & {b} \\\\")
    if CL:
        o = CL["main"]["oracle"]
        out.append(f"Oracle (perfect decoder) & -- & -- & -- & {_pm(o['mean'], o['std'], 2)} \\\\")
    out += ["\\bottomrule", "\\end{tabular}"]
    (tables / "main.tex").write_text("\n".join(out) + "\n")
    # ---- Table 2: ablations
    abl = [("SLD (full)", "sld"), ("w/o input augmentation", "sld_no_aug"), ("w/o masked-spike loss", "sld_no_ssl"),
           ("w/o day adapters", "sld_no_adapters"), ("transformer backbone", "sld_transformer"),
           ("GRU baseline", "gru"), ("GRU + augmentation", "gru_aug")]
    out = ["\\begin{tabular}{lccccc}", "\\toprule",
           f"Model & Within-day & \\multicolumn{{2}}{{c}}{{Cross-day, {len(keep)} days}} & "
           f"\\multicolumn{{2}}{{c}}{{Cross-day, {len([i for i, v in enumerate(M.get('sld_no_aug_aligned', [])) if not np.all(np.isnan(v))])}-day subset}} \\\\",
           " & & fixed & re-centred & aligned & re-centred \\\\", "\\midrule"]
    sub = [i for i, v in enumerate(M.get("sld_no_aug_aligned", [])) if not np.all(np.isnan(v))]
    for name, k in abl:
        if k not in W:
            continue
        cells = [_pm(W[k]["mean"], W[k]["std"]) if W[k]["n"] > 1 else _f(W[k]["mean"])]
        for v in ("fixed", "rc"):
            key = f"{k}_{v}"
            cells.append(_f(float(np.nanmean(_per_day(M[key])[keep]))) if key in M else "--")
        key = f"{k}_aligned"
        cells.append(_f(float(np.mean([np.mean(M[key][i]) for i in sub]))) if key in M and sub else "--")
        key = f"{k}_rc"
        cells.append(_f(float(np.mean([np.mean(M[key][i]) for i in sub]))) if key in M and sub else "--")
        out.append(f"{name} & " + " & ".join(cells) + " \\\\")
    macros["ablSubsetDays"] = ", ".join(str(C["day_index"][i]) for i in sub)
    out += ["\\bottomrule", "\\end{tabular}"]
    (tables / "ablation.tex").write_text("\n".join(out) + "\n")
    # ---- Table 3: few-shot
    if F:
        Ns = F["trials"]
        names = {"wiener_scratch": "Wiener, from scratch", "wiener_prior": "Wiener, prior-centred ridge",
                 "gru_finetune": "GRU, fine-tune all", "sld_fewshot": "SLD, adapter only",
                 "sld_align_fewshot": "SLD, align + adapter"}
        out = ["\\begin{tabular}{l" + "c" * len(Ns) + "}", "\\toprule",
               "Method & " + " & ".join(f"$N={n}$" for n in Ns) + " \\\\", "\\midrule"]
        for k, nm in names.items():
            if k not in F["methods"]:
                continue
            d = F["methods"][k]
            out.append(nm + " & " + " & ".join(_pm(np.mean(d[str(n)]), np.std(d[str(n)]), 2) if str(n) in d else "--"
                                              for n in Ns) + " \\\\")
        out += ["\\bottomrule", "\\end{tabular}"]
        (tables / "fewshot.tex").write_text("\n".join(out) + "\n")
    # ---- Table 4: closed-loop details
    if CL:
        out = ["\\begin{tabular}{lccccc}", "\\toprule",
               "Decoder & gain & smooth & hold (s) & click bits/s & dwell bits/s \\\\", "\\midrule"]
        for k in [k for k in ("oracle", "wiener", "kalman_v1", "kalman", "gru", "sld") if k in CL["tuned"]]:
            t = CL["tuned"][k]
            out.append(f"{DISPLAY[k]} & {t['gain']:g} & {t['smooth']:g} & {t['hold']:g} & "
                       f"{_pm(CL['main'][k]['mean'], CL['main'][k]['std'], 2)} & "
                       f"{_pm(CL['dwell'][k]['mean'], CL['dwell'][k]['std'], 2)} \\\\")
        for k in ("wiener", "kalman"):
            if k in CL.get("refit", {}):
                r = CL["refit"][k]
                out.append(f"{DISPLAY[k]} + ReFIT & {r['gain']:g} & {r['smooth']:g} & {r['hold']:g} & "
                           f"{_pm(r['after']['mean'], r['after']['std'], 2)} & -- \\\\")
        out += ["\\bottomrule", "\\end{tabular}"]
        (tables / "closed_loop.tex").write_text("\n".join(out) + "\n")
    # ---- latency
    if L:
        out = ["\\begin{tabular}{lc}", "\\toprule", "Decoder & ms per 20\\,ms bin \\\\", "\\midrule"]
        for k, v in L.items():
            out.append(f"{LAT_NAMES.get(k, k)} & {v:.3f} \\\\")
        out += ["\\bottomrule", "\\end{tabular}"]
        (tables / "latency.tex").write_text("\n".join(out) + "\n")
    lines = [f"\\newcommand{{\\{_macro_name(k)}}}{{{v}}}" for k, v in sorted(macros.items())]
    (paper / "numbers.tex").write_text("% generated by python -m telepathy.report\n" + "\n".join(lines) + "\n")
    # ---- figures
    figs = paper / "figures"
    figs.mkdir(exist_ok=True)
    for p in (res / "figures").glob("*.png"):
        shutil.copy(p, figs / p.name)
    print(f"wrote {paper / 'numbers.tex'} and {len(list(tables.glob('*.tex')))} tables")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", default=str(PROJECT / "results"))
    ap.add_argument("--paper", default=str(PROJECT / "paper"))
    a = ap.parse_args()
    build(Path(a.results), Path(a.paper))


if __name__ == "__main__":
    main()
