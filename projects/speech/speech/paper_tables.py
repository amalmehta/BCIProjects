"""Generate LaTeX tables and number macros for the paper from ``results/metrics.json``.

Every number in the paper's tables comes from this script, so it can be
re-generated after any rerun::

    python -m speech.paper_tables --config configs/cpu.yaml
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

from .config import PROJECT_DIR, load_config

NAMES = {
    "gru_baseline": "GRU baseline (published recipe)",
    "gru_aug": "GRU + time mask + electrode dropout",
    "conformer_aug": "Causal Conformer + SC-CTC",
    "gru_bi_aug": "Bidirectional GRU + new aug.",
    "gru_noday": "GRU + new aug., no day layers",
    "gru_noaug": "GRU, no augmentation",
}
KEYS = ["per_greedy", "wer_lexicon", "wer_ngram", "wer_llm"]


def pct(x):
    return f"{100 * x:.1f}"


def cell(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return "--"
    if len(vals) == 1:
        return pct(vals[0])
    return f"{pct(np.mean(vals))}\\,$\\pm$\\,{100 * np.std(vals, ddof=1):.1f}"


def main_table(m) -> str:
    lines = [r"\begin{tabular}{lcccc}", r"\toprule",
             r"Model & PER & \multicolumn{3}{c}{WER} \\ \cmidrule(lr){2-2}\cmidrule(lr){3-5}",
             r" & greedy & no LM & 3-gram & +GPT-2 \\", r"\midrule"]
    for run in ("gru_baseline", "gru_aug", "conformer_aug", "gru_bi_aug"):
        seeds = m["runs"].get(run)
        if not seeds:
            continue
        n = len(seeds)
        row = []
        for k in KEYS:
            vals = [s.get(k) for s in seeds.values()]
            have = [v for v in vals if v is not None]
            c = cell(vals)
            if have and len(have) < len(vals):  # e.g. GPT-2 rescoring run on seed 0 only
                c += r"$^\dagger$"
            row.append(c)
        lines.append(f"{NAMES[run]} ({n} seed{'s' if n > 1 else ''}) & " + " & ".join(row) + r" \\")
        ens = m["ensembles"].get(f"{run}_ens")
        if ens:
            lines.append(rf"\quad ensemble of {n} & " + " & ".join(pct(ens[k]) if ens.get(k) is not None else "--" for k in KEYS) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def ablation_table(m) -> str:
    """Component analysis: augmentation, ensembling, day layers (test-time swap), LM stack."""
    runs, ens = m["runs"], m["ensembles"]

    def seeds(run, key):
        return [s.get(key) for s in runs.get(run, {}).values()]

    lines = [r"\begin{tabular}{lcc}", r"\toprule", r"Configuration & PER & WER (3-gram) \\", r"\midrule",
             r"\multicolumn{3}{l}{\emph{Per seed (s0 / s1 / s2)}} \\"]
    for run, name in (("gru_baseline", "GRU baseline"), ("conformer_aug", "Conformer")):
        per, w = seeds(run, "per_greedy"), seeds(run, "wer_ngram")
        if per:
            lines.append(f"\\quad {name} & {' / '.join(pct(v) for v in per)} & {' / '.join(pct(v) for v in w)} \\\\")
    lines.append(r"\multicolumn{3}{l}{\emph{Ensembling (posterior average of all seeds)}} \\")
    for run, name in (("gru_baseline", "GRU baseline"), ("conformer_aug", "Conformer")):
        e = ens.get(f"{run}_ens")
        if e:
            lines.append(f"\\quad {name} & {pct(e['per_greedy'])} & {pct(e['wer_ngram'])} \\\\")
    swap = (m.get("robustness") or {}).get("day_swap", {})
    if swap:
        lines.append(r"\multicolumn{3}{l}{\emph{Day-specific input layer at test time (PER only)}} \\")
        names = {"gru_aug_s0": "GRU + new aug.", "gru_baseline_s0": "GRU baseline", "conformer_aug_s0": "Conformer"}
        conds = list(next(iter(swap.values())).keys())
        for c in conds:
            vals = " / ".join(pct(swap[r][c]) for r in swap)
            lines.append(f"\\quad {c} & {vals} & -- \\\\")
        lines.append(r"\quad\footnotesize (" + " / ".join(names.get(r, r) for r in swap) + r") & & \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def lm_table(m) -> str:
    """WER of each decoding stage for each system, plus the 10-best oracle."""
    lines = [r"\begin{tabular}{lccccc}", r"\toprule",
             r"System & greedy PER & no LM & 3-gram & +GPT-2 & oracle@10 \\", r"\midrule"]
    rows = [(f"{r}_s{k}", f"{n} (s{k})") for r, n in (("conformer_aug", "Conformer"), ("gru_baseline", "GRU baseline")) for k in range(3)]
    rows += [("conformer_aug_ens", "Conformer ens."), ("gru_baseline_ens", "GRU baseline ens.")]
    allm = {**{f"{r}_s{k}": v for r, ss in m["runs"].items() for k, v in ss.items()}, **m["ensembles"]}
    for key, name in rows:
        x = allm.get(key)
        if not x:
            continue
        cells = [pct(x[k]) if x.get(k) is not None else "--" for k in ("per_greedy", "wer_lexicon", "wer_ngram", "wer_llm", "wer_nbest_oracle")]
        lines.append(f"{name} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def tuning_table(m) -> str:
    """Decoder settings chosen on dev for every system; * = optimum on a hard bound of the grid."""
    t = m.get("decoder_tuning", {})
    lines = [r"\begin{tabular}{lrrrrrr}", r"\toprule",
             r"System & beam & $\alpha$ & $\beta$ & $\beta_{\text{lex}}$ & $\gamma$ & dev WER \\", r"\midrule"]
    order = [f"{r}_s{k}" for r in ("conformer_aug", "gru_baseline") for k in range(3)] + ["conformer_aug_ens", "gru_baseline_ens"]
    lab = {"conformer_aug": "Conformer", "gru_baseline": "GRU"}
    for key in order:
        x = t.get(key)
        if not x:
            continue
        run, sd = key.rsplit("_", 1)
        name = f"{lab[run]} {'ens.' if sd == 'ens' else sd}"
        inter = x.get("interior") or {}

        def f(k, fmt):
            v = x.get(k)
            if v is None:
                return "--"
            return fmt.format(v) + ("" if inter.get(k, True) else "$^*$")
        lines.append(f"{name} & {f('beam', '{:d}')} & {f('alpha', '{:g}')} & {f('beta', '{:g}')} & {f('beta_lexicon', '{:g}')} & "
                     f"{f('gamma', '{:g}')} & {pct(x['dev_wer_llm']) if x.get('dev_wer_llm') is not None else '--'} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def calibration_table(m) -> str:
    cal = m.get("calibration")
    if not cal:
        return ""
    names = [("reuse", 0, "reuse last training day's layer"), ("tta_moments", 0, "unsup.: mean/std matching"),
             ("tta_coral", 0, "unsup.: CORAL"), ("tta_self_train", 0, "unsup.: self-training"),
             ("tta_coral+self_train", 0, "unsup.: CORAL + self-training")]
    names += [("day_layer", n, f"supervised day layer, {n} sent.") for n in sorted({r["n_sentences"] for r in cal["rows"] if r["mode"] == "day_layer"})]
    lines = [r"\begin{tabular}{lcc}", r"\toprule", r"New-day method & PER & WER \\", r"\midrule"]
    for mode, n, name in names:
        rs = [r for r in cal["rows"] if r["mode"] == mode and r["n_sentences"] == n]
        if rs:
            lines.append(f"{name} & {pct(np.mean([r['per'] for r in rs]))} & {pct(np.mean([r['wer'] for r in rs]))} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def latency_table(m) -> str:
    lat = m.get("latency", {}).get("models", {})
    lines = [r"\begin{tabular}{lrrrr}", r"\toprule",
             r"Model & params & full fwd & per frame & lookahead \\",
             r" & (M) & (ms) & (ms) & (ms) \\", r"\midrule"]
    for run, r in lat.items():
        base = run.rsplit("_s", 1)[0]
        pf = f"{r['per_frame_ms']:.1f}" if r.get("per_frame_ms") is not None else "--"
        la = r["algorithmic_lookahead_ms"]
        la = f"{la:.0f}" if isinstance(la, (int, float)) else "sentence"
        lines.append(f"{NAMES.get(base, base)} & {r['n_params'] / 1e6:.2f} & {r['full_forward_ms']:.0f} & {pf} & {la} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def macros(m, cfg) -> str:
    """Scalar results used in the running text, as \\newcommand macros."""
    out = {}

    def put(name, val):
        out[name] = val

    lm = m.get("language_model", {})
    put("LMVocab", f"{lm.get('vocab_size', 0):,}".replace(",", "{,}"))
    put("LMWords", f"{lm.get('corpus_words', 0) / 1e6:.1f}")
    put("LMTrigrams", f"{lm.get('trigrams', 0) / 1e6:.1f}")
    put("LMOOV", f"{100 * lm.get('eval_oov_rate', 0):.2f}")
    put("LMRemoved", str(lm.get("removed_eval_overlap", 0)))
    for run, tag in (("gru_baseline", "Base"), ("gru_aug", "Aug"), ("conformer_aug", "Conf"), ("gru_bi_aug", "Bi"),
                     ("gru_noday", "NoDay"), ("gru_noaug", "NoAug")):
        seeds = m["runs"].get(run, {})
        for k, kt in (("per_greedy", "PER"), ("wer_lexicon", "WERlex"), ("wer_ngram", "WERng"), ("wer_llm", "WERllm"),
                      ("wer_nbest_oracle", "Oracle")):
            vals = [s.get(k) for s in seeds.values() if s.get(k) is not None]
            if vals:
                put(f"{tag}{kt}", pct(np.mean(vals)))
        if len(seeds) == 1:
            only = next(iter(seeds.values()))
            for k, kt in (("wer_ngram_ci95", "NG"), ("wer_llm_ci95", "LLM")):
                if only.get(k):
                    put(f"{tag}{kt}lo", pct(only[k][0]))
                    put(f"{tag}{kt}hi", pct(only[k][1]))
        for sd, v in seeds.items():
            for k, kt in (("per_greedy", "PER"), ("wer_ngram", "WERng")):
                if v.get(k) is not None:
                    put(f"{tag}S{['Zero', 'One', 'Two'][int(sd)]}{kt}", pct(v[k]))
        ens = m["ensembles"].get(f"{run}_ens", {})
        for k, kt in (("per_greedy", "PER"), ("wer_lexicon", "WERlex"), ("wer_ngram", "WERng"), ("wer_llm", "WERllm"),
                      ("wer_nbest_oracle", "Oracle")):
            if ens.get(k) is not None:
                put(f"{tag}Ens{kt}", pct(ens[k]))
        if ens.get("wer_llm_ci95"):
            put(f"{tag}EnsLLMlo", pct(ens["wer_llm_ci95"][0]))
            put(f"{tag}EnsLLMhi", pct(ens["wer_llm_ci95"][1]))
        if ens.get("wer_ngram_ci95"):
            put(f"{tag}EnsNGlo", pct(ens["wer_ngram_ci95"][0]))
            put(f"{tag}EnsNGhi", pct(ens["wer_ngram_ci95"][1]))
        steps = [s.get("steps") for s in seeds.values()]
        mins = [s.get("train_minutes") for s in seeds.values()]
        if steps:
            put(f"{tag}Minutes", f"{np.mean(mins):.0f}")
            put(f"{tag}Params", f"{list(seeds.values())[0]['n_params'] / 1e6:.1f}")
        for k, kt in (("per_greedy", "PER"), ("wer_ngram", "WERng"), ("wer_llm", "WERllm"), ("wer_lexicon", "WERlex")):
            vals = [s.get(k) for s in seeds.values() if s.get(k) is not None]
            if len(vals) > 1:
                put(f"{tag}{kt}sd", f"{100 * np.std(vals, ddof=1):.1f}")
                put(f"{tag}{kt}best", pct(min(vals)))
                put(f"{tag}{kt}worst", pct(max(vals)))
        for k, kt in (("per_greedy_ci95", "PERci"), ("wer_ngram_ci95", "NGci"), ("wer_llm_ci95", "LLMci")):
            if ens.get(k):
                put(f"{tag}Ens{kt}lo", pct(ens[k][0]))
                put(f"{tag}Ens{kt}hi", pct(ens[k][1]))
        if ens.get("beam") is not None:
            put(f"{tag}EnsBeam", str(ens["beam"]))
            put(f"{tag}EnsGamma", f"{ens['gamma']:g}" if ens.get("gamma") is not None else "--")
    words_n = {0: "Zero", 1: "One", 2: "Two"}
    for name, tag in (("conformer_vs_gru_seedavg", "CmpSeed"), ("conformer_vs_gru_ensemble", "CmpEns"),
                      ("conformer_ens_vs_best_member", "CmpConfEnsMember")):
        c = m.get("comparisons", {}).get(name)
        if not c:
            continue
        for k, kt in (("per_greedy", "PER"), ("wer_ngram", "NG"), ("wer_llm", "LLM")):
            if k in c:
                put(f"{tag}{kt}diff", f"{100 * c[k]['diff']:.1f}")
                put(f"{tag}{kt}lo", f"{100 * c[k]['ci95'][0]:.1f}")
                put(f"{tag}{kt}hi", f"{100 * c[k]['ci95'][1]:.1f}")
                p = c[k]["p_two_sided"]
                put(f"{tag}{kt}p", "<10^{-4}" if p < 1e-4 else f"{p:.3f}")
    ev = m.get("ensemble_vs_members", {})
    for run, tag in (("gru_baseline_ens", "Base"), ("conformer_aug_ens", "Conf")):
        for k, kt in (("per_greedy", "PER"), ("wer_llm", "WERllm")):
            if ev.get(run, {}).get(k):
                put(f"{tag}BestMember{kt}", pct(ev[run][k]["best_member"]))
    st = m.get("stability", {})
    for d, tag in (("runs", ""), ("probes", "Probe")):
        for name, r in st.get(d, {}).items():
            run, sd = name.rsplit("_s", 1)
            t = {"gru_baseline": "Base", "conformer_aug": "Conf", "gru_old": "Old", "gru_fixed": "Fixed"}.get(run)
            if not t:
                continue
            key = f"Stab{tag}{t}S{words_n.get(int(sd), sd)}"
            put(key + "Plateau", str(r["plateau_steps"]))
            put(key + "Escape", str(r["escape_step"]) if r["escape_step"] is not None else "never")
            put(key + "GradMed", f"{r['grad_norm']['median']:.2f}")
            put(key + "GradMax", f"{r['grad_norm']['max']:.1f}")
            if r.get("grad_norm", {}).get("frac_clipped") is not None:
                put(key + "Clipped", f"{100 * r['grad_norm']['frac_clipped']:.0f}")
            put(key + "InitLoss", f"{r['initial_loss']:.1f}")
            curve = dict((s_, v) for s_, v in r["dev_per_curve"])
            for s_ in (50, 100, 150):
                if s_ in curve:
                    put(key + {50: "DevFifty", 100: "DevHundred", 150: "DevOneFifty"}[s_], pct(curve[s_]))
    pr = m.get("previous_round", {})
    for run, tag in (("gru_baseline", "Base"), ("gru_aug", "Aug"), ("conformer_aug", "Conf")):
        for sd, v in pr.get("runs", {}).get(run, {}).items():
            for k, kt in (("per_greedy", "PER"), ("wer_ngram", "WERng"), ("wer_llm", "WERllm"), ("wer_lexicon", "WERlex")):
                if v.get(k) is not None:
                    put(f"Prev{tag}S{words_n[int(sd)]}{kt}", pct(v[k]))
            if v.get("best_dev_per") is not None:
                put(f"Prev{tag}S{words_n[int(sd)]}DevPER", pct(v["best_dev_per"]))
        e = pr.get("ensembles", {}).get(f"{run}_ens")
        if e:
            put(f"Prev{tag}EnsPER", pct(e["per_greedy"]))
            put(f"Prev{tag}EnsWERng", pct(e["wer_ngram"]))
    byc = m["runs"].get("conformer_aug", {}).get("0", {}).get("wer_ngram_by_corpus", {})
    for c, v in byc.items():
        put("Corpus" + "".join(ch for ch in c.title() if ch.isalpha()), pct(v["wer"]))
        put("CorpusN" + "".join(ch for ch in c.title() if ch.isalpha()), str(v["n"]))
    cal = m.get("calibration")
    if cal:
        def calv(mode, n, key):
            v = [r[key] for r in cal["rows"] if r["mode"] == mode and r["n_sentences"] == n]
            return pct(np.mean(v)) if v else "--"
        put("CalReusePER", calv("reuse", 0, "per"))
        put("CalReuseWER", calv("reuse", 0, "wer"))
        for mode, t in (("tta_moments", "Moments"), ("tta_coral", "Coral"), ("tta_self_train", "Self"),
                        ("tta_coral+self_train", "CoralSelf")):
            put(f"Cal{t}PER", calv(mode, 0, "per"))
            put(f"Cal{t}WER", calv(mode, 0, "wer"))
        put("CalBase", {"conformer_aug": "Conformer", "gru_baseline": "GRU baseline"}.get(cal["base_run"].rsplit("_s", 1)[0], cal["base_run"]))
        put("CalPool", str(cal.get("pool_size")))
        for n in cfg["calibration"]["n_sentences"]:
            for mode, t in (("day_layer", "Day"), ("full", "Full")):
                w = {"10": "Ten", "25": "TwentyFive", "50": "Fifty", "100": "Hundred", "200": "TwoHundred"}.get(str(n), str(n))
                put(f"Cal{t}{w}PER", calv(mode, n, "per"))
                put(f"Cal{t}{w}WER", calv(mode, n, "wer"))
    rob = m.get("robustness")
    if rob:
        fr = rob["drop_fractions"]
        for run, tag in (("gru_baseline_s0", "Base"), ("gru_aug_s0", "Aug"), ("conformer_aug_s0", "Conf")):
            if run in rob["per"]:
                for f, v in zip(fr, rob["per"][run]):
                    word = {0: "Zero", 10: "Ten", 20: "Twenty", 30: "Thirty", 50: "Fifty"}[int(round(100 * f))]
                    put(f"Rob{tag}{word}", pct(v))
    if rob and rob.get("day_swap"):
        keys = {"own day": "Own", "previous session's layer": "Prev", "most distant session's layer": "Far",
                "identity (no day layer)": "Ident"}
        for run, tag in (("gru_aug_s0", "Aug"), ("gru_baseline_s0", "Base"), ("conformer_aug_s0", "Conf")):
            for k, v in rob["day_swap"].get(run, {}).items():
                put(f"Swap{tag}{keys[k]}", pct(v))
    lat = m.get("latency")
    if lat:
        put("BeamMs", f"{lat['beam_search_ms_per_sentence']:.0f}")
        if lat.get("gpt2_rescore_ms_per_sentence"):
            put("GPTMs", f"{lat['gpt2_rescore_ms_per_sentence']:.0f}")
        for run, r in lat["models"].items():
            tag = {"gru_aug": "Aug", "gru_baseline": "Base", "gru_bi_aug": "Bi", "conformer_aug": "Conf"}.get(run.rsplit("_s", 1)[0])
            if tag:
                put(f"Lat{tag}Full", f"{r['full_forward_ms']:.0f}")
                if r.get("per_frame_ms") is not None:
                    put(f"Lat{tag}Frame", f"{r['per_frame_ms']:.1f}")
                put(f"Lat{tag}RTF", f"{r['real_time_factor']:.3f}")
                if r.get("per_frame_ms_recompute_window64") is not None:
                    put(f"Lat{tag}Recompute", f"{r['per_frame_ms_recompute_window64']:.0f}")
    return "\n".join(f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in sorted(out.items())) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    res = Path(cfg["results_dir"])
    m = json.loads((res / "metrics.json").read_text())
    paper = PROJECT_DIR / "paper"
    (paper / "tab_main.tex").write_text(main_table(m))
    (paper / "tab_ablation.tex").write_text(ablation_table(m))
    (paper / "tab_latency.tex").write_text(latency_table(m))
    (paper / "tab_lm.tex").write_text(lm_table(m))
    (paper / "tab_tuning.tex").write_text(tuning_table(m))
    (paper / "tab_calib.tex").write_text(calibration_table(m))
    (paper / "numbers.tex").write_text(macros(m, cfg))
    figs = paper / "figures"
    figs.mkdir(exist_ok=True)
    for f in (res / "figures").glob("*.png"):
        shutil.copy(f, figs / f.name)
    print("[paper] tables, macros and figures written to", paper)


if __name__ == "__main__":
    main()
