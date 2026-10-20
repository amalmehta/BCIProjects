"""Aggregate all experiment outputs into ``results/metrics.json``, figures and the site export.

Usage::

    python -m speech.report --config configs/cpu.yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .config import lm_paths, load_config
from .ctc import greedy_with_times
from .metrics import wer, words
from .phonemes import ALPHABET
from .train import load_posteriors

LABELS = {
    "gru_baseline": "GRU baseline (published recipe)",
    "gru_aug": "GRU + time mask + electrode dropout",
    "conformer_aug": "Causal Conformer + self-cond. CTC",
    "gru_noaug": "GRU, no augmentation",
    "gru_noday": "GRU, shared input layer (no day layers)",
    "gru_bi_aug": "Bidirectional GRU + new aug.",
}
SETTINGS = [("per_greedy", "greedy (PER)"), ("wer_lexicon", "lexicon, no LM"),
            ("wer_ngram", "lexicon + 3-gram"), ("wer_llm", "3-gram + GPT-2 rescoring")]


def _ms(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    return {"mean": float(np.mean(vals)), "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else None, "n": len(vals)}


def build_metrics(cfg: dict, res: Path) -> dict:
    """Merge per-system decode metrics, training summaries and side studies into one dict."""
    systems = json.loads((res / "metrics_systems.json").read_text())
    runs: dict[str, dict] = {}
    for run, seed in cfg["schedule"]:
        s = res / "runs" / f"{run}_s{seed}" / "summary.json"
        if s.exists() and f"{run}_s{seed}" in systems:
            runs.setdefault(run, {})[seed] = {**json.loads(s.read_text()), **systems[f"{run}_s{seed}"]}
    table = []
    for run, seeds in runs.items():
        for key, setting in SETTINGS:
            agg = _ms([v.get(key) for v in seeds.values()])
            if agg:
                table.append({"model": run, "label": LABELS.get(run, run), "system": "single model",
                              "setting": setting, "metric": "PER" if key == "per_greedy" else "WER", **agg,
                              "per_seed_ci95": {str(sd): v.get(key + "_ci95") for sd, v in seeds.items()}})
            ens = systems.get(f"{run}_ens")
            if ens and ens.get(key) is not None:
                table.append({"model": run, "label": LABELS.get(run, run), "system": f"ensemble of {len(seeds)}",
                              "setting": setting, "metric": "PER" if key == "per_greedy" else "WER",
                              "mean": ens[key], "std": None, "n": 1, "ci95": ens.get(key + "_ci95")})
    out = {
        "note": "All numbers computed in this repository on the Brain-to-Text '25 (Card et al. 2024) data subset in configs/cpu.yaml. "
                "PER/WER are corpus-level on the released 'val' split of the 10 training sessions (426 sentences).",
        "table": table,
        "runs": {r: {str(k): v for k, v in s.items()} for r, s in runs.items()},
        "ensembles": {k: v for k, v in systems.items() if k.endswith("_ens")},
        "comparisons": systems.get("_comparisons", {}),
        "decoder_tuning": {k: {h: v.get(h) for h in ("n_dev_tune", "beam", "alpha", "beta", "beta_lexicon", "gamma", "beta_llm",
                                                    "dev_wer_lexicon", "dev_wer_ngram", "dev_wer_llm", "interior")}
                           for k, v in systems.items() if not k.startswith("_")},
    }
    out["ensemble_vs_members"] = {}
    for k, v in out["ensembles"].items():
        mem = [systems[n] for n in v.get("members", []) if n in systems]
        if mem:
            out["ensemble_vs_members"][k] = {key: {"ensemble": v.get(key), "best_member": min(x[key] for x in mem),
                                                   "mean_member": float(np.mean([x[key] for x in mem]))}
                                             for key in ("per_greedy", "wer_ngram", "wer_llm") if v.get(key) is not None}
    stab = res / "stability.json"
    if stab.exists():
        out["stability"] = json.loads(stab.read_text())
    prev = res / "round1" / "metrics.json"
    if prev.exists():
        pm = json.loads(prev.read_text())
        out["previous_round"] = {"note": "Round-1 numbers (480 steps, previous recipe, decoder tuned on 100 dev sentences).",
                                 "runs": {r: {sd: {k: v.get(k) for k in ("per_greedy", "wer_lexicon", "wer_ngram", "wer_llm", "best_dev_per")}
                                              for sd, v in ss.items()} for r, ss in pm["runs"].items()},
                                 "ensembles": {k: {kk: v.get(kk) for kk in ("per_greedy", "wer_lexicon", "wer_ngram")}
                                               for k, v in pm.get("ensembles", {}).items()}}
    for name in ("calibration", "robustness", "latency"):
        p = res / f"{name}.json"
        if p.exists():
            out[name] = json.loads(p.read_text())
    lm_stats = lm_paths(cfg)[1] / "lm_stats.json"
    if lm_stats.exists():
        out["language_model"] = json.loads(lm_stats.read_text())
    return out


# --------------------------------------------------------------------------- site export
def export_site(cfg: dict, res: Path, metrics: dict) -> None:
    """Write the small JSON files the project website animates."""
    site = res / "site_export"
    site.mkdir(parents=True, exist_ok=True)
    main_sys = cfg["report"]["main_system"]
    key_of = {label: key for key, label in SETTINGS}
    rows = []
    for r in metrics["table"]:
        row = {k: r[k] for k in ("model", "label", "system", "setting", "metric", "mean", "std", "n")}
        for extra in ("ci95", "per_seed_ci95"):
            if r.get(extra) is not None:
                row[extra] = r[extra]
        if r["system"] == "single model":
            row["per_seed"] = {sd: v.get(key_of[r["setting"]]) for sd, v in metrics["runs"][r["model"]].items()}
        rows.append(row)
    (site / "metrics.json").write_text(json.dumps({
        "dataset": "Brain-to-Text '25 (Card et al., NEJM 2024), participant T15, 10 training sessions",
        "test_sentences": 426,
        "note": "Error rates are fractions (0.388 = 38.8%). Three seeds per model with the stabilised CTC recipe; "
                "ci95 = sentence-level bootstrap 95% interval. Every decoder setting was tuned on the dev split "
                "(215 held-out training sentences) and GPT-2 rescoring was run for every system.",
        "rows": rows,
        "comparisons": metrics.get("comparisons", {}),
    }, indent=1))

    dec = json.loads((res / "decodes" / f"{main_sys}.json").read_text())
    d, n = dec["decodes"], len(dec["sentences"])
    rng = np.random.default_rng(7)
    idx = sorted(rng.choice(n, size=min(25, n), replace=False).tolist())
    rows = []
    for i in idx:
        ref = dec["sentences"][i]
        row = {
            "id": int(i), "session": dec["sessions"][i], "truth": ref,
            "truth_phonemes": " ".join(ALPHABET[p] for p in dec["phones"][i]),
            "greedy_phonemes": " ".join(ALPHABET[p] for p in d["greedy_phones"][i]),
            "no_lm": d["lexicon"][i], "ngram": d["ngram"][i],
            "wer_no_lm": wer([ref], [d["lexicon"][i]]), "wer_ngram": wer([ref], [d["ngram"][i]]),
        }
        if "llm" in d:
            row.update(llm_rescored=d["llm"][i], wer_llm=wer([ref], [d["llm"][i]]))
        rows.append(row)
    (site / "decodes.json").write_text(json.dumps({"system": main_sys, "label": LABELS.get(main_sys.rsplit("_", 1)[0], main_sys),
                                                   "sentences": rows}, indent=1))

    # posteriors for a live-decoding replay
    run_dir = res / "runs" / cfg["report"]["posterior_run"]
    post = load_posteriors(run_dir / "posteriors_test.npz")
    hop_ms = 20 * cfg["data"]["bin_factor"] * cfg["models"]["gru"]["patch_stride"]
    picks = [i for i in idx if 5 <= len(words(post["sentences"][i])) <= 9][:4]
    items = []
    for i in picks:
        lp = post["log_probs"][i]
        T = len(lp)
        step = int(np.ceil(T / 150))
        p = np.exp(lp)
        # downsample by max-pooling probabilities over blocks of `step` frames, keep top-5 per frame
        pooled = np.stack([p[s : s + step].max(0) for s in range(0, T, step)])
        top = np.argsort(-pooled, axis=1)[:, :5]
        frames = [[[int(c), round(float(pooled[t, c]), 3)] for c in top[t]] for t in range(len(pooled))]
        emitted = [{"phoneme": ALPHABET[c], "frame": int(t // step), "time_s": round(t * hop_ms / 1000, 2)}
                   for c, t in greedy_with_times(lp)]
        items.append({
            "id": int(i), "truth": post["sentences"][i],
            "decoded_ngram": d["ngram"][i] if dec["sentences"][i] == post["sentences"][i] else None,
            "frame_ms": hop_ms * step, "n_frames": len(pooled), "duration_s": round(T * hop_ms / 1000, 2),
            "topk": frames, "emissions": emitted,
        })
    (site / "posteriors.json").write_text(json.dumps({
        "alphabet": ALPHABET, "blank_index": 0, "word_boundary_index": len(ALPHABET) - 1,
        "model": cfg["report"]["posterior_run"], "format": "topk[frame] = [[class_index, probability], ...] (top-5, max-pooled over frame_ms)",
        "sentences": items}, separators=(",", ":")))

    if "calibration" in metrics:
        cal = metrics["calibration"]
        agg = {}
        for r in cal["rows"]:
            agg.setdefault((r["mode"], r["n_sentences"]), []).append(r)
        curve = [{"mode": m, "n_sentences": n, "wer": float(np.mean([r["wer"] for r in rs])),
                  "per": float(np.mean([r["per"] for r in rs])), "repeats": len(rs)}
                 for (m, n), rs in sorted(agg.items(), key=lambda kv: (kv[0][0], kv[0][1]))]
        (site / "calibration.json").write_text(json.dumps({
            "heldout_session": cal["heldout_session"], "base_model": cal["base_run"],
            "test_sentences": cal["n_test_sentences"], "curve": curve,
            "modes": {"reuse": "no new data: reuse the latest day's input layer",
                      "day_layer": "fine-tune only the new day's input layer",
                      "full": "fine-tune the whole network",
                      "tta_moments": "unsupervised: match per-feature mean/std of the new day to the last training day",
                      "tta_coral": "unsupervised: match the full feature covariance (CORAL) to the last training day",
                      "tta_self_train": "unsupervised: pseudo-label self-training of the day layer (LM decodes, confidence-filtered)",
                      "tta_coral+self_train": "unsupervised: CORAL initialisation, then pseudo-label self-training"},
            "n_unlabelled_sentences": cal.get("pool_size")}, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--results-dir", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.results_dir:
        cfg["results_dir"] = str(Path(args.results_dir).resolve())
    res = Path(cfg["results_dir"])
    metrics = build_metrics(cfg, res)
    (res / "metrics.json").write_text(json.dumps(metrics, indent=2))
    export_site(cfg, res, metrics)
    from .figures import make_all

    make_all(cfg, res, metrics)
    print(f"[report] wrote {res / 'metrics.json'}, figures and site export")


if __name__ == "__main__":
    main()
