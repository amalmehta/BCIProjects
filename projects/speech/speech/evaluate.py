"""Decode saved CTC posteriors into text and compute PER / WER for every system.

Systems are single (run, seed) models plus, for runs with several seeds, a
posterior-averaging ensemble. For each system and each decoding setting:

* ``greedy``   best-path phonemes (PER only)
* ``lexicon``  CTC beam search constrained to the vocabulary, no LM (alpha = 0)
* ``ngram``    + Kneser-Ney word trigram
* ``ngram+llm`` n-best list rescored with GPT-2 small (main systems only)

LM weights are tuned on a fixed subset of the *dev* set (held-out training
trials), never on test. Results go to ``results/metrics.json`` and per-system
decodes to ``results/decodes/``.

Usage::

    python -m speech.evaluate --config configs/cpu.yaml
"""

from __future__ import annotations

import argparse
import csv
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from .config import lm_paths, load_config
from .ctc import greedy_decode
from .lm import KneserNeyLM, Trie, load_cmudict
from .beam import BeamDecoder
from .metrics import bootstrap_ci_edits, error_rate, paired_bootstrap, sentence_edits, wer, words
from .train import load_posteriors

_G: dict = {}  # per-worker globals (LM and trie are loaded once per process)


def _init_worker(lm_dir: str, built_dir: str, beam: int) -> None:
    lm = KneserNeyLM.load(Path(built_dir) / "kn3.npz")
    vocab = set((Path(built_dir) / "vocab.txt").read_text().split())
    trie = Trie(load_cmudict(Path(lm_dir) / "cmudict.dict"), vocab)
    _G.update(lm=lm, trie=trie, beam=beam)


def _decode_one(args):
    lp, alpha, beta, nbest, beam = args
    dec = BeamDecoder(_G["trie"], _G["lm"], alpha=alpha, beta=beta, beam=beam or _G["beam"])
    return [(" ".join(r.words), r.acoustic, r.lm_score, r.score) for r in dec.decode(lp, nbest=nbest)]


class Decoder:
    """Process pool that decodes lists of posteriors with given LM weights and beam width."""

    def __init__(self, cfg: dict):
        lm_dir, built_dir = lm_paths(cfg)
        e = cfg["eval"]
        self.pool = Pool(e["workers"], initializer=_init_worker, initargs=(str(lm_dir), str(built_dir), e["beam"]))

    def __call__(self, lps, alpha, beta, nbest=1, beam=None):
        return self.pool.map(_decode_one, [(lp, alpha, beta, nbest, beam) for lp in lps], chunksize=4)

    def close(self):
        self.pool.close()


def ensemble(posts: list[dict]) -> dict:
    """Average per-frame probabilities of several models (same frame grid) and renormalise."""
    out = dict(posts[0])
    lps = []
    for i in range(len(posts[0]["log_probs"])):
        p = np.mean([np.exp(P["log_probs"][i]) for P in posts], axis=0)
        lps.append(np.log(np.maximum(p, 1e-12)))
    out["log_probs"] = lps
    return out


def corpus_map(csv_path: Path) -> dict[tuple[str, int], str]:
    """(session, block) -> cue-sentence corpus, from the dataset's block description file."""
    m = {}
    if csv_path.exists():
        for row in csv.DictReader(open(csv_path)):
            m[("t15." + row["Date"].replace("-", "."), int(row["Block number"]))] = row["Corpus"]
    return m


# --------------------------------------------------------------------------- hyper-parameter search on dev
def _extend(grid: list[float], at_low: bool, mode: str, lo: float | None, hi: float | None):
    """One more grid point beyond the low/high edge (additive or multiplicative step); None if blocked by a bound."""
    g = sorted(grid)
    if mode == "mul":
        v = g[0] / 2 if at_low else g[-1] * 2
        if at_low and g[0] <= 0:
            return None
    else:
        step = (g[1] - g[0]) if at_low else (g[-1] - g[-2])
        v = g[0] - step if at_low else g[-1] + step
    if (lo is not None and v < lo) or (hi is not None and v > hi):
        return None
    return round(float(v), 6)


def search_1d(f, grid, mode="add", lo=None, hi=None, max_expand=3):
    """Minimise ``f`` over ``grid``; while the optimum sits on an edge, extend the grid past it.

    Returns ``(best, {value: f(value)}, edge)`` where ``edge`` is ``None`` when the optimum is
    interior, or ``"low"``/``"high"`` when it is on an edge that could not be extended (a hard
    bound such as gamma >= 0, or ``max_expand`` reached).
    """
    vals: dict[float, float] = {}
    for v in grid:
        vals[float(v)] = f(float(v))
    for _ in range(max_expand + 1):
        g = sorted(vals)
        best = min(g, key=lambda v: (vals[v], abs(v)))
        if len(g) < 2 or g[0] < best < g[-1]:
            return best, vals, None
        nv = _extend(g, best == g[0], mode, lo, hi)
        if nv is None or nv in vals or _ == max_expand:
            return best, vals, ("low" if best == g[0] else "high")
        vals[nv] = f(nv)
    return best, vals, "high"


def tune_decoder(dec: Decoder, lps, refs, ecfg):
    """Tune the lexicon-only word bonus, then (alpha, beta) by coordinate descent, then the beam width.

    Every choice uses only the dev sentences. Returns the chosen values plus the full search log.
    """
    cache: dict[tuple, float] = {}

    def run(a, b, beam):
        k = (round(a, 6), round(b, 6), beam)
        if k not in cache:
            cache[k] = wer(refs, [h[0][0] for h in dec(lps, a, b, beam=beam)])
        return cache[k]

    bt = ecfg.get("beam_tune", ecfg["beam"])
    log: dict = {}
    b_lex, g, edge = search_1d(lambda b: run(0.0, b, bt), ecfg["betas_lexicon"], "add")
    log["lexicon_beta"] = {"grid": g, "edge": edge}
    betas, alphas = ecfg["betas"], ecfg["alphas"]
    b = betas[len(betas) // 2]
    a = None
    for r in range(ecfg.get("tune_rounds", 2)):
        a, ga, ea = search_1d(lambda v: run(v, b, bt), alphas, "mul", lo=0.0)
        b, gb, eb = search_1d(lambda v: run(a, v, bt), betas, "add")
        log[f"round{r}"] = {"alpha": {"grid": ga, "edge": ea}, "beta": {"grid": gb, "edge": eb}}
    beam, gbeam, ebeam = search_1d(lambda v: run(a, b, int(v)), ecfg.get("beams", [ecfg["beam"]]), "mul", lo=1, hi=ecfg.get("beam_max", 128))
    log["beam"] = {"grid": gbeam, "edge": ebeam}
    # the lexicon-only decoder uses the same beam width
    out = {"beta_lexicon": b_lex, "dev_wer_lexicon": run(0.0, b_lex, bt), "alpha": a, "beta": b, "beam": int(beam),
           "dev_wer_ngram": run(a, b, int(beam)), "n_decodes": len(cache)}
    out["interior"] = {"beta_lexicon": log["lexicon_beta"]["edge"] is None,
                       "alpha": log[f"round{ecfg.get('tune_rounds', 2) - 1}"]["alpha"]["edge"] is None,
                       "beta": log[f"round{ecfg.get('tune_rounds', 2) - 1}"]["beta"]["edge"] is None,
                       "beam": ebeam is None}
    return out, {k: _jsonable(v) for k, v in log.items()}


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    return o


def _combined(h, a, b, g, llm):
    text, ac, ng, _ = h
    return ac + a * ng + b * len(text.split()) + g * llm


def rescore_apply(nbests, scorer, a, b, g):
    """Pick the best hypothesis of each n-best list under the combined score."""
    flat = [h[0] for nb in nbests for h in nb]
    llm = dict(zip(flat, scorer.score(flat)))
    return [max(nb, key=lambda h: _combined(h, a, b, g, llm[h[0]]))[0] for nb in nbests]


def rescore_tune(nbests, refs, scorer, a, b, gammas, betas=None):
    """Choose the GPT-2 weight gamma (and, optionally, a re-tuned word bonus) on dev n-best lists.

    Returns ``(gamma, beta, dev_wer, log)``. Gamma is searched with edge extension (bounded below by 0);
    for each gamma the word bonus is re-optimised over ``betas`` around the n-gram value.
    """
    log = {}

    def best_beta(g):
        cand = sorted(set([b] + [b + d for d in (betas or [])]))
        scores = {bb: wer(refs, rescore_apply(nbests, scorer, a, bb, g)) for bb in cand}
        bb = min(cand, key=lambda v: (scores[v], abs(v - b)))
        log[g] = {"beta": bb, "dev_wer": scores[bb]}
        return scores[bb], bb

    gbest, gvals, edge = search_1d(lambda g: best_beta(g)[0], gammas, "mul", lo=0.0)
    if gbest == 0.0 and edge == "low":
        edge = "bound (gamma = 0: GPT-2 not used)"
    return gbest, log[gbest]["beta"], log[gbest]["dev_wer"], {"grid": _jsonable(log), "edge": edge}


def evaluate_system(name, dev, test, dec, ecfg, scorer=None):
    """Full decode of one system; returns (metrics dict, per-sentence decodes).

    All hyper-parameters (beam, alpha, beta, lexicon-only beta, gamma) are tuned on the dev split
    (held-out training-session trials, disjoint from the 426 test sentences); test is decoded once.
    """
    n_tune = ecfg["n_tune"]
    n_tune = len(dev["log_probs"]) if n_tune in (None, "all") else int(n_tune)
    dev_lps, dev_refs = dev["log_probs"][:n_tune], dev["sentences"][:n_tune]
    t_refs_w = [words(s) for s in test["sentences"]]
    greedy = [greedy_decode(lp) for lp in test["log_probs"]]
    e_ph, t_ph = sentence_edits(test["phones"], greedy)
    out = {"per_greedy": float(e_ph.sum() / t_ph.sum()), "per_greedy_ci95": bootstrap_ci_edits(e_ph, t_ph),
           "n_dev_tune": n_tune, "n_test": len(test["sentences"])}
    decodes = {"greedy_phones": greedy}
    edits = {"per_greedy": e_ph.tolist(), "n_phones": t_ph.tolist()}
    tuned, log = tune_decoder(dec, dev_lps, dev_refs, ecfg)
    out.update(tuned)
    out["tuning_log"] = log
    beam = tuned["beam"]

    def score(key, hyps):
        e, t = sentence_edits(t_refs_w, [words(h) for h in hyps])
        out[key] = float(e.sum() / t.sum())
        out[key + "_ci95"] = bootstrap_ci_edits(e, t)
        edits[key] = e.tolist()
        edits["n_words"] = t.tolist()

    hyp_lex = [h[0][0] for h in dec(test["log_probs"], 0.0, tuned["beta_lexicon"], beam=beam)]
    score("wer_lexicon", hyp_lex)
    decodes["lexicon"] = hyp_lex
    nb_test = dec(test["log_probs"], tuned["alpha"], tuned["beta"], nbest=ecfg["nbest"], beam=beam)
    hyp_ng = [h[0][0] for h in nb_test]
    score("wer_ngram", hyp_ng)
    decodes["ngram"] = hyp_ng
    if scorer is not None:
        nb_dev = dec(dev_lps, tuned["alpha"], tuned["beta"], nbest=ecfg["nbest"], beam=beam)
        g, b_llm, dev_llm, glog = rescore_tune(nb_dev, dev_refs, scorer, tuned["alpha"], tuned["beta"], ecfg["gammas"],
                                               ecfg.get("rescore_beta_offsets"))
        hyp_llm = rescore_apply(nb_test, scorer, tuned["alpha"], b_llm, g)
        out.update(gamma=g, beta_llm=b_llm, dev_wer_llm=dev_llm)
        out["interior"]["gamma"] = glog["edge"] is None
        out["tuning_log"]["gamma"] = glog
        score("wer_llm", hyp_llm)
        decodes["llm"] = hyp_llm
        # oracle: best hypothesis in the n-best list (upper bound for any rescoring)
        oracle = [min(nb, key=lambda h: wer([r], [h[0]]))[0] for nb, r in zip(nb_test, test["sentences"])]
        out["wer_nbest_oracle"] = wer(test["sentences"], oracle)
    decodes["edits"] = edits
    brief = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in out.items() if k != "tuning_log"}
    print(f"[eval] {name}: " + json.dumps(brief), flush=True)
    return out, decodes


def compare_systems(cfg: dict, res: Path, all_metrics: dict) -> dict:
    """Paired sentence-level bootstrap tests for the pairs listed in ``eval.compare``.

    Each side is a list of systems; with several systems (seeds) the per-sentence edit counts are
    averaged over them, so the test asks whether the *recipe* differs, not one lucky seed.
    """
    out = {}
    for spec in cfg["eval"].get("compare", []):
        sides = []
        for names in (spec["a"], spec["b"]):
            names = [n for n in names if (res / "decodes" / f"{n}.json").exists()]
            sides.append([json.loads((res / "decodes" / f"{n}.json").read_text())["decodes"]["edits"] for n in names])
        if not sides[0] or not sides[1]:
            continue
        r = {"a": spec["a"], "b": spec["b"]}
        for key, tkey in (("per_greedy", "n_phones"), ("wer_ngram", "n_words"), ("wer_llm", "n_words")):
            if all(key in e for side in sides for e in side):
                ea, eb = (np.mean([e[key] for e in side], axis=0) for side in sides)
                r[key] = paired_bootstrap(ea, eb, sides[0][0][tkey], n=cfg["eval"].get("n_bootstrap", 10000))
        out[spec["name"]] = r
        print(f"[eval] compare {spec['name']}: " + json.dumps({k: v for k, v in r.items() if k not in ('a', 'b')}), flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--results-dir", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.results_dir:
        cfg["results_dir"] = str(Path(args.results_dir).resolve())
    res = Path(cfg["results_dir"])
    ecfg = cfg["eval"]
    corpora = corpus_map(Path(cfg["data_dir"]) / "t15_copyTaskData_description.csv")

    # collect systems
    systems: dict[str, tuple[dict, dict]] = {}
    seeds_of: dict[str, list[int]] = {}
    for run, seed in cfg["schedule"]:
        d = res / "runs" / f"{run}_s{seed}"
        if (d / "summary.json").exists():
            systems[f"{run}_s{seed}"] = (load_posteriors(d / "posteriors_dev.npz"), load_posteriors(d / "posteriors_test.npz"))
            seeds_of.setdefault(run, []).append(seed)
    system_members = {name: [name] for name in systems}
    for run, seeds in seeds_of.items():
        if len(seeds) > 1:
            system_members[f"{run}_ens"] = [f"{run}_s{s}" for s in seeds]
            members = [systems[f"{run}_s{s}"] for s in seeds]
            systems[f"{run}_ens"] = (ensemble([m[0] for m in members]), ensemble([m[1] for m in members]))

    cache_dir = res / "decodes"
    cache_dir.mkdir(parents=True, exist_ok=True)
    scorer = None
    llm_systems = set(systems) if ecfg["llm_systems"] == "all" else set(ecfg["llm_systems"])
    settings = {k: ecfg[k] for k in sorted(ecfg) if k not in ("workers", "llm_systems")}
    if any(s in systems for s in llm_systems):
        from .rescore import GPT2Scorer

        scorer = GPT2Scorer(Path(cfg["data_dir"]) / "gpt2")
    dec = Decoder(cfg)
    all_metrics = {}
    for name, (dev, test) in systems.items():
        cache = cache_dir / f"{name}.json"
        if cache.exists():
            m = json.loads(cache.read_text())
            fresh = m.get("members") == system_members[name] and m.get("settings") == settings
            if fresh and (name not in llm_systems or "wer_llm" in m["metrics"]):
                all_metrics[name] = m["metrics"]
                continue
        met, decs = evaluate_system(name, dev, test, dec, ecfg, scorer if name in llm_systems else None)
        # per-corpus WER for the n-gram decodes
        by = {}
        for i, s in enumerate(test["sentences"]):
            by.setdefault(corpora.get((test["sessions"][i], test["blocks"][i]), "unknown"), []).append(i)
        met["wer_ngram_by_corpus"] = {
            c: {"wer": wer([test["sentences"][i] for i in idx], [decs["ngram"][i] for i in idx]), "n": len(idx)}
            for c, idx in by.items()
        }
        met["members"] = system_members[name]
        all_metrics[name] = met
        cache.write_text(json.dumps({"metrics": met, "decodes": decs, "members": system_members[name], "settings": settings,
                                     "sentences": test["sentences"], "phones": test["phones"], "sessions": test["sessions"],
                                     "blocks": test["blocks"]}))
    all_metrics["_comparisons"] = compare_systems(cfg, res, all_metrics)
    dec.close()
    (res / "metrics_systems.json").write_text(json.dumps(all_metrics, indent=2))
    print(f"[eval] wrote {res / 'metrics_systems.json'}")


if __name__ == "__main__":
    main()
