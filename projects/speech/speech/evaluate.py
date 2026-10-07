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
from .metrics import bootstrap_ci, error_rate, wer, words
from .train import load_posteriors

_G: dict = {}  # per-worker globals (LM and trie are loaded once per process)


def _init_worker(lm_dir: str, built_dir: str, beam: int) -> None:
    lm = KneserNeyLM.load(Path(built_dir) / "kn3.npz")
    vocab = set((Path(built_dir) / "vocab.txt").read_text().split())
    trie = Trie(load_cmudict(Path(lm_dir) / "cmudict.dict"), vocab)
    _G.update(lm=lm, trie=trie, beam=beam)


def _decode_one(args):
    lp, alpha, beta, nbest = args
    dec = BeamDecoder(_G["trie"], _G["lm"], alpha=alpha, beta=beta, beam=_G["beam"])
    return [(" ".join(r.words), r.acoustic, r.lm_score, r.score) for r in dec.decode(lp, nbest=nbest)]


class Decoder:
    """Process pool that decodes lists of posteriors with given LM weights."""

    def __init__(self, cfg: dict):
        lm_dir, built_dir = lm_paths(cfg)
        e = cfg["eval"]
        self.pool = Pool(e["workers"], initializer=_init_worker, initargs=(str(lm_dir), str(built_dir), e["beam"]))

    def __call__(self, lps, alpha, beta, nbest=1):
        return self.pool.map(_decode_one, [(lp, alpha, beta, nbest) for lp in lps], chunksize=4)

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


def tune(dec: Decoder, lps, refs, alphas, betas, lexicon_only=False):
    """Coordinate search for (alpha, beta) minimising dev WER."""
    results = {}

    def run(a, b):
        if (a, b) not in results:
            hyps = [h[0][0] for h in dec(lps, a, b)]
            results[(a, b)] = wer(refs, hyps)
        return results[(a, b)]

    if lexicon_only:
        best_b = min(betas, key=lambda b: run(0.0, b))
        return 0.0, best_b, results
    b0 = betas[len(betas) // 2]
    best_a = min(alphas, key=lambda a: run(a, b0))
    best_b = min(betas, key=lambda b: run(best_a, b))
    return best_a, best_b, results


def evaluate_system(name, dev, test, dec, ecfg, scorer=None):
    """Full decode of one system; returns (metrics dict, per-sentence decodes)."""
    n_tune = ecfg["n_tune"]
    dev_lps, dev_refs = dev["log_probs"][:n_tune], dev["sentences"][:n_tune]
    out = {"per_greedy": error_rate(test["phones"], [greedy_decode(lp) for lp in test["log_probs"]])}
    decodes = {"greedy_phones": [greedy_decode(lp) for lp in test["log_probs"]]}
    # lexicon only
    _, b_lex, _ = tune(dec, dev_lps, dev_refs, [0.0], ecfg["betas_lexicon"], lexicon_only=True)
    hyp_lex = [h[0][0] for h in dec(test["log_probs"], 0.0, b_lex)]
    out["wer_lexicon"], out["beta_lexicon"] = wer(test["sentences"], hyp_lex), b_lex
    decodes["lexicon"] = hyp_lex
    # n-gram
    a, b, grid = tune(dec, dev_lps, dev_refs, ecfg["alphas"], ecfg["betas"])
    out.update(alpha=a, beta=b, dev_wer_ngram=grid[(a, b)])
    nb_test = dec(test["log_probs"], a, b, nbest=ecfg["nbest"])
    hyp_ng = [h[0][0] for h in nb_test]
    out["wer_ngram"] = wer(test["sentences"], hyp_ng)
    decodes["ngram"] = hyp_ng
    refs_w = [words(s) for s in test["sentences"]]
    out["wer_ngram_ci95"] = bootstrap_ci(refs_w, [h.split() for h in hyp_ng])
    if scorer is not None:
        nb_dev = dec(dev_lps, a, b, nbest=ecfg["nbest"])
        g = rescore_tune(nb_dev, dev_refs, scorer, a, b, ecfg["gammas"])
        hyp_llm = rescore_apply(nb_test, scorer, a, b, g)
        out["wer_llm"], out["gamma"] = wer(test["sentences"], hyp_llm), g
        out["wer_llm_ci95"] = bootstrap_ci(refs_w, [h.split() for h in hyp_llm])
        decodes["llm"] = hyp_llm
        # oracle: best hypothesis in the n-best list (upper bound for any rescoring)
        oracle = [min(nb, key=lambda h: wer([r], [h[0]]))[0] for nb, r in zip(nb_test, test["sentences"])]
        out["wer_nbest_oracle"] = wer(test["sentences"], oracle)
    print(f"[eval] {name}: " + json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in out.items()}), flush=True)
    return out, decodes


def _combined(h, a, b, g, llm):
    text, ac, ng, _ = h
    return ac + a * ng + b * len(text.split()) + g * llm


def rescore_apply(nbests, scorer, a, b, g):
    """Pick the best hypothesis of each n-best list under the combined score."""
    flat = [h[0] for nb in nbests for h in nb]
    llm = dict(zip(flat, scorer.score(flat)))
    return [max(nb, key=lambda h: _combined(h, a, b, g, llm[h[0]]))[0] for nb in nbests]


def rescore_tune(nbests, refs, scorer, a, b, gammas):
    """Choose the LLM weight on dev."""
    return min(gammas, key=lambda g: wer(refs, rescore_apply(nbests, scorer, a, b, g)))


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
    llm_systems = set(ecfg["llm_systems"])
    if any(s in systems for s in llm_systems):
        from .rescore import GPT2Scorer

        scorer = GPT2Scorer(Path(cfg["data_dir"]) / "gpt2")
    dec = Decoder(cfg)
    all_metrics = {}
    for name, (dev, test) in systems.items():
        cache = cache_dir / f"{name}.json"
        if cache.exists():
            m = json.loads(cache.read_text())
            fresh = m.get("members") == system_members[name]  # an ensemble gains members as seeds finish
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
        all_metrics[name] = met
        cache.write_text(json.dumps({"metrics": met, "decodes": decs, "members": system_members[name], "sentences": test["sentences"],
                                     "phones": test["phones"], "sessions": test["sessions"], "blocks": test["blocks"]}))
    dec.close()
    (res / "metrics_systems.json").write_text(json.dumps(all_metrics, indent=2))
    print(f"[eval] wrote {res / 'metrics_systems.json'}")


if __name__ == "__main__":
    main()
