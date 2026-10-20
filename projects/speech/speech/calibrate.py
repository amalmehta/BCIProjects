"""Calibration study: how much data does a new recording day need?

A base model is trained without the held-out session (the latest one in the
config, recorded ~2.5 months after the last training session). For the new
day we compare:

* ``reuse``      no new data; the new day reuses the most recent day's input layer
* ``day_layer``  fine-tune only the new day's input layer on N sentences
                 (initialised from the most recent day's layer)
* ``full``       fine-tune the whole network on N sentences

and, without any labels from the new day (``speech.adapt``):

* ``tta_moments`` / ``tta_coral``  match the new day's feature statistics to the last training day
* ``tta_self_train``               pseudo-label self-training of the day layer (LM-constrained decodes,
                                   confidence filtering), starting from ``reuse``
* ``tta_coral+self_train``         self-training starting from the CORAL-initialised layer

The unlabelled adaptation data is the new day's ``train`` split (the same pool the supervised
modes draw labelled sentences from); the test sentences are never used for adaptation.
We report greedy PER and n-gram WER on that day's held-out (``val``) sentences.

Usage::

    python -m speech.calibrate --config configs/cpu.yaml
"""

from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path

import numpy as np
import torch

from .config import get_device, load_config
from .adapt import self_train, stats_adapt
from .config import lm_paths
from .evaluate import Decoder
from .lm import load_cmudict
from .metrics import wer
from .models import build_model
from .train import greedy_per, posteriors, prepare_data, train_loop


def finetune(base, trials, new_day, prev_day, mode, ccfg, rc, device, seed):
    """Copy ``base``, initialise the new day's layer from ``prev_day`` and fine-tune."""
    model = copy.deepcopy(base)
    with torch.no_grad():
        model.day.weight[new_day] = model.day.weight[prev_day]
        model.day.bias[new_day] = model.day.bias[prev_day]
    if mode == "reuse" or not trials:
        return model
    for n, p in model.named_parameters():
        p.requires_grad = (mode == "full") or n.startswith("day.")
    rc = copy.deepcopy(rc)
    rc["train"].update(
        max_steps=ccfg["steps"], max_minutes=1e9, schedule="steps", warmup_steps=5,
        lr=ccfg["lr_full"] if mode == "full" else ccfg["lr_day"],
        batch_size=min(ccfg["batch_size"], len(trials)), eval_every=10**9, lr_min_ratio=0.1,
    )
    # no early stopping on the new day: there is no separate dev data at calibration time
    _, state = train_loop(model, trials, trials[: min(len(trials), 10)], rc, device, seed)
    model.load_state_dict(state)
    for p in model.parameters():
        p.requires_grad = True
    return model


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--results-dir", default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.results_dir:
        cfg["results_dir"] = str(Path(args.results_dir).resolve())
    torch.set_num_threads(cfg.get("threads", 2))
    device = get_device(cfg.get("device", "auto"))
    ccfg = cfg["calibration"]
    res = Path(cfg["results_dir"])
    if not args.force and (res / "calibration.json").exists() \
            and json.loads((res / "calibration.json").read_text()).get("settings") == ccfg:
        print("[calibrate] up to date (use --force to rerun)")
        return
    ck = torch.load(res / "runs" / ccfg["base_run"] / "model.pt", map_location="cpu")
    print(f"[calibrate] base model {ccfg['base_run']}", flush=True)
    rc = ck["run_config"]
    data = prepare_data(rc["data"], cfg["data_dir"])
    base = build_model(rc["model"], ck["n_days"])
    base.load_state_dict(ck["state_dict"])
    base.to(device)
    new_day, prev_day = data["n_days"] - 1, data["n_days"] - 2
    pool, test = data["heldout_train"], data["heldout_test"]
    refs = [t.sentence for t in test]

    sys_cache = res / "decodes" / f"{ccfg['base_run']}.json"
    m = json.loads(sys_cache.read_text())["metrics"] if sys_cache.exists() else {"alpha": 0.8, "beta": 1.0}
    beam = m.get("beam")
    dec = Decoder(cfg)

    def decode_fn(lps):
        return [h[0][0] for h in dec(lps, m["alpha"], m["beta"], beam=beam)]

    def evaluate_model(model, row):
        lps = posteriors(model, test, device)
        row.update(per=greedy_per(lps, test), wer=wer(refs, decode_fn(lps)))
        rows.append(row)
        print(json.dumps(row), flush=True)

    rows = []
    # ---- unsupervised test-time adaptation (no new-day labels)
    tcfg = ccfg.get("tta") or {}
    if tcfg:
        lexicon = load_cmudict(lm_paths(cfg)[0] / "cmudict.dict")
        ref = [t for t in data["train"] if t.day == prev_day]
        reuse = finetune(base, [], new_day, prev_day, "reuse", ccfg, rc, device, 0)
        coral = stats_adapt(base, pool, ref, new_day, prev_day, "coral", tcfg.get("coral_shrink", 0.1))
        for name, model in (("tta_moments", stats_adapt(base, pool, ref, new_day, prev_day, "moments")), ("tta_coral", coral)):
            evaluate_model(model, {"mode": name, "n_sentences": 0, "n_unlabelled": len(pool), "repeat": 0})
        for start_name, start in (("tta_self_train", reuse), ("tta_coral+self_train", coral)):
            log: list = []
            torch.manual_seed(0)
            model = self_train(start, pool, decode_fn, lexicon, rc, device, rounds=tcfg.get("rounds", 2),
                               keep_frac=tcfg.get("keep_frac", 0.5), steps=tcfg.get("steps", ccfg["steps"]),
                               lr=tcfg.get("lr", ccfg["lr_day"]), batch_size=ccfg["batch_size"], seed=0, log=log)
            evaluate_model(model, {"mode": start_name, "n_sentences": 0, "n_unlabelled": len(pool), "repeat": 0,
                                   "self_train_log": log})
    for rep in range(ccfg["repeats"]):
        rng = random.Random(100 + rep)
        order = list(range(len(pool)))
        rng.shuffle(order)
        for mode in ccfg.get("modes", ("reuse", "day_layer", "full")):
            for n in ([0] if mode == "reuse" else ccfg["n_sentences"]):
                if mode == "reuse" and rep > 0:
                    continue
                trials = [pool[i] for i in order[: min(n, len(pool))]]
                torch.manual_seed(rep)
                model = finetune(base, trials, new_day, prev_day, mode, ccfg, rc, device, seed=rep)
                evaluate_model(model, {"mode": mode, "n_sentences": len(trials), "repeat": rep})
    dec.close()
    out = {"settings": ccfg, "base_run": ccfg["base_run"], "decoder": {k: m.get(k) for k in ("alpha", "beta", "beam")}, "heldout_session": rc["data"]["heldout_session"],
           "n_test_sentences": len(test), "pool_size": len(pool), "rows": rows}
    (res / "calibration.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
