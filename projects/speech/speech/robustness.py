"""Robustness studies at test time (no retraining).

1. Electrode loss: PER when a fraction of electrodes is silenced.
2. Day-layer swap: PER when each session is decoded through the wrong day's input layer
   (previous session, most distant session) or through an identity layer. This measures
   how much the decoder relies on day-specific input layers.

Electrode loss details:

Chronic implants lose channels over months (broken threads, rising impedance).
We zero both features (TX and SBP) of a random subset of electrodes for every
test trial (three fixed random masks per fraction, averaged) and measure greedy
PER, without retraining.

Usage::

    python -m speech.robustness --config configs/cpu.yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .config import get_device, load_config
from .data import Trial
from .models import build_model
from .train import greedy_per, posteriors, prepare_data


def drop_electrodes(trials: list[Trial], frac: float, rng: np.random.Generator, n_el: int = 256) -> list[Trial]:
    """Copy of ``trials`` with the same random electrode subset zeroed in all of them."""
    if frac == 0:
        return trials
    lost = rng.choice(n_el, int(round(frac * n_el)), replace=False)
    cols = np.concatenate([lost, lost + n_el])
    out = []
    for t in trials:
        x = t.features.copy()
        x[:, cols] = 0
        out.append(Trial(x, t.phones, t.sentence, t.day, t.session, t.block))
    return out


def day_layer_swap(model, test: list[Trial], n_train_days: int, device) -> dict[str, float]:
    """Greedy PER with the correct, the previous, the most distant and an identity day layer."""
    import copy

    def remap(f):
        return [Trial(t.features, t.phones, t.sentence, f(t.day), t.session, t.block) for t in test]

    out = {"own day": greedy_per(posteriors(model, test, device), test)}
    prev = remap(lambda d: d - 1 if d > 0 else 1)
    out["previous session's layer"] = greedy_per(posteriors(model, prev, device), prev)
    far = remap(lambda d: n_train_days - 1 - d if n_train_days - 1 - d != d else 0)
    out["most distant session's layer"] = greedy_per(posteriors(model, far, device), far)
    ident = copy.deepcopy(model)
    with torch.no_grad():
        ident.day.weight.copy_(torch.eye(ident.day.weight.shape[-1]).expand_as(ident.day.weight))
        ident.day.bias.zero_()
    out["identity (no day layer)"] = greedy_per(posteriors(ident, test, device), test)
    return out


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
    res = Path(cfg["results_dir"])
    rcfg = cfg["robustness"]
    if not args.force and (res / "robustness.json").exists() \
            and json.loads((res / "robustness.json").read_text()).get("settings") == rcfg:
        print("[robustness] up to date (use --force to rerun)")
        return
    out = {"settings": rcfg, "drop_fractions": rcfg["drop_fractions"], "n_masks": rcfg.get("n_masks", 3), "per": {}}
    for run in rcfg["runs"]:
        path = res / "runs" / run / "model.pt"
        if not path.exists():
            continue
        ck = torch.load(path, map_location="cpu")
        rc = ck["run_config"]
        model = build_model(rc["model"], ck["n_days"])
        model.load_state_dict(ck["state_dict"])
        model.to(device)
        test = prepare_data(rc["data"], cfg["data_dir"])["test"]
        pers = []
        for frac in rcfg["drop_fractions"]:
            vals = []
            for k in range(1 if frac == 0 else out["n_masks"]):
                trials = drop_electrodes(test, frac, np.random.default_rng(1000 + k))
                vals.append(greedy_per(posteriors(model, trials, device), trials))
            pers.append(float(np.mean(vals)))
            print(json.dumps({"run": run, "drop": frac, "per": pers[-1]}), flush=True)
        out["per"][run] = pers
        if run in rcfg.get("day_swap_runs", []):
            n_train = len(rc["data"]["sessions"])
            out.setdefault("day_swap", {})[run] = day_layer_swap(model, test, n_train, device)
            print(json.dumps({"run": run, "day_swap": out["day_swap"][run]}), flush=True)
    (res / "robustness.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
