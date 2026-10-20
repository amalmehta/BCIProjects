"""Summarise CTC training dynamics of every run from its per-step trace (``trace.jsonl``).

For each run directory under ``results/runs`` (and the diagnostic probes in ``results/diag/runs``):

* ``plateau_steps``    steps (after the first 10) during which the batch output is >= 95% blank by argmax
                       (the "blank plateau": the model emits nothing but blanks)
* ``escape_step``      first evaluation step with dev greedy PER below 0.80 (the model emits useful phonemes)
* ``loss_at``          10-step mean training loss at steps 50/100/150/300/480
* ``grad_norm``        median and max pre-clip gradient norm, and the fraction of steps that were clipped
* ``best_dev_per``     best dev greedy PER

Usage::

    python -m speech.stability --config configs/cpu.yaml   # writes results/stability.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .config import load_config


def summarise(run_dir: Path, grad_clip: float | None = None) -> dict | None:
    tp, lp = run_dir / "trace.jsonl", run_dir / "log.jsonl"
    if not tp.exists():
        return None
    tr = [json.loads(l) for l in tp.read_text().splitlines() if l.strip()]
    log = [json.loads(l) for l in lp.read_text().splitlines() if l.strip()] if lp.exists() else []
    loss = np.array([r["loss"] for r in tr])
    gn = np.array([r["grad_norm"] for r in tr])
    blank = np.array([r["blank_argmax_frac"] for r in tr])
    out = {
        "steps": len(tr),
        "plateau_steps": int((blank[10:] >= 0.95).sum()),
        "escape_step": next((r["step"] for r in log if r["dev_per"] < 0.80), None),
        "loss_at": {str(s): float(loss[max(0, s - 10):s].mean()) for s in (50, 100, 150, 300, 480) if s <= len(loss)},
        "grad_norm": {"median": float(np.median(gn)), "max": float(gn.max()),
                      "frac_clipped": float((gn > grad_clip).mean()) if grad_clip else None},
        "initial_loss": float(loss[:5].mean()),
        "best_dev_per": min((r["dev_per"] for r in log), default=None),
        "dev_per_curve": [[r["step"], r["dev_per"]] for r in log],
    }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    res = Path(cfg["results_dir"])
    out = {"note": "Per-run CTC training diagnostics from per-step traces (see speech/stability.py).", "runs": {}, "probes": {}}
    for key, root in (("runs", res / "runs"), ("probes", res / "diag" / "runs")):
        for d in sorted(root.glob("*_s*")):
            clip = None
            s = d / "model.pt"
            if s.exists():
                import torch

                clip = torch.load(s, map_location="cpu")["run_config"]["train"].get("grad_clip")
            r = summarise(d, clip)
            if r:
                out[key][d.name] = r
    (res / "stability.json").write_text(json.dumps(out, indent=2))
    for k in ("runs", "probes"):
        for n, r in out[k].items():
            print(k, n, {kk: r[kk] for kk in ("plateau_steps", "escape_step", "best_dev_per", "initial_loss")}, r["grad_norm"])


if __name__ == "__main__":
    main()
