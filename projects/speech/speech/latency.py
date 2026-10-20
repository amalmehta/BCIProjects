"""Latency and compute on CPU (2 threads): causal vs bidirectional decoders.

For each model we measure, on a median-length test sentence (batch size 1):

* ``full_forward_ms``: one forward pass over the whole sentence.
* ``per_frame_ms``: streaming cost per 120 ms output frame. For the GRU this
  is a true incremental step (with the frontend for that frame). For the
  causal Conformer it is the amortised full-forward cost, which is what a
  key/value-cached implementation would pay; we also report the cost of
  naively recomputing a 64-frame window every frame.
* ``algorithmic_lookahead_ms``: future input a frame needs before it can be
  emitted: the smoothing kernel's half-width for causal models, the whole
  remaining sentence for bidirectional ones.

We also time the language-model stages: beam search per sentence, and GPT-2
rescoring per n-best list.

Usage::

    python -m speech.latency --config configs/cpu.yaml
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from .config import lm_paths, load_config, run_config
from .models import GRUDecoder, build_model
from .train import make_batch, prepare_data


def _timeit(fn, n=7):
    """Minimum wall time over ``n`` repeats (ms): the least contention-affected estimate on a shared machine."""
    fn()
    ts = []
    for _ in range(n):
        t = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t)
    return float(np.min(ts) * 1000)


@torch.no_grad()
def measure(model, trial_batch, bin_ms: float) -> dict:
    x, x_len, _, _, day, _ = trial_batch
    model.eval()
    out = model(x, x_len, day)
    frames = int(out["lengths"][0])
    hop_ms = bin_ms * model.patch_stride
    full = _timeit(lambda: model(x, x_len, day))
    res = {"frames": frames, "hop_ms": hop_ms, "full_forward_ms": full,
           "sentence_ms": float(x.shape[1] * bin_ms), "real_time_factor": full / (x.shape[1] * bin_ms)}
    half_kernel_ms = (len(model.kernel) - 1) / 2 * bin_ms
    if isinstance(model, GRUDecoder) and model.causal:
        feats = model.features(x, day)
        h = None

        def step_all():
            nonlocal h
            h = None
            for t in range(feats.shape[1]):
                _, h = model.gru(feats[:, t : t + 1], h)

        total = _timeit(step_all, n=3)
        frontend = _timeit(lambda: model.features(x, day), n=3) / frames
        res["per_frame_ms"] = total / frames + frontend
        res["algorithmic_lookahead_ms"] = half_kernel_ms
    elif model.causal:
        res["per_frame_ms"] = full / frames
        W = 64
        nbins = (W - 1) * model.patch_stride + model.patch_size
        xw, lw = x[:, :nbins], torch.tensor([nbins])
        res["per_frame_ms_recompute_window64"] = _timeit(lambda: model(xw, lw, day))
        res["algorithmic_lookahead_ms"] = half_kernel_ms
    else:
        res["per_frame_ms"] = None
        res["algorithmic_lookahead_ms"] = "end of sentence"
        res["latency_after_sentence_end_ms"] = full
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--results-dir", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.results_dir:
        cfg["results_dir"] = str(Path(args.results_dir).resolve())
    torch.set_num_threads(cfg.get("threads", 2))
    res = Path(cfg["results_dir"])
    out = {"threads": cfg.get("threads", 2), "models": {}}
    for run in cfg["latency"]["runs"]:
        path = res / "runs" / run / "model.pt"
        if path.exists():
            ck = torch.load(path, map_location="cpu")
            rc = ck["run_config"]
            model = build_model(rc["model"], ck["n_days"])
            model.load_state_dict(ck["state_dict"])
            trained = True
        else:  # compute cost does not depend on the weights: time an untrained instance
            rc = run_config(cfg, run.rsplit("_s", 1)[0])
            data = prepare_data(rc["data"], cfg["data_dir"])
            model = build_model(rc["model"], data["n_days"])
            trained = False
        test = prepare_data(rc["data"], cfg["data_dir"])["test"]
        lens = [t.features.shape[0] for t in test]
        i = int(np.argsort(lens)[len(lens) // 2])
        r = measure(model, make_batch(test, [i]), 20.0 * rc["data"]["bin_factor"])
        r["n_params"] = sum(p.numel() for p in model.parameters())
        r["trained"] = trained
        out["models"][run] = r
        print(run, json.dumps(r), flush=True)

    # language-model stages
    from .evaluate import _decode_one, _init_worker
    from .train import load_posteriors

    sys = res / "decodes" / f"{cfg['calibration']['base_run']}.json"
    weights = json.loads(sys.read_text())["metrics"] if sys.exists() else {"alpha": 0.8, "beta": 1.0}
    post = load_posteriors(res / "runs" / cfg["calibration"]["base_run"] / "posteriors_test.npz")
    _init_worker(*map(str, lm_paths(cfg)), cfg["eval"]["beam"])
    lps = post["log_probs"][:30]
    t = time.perf_counter()
    beam = weights.get("beam") or cfg["eval"]["beam"]
    out["beam_width"] = beam
    nb = [_decode_one((lp, weights["alpha"], weights["beta"], cfg["eval"]["nbest"], beam)) for lp in lps]
    out["beam_search_ms_per_sentence"] = (time.perf_counter() - t) / len(lps) * 1000
    try:
        from .rescore import GPT2Scorer

        scorer = GPT2Scorer(Path(cfg["data_dir"]) / "gpt2")
        t = time.perf_counter()
        for hyps in nb:
            scorer.score([h[0] for h in hyps])
        out["gpt2_rescore_ms_per_sentence"] = (time.perf_counter() - t) / len(nb) * 1000
    except (ImportError, OSError) as e:  # transformers or weights missing
        out["gpt2_rescore_ms_per_sentence"] = None
        print("GPT-2 rescoring not timed:", e)
    (res / "latency.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
