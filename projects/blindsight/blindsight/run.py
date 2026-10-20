"""Command-line pipeline: data -> validate -> train -> evaluate -> ablate -> adapt -> figures -> export.

Examples::

    python -m blindsight.run --config configs/cpu.yaml --stage data
    python -m blindsight.run --config configs/cpu.yaml --stage train --shard 0/2
    python -m blindsight.run --config configs/cpu.yaml --stage evaluate --dataset mnist
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from .config import build_setup, load_config, variant_config
from .data import load, prepare
from .evaluate import (adapt_encoder, baseline_factories, evaluate_factory, evaluate_oracle, image_ceiling)
from .metrics import psnr, ssim
from .train import load_encoder, train_encoder


def _log_to(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)

    def log(msg: str) -> None:
        line = f"{time.strftime('%H:%M:%S')} {msg}"
        print(line, flush=True)
        with open(path, "a") as f:
            f.write(line + "\n")
    return log


def run_dir(cfg: dict, dataset: str, variant: str, seed: int) -> Path:
    return Path(cfg["results_dir"]) / "runs" / f"{dataset}_{variant}_s{seed}"


def encoder_variants(cfg: dict) -> list[str]:
    """Variants trained on every dataset and evaluated next to the main encoder (they carry _datasets)."""
    return [v for v, spec in cfg["variants"].items() if "_datasets" in spec]


def ablation_variants(cfg: dict) -> list[str]:
    return [v for v in cfg["variants"] if v not in encoder_variants(cfg)]


def training_jobs(cfg: dict, ablations: bool = True) -> list[tuple[str, str, int]]:
    """All (dataset, variant, seed) encoders the pipeline needs, ordered seed-major so that a
    first complete set (seed 0 of every encoder) finishes before the extra seeds."""
    jobs = []
    for s in sorted(set(cfg["seeds"])):
        jobs += [(d, "main", s) for d in cfg["datasets"]]
        for v in encoder_variants(cfg):
            spec = cfg["variants"][v]
            if s in spec["_seeds"]:
                jobs += [(d, v, s) for d in spec["_datasets"] if d in cfg["datasets"]]
    if ablations:
        jobs += [(cfg["ablation_dataset"], v, s) for v in ablation_variants(cfg) for s in cfg["ablation_seeds"]]
    return jobs


def stage_train(cfg: dict, shard: str, log, ablations: bool = True) -> None:
    i, n = (int(v) for v in shard.split("/"))
    for k, (d, v, s) in enumerate(training_jobs(cfg, ablations)):
        out = run_dir(cfg, d, v, s)
        if k % n != i or (out / "encoder.pt").exists():
            continue
        train_encoder(cfg, d, s, v, out, log=log)


def _strip(res: dict) -> dict:
    return {k: v for k, v in res.items() if not k.startswith("_")}


def stage_evaluate(cfg: dict, dataset: str, log) -> None:
    """Main comparison on one dataset: baselines, learned (per seed), adapted (per seed), oracle."""
    setup = build_setup(cfg)
    ec = cfg["eval"]
    train_set, test_set = load(dataset, cfg["data_root"], setup.params.resolution,
                               n_train=ec["n_clf_train"], n_test=ec["n_test"], seed=1)
    out = {"dataset": dataset, "layout": setup.layout.name, "n_electrodes": setup.n,
           "test_patients": cfg["patients"]["test_seeds"], "rows": []}
    out["image_ceiling_acc"] = image_ceiling(dataset, train_set, test_set, cfg)
    log(f"{dataset}: classifier on original images acc={out['image_ceiling_acc']:.3f}")
    examples = {}
    for name, fac in baseline_factories(setup).items():
        r = evaluate_factory(name, fac, setup, dataset, train_set, test_set, cfg, log, ec["dropout_levels"], keep_examples=12)
        examples[name] = r.pop("_examples")
        out["rows"].append({**r, "seed": None})
    # Unlabeled "shots" for adaptation come from a disjoint part of the training split.
    shots_set, _ = load(dataset, cfg["data_root"], setup.params.resolution, n_train=cfg["adapt"]["main_shots"], n_test=1, seed=7)
    shots = shots_set.all()[0].to(setup.device)
    for seed in cfg["seeds"]:
        enc, dec, _ = load_encoder(cfg, "main", run_dir(cfg, dataset, "main", seed))
        r = evaluate_factory("learned", lambda p: enc, setup, dataset, train_set, test_set, cfg, log,
                             ec["dropout_levels"], decoder=dec, keep_examples=12)
        examples.setdefault("learned", r.pop("_examples"))
        out["rows"].append({**r, "seed": seed})
        cache: dict[str, torch.nn.Module] = {}

        def adapted(p, enc=enc, dec=dec, cache=cache, seed=seed):
            key = p.name.split("_drop")[0]
            if key not in cache:
                cache[key] = adapt_encoder(enc, dec, setup, p, shots, cfg, mode="adapter", seed=seed)
            return cache[key]
        r = evaluate_factory("learned_adapted", adapted, setup, dataset, train_set, test_set, cfg, log,
                             ec["dropout_levels"], decoder=dec, keep_examples=12)
        examples.setdefault("learned_adapted", r.pop("_examples"))
        out["rows"].append({**r, "seed": seed})
    out["rows"].append({**evaluate_oracle(setup, test_set, cfg, log), "seed": None})
    x_ex, y_ex = test_set.batch(torch.arange(12))
    examples["input"] = x_ex
    ev = Path(cfg["results_dir"]) / "eval"
    ev.mkdir(parents=True, exist_ok=True)
    (ev / f"{dataset}.json").write_text(json.dumps(out, indent=1))
    torch.save({k: v.clone() for k, v in examples.items()}, ev / f"{dataset}_examples.pt")


def stage_ablate(cfg: dict, log) -> None:
    """Training-variant ablations (loss terms, safety constraints, electrode layouts) vs. the intensity baseline.

    Each variant's rows are cached in results/eval/ablations/<variant>.json, so the stage can be re-run
    as more variants finish training; results/eval/ablations.json collects whatever is available.
    """
    d = cfg["ablation_dataset"]
    ec = cfg["eval"]
    cache_dir = Path(cfg["results_dir"]) / "eval" / "ablations"
    cache_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for variant in ["main", *ablation_variants(cfg)]:
        cache = cache_dir / f"{variant}.json"
        runs = [run_dir(cfg, d, variant, s) for s in cfg["ablation_seeds"]]
        if not cache.exists() and all((r / "encoder.pt").exists() for r in runs):
            vcfg = variant_config(cfg, variant)
            vrows = []
            for seed, rd in zip(cfg["ablation_seeds"], runs):
                enc, dec, setup = load_encoder(cfg, variant, rd)
                train_set, test_set = load(d, cfg["data_root"], setup.params.resolution,
                                           n_train=ec["n_clf_train"], n_test=ec["n_test"], seed=1)
                r = evaluate_factory("learned", lambda p: enc, setup, d, train_set, test_set, vcfg, log, decoder=dec)
                vrows.append({**r, "variant": variant, "seed": seed, "layout": setup.layout.name, "n_electrodes": setup.n})
            if variant == "main" or variant.startswith(("threads", "utah")):
                fac = baseline_factories(setup)["intensity"]
                r = evaluate_factory("intensity", fac, setup, d, train_set, test_set, vcfg, log)
                vrows.append({**r, "variant": variant, "seed": None, "layout": setup.layout.name, "n_electrodes": setup.n})
            cache.write_text(json.dumps(vrows, indent=1))
        if cache.exists():
            rows += json.loads(cache.read_text())
    (Path(cfg["results_dir"]) / "eval" / "ablations.json").write_text(json.dumps({"dataset": d, "rows": rows}, indent=1))


def stage_adapt_shots(cfg: dict, log) -> None:
    """Few-shot adaptation curve: SSIM vs. number of unlabeled images, adapter-only vs. full fine-tuning."""
    d = cfg["ablation_dataset"]
    setup = build_setup(cfg)
    _, test_set = load(d, cfg["data_root"], setup.params.resolution, n_train=1, n_test=500, seed=1)
    x_te = test_set.all()[0].to(setup.device)
    rows = []
    for seed in cfg["ablation_seeds"]:
        enc, dec, _ = load_encoder(cfg, "main", run_dir(cfg, d, "main", seed))
        for sd in cfg["patients"]["test_seeds"][: cfg["adapt"]["curve_patients"]]:
            patient = setup.patient(sd)
            with torch.no_grad():
                p0 = setup.sim(setup.safety(enc(x_te)), patient)
            base = float(ssim(p0, x_te).mean())
            rows.append({"seed": seed, "patient": sd, "mode": "generic", "shots": 0, "ssim": base,
                         "psnr": float(psnr(p0, x_te).mean())})
            for shots in cfg["adapt"]["shots"]:
                shot_set, _ = load(d, cfg["data_root"], setup.params.resolution, n_train=shots, n_test=1, seed=7)
                xs = shot_set.all()[0].to(setup.device)
                for mode in ("adapter", "full"):
                    a = adapt_encoder(enc, dec, setup, patient, xs, cfg, mode=mode, seed=seed)
                    with torch.no_grad():
                        p = setup.sim(setup.safety(a(x_te)), patient)
                    rows.append({"seed": seed, "patient": sd, "mode": mode, "shots": shots,
                                 "ssim": float(ssim(p, x_te).mean()), "psnr": float(psnr(p, x_te).mean())})
                    log(f"adapt patient={sd} shots={shots} mode={mode} ssim={rows[-1]['ssim']:.3f} (generic {base:.3f})")
    ev = Path(cfg["results_dir"]) / "eval"
    (ev / "adapt_shots.json").write_text(json.dumps({"dataset": d, "rows": rows}, indent=1))


def stage_diagnostics(cfg: dict, log) -> None:
    """How encoders use their stimulation budget: share of active electrodes driven above the patient's
    detection threshold, and median current (seed-0 encoders, 300 test images, all test patients)."""
    out = {}
    for d in cfg["datasets"]:
        enc, dec, setup = load_encoder(cfg, "main", run_dir(cfg, d, "main", 0))
        _, test_set = load(d, cfg["data_root"], setup.params.resolution, n_train=1, n_test=300, seed=1)
        x = test_set.all()[0].to(setup.device)
        shots_set, _ = load(d, cfg["data_root"], setup.params.resolution, n_train=cfg["adapt"]["main_shots"], n_test=1, seed=7)
        shots = shots_set.all()[0].to(setup.device)
        acc: dict[str, list] = {"intensity": [], "learned": [], "learned_adapted": []}
        for sd in cfg["patients"]["test_seeds"]:
            patient = setup.patient(sd)
            thr = patient.threshold_current_ua(setup.params)
            encs = {"intensity": baseline_factories(setup)["intensity"](patient), "learned": enc,
                    "learned_adapted": adapt_encoder(enc, dec, setup, patient, shots, cfg, mode="adapter", seed=0)}
            with torch.no_grad():
                for k, e in encs.items():
                    a = setup.safety(e(x)) * patient.alive
                    active = a > 0
                    acc[k].append(((a - thr)[active] > 0).float().mean().item())
                    acc[k].append(a[active].median().item())
        out[d] = {k: {"frac_active_above_threshold": float(np.mean(v[0::2])), "median_active_current_ua": float(np.mean(v[1::2]))}
                  for k, v in acc.items()}
        log(f"diagnostics {d}: {out[d]}")
    ev = Path(cfg["results_dir"]) / "eval"
    (ev / "diagnostics.json").write_text(json.dumps(out, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--stage", required=True,
                    choices=["data", "validate", "train", "evaluate", "ablate", "adapt", "diagnostics", "summarize", "figures",
                             "export"])
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--shard", default="0/1", help="i/n: run every n-th training job starting at i")
    ap.add_argument("--force", action="store_true", help="recompute evaluate/adapt/diagnostics even if outputs exist")
    args = ap.parse_args()
    cfg = load_config(args.config)
    torch.set_num_threads(cfg.get("threads", 2))
    torch.set_flush_denormal(True)
    if torch.cuda.is_available():
        print("note: this pipeline's simulator/encoder code runs on CPU tensors; see docs/DESIGN.md")
    log = _log_to(Path(cfg["results_dir"]) / "logs" / f"{args.stage}{'_' + args.dataset if args.dataset else ''}.log")
    np.random.seed(0)
    ev_dir = Path(cfg["results_dir"]) / "eval"  # finished stages are skipped on re-runs unless --force
    if args.stage == "data":
        for d in cfg["datasets"]:
            log(f"prepared {prepare(d, cfg['data_root'])}")
    elif args.stage == "validate":
        from .validation import run_validation
        run_validation(cfg, log)
    elif args.stage == "train":
        stage_train(cfg, args.shard, log)
    elif args.stage == "evaluate":
        for d in ([args.dataset] if args.dataset else cfg["datasets"]):
            if args.force or not (ev_dir / f"{d}.json").exists():
                stage_evaluate(cfg, d, log)
    elif args.stage == "ablate":
        stage_ablate(cfg, log)
    elif args.stage == "adapt":
        if args.force or not (ev_dir / "adapt_shots.json").exists():
            stage_adapt_shots(cfg, log)
    elif args.stage == "diagnostics":
        if args.force or not (ev_dir / "diagnostics.json").exists():
            stage_diagnostics(cfg, log)
    elif args.stage == "summarize":
        from .summarize import summarize
        summarize(cfg, log)
    elif args.stage == "figures":
        from .figures import make_figures
        make_figures(cfg, log)
    elif args.stage == "export":
        from .export import export_site
        export_site(cfg, log)


if __name__ == "__main__":
    main()
