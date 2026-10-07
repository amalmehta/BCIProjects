"""End-to-end experiment pipeline, organised in cached stages.

    python -m telepathy.experiments --config configs/cpu.yaml --stage <stage> [--models sld:0,gru:0]

Stages (each writes JSON under ``results/`` and can be re-run independently):

* ``prep``      choose the neural lag, fit classical decoders (Wiener, Kalman, FA stabiliser)
* ``train``     train neural decoders (GRU baseline, proposed SLD, ablations) -> checkpoints
* ``eval``      within-day R^2, cross-day R^2 (fixed / re-centred / aligned), latency
* ``fewshot``   few-shot recalibration curves on held-out days
* ``stress``    electrode dropout and channel-shift stress tests
* ``sim``       closed-loop WebGrid simulator: tuning, bitrate, grid sweep, cross-day, ReFIT
* ``export``    metrics.json, figures and the website export (see :mod:`telepathy.export`)
"""
from __future__ import annotations

import argparse
import copy
import dataclasses
import json
import pickle
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from .align import FAStabilizedDecoder
from .data import load_sessions
from .features import DayData, channel_stats, lagged, make_day
from .linear import Decoder, KalmanFilter, WienerFilter
from .metrics import r2_score
from .simulator import (UserModel, WebGridConfig, calibrate_noise, fit_encoding_model, fit_user_speed_profile,
                        run_webgrid)
from .train import (AdaptConfig, TorchDecoder, TrainConfig, adapt_sld, finetune_gru, latent_stats, train_neural)

PROJECT = Path(__file__).resolve().parents[1]


# ----------------------------------------------------------------------------------------
# Context: configuration, data, cached artefacts
# ----------------------------------------------------------------------------------------
class Ctx:
    """Loads the config and data once; gives paths for results and checkpoints."""

    def __init__(self, config: str):
        self.cfg = yaml.safe_load(open(config))
        torch.set_num_threads(int(self.cfg.get("torch_threads", 2)))
        self.res = PROJECT / self.cfg["results_dir"]
        self.ckpt = self.res / "checkpoints"
        for p in (self.res, self.ckpt, self.res / "logs"):
            p.mkdir(parents=True, exist_ok=True)
        self.sessions = load_sessions(PROJECT / self.cfg["data_dir"])
        if not self.sessions:
            raise SystemExit("No binned sessions found - run the data download first.")
        self.t0 = self.sessions[0].day
        prep = self.res / "prep.json"
        self.lag = json.loads(prep.read_text())["lag_bins"] if prep.exists() else None
        if self.lag is not None:
            self._make_days()

    def _make_days(self) -> None:
        self.days = [make_day(s, self.lag, int((s.day - self.t0).astype(int))) for s in self.sessions]
        n = self.cfg["n_anchor_days"]
        self.anchors, self.tests = self.days[:n], self.days[n:]
        self.vscale = float(np.concatenate([d.Y[d.train] for d in self.anchors]).std())
        self.anchor_stats = [channel_stats(d.X[d.train]) for d in self.anchors]

    def log(self, msg: str, name: str = "pipeline") -> None:
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        with open(self.res / "logs" / f"{name}.log", "a") as f:
            f.write(line + "\n")

    def save_json(self, name: str, obj) -> None:
        (self.res / name).write_text(json.dumps(obj, indent=1, default=_json_default))

    def load_json(self, name: str):
        return json.loads((self.res / name).read_text())

    # -- checkpoints ----------------------------------------------------------------------
    def save_model(self, name: str, payload: dict) -> None:
        torch.save(payload, self.ckpt / f"{name}.pt")

    def load_model(self, name: str) -> dict:
        return torch.load(self.ckpt / f"{name}.pt", weights_only=False)

    def linear(self) -> dict:
        return pickle.load(open(self.ckpt / "linear.pkl", "rb"))

    def aligned_cache(self) -> dict:
        if not hasattr(self, "_aligned"):
            p = self.ckpt / "aligned.pkl"
            self._aligned = pickle.load(open(p, "rb")) if p.exists() else {}
        return self._aligned

    def save_aligned_cache(self) -> None:
        pickle.dump(self.aligned_cache(), open(self.ckpt / "aligned.pkl", "wb"))


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def unlabelled(d: DayData) -> np.ndarray:
    """Spike counts of a day's train block, used *without* labels for adaptation."""
    return d.X[d.train]


def test_r2(dec: Decoder, d: DayData, ctx_bins: int = 300) -> float:
    """R^2 on the test block, decoding it causally with ``ctx_bins`` (6 s) of preceding history."""
    a = max(0, d.test.start - ctx_bins)
    return r2_score(d.Y[d.test], dec.predict(d.X[a:])[d.test.start - a :])


# ----------------------------------------------------------------------------------------
# Stage: prep (lag + classical decoders)
# ----------------------------------------------------------------------------------------
def fit_pooled_wiener(days: list[DayData], stats, n_lags: int, alpha: float) -> WienerFilter:
    """Wiener filter trained on several days, each z-scored with its own statistics."""
    w = WienerFilter(n_lags, alpha)
    Z, Y = [], []
    for d, (mu, sd) in zip(days, stats):
        w.mu, w.sd = mu, sd
        Z.append(w._design(d.X[d.train]))
        Y.append(d.Y[d.train])
    Z, Y = np.concatenate(Z).astype(np.float64), np.concatenate(Y).astype(np.float64)
    reg = alpha * np.eye(Z.shape[1])
    reg[-1, -1] = 0
    w.W = np.linalg.solve(Z.T @ Z + reg, Z.T @ Y).astype(np.float32)
    w.mu, w.sd = stats[-1]
    return w


def fit_pooled_kalman(days: list[DayData], stats, smooth: int, q: float) -> KalmanFilter:
    """Kalman filter trained on several days; per-day z-scoring is folded into the inputs."""
    segs = [(((d.X[d.train] - mu) / sd).astype(np.float32), d.Y[d.train]) for d, (mu, sd) in zip(days, stats)]
    C = days[0].X.shape[1]
    k = KalmanFilter(smooth, q_scale=q).fit(segs, np.zeros(C, np.float32), np.ones(C, np.float32))
    k.mu, k.sd = stats[-1]
    return k


def stage_prep(ctx: Ctx) -> None:
    cfg = ctx.cfg
    n = cfg["n_anchor_days"]
    wc = cfg["wiener"]
    # 1. lag: mean validation R^2 of per-day Wiener filters on the anchor days
    lag_scores = {}
    for lag in cfg["lag_candidates"]:
        sc = []
        for s in ctx.sessions[:n]:
            d = make_day(s, lag, 0)
            mu, sd = channel_stats(d.X[d.train])
            w = WienerFilter(wc["n_lags"], 1e3).fit([(d.X[d.train], d.Y[d.train])], mu, sd)
            sc.append(r2_score(d.Y[d.val], w.predict(d.X)[d.val]))
        lag_scores[lag] = float(np.mean(sc))
        ctx.log(f"lag {lag} bins: anchor val R2 {lag_scores[lag]:.4f}")
    ctx.lag = int(max(lag_scores, key=lag_scores.get))
    ctx._make_days()
    A, S = ctx.anchors, ctx.anchor_stats
    # 2. classical decoders, hyper-parameters chosen on anchor validation blocks
    def val(dec, d, st):
        return r2_score(d.Y[d.val], dec.with_stats(*st).predict(d.X)[d.val])

    per_day_w, per_day_k = [], []
    for d, st in zip(A, S):
        seg = [(d.X[d.train], d.Y[d.train])]
        per_day_w.append(max((WienerFilter(wc["n_lags"], a).fit(seg, *st) for a in wc["alphas"]),
                             key=lambda w: val(w, d, st)))
        kc = cfg["kalman"]
        per_day_k.append(max((KalmanFilter(sb, q_scale=q).fit(seg, *st) for sb in kc["smooth_bins"]
                              for q in kc["q_scales"]), key=lambda k: val(k, d, st)))
    pooled_w = max((fit_pooled_wiener(A, S, wc["n_lags"], a) for a in wc["alphas"]),
                   key=lambda w: np.mean([val(w, d, st) for d, st in zip(A, S)]))
    kc = cfg["kalman"]
    pooled_k = max((fit_pooled_kalman(A, S, sb, q) for sb in kc["smooth_bins"] for q in kc["q_scales"]),
                   key=lambda k: np.mean([val(k, d, st) for d, st in zip(A, S)]))
    fc = cfg["fa"]
    fa = FAStabilizedDecoder(fc["q"], fc["smooth_bins"], fc["n_lags"], stable_frac=fc["stable_frac"]).fit(A)
    pickle.dump({"wiener_day": per_day_w, "kalman_day": per_day_k, "wiener": pooled_w, "kalman": pooled_k,
                 "fa": fa}, open(ctx.ckpt / "linear.pkl", "wb"))
    vmax, d0 = fit_user_speed_profile(A)
    ctx.save_json("prep.json", {
        "lag_bins": ctx.lag, "lag_ms": ctx.lag * 20, "lag_scores": lag_scores, "vscale": ctx.vscale,
        "anchor_days": [d.name for d in A], "test_days": [d.name for d in ctx.tests],
        "test_day_index": [d.day_index for d in ctx.tests],
        "kalman_pooled": {"smooth_bins": pooled_k.smooth_bins, "q_scale": pooled_k.q_scale},
        "wiener_pooled_alpha": pooled_w.alpha, "user_vmax": vmax, "user_d0": d0,
        "n_bins": {d.name: d.T for d in ctx.days}, "n_trials": {d.name: len(d.trial_starts) for d in ctx.days},
    })
    ctx.log(f"prep done: lag={ctx.lag} bins; pooled Wiener alpha={pooled_w.alpha}; "
            f"KF smooth={pooled_k.smooth_bins} q={pooled_k.q_scale}; user vmax={vmax:.1f} d0={d0:.1f}")


# ----------------------------------------------------------------------------------------
# Stage: train neural decoders
# ----------------------------------------------------------------------------------------
def model_specs(cfg: dict) -> dict[str, dict]:
    """name -> {kind, train overrides, model kwargs} for the main models and ablations."""
    specs = {"gru": {"kind": "gru", "train": {"augment": False}, "model": dict(cfg["gru"])},
             "sld": {"kind": "sld", "train": {}, "model": dict(cfg["sld"])}}
    for name, ab in cfg.get("ablations", {}).items():
        kind = ab.get("kind", "sld")
        model = dict(cfg[kind])
        model.update(ab.get("model", {}))
        tr = {k: v for k, v in ab.items() if k not in ("kind", "model")}
        specs[name] = {"kind": kind, "train": tr, "model": model}
    return specs


def stage_train(ctx: Ctx, models: list[str]) -> None:
    specs = model_specs(ctx.cfg)
    for item in models:
        name, seed = item.split(":")
        seed = int(seed)
        tag = f"{name}_s{seed}"
        if (ctx.ckpt / f"{tag}.pt").exists():
            ctx.log(f"skip {tag} (exists)")
            continue
        sp = specs[name]
        tc = TrainConfig(**{**ctx.cfg["train"], **sp["train"], "seed": seed})
        t0 = time.time()
        model, vscale, stats, hist = train_neural(sp["kind"], ctx.anchors, tc, sp["model"],
                                                  log=lambda m: ctx.log(m, f"train_{tag}"))
        ctx.save_model(tag, {"model": model, "vscale": vscale, "stats": stats, "hist": hist, "kind": sp["kind"],
                             "train_seconds": time.time() - t0, "n_params": sum(p.numel() for p in model.parameters())})
        ctx.log(f"trained {tag} in {time.time() - t0:.0f}s, best val R2 {max(hist['val_r2']):.4f}")


# ----------------------------------------------------------------------------------------
# Neural decoder helpers
# ----------------------------------------------------------------------------------------
def neural_fixed(ck: dict, day: int = -1) -> TorchDecoder:
    """Decoder as left after calibration: last anchor day's slot and statistics."""
    mu, sd = ck["stats"][day]
    slot = (len(ck["stats"]) + day) % len(ck["stats"]) if ck["kind"] == "sld" else 0
    return TorchDecoder(ck["model"], slot, mu, sd, ck["vscale"])


def neural_rc(ck: dict, X_unlab: np.ndarray) -> TorchDecoder:
    """Re-centred decoder: new day's own statistics (SLD: fresh identity adapter slot)."""
    mu, sd = channel_stats(X_unlab)
    if ck["kind"] == "sld":
        m = copy.deepcopy(ck["model"])
        k = m.add_day()
        return TorchDecoder(m, k, mu, sd, ck["vscale"])
    return TorchDecoder(ck["model"], 0, mu, sd, ck["vscale"])


def sld_ref(ctx: Ctx, ck: dict):
    if "ref" not in ck:
        ck["ref"] = latent_stats(ck["model"], ctx.anchors, ck["stats"], list(range(len(ctx.anchors))))
    return ck["ref"]


def neural_aligned(ctx: Ctx, ck: dict, X_unlab: np.ndarray, seed: int = 0, cache_key: str | None = None,
                   **over) -> TorchDecoder:
    """SLD with an unsupervised, NoMAD-style aligned adapter for the new day.

    With ``cache_key`` the result is memoised in ``results/checkpoints/aligned.pkl`` so the
    few-shot and simulator stages reuse the adapters fitted during ``eval``."""
    cache = ctx.aligned_cache()
    if cache_key is not None and cache_key in cache:
        return cache[cache_key]
    mu, sd = channel_stats(X_unlab)
    ac = AdaptConfig(**{**ctx.cfg["adapt"], **over, "seed": seed})
    m, k = adapt_sld(ck["model"], sld_ref(ctx, ck), X_unlab, mu, sd, ac, vscale=ck["vscale"])
    dec = TorchDecoder(m, k, mu, sd, ck["vscale"])
    if cache_key is not None:
        cache[cache_key] = dec
    return dec


def available_models(ctx: Ctx) -> dict[str, list[dict]]:
    """name -> list of checkpoints over seeds (in seed order) that exist on disk."""
    out: dict[str, list[dict]] = {}
    for p in sorted(ctx.ckpt.glob("*_s*.pt")):
        name, seed = p.stem.rsplit("_s", 1)
        ck = ctx.load_model(p.stem)
        ck["tag"], ck["seed"] = p.stem, int(seed)
        out.setdefault(name, []).append(ck)
    return out


def _ms(xs) -> dict:
    xs = [float(x) for x in xs]
    return {"mean": float(np.mean(xs)), "std": float(np.std(xs)), "n": len(xs), "values": xs}


# ----------------------------------------------------------------------------------------
# Stage: eval (within-day, cross-day, latency)
# ----------------------------------------------------------------------------------------
def stage_eval(ctx: Ctx) -> None:
    lin = ctx.linear()
    models = available_models(ctx)
    A, S = ctx.anchors, ctx.anchor_stats
    # --- within-day: test blocks of the anchor days
    within: dict[str, dict] = {}
    within["wiener_per_day"] = _ms([test_r2(w, d) for w, d in zip(lin["wiener_day"], A)])
    within["kalman_per_day"] = _ms([test_r2(k, d) for k, d in zip(lin["kalman_day"], A)])
    within["wiener"] = _ms([test_r2(lin["wiener"].with_stats(*st), d) for d, st in zip(A, S)])
    within["kalman"] = _ms([test_r2(lin["kalman"].with_stats(*st), d) for d, st in zip(A, S)])
    within["fa"] = _ms([test_r2(lin["fa"].with_stats(*st), d) for d, st in zip(A, S)])
    for name, cks in models.items():
        per_seed = []
        for ck in cks:
            per_seed.append(np.mean([test_r2(neural_fixed(ck, i), d) for i, d in enumerate(A)]))
        within[name] = _ms(per_seed)
        within[name]["per_day"] = [float(np.mean([test_r2(neural_fixed(ck, i), d) for ck in cks]))
                                   for i, d in enumerate(A)]
        ctx.log(f"within-day {name}: {within[name]['mean']:.4f} +- {within[name]['std']:.4f}")
    ctx.save_json("within_day.json", within)
    # --- cross-day: held-out later days
    cross = {"days": [d.name for d in ctx.tests], "day_index": [d.day_index for d in ctx.tests], "methods": {}}
    M = cross["methods"]

    def add(name, vals):
        M.setdefault(name, []).append(vals)

    for d in ctx.tests:
        Xu = unlabelled(d)
        mu, sd = channel_stats(Xu)
        add("wiener_fixed", [test_r2(lin["wiener"], d)])
        add("wiener_rc", [test_r2(lin["wiener"].with_stats(mu, sd), d)])
        add("kalman_fixed", [test_r2(lin["kalman"], d)])
        add("kalman_rc", [test_r2(lin["kalman"].with_stats(mu, sd), d)])
        add("fa_fixed", [test_r2(lin["fa"], d)])
        add("fa_aligned", [test_r2(lin["fa"].aligned_to(Xu), d)])
        is_strided = (ctx.tests.index(d) % ctx.cfg.get("ablation_day_stride", 1)) == 0
        for name, cks in models.items():
            add(f"{name}_fixed", [test_r2(neural_fixed(ck), d) for ck in cks])
            add(f"{name}_rc", [test_r2(neural_rc(ck, Xu), d) for ck in cks])
            can_align = cks[0]["kind"] == "sld" and cks[0]["model"].use_adapters
            if can_align and (name == "sld" or is_strided):
                add(f"{name}_aligned", [test_r2(neural_aligned(ctx, ck, Xu, seed=ck["seed"],
                                                               cache_key=f"{ck['tag']}|{d.name}"), d) for ck in cks])
            elif can_align:
                add(f"{name}_aligned", [float("nan")])
        ctx.log(f"cross-day {d.name} (day {d.day_index}): " + ", ".join(
            f"{k}={np.mean(v[-1]):.3f}" for k, v in M.items() if k.split('_')[0] in ('wiener', 'gru', 'sld')))
        ctx.save_json("cross_day.json", cross)
        ctx.save_aligned_cache()
    # --- latency (single-bin streaming step, CPU, 2 threads)
    lat = {}
    x = np.random.default_rng(0).poisson(1.0, (300, A[0].X.shape[1])).astype(np.float32)
    decs = {"wiener": lin["wiener"], "kalman": lin["kalman"], "fa": lin["fa"]}
    for name in ("gru", "sld", "sld_transformer"):
        if name in models:
            decs[name] = neural_fixed(models[name][0])
    for name, dec in decs.items():
        dec.reset()
        for xx in x[:50]:
            dec.step(xx)
        t = time.perf_counter()
        for xx in x[50:]:
            dec.step(xx)
        lat[name] = (time.perf_counter() - t) / 250 * 1e3
    ctx.save_json("latency.json", lat)
    ctx.log(f"latency (ms/bin): {lat}")


# ----------------------------------------------------------------------------------------
# Stage: few-shot recalibration
# ----------------------------------------------------------------------------------------
def stage_fewshot(ctx: Ctx) -> None:
    lin = ctx.linear()
    models = available_models(ctx)
    cfg = ctx.cfg
    Ns = cfg["fewshot_trials"]
    days = ctx.tests[:: cfg["fewshot_day_stride"]][: cfg.get("fewshot_max_days")]
    seeds = cfg["fewshot_seeds"]
    out = {"trials": Ns, "days": [d.name for d in days], "day_index": [d.day_index for d in days], "methods": {}}
    M = out["methods"]
    alpha_fs = 1e4
    for d in days:
        Xu = unlabelled(d)
        mu, sd = channel_stats(Xu)
        for N in Ns:
            sl = d.first_trials(N)
            Xl, Yl = d.X[sl], d.Y[sl]
            row = {}
            if N > 0:
                mus, sds = channel_stats(Xl)
                row["wiener_scratch"] = test_r2(WienerFilter(lin["wiener"].n_lags, alpha_fs).fit([(Xl, Yl)], mus, sds), d)
                row["wiener_prior"] = test_r2(WienerFilter(lin["wiener"].n_lags, alpha_fs).fit(
                    [(Xl, Yl)], mu, sd, W0=lin["wiener"].W), d)
            else:
                row["wiener_prior"] = test_r2(lin["wiener"].with_stats(mu, sd), d)
            for s in seeds:
                ck_g, ck_s = models["gru"][s], models["sld"][s]
                if N > 0:
                    g = finetune_gru(ck_g["model"], Xl, Yl, mu, sd, ck_g["vscale"], seed=s)
                    row.setdefault("gru_finetune", []).append(test_r2(TorchDecoder(g, 0, mu, sd, ck_g["vscale"]), d))
                    ac = AdaptConfig(**{**cfg["adapt"], "seed": s})
                    m, k = adapt_sld(ck_s["model"], sld_ref(ctx, ck_s), Xu, mu, sd, ac, labelled=(Xl, Yl),
                                     vscale=ck_s["vscale"], unsupervised=False)
                    row.setdefault("sld_fewshot", []).append(test_r2(TorchDecoder(m, k, mu, sd, ck_s["vscale"]), d))
                    al = neural_aligned(ctx, ck_s, Xu, seed=s, cache_key=f"{ck_s['tag']}|{d.name}")
                    m, k = adapt_sld(al.model, sld_ref(ctx, ck_s), Xu, mu, sd, ac, labelled=(Xl, Yl),
                                     vscale=ck_s["vscale"], unsupervised=True, day=al.day)
                    row.setdefault("sld_align_fewshot", []).append(test_r2(TorchDecoder(m, k, mu, sd, ck_s["vscale"]), d))
                else:
                    row.setdefault("gru_finetune", []).append(test_r2(neural_rc(ck_g, Xu), d))
                    row.setdefault("sld_fewshot", []).append(test_r2(neural_rc(ck_s, Xu), d))
                    row.setdefault("sld_align_fewshot", []).append(
                        test_r2(neural_aligned(ctx, ck_s, Xu, seed=s, cache_key=f"{ck_s['tag']}|{d.name}"), d))
            for k_, v in row.items():
                M.setdefault(k_, {}).setdefault(str(N), []).append(float(np.mean(v)))
            ctx.log(f"few-shot {d.name} N={N}: " + ", ".join(f"{k_}={np.mean(v):.3f}" for k_, v in row.items()))
        ctx.save_json("fewshot.json", out)


# ----------------------------------------------------------------------------------------
# Stage: stress tests
# ----------------------------------------------------------------------------------------
def corrupt(X: np.ndarray, frac: float, mode: str, rng: np.random.Generator) -> np.ndarray:
    """Silence (``drop``) or scramble the identity of (``shift``) a fraction of channels."""
    C = X.shape[1]
    n = int(round(frac * C))
    if n == 0:
        return X
    idx = rng.choice(C, n, replace=False)
    X = X.copy()
    if mode == "drop":
        X[:, idx] = 0
    else:
        X[:, idx] = X[:, np.roll(idx, 1)]
    return X


def stage_stress(ctx: Ctx) -> None:
    lin = ctx.linear()
    models = available_models(ctx)
    sc = ctx.cfg["stress"]
    d = ctx.anchors[-1]
    out = {"day": d.name, "results": {}}
    for mode, fracs in (("drop", sc["drop_fracs"]), ("shift", sc["shift_fracs"])):
        R = out["results"].setdefault(mode, {"fracs": fracs, "methods": {}})
        for f in fracs:
            vals: dict[str, list] = {}
            for rep in range(sc["repeats"]):
                Xc = corrupt(d.X, f, mode, np.random.default_rng(1000 + rep))
                dc = copy.copy(d)
                dc.X = Xc
                mu, sd = channel_stats(unlabelled(dc))
                vals.setdefault("wiener_naive", []).append(test_r2(lin["wiener"], dc))
                vals.setdefault("wiener_rc", []).append(test_r2(lin["wiener"].with_stats(mu, sd), dc))
                vals.setdefault("fa_aligned", []).append(test_r2(lin["fa"].aligned_to(unlabelled(dc)), dc))
                for name in ("gru", "gru_aug", "sld"):
                    if name not in models:
                        continue
                    ck = models[name][0]
                    slot = len(ck["stats"]) - 1 if ck["kind"] == "sld" else 0
                    vals.setdefault(f"{name}_naive", []).append(test_r2(neural_fixed(ck), dc))
                    vals.setdefault(f"{name}_rc", []).append(
                        test_r2(TorchDecoder(ck["model"], slot, mu, sd, ck["vscale"]), dc))
                    if ck["kind"] == "sld":
                        # anchor day: continue from its trained adapter rather than the identity
                        ac = AdaptConfig(**{**ctx.cfg["adapt"], "seed": rep})
                        m, k = adapt_sld(ck["model"], sld_ref(ctx, ck), unlabelled(dc), mu, sd, ac,
                                         vscale=ck["vscale"], day=slot)
                        vals.setdefault(f"{name}_aligned", []).append(
                            test_r2(TorchDecoder(m, k, mu, sd, ck["vscale"]), dc))
            for k, v in vals.items():
                R["methods"].setdefault(k, []).append(_ms(v))
            ctx.log(f"stress {mode} {f:.2f}: " + ", ".join(f"{k}={np.mean(v):.3f}" for k, v in vals.items()))
            ctx.save_json("stress.json", out)


# ----------------------------------------------------------------------------------------
# Stage: closed-loop simulator
# ----------------------------------------------------------------------------------------
def sim_encoder(ctx: Ctx, d: DayData, calib: Decoder):
    sc = ctx.cfg["sim"]
    enc = fit_encoding_model(d, ctx.vscale, steps=sc["encoder_steps"], hidden=sc["encoder_hidden"])
    scale, info = calibrate_noise(enc, d, calib, grid=tuple(sc["noise_grid"]))
    return enc, {"noise_scale": scale, **info}


def _sessions(dec, enc, user, wcfg, gain, smooth, seeds, hold: float | None = None) -> list:
    if hold is not None:
        wcfg = dataclasses.replace(wcfg, click_hold_s=hold)
    return [run_webgrid(copy.deepcopy(dec) if dec is not None else None, enc, user, wcfg, gain, smooth, seed=s)
            for s in seeds]


def _run(dec, enc, user, wcfg, t: dict, seeds) -> list:
    """Sessions with a decoder's tuned (gain, smooth, click hold)."""
    return _sessions(dec, enc, user, wcfg, t["gain"], t["smooth"], seeds, t["hold"])


def refit_linear(kind: str, base: Decoder, runs: list, mu, sd, cfg: dict) -> Decoder:
    """ReFIT-style recalibration (Gilja et al. 2012) on closed-loop simulator data.

    Intended velocity is estimated from the decoded velocity by rotating it toward the
    target (speed kept) and setting it to zero while the cursor is on the target."""
    segs = []
    for r in runs:
        to_t = r.target_pos - r.cursor
        dist = np.linalg.norm(to_t, axis=1, keepdims=True)
        speed = np.linalg.norm(r.decoded, axis=1, keepdims=True)
        y = speed * to_t / np.maximum(dist, 1e-6)
        cell = cfg["side_mm"] / r.grid
        y[np.max(np.abs(to_t), axis=1) < cell / 2] = 0
        segs.append((r.counts, y.astype(np.float32)))
    if kind == "wiener":
        return WienerFilter(base.n_lags, base.alpha).fit(segs, mu, sd)
    return KalmanFilter(base.smooth_bins, base.ridge, base.q_scale).fit(segs, mu, sd)


def stage_sim(ctx: Ctx) -> None:
    sc = ctx.cfg["sim"]
    lin = ctx.linear()
    models = available_models(ctx)
    prep = ctx.load_json("prep.json")
    user = UserModel(vmax=prep["user_vmax"], d0=prep["user_d0"])
    d = ctx.anchors[-1]
    enc, enc_info = sim_encoder(ctx, d, lin["wiener"])
    ctx.log(f"sim encoder on {d.name}: {enc_info}")
    decs: dict[str, Decoder | None] = {"oracle": None, "wiener": lin["wiener"], "kalman": lin["kalman"]}
    for name in ("gru", "sld"):
        if name in models:
            decs[name] = neural_fixed(models[name][0])
    # Simulator fidelity: open-loop R^2 on real vs simulated spikes for the same velocities.
    te = d.test
    Xs = enc.sample(d.Y[te], d.pos[te], np.random.default_rng(7))
    fidelity = {n: {"real": test_r2(dec, d), "sim": r2_score(d.Y[te], dec.predict(Xs))}
                for n, dec in decs.items() if dec is not None}
    ctx.log(f"fidelity: {fidelity}")
    base = WebGridConfig(grid=sc["grid"], seconds=sc["tune_seconds"])
    tuned = {}
    for name, dec in decs.items():
        if dec is None:
            tuned[name] = {"gain": 1.0, "smooth": 0.0, "hold": 0.1}
            continue
        best = None
        for g in sc["gains"]:
            for s in sc["smooths"]:
                for h in sc["click_holds"]:
                    b = np.mean([r.bps for r in _sessions(dec, enc, user, base, g, s, [101, 102], h)])
                    if best is None or b > best[0]:
                        best = (b, g, s, h)
        tuned[name] = {"gain": best[1], "smooth": best[2], "hold": best[3], "tune_bps": best[0]}
        ctx.log(f"tuned {name}: {tuned[name]}")
    evalc = WebGridConfig(grid=sc["grid"], seconds=sc["seconds"])
    main, traces = {}, {}
    seeds = list(range(sc["eval_sessions"]))
    for name, dec in decs.items():
        runs = _run(dec, enc, user, evalc, tuned[name], seeds)
        main[name] = {**_ms([r.bps for r in runs]), "n_correct": [r.n_correct for r in runs],
                      "n_incorrect": [r.n_incorrect for r in runs]}
        traces[name] = runs[0]
        ctx.log(f"WebGrid {sc['grid']}x{sc['grid']} {name}: {main[name]['mean']:.2f} +- {main[name]['std']:.2f} bps")
    pickle.dump(traces, open(ctx.ckpt / "webgrid_traces.pkl", "wb"))
    sweep = {}
    for g in sc["grid_sweep"]:
        wc = WebGridConfig(grid=g, seconds=sc["tune_seconds"])
        for name, dec in decs.items():
            runs = _run(dec, enc, user, wc, tuned[name], seeds)
            sweep.setdefault(name, []).append(_ms([r.bps for r in runs]))
        ctx.log(f"grid {g}: " + ", ".join(f"{n}={v[-1]['mean']:.2f}" for n, v in sweep.items()))
    dwell = {}
    dc = WebGridConfig(grid=sc["grid"], seconds=sc["seconds"], selection="dwell")
    for name, dec in decs.items():
        dwell[name] = _ms([r.bps for r in _run(dec, enc, user, dc, tuned[name], seeds)])
    ctx.log(f"dwell selection: { {k: round(v['mean'], 2) for k, v in dwell.items()} }")
    # ReFIT-style closed-loop recalibration of the linear decoders.
    refit = {}
    mu, sd = lin["wiener"].mu, lin["wiener"].sd
    calib_cfg = WebGridConfig(grid=sc["grid"], seconds=sc["refit_seconds"])
    for name in ("wiener", "kalman"):
        dec = decs[name]
        runs = _run(dec, enc, user, calib_cfg, tuned[name], [201])
        new = refit_linear(name, dec, runs, mu, sd, {"side_mm": calib_cfg.side_mm})
        best = max(({"gain": g, "smooth": s, "hold": h} for g in sc["gains"] for s in sc["smooths"]
                    for h in sc["click_holds"]),
                   key=lambda t: np.mean([r.bps for r in _run(new, enc, user, base, t, [101, 102])]))
        r2 = _ms([r.bps for r in _run(new, enc, user, evalc, best, seeds)])
        refit[name] = {"before": main[name], "after": r2, **best}
        ctx.log(f"ReFIT {name}: {main[name]['mean']:.2f} -> {r2['mean']:.2f} bps")
    out = {"grid": sc["grid"], "seconds": sc["seconds"], "encoder": enc_info, "fidelity": fidelity, "tuned": tuned,
           "main": main, "grid_sweep": {"grids": sc["grid_sweep"], "methods": sweep}, "dwell": dwell,
           "refit": refit, "user": {"vmax": user.vmax, "d0": user.d0, "delay_ms": user.delay_bins * 20}}
    ctx.save_json("closed_loop.json", out)
    # Cross-day closed loop: users on later days, decoders not re-tuned.
    cross = {"days": [], "day_index": [], "methods": {}}
    cseeds = list(range(sc["crossday_sessions"]))
    for name in sc["crossday_days"]:
        dd = next(x for x in ctx.days if x.name == name)
        Xu = unlabelled(dd)
        mu2, sd2 = channel_stats(Xu)
        enc2, info2 = sim_encoder(ctx, dd, lin["wiener"].with_stats(mu2, sd2))
        cand = {"oracle": (None, "oracle"), "wiener_rc": (lin["wiener"].with_stats(mu2, sd2), "wiener"),
                "kalman_rc": (lin["kalman"].with_stats(mu2, sd2), "kalman")}
        if "gru" in models:
            cand["gru_fixed"] = (neural_fixed(models["gru"][0]), "gru")
            cand["gru_rc"] = (neural_rc(models["gru"][0], Xu), "gru")
        if "sld" in models:
            cand["sld_fixed"] = (neural_fixed(models["sld"][0]), "sld")
            cand["sld_rc"] = (neural_rc(models["sld"][0], Xu), "sld")
            cand["sld_aligned"] = (neural_aligned(ctx, models["sld"][0], Xu,
                                                  cache_key=f"{models['sld'][0]['tag']}|{name}"), "sld")
        cross["days"].append(name)
        cross["day_index"].append(dd.day_index)
        for k, (dec, base_name) in cand.items():
            cross["methods"].setdefault(k, []).append(
                _ms([r.bps for r in _run(dec, enc2, user, evalc, tuned[base_name], cseeds)]))
        ctx.log(f"cross-day sim {name}: " + ", ".join(f"{k}={v[-1]['mean']:.2f}" for k, v in cross["methods"].items()))
        out["crossday"] = cross
        ctx.save_json("closed_loop.json", out)


def stage_sim_ablation(ctx: Ctx) -> None:
    """Closed-loop bitrate of the ablation models (same encoder, user and tuning protocol)."""
    sc = ctx.cfg["sim"]
    lin = ctx.linear()
    models = available_models(ctx)
    prep = ctx.load_json("prep.json")
    user = UserModel(vmax=prep["user_vmax"], d0=prep["user_d0"])
    d = ctx.anchors[-1]
    enc, _ = sim_encoder(ctx, d, lin["wiener"])
    base = WebGridConfig(grid=sc["grid"], seconds=sc["tune_seconds"])
    evalc = WebGridConfig(grid=sc["grid"], seconds=sc["seconds"])
    out = {}
    for name in sc.get("ablations", []):
        if name not in models:
            continue
        dec = neural_fixed(models[name][0])
        grid = [{"gain": g, "smooth": s, "hold": h} for g in sc["gains"] for s in sc["smooths"] for h in sc["click_holds"]]
        best = max(grid, key=lambda t: np.mean([r.bps for r in _run(dec, enc, user, base, t, [101, 102])]))
        res = _ms([r.bps for r in _run(dec, enc, user, evalc, best, list(range(sc["eval_sessions"])))])
        out[name] = {**res, "tuned": best, "fidelity_real": test_r2(dec, d)}
        ctx.log(f"sim ablation {name}: {res['mean']:.2f} +- {res['std']:.2f} bps ({best})")
        ctx.save_json("closed_loop_ablation.json", out)


# ----------------------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--stage", required=True,
                    choices=["prep", "train", "eval", "fewshot", "stress", "sim", "simabl", "export"])
    ap.add_argument("--models", default="", help="comma list name:seed for --stage train")
    a = ap.parse_args()
    ctx = Ctx(a.config)
    if a.stage != "train":
        # Evaluation and simulation run many tiny sequential ops; on a shared CPU, intra-op
        # threading makes these ~15x slower, so they use a single thread.
        torch.set_num_threads(int(ctx.cfg.get("eval_threads", 1)))
    if a.stage != "prep" and ctx.lag is None:
        raise SystemExit("run --stage prep first")
    t = time.time()
    if a.stage == "prep":
        stage_prep(ctx)
    elif a.stage == "train":
        stage_train(ctx, [m for m in a.models.split(",") if m])
    elif a.stage == "eval":
        stage_eval(ctx)
    elif a.stage == "fewshot":
        stage_fewshot(ctx)
    elif a.stage == "stress":
        stage_stress(ctx)
    elif a.stage == "sim":
        stage_sim(ctx)
    elif a.stage == "simabl":
        stage_sim_ablation(ctx)
    elif a.stage == "export":
        from .export import export_all

        export_all(ctx)
    ctx.log(f"stage {a.stage} finished in {time.time() - t:.0f}s")


if __name__ == "__main__":
    main()
