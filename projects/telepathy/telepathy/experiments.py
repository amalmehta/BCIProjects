"""End-to-end experiment pipeline, organised in cached stages.

    python -m telepathy.experiments --config configs/cpu.yaml --stage <stage> [--models sld:0,gru:0]

Stages (each writes JSON under ``results/`` and can be re-run independently):

* ``prep``      choose the neural lag, fit classical decoders (Wiener, Kalman, FA stabiliser)
* ``train``     train neural decoders (GRU baseline, proposed SLD, ablations) -> checkpoints
* ``eval``      within-day R^2, cross-day R^2 (fixed / re-centred / aligned), latency
* ``fewshot``   few-shot recalibration curves on held-out days
* ``stress``    electrode dropout and channel-shift stress tests
* ``kalman``    strong position-velocity(-acceleration) Kalman baseline (tuned; old filter kept as kalman_v1)
* ``remap``     label-free channel re-matching on re-mapping days + synthetic permutation tests
* ``sim``       closed-loop WebGrid simulator (vectorised, paired seeds): tuning, bitrate with CIs and
                paired tests, grid sweep, dwell, cross-day, ReFIT
* ``clda``      ReFIT-style closed-loop decoder adaptation of the GRU / SLD (and linear decoders)
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
from .clda import CLDAConfig, finetune_closed_loop, refit_labels
from .linear import Decoder, KalmanFilter, PVKalmanFilter, WienerFilter
from .remap import ChannelMatcher, partial_shuffle
from .stats import mean_ci, paired_test, pairwise
from .metrics import r2_score
from .simulator import (UserModel, WebGridConfig, calibrate_noise, fit_encoding_model, fit_user_speed_profile,
                        run_webgrid_batch)
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


TUNE_SEED0 = 1000  # tuning seeds are disjoint from evaluation seeds (0..N-1) and CLDA seeds (300+)


def _batch(dec, enc, user, wcfg, t: dict, seeds) -> list:
    """Batched sessions with a decoder's tuned (gain, smooth, click hold); same seeds = paired."""
    return run_webgrid_batch(dec, enc, user, wcfg, seeds, t["gain"], t["smooth"], t["hold"])


def tune(dec, enc, user, sc: dict, n_seeds: int | None = None) -> dict:
    """Grid-search gain x smoothing x click hold on separate 30 s tuning sessions (one batch)."""
    if dec is None:
        return {"gain": 1.0, "smooth": 0.0, "hold": 0.1}
    grid = [(g, s, h) for g in sc["gains"] for s in sc["smooths"] for h in sc["click_holds"]]
    k = n_seeds or sc.get("tune_sessions", 2)
    seeds = [TUNE_SEED0 + i for i in range(k)]
    wc = WebGridConfig(grid=sc["grid"], seconds=sc["tune_seconds"])
    runs = run_webgrid_batch(dec, enc, user, wc, seeds * len(grid), np.repeat([g[0] for g in grid], k),
                             np.repeat([g[1] for g in grid], k), np.repeat([g[2] for g in grid], k))
    bps = np.array([r.bps for r in runs]).reshape(len(grid), k).mean(1)
    b = int(np.argmax(bps))
    return {"gain": grid[b][0], "smooth": grid[b][1], "hold": grid[b][2], "tune_bps": float(bps[b])}


def _summ(runs) -> dict:
    v = [r.bps for r in runs]
    return {**_ms(v), **{"ci95": mean_ci(v)["ci95"]}, "n_correct": [r.n_correct for r in runs],
            "n_incorrect": [r.n_incorrect for r in runs]}


def refit_linear(kind: str, base: Decoder, runs: list, mu, sd, cfg: dict) -> Decoder:
    """ReFIT-style recalibration (Gilja et al. 2012) on closed-loop simulator data: intended
    velocity = decoded velocity rotated toward the target (speed kept), zero on the target."""
    segs = refit_labels(runs, cfg["side_mm"])
    if kind == "wiener":
        return WienerFilter(base.n_lags, base.alpha).fit(segs, mu, sd)
    if isinstance(base, PVKalmanFilter):
        return PVKalmanFilter(base.smooth_bins, base.obs_delay, base.order, base.q_scale, base.sqrt,
                              base.ridge).fit(segs, mu, sd, vscale=base.vscale)
    return KalmanFilter(base.smooth_bins, base.ridge, base.q_scale).fit(segs, mu, sd)


def sim_decoders(ctx: Ctx, lin: dict, models: dict) -> dict:
    decs: dict[str, Decoder | None] = {"oracle": None, "wiener": lin["wiener"], "kalman": lin["kalman"]}
    if isinstance(lin["kalman"], PVKalmanFilter):  # ablation: same filter without cursor-position feedback
        k = copy.deepcopy(lin["kalman"])
        k.position_feedback = False
        decs["kalman_nofb"] = k
    if "kalman_v1" in lin:
        decs["kalman_v1"] = lin["kalman_v1"]
    for name in ("gru", "sld"):
        if name in models:
            decs[name] = neural_fixed(models[name][0])
    return decs


def stage_sim(ctx: Ctx) -> None:
    sc = ctx.cfg["sim"]
    lin = ctx.linear()
    models = available_models(ctx)
    prep = ctx.load_json("prep.json")
    user = UserModel(vmax=prep["user_vmax"], d0=prep["user_d0"])
    d = ctx.anchors[-1]
    enc, enc_info = sim_encoder(ctx, d, lin["wiener"])
    ctx.log(f"sim encoder on {d.name}: {enc_info}")
    decs = sim_decoders(ctx, lin, models)
    # Simulator fidelity: open-loop R^2 on real vs simulated spikes for the same velocities.
    te = d.test
    Xs = enc.sample(d.Y[te], d.pos[te], np.random.default_rng(7))
    fidelity = {n: {"real": test_r2(dec, d), "sim": r2_score(d.Y[te], dec.predict(Xs))}
                for n, dec in decs.items() if dec is not None}
    ctx.log(f"fidelity: {fidelity}")
    tuned = {}
    for name, dec in decs.items():
        tuned[name] = tune(dec, enc, user, sc)
        ctx.log(f"tuned {name}: {tuned[name]}")
    evalc = WebGridConfig(grid=sc["grid"], seconds=sc["seconds"])
    main, traces, per = {}, {}, {}
    seeds = list(range(sc["eval_sessions"]))
    for name, dec in decs.items():
        runs = _batch(dec, enc, user, evalc, tuned[name], seeds)
        main[name] = _summ(runs)
        per[name] = main[name]["values"]
        traces[name] = runs[0]
        ctx.log(f"WebGrid {sc['grid']}x{sc['grid']} {name}: {main[name]['mean']:.2f} "
                f"[{main[name]['ci95'][0]:.2f}, {main[name]['ci95'][1]:.2f}] bps (n={len(seeds)})")
    pickle.dump(traces, open(ctx.ckpt / "webgrid_traces.pkl", "wb"))
    stats = pairwise({k: v for k, v in per.items() if k != "oracle"})
    for k, v in stats.items():
        ctx.log(f"paired {k}: diff {v['diff_mean']:+.2f} [{v['diff_ci95'][0]:+.2f}, {v['diff_ci95'][1]:+.2f}] "
                f"p_holm={v['p_holm']:.3g}")
    out = {"grid": sc["grid"], "seconds": sc["seconds"], "n_sessions": len(seeds), "encoder": enc_info,
           "fidelity": fidelity, "tuned": tuned, "main": main, "paired_tests": stats,
           "user": {"vmax": user.vmax, "d0": user.d0, "delay_ms": user.delay_bins * 20}}
    ctx.save_json("closed_loop.json", out)
    sweep = {}
    sw_seeds = list(range(sc.get("sweep_sessions", len(seeds))))
    for g in sc["grid_sweep"]:
        wc = WebGridConfig(grid=g, seconds=sc["tune_seconds"])
        for name, dec in decs.items():
            sweep.setdefault(name, []).append(_ms([r.bps for r in _batch(dec, enc, user, wc, tuned[name], sw_seeds)]))
        ctx.log(f"grid {g}: " + ", ".join(f"{n}={v[-1]['mean']:.2f}" for n, v in sweep.items()))
    out["grid_sweep"] = {"grids": sc["grid_sweep"], "methods": sweep, "n_sessions": len(sw_seeds)}
    dwell = {}
    dc = WebGridConfig(grid=sc["grid"], seconds=sc["seconds"], selection="dwell")
    for name, dec in decs.items():
        dwell[name] = _ms([r.bps for r in _batch(dec, enc, user, dc, tuned[name], seeds)])
    ctx.log(f"dwell selection: { {k: round(v['mean'], 2) for k, v in dwell.items()} }")
    out["dwell"] = dwell
    # ReFIT-style closed-loop recalibration of the linear decoders (one round, 120 s).
    refit = {}
    mu, sd = lin["wiener"].mu, lin["wiener"].sd
    calib_cfg = WebGridConfig(grid=sc["grid"], seconds=sc["refit_seconds"])
    for name in ("wiener", "kalman"):
        dec = decs[name]
        runs = _batch(dec, enc, user, calib_cfg, tuned[name], [201])
        new = refit_linear(name, dec, runs, mu, sd, {"side_mm": calib_cfg.side_mm})
        best = tune(new, enc, user, sc)
        after = _summ(_batch(new, enc, user, evalc, best, seeds))
        refit[name] = {"before": main[name], "after": after, **best,
                       "paired": paired_test(after["values"], main[name]["values"])}
        ctx.log(f"ReFIT {name}: {main[name]['mean']:.2f} -> {after['mean']:.2f} bps "
                f"(p={refit[name]['paired']['p_wilcoxon']:.3g})")
    out["refit"] = refit
    ctx.save_json("closed_loop.json", out)
    # Cross-day closed loop: users on later days, decoders not re-tuned.
    cross = {"days": [], "day_index": [], "methods": {}, "n_sessions": sc["crossday_sessions"]}
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
            cross["methods"].setdefault(k, []).append(_ms([r.bps for r in _batch(dec, enc2, user, evalc,
                                                                                 tuned[base_name], cseeds)]))
        ctx.log(f"cross-day sim {name}: " + ", ".join(f"{k}={v[-1]['mean']:.2f}" for k, v in cross["methods"].items()))
        out["crossday"] = cross
        ctx.save_json("closed_loop.json", out)
    ctx.save_aligned_cache()


def stage_sim_ablation(ctx: Ctx) -> None:
    """Closed-loop bitrate of the ablation models (same encoder, user, tuning protocol and seeds)."""
    sc = ctx.cfg["sim"]
    lin = ctx.linear()
    models = available_models(ctx)
    prep = ctx.load_json("prep.json")
    user = UserModel(vmax=prep["user_vmax"], d0=prep["user_d0"])
    d = ctx.anchors[-1]
    enc, _ = sim_encoder(ctx, d, lin["wiener"])
    evalc = WebGridConfig(grid=sc["grid"], seconds=sc["seconds"])
    seeds = list(range(sc["eval_sessions"]))
    out = {}
    for name in sc.get("ablations", []):
        if name not in models:
            continue
        dec = neural_fixed(models[name][0])
        best = tune(dec, enc, user, sc)
        res = _summ(_batch(dec, enc, user, evalc, best, seeds))
        out[name] = {**res, "tuned": best, "fidelity_real": test_r2(dec, d)}
        ctx.log(f"sim ablation {name}: {res['mean']:.2f} +- {res['std']:.2f} bps ({best})")
        ctx.save_json("closed_loop_ablation.json", out)
    # paired against the main decoders when available (same seeds, same encoder seed)
    p = ctx.res / "closed_loop.json"
    if p.exists():
        main = json.loads(p.read_text())["main"]
        if all(k in main and len(main[k]["values"]) == len(seeds) for k in ("sld", "gru")):
            per = {k: v["values"] for k, v in out.items()}
            per.update({"sld": main["sld"]["values"], "gru": main["gru"]["values"]})
            out["_paired_vs_sld"] = {k: paired_test(per[k], per["sld"]) for k in per if k != "sld"}
            ctx.save_json("closed_loop_ablation.json", out)


# ----------------------------------------------------------------------------------------
# Stage: strong Kalman / ReFIT-KF baseline
# ----------------------------------------------------------------------------------------
def stage_kalman(ctx: Ctx) -> None:
    """Fit and tune the position-velocity(-acceleration) Kalman filter; evaluate within and across
    days; the previous velocity-only filter is kept under ``kalman_v1`` keys."""
    kc = ctx.cfg["kalman_pv"]
    A, S = ctx.anchors, ctx.anchor_stats
    grid = [dict(smooth_bins=sb, obs_delay=dl, order=o, q_scale=q, sqrt=sq) for sb in kc["smooth_bins"]
            for dl in kc["obs_delay"] for o in kc["order"] for q in kc["q_scales"] for sq in kc["sqrt"]]

    def val(dec, d, st, ctx_bins: int = 300):
        a = d.val.start - ctx_bins
        return r2_score(d.Y[d.val], dec.with_stats(*st).predict(d.X[a : d.val.stop])[ctx_bins:])

    t0 = time.time()
    pooled_scores, pooled_fits = [], []
    for g in grid:
        k = PVKalmanFilter(**g).fit([(d.X[d.train], d.Y[d.train]) for d in A], *S[-1], vscale=ctx.vscale,
                                    seg_stats=S)
        pooled_scores.append(float(np.mean([val(k, d, st) for d, st in zip(A, S)])))
        pooled_fits.append(k)
        ctx.log(f"KF-PV pooled {g}: anchor val R2 {pooled_scores[-1]:.4f} ({time.time() - t0:.0f}s)", "kalman")
    b = int(np.argmax(pooled_scores))
    pooled = pooled_fits[b]
    best = grid[b]
    # per-day filters with the pooled hyper-parameters (and per-day tuning of q only)
    per_day = []
    for d, st in zip(A, S):
        cands = [PVKalmanFilter(**{**best, "q_scale": q}).fit([(d.X[d.train], d.Y[d.train])], *st, vscale=ctx.vscale)
                 for q in kc["q_scales"]]
        per_day.append(max(cands, key=lambda k: val(k, d, st)))
    # diagnosis: best validation score for each state order, all else tuned
    by_order = {str(o): max(sc for g, sc in zip(grid, pooled_scores) if g["order"] == o) for o in kc["order"]}
    by_sqrt = {str(v): max(sc for g, sc in zip(grid, pooled_scores) if g["sqrt"] == v) for v in kc["sqrt"]}
    by_smooth = {str(v): max(sc for g, sc in zip(grid, pooled_scores) if g["smooth_bins"] == v) for v in kc["smooth_bins"]}
    by_delay = {str(v): max(sc for g, sc in zip(grid, pooled_scores) if g["obs_delay"] == v) for v in kc["obs_delay"]}
    lin = ctx.linear()
    if "kalman_v1" not in lin:
        lin["kalman_v1"], lin["kalman_v1_day"] = lin["kalman"], lin["kalman_day"]
    lin["kalman"], lin["kalman_day"] = pooled, per_day
    pickle.dump(lin, open(ctx.ckpt / "linear.pkl", "wb"))
    # within-day
    W = ctx.load_json("within_day.json")
    if "kalman_v1" not in W:
        W["kalman_v1"], W["kalman_v1_per_day"] = W["kalman"], W["kalman_per_day"]
    W["kalman"] = _ms([test_r2(pooled.with_stats(*st), d) for d, st in zip(A, S)])
    W["kalman_per_day"] = _ms([test_r2(k, d) for k, d in zip(per_day, A)])
    ctx.save_json("within_day.json", W)
    # cross-day
    Cj = ctx.load_json("cross_day.json")
    M = Cj["methods"]
    for v in ("fixed", "rc"):
        if f"kalman_v1_{v}" not in M:
            M[f"kalman_v1_{v}"] = M[f"kalman_{v}"]
    M["kalman_fixed"] = [[test_r2(pooled, d)] for d in ctx.tests]
    M["kalman_rc"] = [[test_r2(pooled.with_stats(*channel_stats(unlabelled(d))), d)] for d in ctx.tests]
    ctx.save_json("cross_day.json", Cj)
    # latency
    L = ctx.load_json("latency.json")
    x = np.random.default_rng(0).poisson(1.0, (300, A[0].X.shape[1])).astype(np.float32)
    pooled.reset()
    for xx in x[:50]:
        pooled.step(xx)
    t = time.perf_counter()
    for xx in x[50:]:
        pooled.step(xx)
    if "kalman_v1" not in L:
        L["kalman_v1"] = L["kalman"]
    L["kalman"] = (time.perf_counter() - t) / 250 * 1e3
    ctx.save_json("latency.json", L)
    flagged = {d.name for d in ctx.tests if np.corrcoef(A[-1].X[A[-1].train].mean(0), d.X[d.train].mean(0))[0, 1] < 0.2}
    keep = [i for i, d in enumerate(ctx.tests) if d.name not in flagged]
    res = {"best": best, "best_val_r2": pooled_scores[b], "grid": grid, "val_r2": pooled_scores,
           "best_val_by_order": by_order, "best_val_by_sqrt": by_sqrt, "best_val_by_smooth_bins": by_smooth,
           "best_val_by_obs_delay": by_delay,
           "within_day": W["kalman"], "within_day_per_day": W["kalman_per_day"], "within_day_v1": W["kalman_v1"],
           "cross_day_rc_mean_regular": float(np.mean([M["kalman_rc"][i][0] for i in keep])),
           "cross_day_rc_v1_mean_regular": float(np.mean([M["kalman_v1_rc"][i][0] for i in keep])),
           "per_day_q": [k.q_scale for k in per_day]}
    ctx.save_json("kalman.json", res)
    ctx.log(f"KF-PV best {best}: within {W['kalman']['mean']:.3f} (per-day {W['kalman_per_day']['mean']:.3f}; "
            f"v1 {W['kalman_v1']['mean']:.3f}); cross rc {res['cross_day_rc_mean_regular']:.3f}", "kalman")


# ----------------------------------------------------------------------------------------
# Stage: label-free channel re-matching
# ----------------------------------------------------------------------------------------
def _permuted(d: DayData, perm: np.ndarray) -> DayData:
    dc = copy.copy(d)
    dc.X = d.X[:, perm]
    return dc


def stage_remap(ctx: Ctx) -> None:
    rc = ctx.cfg["remap"]
    lin = ctx.linear()
    models = available_models(ctx)
    A = ctx.anchors
    matcher = ChannelMatcher([d.X[d.train] for d in A], corr_weight=rc["corr_weight"], margin=rc["margin"])
    C = A[0].X.shape[1]
    cs = ctx.load_json("cross_day.json")
    M = cs["methods"]
    flag_corr = {d.name: float(np.corrcoef(A[-1].X[A[-1].train].mean(0), d.X[d.train].mean(0))[0, 1]) for d in ctx.tests}
    out = {"n_candidates": len(matcher.cands), "margin": rc["margin"], "days": {}, "synthetic": {}}

    def scores(dd, seeds_all=True):
        Xu = unlabelled(dd)
        mu, sd = channel_stats(Xu)
        r = {"wiener_rc": test_r2(lin["wiener"].with_stats(mu, sd), dd),
             "kalman_rc": test_r2(lin["kalman"].with_stats(mu, sd), dd)}
        for name in ("gru", "sld"):
            cks = models[name] if seeds_all else models[name][:1]
            r[f"{name}_rc"] = [test_r2(neural_rc(ck, Xu), dd) for ck in cks]
        return r

    # 1. every held-out day: matcher decision; full evaluation where it changes the mapping
    new_methods = {k: [] for k in ("wiener_rc_remap", "kalman_rc_remap", "gru_rc_remap", "sld_rc_remap",
                                   "sld_aligned_remap")}
    for i, d in enumerate(ctx.tests):
        t = time.time()
        m = matcher.match(unlabelled(d))
        rec = {k: v for k, v in m.items() if k != "perm"}
        rec["rate_corr_flag"] = flag_corr[d.name]
        rec["flagged"] = flag_corr[d.name] < 0.2
        if m["changed"]:
            dd = _permuted(d, m["perm"])
            after = scores(dd)
            after["sld_aligned"] = [test_r2(neural_aligned(ctx, ck, unlabelled(dd), seed=ck["seed"],
                                                           cache_key=f"{ck['tag']}|{d.name}|remap"), dd)
                                    for ck in models["sld"]]
            rec["after"] = after
            rec["before"] = {k: M[k][i] for k in ("wiener_rc", "kalman_rc", "gru_rc", "sld_rc", "sld_aligned")}
            vals = {"wiener_rc_remap": [after["wiener_rc"]], "kalman_rc_remap": [after["kalman_rc"]],
                    "gru_rc_remap": after["gru_rc"], "sld_rc_remap": after["sld_rc"],
                    "sld_aligned_remap": after["sld_aligned"]}
        else:
            vals = {"wiener_rc_remap": M["wiener_rc"][i], "kalman_rc_remap": M["kalman_rc"][i],
                    "gru_rc_remap": M["gru_rc"][i], "sld_rc_remap": M["sld_rc"][i],
                    "sld_aligned_remap": M["sld_aligned"][i]}
        for k, v in vals.items():
            new_methods[k].append(v)
        out["days"][d.name] = rec
        ctx.log(f"remap {d.name} (day {d.day_index}): {m['name']} score {m['score']:.3f} vs identity "
                f"{m['identity_score']:.3f}" + (f"; after: { {k: np.round(np.mean(v), 3) for k, v in rec['after'].items()} }"
                                                 if m['changed'] else "") + f" ({time.time() - t:.0f}s)", "remap")
        ctx.save_json("remap.json", out)
    M.update(new_methods)
    ctx.save_json("cross_day.json", cs)
    # 2. synthetic tests on regular days
    regular = [d for d in ctx.tests if flag_corr[d.name] >= 0.2][:: rc["synth_day_stride"]]
    rng = np.random.default_rng(0)
    names = [k for k in matcher.cands if k != "identity"]
    S = out["synthetic"]
    S["days"] = [d.name for d in regular]
    # 2a. structured re-mappings drawn from the hypothesis family
    rows = []
    for d in regular:
        for rep in range(rc["repeats"]):
            nm = names[rng.integers(len(names))]
            sh = matcher.cands[nm]
            dd = _permuted(d, sh)
            m = matcher.match(unlabelled(dd))
            fixed = _permuted(dd, m["perm"])
            rows.append({"day": d.name, "true": nm, "found": m["name"],
                         "recovery": float(np.mean(sh[m["perm"]] == np.arange(C))),
                         "wiener_rc_shuffled": test_r2(lin["wiener"].with_stats(*channel_stats(unlabelled(dd))), dd),
                         "wiener_rc_matched": test_r2(lin["wiener"].with_stats(*channel_stats(unlabelled(fixed))), fixed),
                         "sld_rc_shuffled": test_r2(neural_rc(models["sld"][0], unlabelled(dd)), dd),
                         "sld_rc_matched": test_r2(neural_rc(models["sld"][0], unlabelled(fixed)), fixed)})
        ctx.log(f"synthetic structured {d.name}: recovery {np.mean([r['recovery'] for r in rows[-rc['repeats']:]]):.2f}", "remap")
    S["structured"] = {"rows": rows, **{k: float(np.mean([r[k] for r in rows])) for k in
                                         ("recovery", "wiener_rc_shuffled", "wiener_rc_matched", "sld_rc_shuffled",
                                          "sld_rc_matched")}}
    # 2b. arbitrary partial shuffles (the stress-test re-wiring), structured matcher vs + Hungarian
    S["random"] = {"fracs": rc["shuffle_fracs"], "methods": {}}
    for f in rc["shuffle_fracs"]:
        acc: dict[str, list] = {}
        for d in regular:
            for rep in range(rc["repeats"]):
                sh, moved = partial_shuffle(C, f, rng)
                dd = _permuted(d, sh)
                Xu = unlabelled(dd)
                acc.setdefault("wiener_rc_shuffled", []).append(test_r2(lin["wiener"].with_stats(*channel_stats(Xu)), dd))
                acc.setdefault("sld_rc_shuffled", []).append(test_r2(neural_rc(models["sld"][0], Xu), dd))
                for tag, refine in (("structured", False), ("hungarian", True)):
                    m = matcher.match(Xu, refine=refine)
                    ok = sh[m["perm"]] == np.arange(C)
                    fixed = _permuted(dd, m["perm"])
                    acc.setdefault(f"{tag}_recovery_moved", []).append(float(np.mean(ok[moved])) if len(moved) else 1.0)
                    acc.setdefault(f"{tag}_recovery_all", []).append(float(np.mean(ok)))
                    acc.setdefault(f"{tag}_wiener_rc", []).append(
                        test_r2(lin["wiener"].with_stats(*channel_stats(unlabelled(fixed))), fixed))
                    acc.setdefault(f"{tag}_sld_rc", []).append(test_r2(neural_rc(models["sld"][0], unlabelled(fixed)), fixed))
        S["random"]["methods"][str(f)] = {k: float(np.mean(v)) for k, v in acc.items()}
        ctx.log(f"synthetic random shuffle {f}: { {k: round(float(np.mean(v)), 3) for k, v in acc.items()} }", "remap")
        ctx.save_json("remap.json", out)
    ctx.save_aligned_cache()
    ctx.save_json("remap.json", out)


# ----------------------------------------------------------------------------------------
# Stage: closed-loop decoder adaptation (ReFIT-style) for neural and linear decoders
# ----------------------------------------------------------------------------------------
def stage_clda(ctx: Ctx) -> None:
    sc, cc = ctx.cfg["sim"], ctx.cfg["clda"]
    cfg = CLDAConfig(**cc)
    lin = ctx.linear()
    models = available_models(ctx)
    prep = ctx.load_json("prep.json")
    user = UserModel(vmax=prep["user_vmax"], d0=prep["user_d0"])
    d = ctx.anchors[-1]
    enc, _ = sim_encoder(ctx, d, lin["wiener"])  # same encoder (seeded) as stage sim
    CL = ctx.load_json("closed_loop.json")
    evalc = WebGridConfig(grid=sc["grid"], seconds=sc["seconds"])
    datac = WebGridConfig(grid=sc["grid"], seconds=cfg.seconds)
    seeds = list(range(sc["eval_sessions"]))
    decs = {k: v for k, v in sim_decoders(ctx, lin, models).items() if k in cc.get("decoders", ["gru", "sld"])}
    out = {"config": dataclasses.asdict(cfg), "n_eval_sessions": len(seeds), "decoders": {}}
    offline = [(d.X[d.train], d.Y[d.train])]
    for name, dec0 in decs.items():
        t0 = time.time()
        variants = [("", cfg)]
        if name == "sld" and cc.get("sld_head_variant", False):
            variants.append(("_head", dataclasses.replace(cfg, sld_mode="adapter_head")))
        for suffix, vc in variants:
            tuned = CL["tuned"][name]
            before = CL["main"][name]["values"]
            rec = {"rounds": [{"round": 0, "bps": mean_ci(before), "offline_r2": test_r2(dec0, d), "tuned": tuned}]}
            dec, segs = dec0, []
            for r in range(1, vc.rounds + 1):
                runs = _batch(dec, enc, user, datac, tuned, [300 + 10 * r + k for k in range(vc.sessions_per_round)])
                segs += refit_labels(runs, datac.side_mm)
                if name in ("gru", "sld"):
                    dec = finetune_closed_loop(dec, segs, dataclasses.replace(vc, seed=r),
                                               offline=offline if vc.rehearsal > 0 else None)
                else:  # ReFIT: re-fit from scratch on all re-labelled closed-loop data so far
                    dec = _refit_segs(name, dec, segs)
                tuned = tune(dec, enc, user, sc)
                after = [x.bps for x in _batch(dec, enc, user, evalc, tuned, seeds)]
                rec["rounds"].append({"round": r, "bps": mean_ci(after), "offline_r2": test_r2(dec, d), "tuned": tuned,
                                      "paired_vs_before": paired_test(after, before), "values": after,
                                      "data_seconds": len(segs) * vc.seconds})
                ctx.log(f"CLDA {name}{suffix} round {r}: {np.mean(after):.2f} bps (before {np.mean(before):.2f}; "
                        f"p={rec['rounds'][-1]['paired_vs_before']['p_wilcoxon']:.3g}), offline R2 "
                        f"{rec['rounds'][-1]['offline_r2']:.3f} ({time.time() - t0:.0f}s)", "clda")
            out["decoders"][name + suffix] = rec
            ctx.save_json("clda.json", out)
    # paired comparison of the final adapted decoders
    final = {k: v["rounds"][-1]["values"] for k, v in out["decoders"].items()}
    out["paired_final"] = pairwise(final)
    ctx.save_json("clda.json", out)


def _refit_segs(kind: str, base: Decoder, segs: list) -> Decoder:
    if kind == "wiener":
        return WienerFilter(base.n_lags, base.alpha).fit(segs, base.mu, base.sd)
    return PVKalmanFilter(base.smooth_bins, base.obs_delay, base.order, base.q_scale, base.sqrt,
                          base.ridge).fit(segs, base.mu, base.sd, vscale=base.vscale)


# ----------------------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--stage", required=True,
                    choices=["prep", "train", "eval", "kalman", "remap", "fewshot", "stress", "sim", "simabl", "clda",
                             "export"])
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
    elif a.stage == "kalman":
        stage_kalman(ctx)
    elif a.stage == "remap":
        stage_remap(ctx)
    elif a.stage == "clda":
        stage_clda(ctx)
    elif a.stage == "export":
        from .export import export_all

        export_all(ctx)
    ctx.log(f"stage {a.stage} finished in {time.time() - t:.0f}s")


if __name__ == "__main__":
    main()
