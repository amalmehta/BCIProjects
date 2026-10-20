"""Offline decoding experiments on FALCON H1.

Within-session (held-in): train on the 13 held-in calibration blocks (6 days), test on
FALCON's public minival split (later portion of the same blocks).

Cross-session (held-out): 7 later days, two blocks each. Block A is the recalibration
block (its neural data may be used unlabelled; only its first ``k`` seconds of labels may
be used), block B is the test block. Days elapsed since the last training day: 6-20.

Usage: ``python -m robotic_arm.run_decoding --config configs/cpu.yaml``
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from . import DOF_NAMES, RESULTS_ROOT
from .config import load_config
from .data import Block, Normalizer, held_out_pairs, load_blocks
from .decoders.linear import KalmanFilter, WienerFilter, wiener_grid
from .decoders.neural import (Augment, SeqDecoder, TrainCfg, align_readin_kl, align_readin_moments,
                              finetune_all, fit_new_readin, latent_stats, mean_readin, pick_device,
                              train_model)
from .stats import add_decoding_stats
from .metrics import block_scores, r2_per_dim, r2_variance_weighted


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


class Scorer:
    """Scores predictions (in z-scored velocity units) against raw cued velocities."""

    def __init__(self, ny: Normalizer):
        self.ny = ny

    def __call__(self, blocks: list[Block], preds_z: list[np.ndarray]) -> dict:
        per_block, ys, ps = [], [], []
        for b, pz in zip(blocks, preds_z):
            p = self.ny.inverse(pz)
            per_block.append(block_scores(b.vel, p, b.eval_mask))
            ys.append(b.vel[b.eval_mask])
            ps.append(p[b.eval_mask])
        vw = [s["r2_vw"] for s in per_block]
        pooled = r2_per_dim(np.concatenate(ys), np.concatenate(ps))
        return {"r2_vw_mean": float(np.mean(vw)), "r2_vw_std": float(np.std(vw)),
                "r2_vw_blocks": vw, "r2_pooled_per_dof": dict(zip(DOF_NAMES, pooled.tolist())),
                "r2_pooled_mean_dof": float(pooled.mean())}


def split_tail(arrs: list[np.ndarray], frac: float) -> tuple[list, list]:
    head, tail = [], []
    for a in arrs:
        k = int(len(a) * (1 - frac))
        head.append(a[:k])
        tail.append(a[k:])
    return head, tail


def run(cfg, parts: set[str] | None = None, seeds: list[int] | None = None) -> dict:
    """Run the experiments in ``parts`` ("linear" and/or neural variant names; all if None).

    ``seeds`` restricts the neural seeds (default: ``decoding.seeds``), so each
    (variant, seed) can run as its own process on a shared machine."""
    torch.set_num_threads(cfg.threads)
    dev = pick_device()
    dc = cfg.decoding
    B = load_blocks()
    train, minival = B["held_in_calib"], B["held_in_minival"]
    pairs = held_out_pairs(B["held_out_calib"])
    days = sorted({b.day for b in train})
    day_id = {d: i for i, d in enumerate(days)}
    last_day = days[-1]
    log(f"train blocks {len(train)} over {len(days)} days; minival {len(minival)}; held-out days {len(pairs)}")

    ny = Normalizer.fit(np.concatenate([b.vel[b.eval_mask] for b in train]))
    score = Scorer(ny)
    global_norm = Normalizer.fit(np.concatenate([b.spikes for b in train]))
    day_norm = {d: Normalizer.fit(np.concatenate([b.spikes for b in train if b.day == d])) for d in days}
    held_norm = {a.day: Normalizer.fit(a.spikes) for a, _ in pairs}  # unlabelled block A

    def z_day(b: Block) -> np.ndarray:
        n = day_norm[b.day] if b.day in day_norm else held_norm[b.day]
        return n(b.spikes).astype(np.float32)

    zs = [z_day(b) for b in train]
    ys = [ny(b.vel).astype(np.float32) for b in train]
    ms = [b.eval_mask for b in train]
    rid = [day_id[b.day] for b in train]
    (zs_tr, zs_va), (ys_tr, ys_va), (ms_tr, ms_va) = (split_tail(a, dc.val_frac) for a in (zs, ys, ms))

    test_B = [b for _, b in pairs]
    out: dict = {"within": {}, "cross": {}, "selection": {}, "meta": {
        "train_days": [d.isoformat() for d in days], "minival_blocks": len(minival),
        "held_out_days": [a.day.isoformat() for a, _ in pairs],
        "days_since_train": [(a.day - last_day).days for a, _ in pairs],
        "recal_seconds": list(dc.recal_seconds), "seeds": list(dc.seeds)}}
    preds_store: dict = {}

    def val_score(preds):  # validation: variance-weighted R² in z units over val tails
        return float(np.mean([r2_variance_weighted(y[m], p[m]) for y, p, m in zip(ys_va, preds, ms_va)]))

    nc = dc.neural
    parts = set(parts) if parts else {"linear", *nc.variants}
    if "linear" in parts:
        _run_linear(out, preds_store, dc, train, minival, pairs, test_B, zs, ys, ms, zs_tr, zs_va, ys_tr, ys_va, ms_tr,
                    ms_va, z_day, ny, score, val_score, global_norm)
    _run_neural(out, preds_store, dc, [v for v in nc.variants if v in parts], dev, days, day_id, minival, pairs,
                test_B, zs, ys, ms, rid, zs_tr, zs_va, ys_tr, ys_va, ms_tr, ms_va, z_day, ny, score,
                global_norm, seeds)
    out["_preds"] = {m: {k: v.tolist() for k, v in d.items()} for m, d in preds_store.items()}
    return out


def _run_linear(out, preds_store, dc, train, minival, pairs, test_B, zs, ys, ms, zs_tr, zs_va, ys_tr, ys_va, ms_tr,
                ms_va, z_day, ny, score, val_score, global_norm) -> None:
    """Wiener and Kalman filters: within-session, zero-shot and few-shot."""
    for norm_name in ("day", "global"):
        if norm_name == "day":
            ztr, zva = zs_tr, zs_va
        else:
            zg = [global_norm(b.spikes).astype(np.float32) for b in train]
            ztr, zva = split_tail(zg, dc.val_frac)
        t0 = time.time()
        wf, grid = wiener_grid(ztr, ys_tr, ms_tr, zva, ys_va, ms_va, max(dc.wiener.lags), dc.wiener.stride,
                               dc.wiener.smooth_tau, list(dc.wiener.alphas), list(dc.wiener.lags), val_score)
        # refit on all training data with the selected hyperparameters
        zall = zs if norm_name == "day" else [global_norm(b.spikes).astype(np.float32) for b in train]
        wf = WienerFilter(wf.n_lags, wf.alpha, wf.smooth_tau, wf.stride).fit(zall, ys, ms)
        out["selection"][f"wiener_{norm_name}"] = {"grid": grid, "chosen": {"n_lags": wf.n_lags, "alpha": wf.alpha}}
        nz = (lambda b: z_day(b)) if norm_name == "day" else (lambda b: global_norm(b.spikes))
        out["within"][f"wiener_{norm_name}norm"] = score(minival, [wf.predict(nz(b)) for b in minival])
        out["cross"][f"wiener_{norm_name}norm_zeroshot"] = score(test_B, [wf.predict(nz(b)) for b in test_B])
        log(f"wiener {norm_name}: lags={wf.n_lags} alpha={wf.alpha} within={out['within'][f'wiener_{norm_name}norm']['r2_vw_mean']:.3f} "
            f"cross0={out['cross'][f'wiener_{norm_name}norm_zeroshot']['r2_vw_mean']:.3f} ({time.time()-t0:.0f}s)")
        if norm_name == "day":
            wf_day = wf
            preds_store["wiener"] = {b.key: wf.predict(z_day(b)) for b in test_B}
            preds_store["wiener_within"] = {b.key: wf.predict(z_day(b)) for b in minival}

    # few-shot Wiener: (a) from scratch on k s of block A; (b) pooled refit held-in + k s
    proto = WienerFilter(wf_day.n_lags, wf_day.alpha, wf_day.smooth_tau, wf_day.stride)
    held_in_stats = proto.stats(zs, ys, ms)
    for k in dc.recal_seconds:
        nb = int(k / 0.02)
        sc_preds, pool_preds = [], []
        for a, b in pairs:
            r = a.slice(0, nb)
            st = proto.stats([z_day(r)], [ny(r.vel)], [r.eval_mask])
            a_s = _scratch_alpha(proto, z_day(r), ny(r.vel), r.eval_mask, list(dc.wiener.scratch_alphas))
            w_s = WienerFilter(wf_day.n_lags, a_s, wf_day.smooth_tau, wf_day.stride).fit_stats(st)
            sc_preds.append(w_s.predict(z_day(b)))
            w_p = WienerFilter(wf_day.n_lags, wf_day.alpha, wf_day.smooth_tau, wf_day.stride).fit_stats(held_in_stats, st)
            pool_preds.append(w_p.predict(z_day(b)))
        out["cross"][f"wiener_scratch_{k}s"] = score(test_B, sc_preds)
        out["cross"][f"wiener_pooled_{k}s"] = score(test_B, pool_preds)
        log(f"wiener few-shot {k}s: scratch={out['cross'][f'wiener_scratch_{k}s']['r2_vw_mean']:.3f} "
            f"pooled={out['cross'][f'wiener_pooled_{k}s']['r2_vw_mean']:.3f}")

    # ------------------------------------------------------------------ Kalman filter
    best = None
    for tau in dc.kalman.smooth_taus:
        for lag in dc.kalman.obs_lags:
            for decay in dc.kalman.decays:
                kf = KalmanFilter(tau, lag, decay=decay).fit(zs_tr, ys_tr, ms_tr)
                s = val_score([kf.predict(z) for z in zs_va])
                if best is None or s > best[0]:
                    best = (s, tau, lag, decay)
    kf = KalmanFilter(best[1], best[2], decay=best[3]).fit(zs, ys, ms)
    out["selection"]["kalman"] = {"smooth_tau": best[1], "obs_lag": best[2], "decay": best[3], "val": best[0]}
    out["within"]["kalman"] = score(minival, [kf.predict(z_day(b)) for b in minival])
    out["cross"]["kalman_zeroshot"] = score(test_B, [kf.predict(z_day(b)) for b in test_B])
    log(f"kalman tau={best[1]} lag={best[2]} decay={best[3]} within={out['within']['kalman']['r2_vw_mean']:.3f} "
        f"cross0={out['cross']['kalman_zeroshot']['r2_vw_mean']:.3f}")



def _run_neural(out, preds_store, dc, names, dev, days, day_id, minival, pairs, test_B, zs, ys, ms, rid, zs_tr, zs_va,
                ys_tr, ys_va, ms_tr, ms_va, z_day, ny, score, global_norm=None, seeds=None) -> None:
    """GRU / Transformer / multi-day GRU variants over seeds, with few-shot recalibration."""
    nc = dc.neural
    variants = {
        "gru": dict(core="gru", per_day=False, aug=True),
        "gru_noaug": dict(core="gru", per_day=False, aug=False),
        "transformer": dict(core="transformer", per_day=False, aug=True),
        "ms_gru": dict(core="gru", per_day=True, aug=True),  # proposed
        "ms_gru_noaug": dict(core="gru", per_day=True, aug=False),
    }
    aug = Augment(nc.aug.gain_sd, nc.aug.offset_sd, nc.aug.channel_drop, nc.aug.noise_sd)
    per_seed: dict = {}
    for name in names:
        v = variants[name]
        run_seeds = dc.seeds if name in nc.multi_seed else dc.seeds[:1]
        if seeds is not None:
            run_seeds = [sd for sd in run_seeds if sd in seeds]
        for seed in run_seeds:
            t0 = time.time()
            torch.manual_seed(seed)
            model = SeqDecoder(zs[0].shape[1], 7, n_readins=len(days) if v["per_day"] else 1,
                               d_in=nc.d_in, hidden=nc.hidden if v["core"] == "gru" else nc.tf_hidden,
                               layers=nc.layers if v["core"] == "gru" else nc.tf_layers, core=v["core"], dropout=nc.dropout,
                               context=nc.tf_context).to(dev)
            r = rid if v["per_day"] else [0] * len(rid)
            win = nc.window if v["core"] == "gru" else nc.tf_context
            tc = TrainCfg(window=win, batch=nc.batch,
                          steps=nc.steps, lr=nc.lr, weight_decay=nc.weight_decay, eval_every=nc.eval_every,
                          patience=nc.patience, augment=aug if v["aug"] else Augment())
            hist = train_model(model, (zs_tr, ys_tr, ms_tr, r), (zs_va, ys_va, ms_va, r), tc, seed=seed,
                               log=lambda m: log(f"{name} s{seed}{m}"))
            res = {"train_time_s": time.time() - t0, "best_val": hist["best_val"]}
            ck = RESULTS_ROOT / "checkpoints"
            ck.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), ck / f"{name}_s{seed}.pt")
            # within-session
            within_preds = [model.predict(z_day(b), day_id[b.day] if v["per_day"] else 0) for b in minival]
            res["within"] = score(minival, within_preds)
            if seed == dc.seeds[0] and name in ("gru", "ms_gru"):
                preds_store[f"{name}_within"] = {b.key: p for b, p in zip(minival, within_preds)}
            # cross-session zero-shot: per-day read-in model uses the mean read-in
            if v["per_day"]:
                zero_readin = fit_new_readin(model, [], [], [], steps=0)  # mean of trained read-ins
                res["cross_zeroshot"] = score(test_B, [model.predict(z_day(b), zero_readin) for b in test_B])
            else:
                res["cross_zeroshot"] = score(test_B, [model.predict(z_day(b), 0) for b in test_B])
            # label-free recalibration (improvement 9): unlabelled block A only
            if nc.get("unsup", {}).get("enabled", True):
                t1 = time.time()
                unsup = _unsup_recal(model, v["per_day"], nc, seed, pairs, zs_tr, rid, z_day, global_norm, score,
                                     test_B, preds_store if seed == dc.seeds[0] else None, name)
                res.update(unsup)
                log(f"{name} seed {seed} label-free: " + " ".join(
                    f"{k.replace('cross_', '')}={unsup[k]['r2_vw_mean']:.3f}" for k in unsup) + f" ({time.time()-t1:.0f}s)")
            # few-shot
            for k in dc.recal_seconds:
                nb = int(k / 0.02)
                preds = []
                for a, b in pairs:
                    rcl = a.slice(0, nb)
                    zr, yr, mr = [z_day(rcl)], [ny(rcl.vel).astype(np.float32)], [rcl.eval_mask]
                    if v["per_day"]:
                        ri = fit_new_readin(model, zr, yr, mr, steps=nc.recal_steps, lr=nc.recal_lr,
                                            anchor=nc.recal_anchor, seed=seed, window=win,
                                            batch=nc.recal_batch)
                        preds.append(model.predict(z_day(b), ri))
                    else:
                        m2 = finetune_all(model, zr, yr, mr, steps=nc.recal_steps, lr=nc.ft_lr, seed=seed,
                                          window=win, batch=nc.recal_batch)
                        preds.append(m2.predict(z_day(b), 0))
                res[f"cross_{k}s"] = score(test_B, preds)
                if name == "ms_gru" and k == max(dc.recal_seconds) and seed == dc.seeds[0]:
                    preds_store["ms_gru"] = {b.key: p for (_, b), p in zip(pairs, preds)}
                if name == "gru" and k == max(dc.recal_seconds) and seed == dc.seeds[0]:
                    preds_store["gru"] = {b.key: p for (_, b), p in zip(pairs, preds)}
            log(f"{name} seed {seed}: within={res['within']['r2_vw_mean']:.3f} zs={res['cross_zeroshot']['r2_vw_mean']:.3f} "
                + " ".join(f"{k}s={res[f'cross_{k}s']['r2_vw_mean']:.3f}" for k in dc.recal_seconds)
                + f" ({res['train_time_s']:.0f}s)")
            per_seed.setdefault(name, []).append(res)
            out.setdefault("neural_runs", {})[f"{name}|{seed}"] = res
            # ablation for the proposed model: fine-tune everything instead of read-in only
            if name == "ms_gru" and nc.get("ablate_ft_all", True) and seed == dc.seeds[0]:
                for k in dc.recal_seconds:
                    preds = []
                    for a, b in pairs:
                        rcl = a.slice(0, int(k / 0.02))
                        ri = fit_new_readin(model, [z_day(rcl)], [ny(rcl.vel).astype(np.float32)], [rcl.eval_mask],
                                            steps=0)
                        m2 = _with_readin(model, ri)
                        m2 = finetune_all(m2, [z_day(rcl)], [ny(rcl.vel).astype(np.float32)], [rcl.eval_mask],
                                          steps=nc.recal_steps, lr=nc.ft_lr, seed=seed, window=win,
                                          batch=nc.recal_batch)
                        preds.append(m2.predict(z_day(b), 0))
                    res[f"cross_ftall_{k}s"] = score(test_B, preds)
                log(f"ms_gru seed {seed} ft-all: " + " ".join(
                    f"{k}s={res[f'cross_ftall_{k}s']['r2_vw_mean']:.3f}" for k in dc.recal_seconds))

    aggregate_neural(out)


def aggregate_neural(out: dict) -> None:
    """Aggregate the raw per-(variant, seed) runs in ``out['neural_runs']`` across seeds."""
    by_name: dict = {}
    for key, res in sorted(out.get("neural_runs", {}).items()):
        by_name.setdefault(key.split("|")[0], []).append(res)
    for name, runs in by_name.items():
        keys = sorted({k for r in runs for k in r if isinstance(r[k], dict)})
        for k in keys:
            have = [r for r in runs if k in r]  # some ablations run on the first seed only
            vals = [r[k]["r2_vw_mean"] for r in have]
            per_dof = {d: float(np.mean([r[k]["r2_pooled_per_dof"][d] for r in have])) for d in DOF_NAMES}
            entry = {"r2_vw_mean": float(np.mean(vals)), "r2_vw_seed_std": float(np.std(vals)),
                     "r2_vw_block_std": float(np.mean([r[k]["r2_vw_std"] for r in have])),
                     "r2_vw_blocks": np.mean([r[k]["r2_vw_blocks"] for r in have], 0).tolist(),
                     "r2_vw_blocks_per_seed": [r[k]["r2_vw_blocks"] for r in have],
                     "r2_pooled_per_dof": per_dof, "n_seeds": len(have),
                     "r2_vw_per_seed": vals}
            if k == "within":
                out.setdefault("within", {})[name] = entry
            else:
                out.setdefault("cross", {})[f"{name}_{k.replace('cross_', '')}"] = entry
        out.setdefault("selection", {})[f"{name}_train_time_s"] = float(np.mean([r["train_time_s"] for r in runs]))


def _unsup_recal(model, per_day, nc, seed, pairs, zs_tr, rid, z_day, global_norm, score, test_B, preds_store,
                 name) -> dict:
    """Label-free recalibration on each held-out day, using block A's neural data only.

    * ``zs_fixednorm``: no per-day re-centring/re-scaling (training-pooled z-scores) - the
      ablation that shows what per-channel re-normalisation to the new day buys.
    * ``unsup_center``: re-centre the read-in output (latent mean matching only).
    * ``unsup_coral``: closed-form latent mean + covariance matching (CORAL).
    * ``unsup_kl``: NoMAD-style KL alignment of read-in output and core-state Gaussians,
      initialised at the CORAL solution.
    All variants except ``zs_fixednorm`` use block-A per-channel z-scoring.
    """
    uc = nc.get("unsup", {})
    win = nc.window if model.core_type == "gru" else nc.tf_context
    burn = min(16, win // 4)
    ref = latent_stats(model, zs_tr, [r if per_day else 0 for r in rid], window=win, burn=burn, seed=seed)
    base = mean_readin(model)
    preds = {k: [] for k in ("zs_fixednorm", "unsup_center", "unsup_coral", "unsup_kl")}
    for a, b in pairs:
        za = [z_day(a)]
        preds["zs_fixednorm"].append(model.predict(global_norm(b.spikes).astype(np.float32), base))
        ri_c = align_readin_moments(model, za, ref, mean_only=True, init=base)
        preds["unsup_center"].append(model.predict(z_day(b), ri_c))
        ri_m = align_readin_moments(model, za, ref, shrink=uc.get("shrink", 0.1), init=base)
        preds["unsup_coral"].append(model.predict(z_day(b), ri_m))
        ri_k = align_readin_kl(model, za, ref, steps=uc.get("kl_steps", 150), lr=uc.get("kl_lr", 2e-3),
                               window=win, burn=burn, batch=uc.get("kl_batch", 24),
                               anchor=uc.get("kl_anchor", 1e-2), w_h=uc.get("kl_w_h", 1.0), init=ri_m, seed=seed)
        preds["unsup_kl"].append(model.predict(z_day(b), ri_k))
    if preds_store is not None:
        for k in ("unsup_coral", "unsup_kl"):
            preds_store[f"{name}_{k}"] = {b.key: p for b, p in zip(test_B, preds[k])}
    return {f"cross_{k}": score(test_B, p) for k, p in preds.items()}


def _scratch_alpha(proto: WienerFilter, z: np.ndarray, y: np.ndarray, m: np.ndarray, alphas: list[float]) -> float:
    """Pick the ridge strength for a from-scratch Wiener filter by 2-fold (first/second
    half) cross-validation inside the labelled recalibration data only."""
    h = len(z) // 2
    halves = [(slice(0, h), slice(h, None)), (slice(h, None), slice(0, h))]
    best = None
    for a in alphas:
        sc = []
        for tr, te in halves:
            w = WienerFilter(proto.n_lags, a, proto.smooth_tau, proto.stride).fit([z[tr]], [y[tr]], [m[tr]])
            sc.append(r2_variance_weighted(y[te][m[te]], w.predict(z[te])[m[te]]))
        if best is None or np.mean(sc) > best[0]:
            best = (float(np.mean(sc)), a)
    return best[1]


def _with_readin(model: SeqDecoder, readin) -> SeqDecoder:
    """Copy of a multi-read-in model collapsed to a single given read-in layer."""
    import copy

    m2 = copy.deepcopy(model)
    m2.readins = torch.nn.ModuleList([copy.deepcopy(readin)])
    return m2


def _merge(parts_dir: Path) -> dict:
    """Merge per-part result files written by ``--parts`` runs."""
    out: dict = {}
    for f in sorted(parts_dir.glob("*.json")):
        d = json.loads(f.read_text())
        for k, v in d.items():
            if isinstance(v, dict):
                out.setdefault(k, {}).update(v)
            else:
                out[k] = v
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/cpu.yaml")
    ap.add_argument("--parts", default=None,
                    help="comma-separated subset, e.g. 'linear,gru' (results go to results/decoding_parts/)")
    ap.add_argument("--seeds", default=None, help="comma-separated neural seeds to run (default: all)")
    ap.add_argument("--merge", action="store_true", help="merge results/decoding_parts/ into the final files")
    args = ap.parse_args()
    res_dir = RESULTS_ROOT
    res_dir.mkdir(parents=True, exist_ok=True)
    parts_dir = res_dir / "decoding_parts"
    if args.merge:
        out = _merge(parts_dir)
        aggregate_neural(out)
        add_decoding_stats(out)
        preds = {}
        for f in sorted(parts_dir.glob("*.npz")):
            d = np.load(f)
            preds.update({k: d[k] for k in d.files})
        np.savez_compressed(res_dir / "decoding_preds.npz", **preds)
    else:
        cfg = load_config(args.config)
        parts = set(args.parts.split(",")) if args.parts else None
        seeds = [int(x) for x in args.seeds.split(",")] if args.seeds else None
        out = run(cfg, parts, seeds)
        preds = out.pop("_preds")
        arrays = {f"{m}|{k}": np.array(v) for m, d in preds.items() for k, v in d.items()}
        if parts is not None:
            parts_dir.mkdir(exist_ok=True)
            tag = "_".join(sorted(parts)) + (f"_s{args.seeds.replace(',', '-')}" if args.seeds else "")
            (parts_dir / f"{tag}.json").write_text(json.dumps(out, indent=1))
            np.savez_compressed(parts_dir / f"{tag}.npz", **arrays)
            log(f"wrote part {tag}")
            return
        np.savez_compressed(res_dir / "decoding_preds.npz", **arrays)
        add_decoding_stats(out)
    with open(res_dir / "decoding_metrics.json", "w") as fh:
        json.dump(out, fh, indent=1)
    log(f"wrote {res_dir / 'decoding_metrics.json'}")


if __name__ == "__main__":
    main()
