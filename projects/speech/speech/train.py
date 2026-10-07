"""Train a neural-to-phoneme CTC decoder and dump its posteriors.

Usage::

    python -m speech.train --config configs/cpu.yaml --run gru_baseline --seed 0

Outputs (``results/runs/<run>_s<seed>/``): ``model.pt`` (best dev checkpoint),
``log.jsonl`` (training curve), ``summary.json`` and ``posteriors_{dev,test}.npz``
(per-trial CTC log-probabilities used by the language-model decoding stage).
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .augment import AugmentConfig, augment_batch
from .config import get_device, load_config, run_config
from .ctc import greedy_decode
from .data import Trial, collate, load_split
from .metrics import error_rate
from .models import build_model


# --------------------------------------------------------------------------- data
def bin_trial(t: Trial, factor: int) -> Trial:
    """Average ``factor`` consecutive 20 ms bins (factor 2 -> 40 ms bins)."""
    if factor == 1:
        return t
    T = (t.features.shape[0] // factor) * factor
    x = t.features[:T].astype(np.float32).reshape(T // factor, factor, -1).mean(1)
    return Trial(x.astype(np.float16), t.phones, t.sentence, t.day, t.session, t.block)


_CACHE: dict = {}


def prepare_data(dcfg: dict, data_dir: str) -> dict[str, list[Trial]]:
    """Build train/dev/test trial lists.

    * train/dev: the released ``train`` split of every training session, with a
      fixed ``dev_fraction`` of each session's trials held out for model
      selection and language-model tuning.
    * test: the released ``val`` split of the same sessions (never used for
      any choice).
    * heldout_train/heldout_test: train/val splits of ``heldout_session``,
      which is excluded from base training (used for the calibration study).

    Day indices follow the order of ``sessions`` (+ the held-out session last).
    """
    key = json.dumps(dcfg, sort_keys=True) + data_dir
    if key in _CACHE:
        return _CACHE[key]
    sessions = list(dcfg["sessions"])
    held = dcfg.get("heldout_session")
    all_sessions = sessions + ([held] if held else [])
    f = dcfg.get("bin_factor", 1)
    train_all = [bin_trial(t, f) for t in load_split(Path(data_dir), sessions, "train")]
    test = [bin_trial(t, f) for t in load_split(Path(data_dir), sessions, "val")]
    rng = random.Random(1234)
    train, dev = [], []
    for day in range(len(sessions)):
        ts = [t for t in train_all if t.day == day]
        rng.shuffle(ts)
        n_dev = max(1, int(round(dcfg.get("dev_fraction", 0.1) * len(ts))))
        dev += ts[:n_dev]
        train += ts[n_dev:]
    out = {"train": train, "dev": dev, "test": test, "heldout_train": [], "heldout_test": []}
    if held:
        day = len(sessions)
        for split, name in (("train", "heldout_train"), ("val", "heldout_test")):
            ts = load_split(Path(data_dir), [held], split)
            for t in ts:
                t.day = day
            out[name] = [bin_trial(t, f) for t in ts]
    out["n_days"] = len(all_sessions)
    out["sessions"] = all_sessions
    _CACHE[key] = out
    return out


def length_batches(trials: list[Trial], batch_size: int, rng: random.Random) -> list[list[int]]:
    """Bucket trials of similar length together to minimise padding, shuffle batch order."""
    order = sorted(range(len(trials)), key=lambda i: trials[i].features.shape[0] + rng.random())
    batches = [order[i : i + batch_size] for i in range(0, len(order), batch_size)]
    rng.shuffle(batches)
    return batches


def make_batch(trials: list[Trial], idx: list[int]):
    """Collate trials ``idx`` into padded float32 tensors."""
    items = [
        (torch.from_numpy(trials[i].features.astype(np.float32)), torch.from_numpy(trials[i].phones), trials[i].day, i)
        for i in idx
    ]
    return collate(items)


# --------------------------------------------------------------------------- loss / eval
def ctc_loss(out: dict, y, y_len, inter_weight: float) -> torch.Tensor:
    """CTC loss (mean over the batch of per-sentence loss / target length), plus intermediate CTC."""

    def one(lp):
        return F.ctc_loss(lp.transpose(0, 1), y, out["lengths"], y_len, blank=0, reduction="mean", zero_infinity=True)

    loss = one(out["log_probs"])
    inter = out.get("inter_log_probs") or []
    if inter and inter_weight > 0:
        loss = (1 - inter_weight) * loss + inter_weight * sum(one(lp) for lp in inter) / len(inter)
    return loss


@torch.no_grad()
def posteriors(model, trials: list[Trial], device, batch_size: int = 32) -> list[np.ndarray]:
    """Per-trial CTC log-probabilities (T', 41) as float32 arrays, in input order."""
    model.eval()
    out: list[np.ndarray | None] = [None] * len(trials)
    order = sorted(range(len(trials)), key=lambda i: trials[i].features.shape[0])
    for s in range(0, len(order), batch_size):
        idx = order[s : s + batch_size]
        x, x_len, _, _, day, _ = make_batch(trials, idx)
        o = model(x.to(device), x_len.to(device), day.to(device))
        lp, L = o["log_probs"].float().cpu().numpy(), o["lengths"].cpu().numpy()
        for j, i in enumerate(idx):
            out[i] = lp[j, : L[j]]
    return out  # type: ignore[return-value]


def greedy_per(lps: list[np.ndarray], trials: list[Trial]) -> float:
    """Phoneme error rate of best-path decoding (word-boundary token included)."""
    return error_rate([list(t.phones) for t in trials], [greedy_decode(lp) for lp in lps])


def save_posteriors(path: Path, lps: list[np.ndarray], trials: list[Trial]) -> None:
    """Store posteriors (float16) with labels so decoding needs no neural data."""
    offs = np.cumsum([0] + [len(lp) for lp in lps])
    np.savez_compressed(
        path,
        log_probs=np.concatenate(lps).astype(np.float16),
        offsets=offs,
        sentences=np.array([t.sentence for t in trials]),
        phones=np.array([" ".join(map(str, t.phones)) for t in trials]),
        sessions=np.array([t.session for t in trials]),
        blocks=np.array([t.block for t in trials]),
    )


def load_posteriors(path: Path) -> dict:
    """Inverse of :func:`save_posteriors`."""
    z = np.load(path)
    o = z["offsets"]
    lp = z["log_probs"].astype(np.float32)
    return {
        "log_probs": [lp[o[i] : o[i + 1]] for i in range(len(o) - 1)],
        "sentences": [str(s) for s in z["sentences"]],
        "phones": [[int(p) for p in s.split()] for s in z["phones"]],
        "sessions": [str(s) for s in z["sessions"]],
        "blocks": [int(b) for b in z["blocks"]],
    }


# --------------------------------------------------------------------------- training
def lr_lambda(step: int, warmup: int, total: int, min_ratio: float) -> float:
    """Linear warm-up then cosine decay to ``min_ratio`` of the peak."""
    if step < warmup:
        return (step + 1) / warmup
    p = min(1.0, (step - warmup) / max(1, total - warmup))
    return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * p))


def make_optimizer(model, tcfg: dict):
    """AdamW with the day-specific input layers in their own group (no weight decay)."""
    day = [p for n, p in model.named_parameters() if n.startswith("day.") and p.requires_grad]
    rest = [p for n, p in model.named_parameters() if not n.startswith("day.") and p.requires_grad]
    groups = []
    if rest:
        groups.append({"params": rest, "weight_decay": tcfg["weight_decay"]})
    if day:
        groups.append({"params": day, "weight_decay": 0.0})
    return torch.optim.AdamW(groups, lr=tcfg["lr"], betas=(0.9, 0.999), eps=tcfg.get("adam_eps", 1e-8))


def train_loop(model, train: list[Trial], dev: list[Trial], rc: dict, device, seed: int, log_path: Path | None = None):
    """Optimise CTC; keep the parameters with the best dev greedy PER. Returns (history, best_state)."""
    tcfg = rc["train"]
    aug = AugmentConfig(**rc["aug"])
    rng = random.Random(seed)
    opt = make_optimizer(model, tcfg)
    total = tcfg["max_steps"]
    by_time = tcfg.get("schedule", "steps") == "time"
    base_lrs = [g["lr"] for g in opt.param_groups]
    use_amp = device.type == "cuda" and tcfg.get("amp", True)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    t0, step, best, best_state, hist = time.time(), 0, float("inf"), None, []
    batches: list[list[int]] = []
    feasible = [t for t in train if (t.features.shape[0] - rc["model"]["patch_size"]) // rc["model"]["patch_stride"] + 1 >= len(t.phones)]
    while step < total and (time.time() - t0) / 60 < tcfg["max_minutes"]:
        if not batches:
            batches = length_batches(feasible, tcfg["batch_size"], rng)
        # Cosine schedule over steps, or (CPU budget mode) over the wall-clock budget,
        # so that a run on a slower/shared machine still completes its schedule.
        if by_time:
            frac = (time.time() - t0) / 60 / tcfg["max_minutes"]
            factor = lr_lambda(step, tcfg["warmup_steps"], 10**9, tcfg["lr_min_ratio"]) if step < tcfg["warmup_steps"] \
                else lr_lambda(int(frac * 10**6), 0, 10**6, tcfg["lr_min_ratio"])
        else:
            factor = lr_lambda(step, tcfg["warmup_steps"], total, tcfg["lr_min_ratio"])
        for g, b in zip(opt.param_groups, base_lrs):
            g["lr"] = b * factor
        model.train()
        x, x_len, y, y_len, day, _ = make_batch(feasible, batches.pop())
        x, x_len = augment_batch(x, x_len, aug)
        x, x_len, y, y_len, day = (v.to(device) for v in (x, x_len, y, y_len, day))
        with torch.autocast(device.type, enabled=use_amp):
            out = model(x, x_len, day)
        out = {k: (v.float() if torch.is_tensor(v) else [u.float() for u in v]) for k, v in out.items()}
        out["lengths"] = out["lengths"].long()
        loss = ctc_loss(out, y, y_len, tcfg.get("inter_ctc_weight", 0.0))
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg["grad_clip"])
        scaler.step(opt)
        scaler.update()
        step += 1
        if step % tcfg["eval_every"] == 0 or step == total:
            per = greedy_per(posteriors(model, dev, device), dev)
            rec = {"step": step, "loss": float(loss), "dev_per": per, "minutes": (time.time() - t0) / 60, "lr": opt.param_groups[0]["lr"]}
            hist.append(rec)
            print(json.dumps(rec), flush=True)
            if log_path:
                with open(log_path, "a") as f:
                    f.write(json.dumps(rec) + "\n")
            if per < best:
                best = per
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    # Final evaluation if the time budget ended between evaluations.
    if not hist or hist[-1]["step"] != step:
        per = greedy_per(posteriors(model, dev, device), dev)
        hist.append({"step": step, "loss": float(loss), "dev_per": per, "minutes": (time.time() - t0) / 60})
        if log_path:
            with open(log_path, "a") as f:
                f.write(json.dumps(hist[-1]) + "\n")
        if per < best:
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    return hist, best_state


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-minutes", type=float, default=None, help="override the time budget (smoke tests)")
    ap.add_argument("--results-dir", default=None, help="override the results directory")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.results_dir:
        cfg["results_dir"] = str(Path(args.results_dir).resolve())
    torch.set_num_threads(cfg.get("threads", 2))
    device = get_device(cfg.get("device", "auto"))
    rc = run_config(cfg, args.run)
    if args.max_minutes is not None:
        rc["train"]["max_minutes"] = args.max_minutes
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    out_dir = Path(cfg["results_dir"]) / "runs" / f"{args.run}_s{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "log.jsonl").unlink(missing_ok=True)
    data = prepare_data(rc["data"], cfg["data_dir"])
    print(f"[{args.run}] train={len(data['train'])} dev={len(data['dev'])} test={len(data['test'])} days={data['n_days']}", flush=True)

    model = build_model(rc["model"], data["n_days"]).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    t0 = time.time()
    hist, best_state = train_loop(model, data["train"], data["dev"], rc, device, args.seed, out_dir / "log.jsonl")
    train_min = (time.time() - t0) / 60
    model.load_state_dict(best_state)
    torch.save({"state_dict": model.state_dict(), "run_config": rc, "n_days": data["n_days"]}, out_dir / "model.pt")

    summary = {"run": args.run, "seed": args.seed, "n_params": n_params, "train_minutes": train_min, "steps": hist[-1]["step"]}
    for split in ("dev", "test"):
        lps = posteriors(model, data[split], device)
        save_posteriors(out_dir / f"posteriors_{split}.npz", lps, data[split])
        summary[f"{split}_per_greedy"] = greedy_per(lps, data[split])
    summary["best_dev_per"] = min(h["dev_per"] for h in hist)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
