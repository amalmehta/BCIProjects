"""Short stimulation sequences under habituation, and a temporal encoder that manages it.

In the simulator (van der Grinten et al. 2024), every pulse train an electrode delivers raises
its habituation trace B, which subtracts from the effective current: I_eff = (I - I0 - B) d.
The trace decays with a time constant of about 2000 s, so within a glance it only grows. An
electrode driven continuously at 100 uA loses its phosphene after roughly a second
(``frame_update`` in simulator.py gives the per-frame closed form).

Sequences here are ``frames`` consecutive 166 ms frames (one pulse train each) showing either
a static image held still or the same image translating slowly. Encoders compared:

* ``intensity``, ``learned`` (generic), ``learned_threshold_aware``: single-frame encoders
  applied frame by frame, unaware of habituation;
* ``threshold_aware_compensated``: the threshold-aware encoder given its threshold plus the
  tracked habituation shift (B in uA) - a no-training habituation-compensation rule;
* ``learned_temporal``: the threshold-aware architecture with two extra per-electrode inputs
  (habituation trace and previous amplitude), fine-tuned through the frame dynamics on
  sequences (backpropagation through time) under the same safety layer and charge cost.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .config import Setup, build_setup, variant_config
from .data import load
from .encoders import Condition, ThresholdAwareEncoder
from .metrics import ssim, ssim_lowpass
from .simulator import Patient, frame_update
from .train import _autocast, build_encoder, load_encoder, patient_condition, percept_loss


def make_sequences(x: torch.Tensor, frames: int, mode: str, speed_px: float, gen: torch.Generator) -> torch.Tensor:
    """(B, 1, H, W) images -> (B, T, 1, H, W) sequences.

    ``static``: the image held still. ``moving``: the image translates in a random direction at
    ``speed_px`` pixels per frame, centred on its original position at mid-sequence (zero padding).
    """
    b, _, h, w = x.shape
    if mode == "static":
        return x.unsqueeze(1).expand(b, frames, 1, h, w).contiguous()
    ang = torch.rand(b, generator=gen) * 2 * math.pi
    t = torch.arange(frames, dtype=torch.float32) - (frames - 1) / 2
    dx = (speed_px * t[None] * torch.cos(ang)[:, None]) * 2 / w  # normalised grid units
    dy = (speed_px * t[None] * torch.sin(ang)[:, None]) * 2 / h
    theta = torch.zeros(b * frames, 2, 3)
    theta[:, 0, 0] = theta[:, 1, 1] = 1.0
    theta[:, 0, 2] = -dx.reshape(-1)
    theta[:, 1, 2] = -dy.reshape(-1)
    xs = x.unsqueeze(1).expand(b, frames, 1, h, w).reshape(b * frames, 1, h, w)
    grid = F.affine_grid(theta.to(x.device), list(xs.shape), align_corners=False)
    out = F.grid_sample(xs, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    return out.view(b, frames, 1, h, w)


def _amplitudes(kind: str, enc, setup: Setup, x: torch.Tensor, cond: Condition | None, trace: torch.Tensor,
                prev: torch.Tensor, train: bool):
    """(gate, level) for one frame; gate is None for unconditioned encoders (amp = level)."""
    imax = setup.params.max_current_ua
    if kind == "frame":  # any image-only encoder (baselines, generic learned)
        return None, setup.safety(enc(x)).clamp(0, imax)
    tr_ua = trace * 1e6
    if kind == "threshold_aware":
        c = cond
    elif kind == "compensated":
        c = Condition(cond.threshold_ua, cond.alive, trace_ua=tr_ua)
    elif kind == "temporal":
        c = Condition(cond.threshold_ua, cond.alive, trace_ua=tr_ua, prev_amp_ua=prev)
    else:
        raise ValueError(kind)
    with _autocast(x.device):
        gate, level = enc.parts(x, c)
    gate, level = gate.float(), level.float().clamp(0.0, imax)
    k = setup.safety.keep_mask(gate.detach() * level)
    return gate, level * k


def rollout(kind: str, enc, setup: Setup, patient: Patient, seq: torch.Tensor, cond: Condition | None,
            train: bool = False, act: torch.Tensor | None = None, trace: torch.Tensor | None = None,
            prev: torch.Tensor | None = None):
    """Run a sequence (B, T, 1, H, W) through encoder + frame dynamics.

    Returns percepts (B, T, 1, H, W), amplitudes (B, T, N), "wasted" masks (B, T, N: stimulated but
    below threshold, i.e. no phosphene) and the final state (act, trace, prev).
    """
    b, frames = seq.shape[:2]
    n = setup.n
    alive = patient.alive
    act = torch.zeros(b, n, device=seq.device) if act is None else act
    trace = torch.zeros(b, n, device=seq.device) if trace is None else trace
    prev = torch.zeros(b, n, device=seq.device) if prev is None else prev
    percepts, amps, wasted = [], [], []
    for t in range(frames):
        gate, level = _amplitudes(kind, enc, setup, seq[:, t], cond, trace, prev, train)
        amp = level * alive if gate is None else gate * level * alive
        act_new, trace_new = frame_update(setup.params, amp, act, trace)
        if gate is not None and train:
            # Brightness gate trick (see PhospheneSimulator.forward): activation the electrode would
            # reach if on, brightness multiplied by the straight-through gate.
            act_lvl, _ = frame_update(setup.params, level * alive, act, trace)
            p = setup.sim(level, patient, hard=False, fast=True, activation=act_lvl, gate=gate)
        else:
            p = setup.sim(amp, patient, hard=not train, fast=train, activation=act_new)
        percepts.append(p)
        amps.append(amp)
        wasted.append((amp > 0) & (act_new <= patient.threshold))
        act, trace, prev = act_new, trace_new, amp
    return torch.stack(percepts, 1), torch.stack(amps, 1), torch.stack(wasted, 1), (act, trace, prev)


def sequence_loss(percepts: torch.Tensor, amps: torch.Tensor, seq: torch.Tensor, w_ssim: float, w_charge: float,
                  imax: float) -> torch.Tensor:
    b, frames = seq.shape[:2]
    pl = percept_loss(percepts.flatten(0, 1), seq.flatten(0, 1), w_ssim)
    return pl + w_charge * (amps / imax).mean()


def run_dir(cfg: dict, dataset: str, variant: str, seed: int) -> Path:
    return Path(cfg["results_dir"]) / "runs" / f"{dataset}_{variant}_s{seed}"


def temporal_encoder_from(ta: ThresholdAwareEncoder, setup: Setup, width: int) -> ThresholdAwareEncoder:
    """Temporal encoder initialised from a trained threshold-aware one (new inputs start at zero weight)."""
    enc = ThresholdAwareEncoder(setup.context(calibrated=False), width=width, temporal=True).to(setup.device)
    sd = ta.state_dict()
    w_old = sd["head.0.weight"]
    w_new = torch.zeros(w_old.shape[0], w_old.shape[1] + 2, device=w_old.device)
    w_new[:, : w_old.shape[1]] = w_old
    sd["head.0.weight"] = w_new
    enc.load_state_dict(sd)
    return enc


def train_temporal(cfg: dict, dataset: str, out_dir: Path, log=print) -> dict:
    """Fine-tune a temporal encoder through the frame dynamics (BPTT over ``frames_train`` frames).

    Each sequence starts from a random habituation state: a warm-up of 0..frames_train frames
    rendered without gradients by the encoder itself, so later-sequence states are seen in training.
    """
    tc_t = cfg["temporal"]
    seed = tc_t["seed"]
    torch.manual_seed(seed)
    vcfg = variant_config(cfg, tc_t["init_variant"])
    ta, _, setup = load_encoder(cfg, tc_t["init_variant"], run_dir(cfg, dataset, tc_t["init_variant"], seed))
    enc = temporal_encoder_from(ta, setup, vcfg["train"]["width"]).train()
    pc, tc = cfg["patients"], cfg["train"]
    train_set, _ = load(dataset, cfg["data_root"], setup.params.resolution, n_train=tc["n_train"], n_test=1, seed=0)
    pool = [setup.patient(pc["train_seed0"] + i) for i in range(pc["train_pool"])]
    conds = {id(p): patient_condition(setup, p) for p in pool}
    opt = torch.optim.Adam(enc.parameters(), lr=tc_t["lr"])
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 0.5 * (1 + math.cos(math.pi * s / tc_t["steps"])))
    rng = np.random.default_rng(seed)
    gen = torch.Generator().manual_seed(seed)
    imax, frames = setup.params.max_current_ua, tc_t["frames_train"]
    history, t0 = [], time.time()
    for step in range(tc_t["steps"]):
        patient = pool[rng.integers(len(pool))]
        cond = conds[id(patient)]
        x, _ = train_set.batch(rng.integers(len(train_set), size=tc_t["batch"]))
        mode = "static" if rng.random() < 0.5 else "moving"
        warm = int(rng.integers(0, frames + 1))
        seq = make_sequences(x.to(setup.device), warm + frames, mode, tc_t["speed_px"], gen)
        state = (None, None, None)
        if warm:
            enc.eval()
            with torch.no_grad():
                *_, state = rollout("temporal", enc, setup, patient, seq[:, :warm], cond, train=False)
            enc.train()
        p, a, _, _ = rollout("temporal", enc, setup, patient, seq[:, warm:], cond, train=True,
                             act=state[0], trace=state[1], prev=state[2])
        loss = sequence_loss(p, a, seq[:, warm:], tc["w_ssim"], tc_t["w_charge"], imax)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(enc.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % 25 == 0 or step == tc_t["steps"] - 1:
            rec = {"step": step, "loss": float(loss), "warm": warm, "mode": mode,
                   "charge": float((a / imax).mean()), "elapsed_s": round(time.time() - t0, 1)}
            history.append(rec)
            log(f"[{dataset}/temporal] " + " ".join(f"{k}={v}" for k, v in rec.items()))
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(enc.state_dict(), out_dir / "encoder.pt")
    summary = {"dataset": dataset, "seed": seed, "variant": "temporal", "init": tc_t["init_variant"],
               "steps": tc_t["steps"], "train_time_s": round(time.time() - t0, 1),
               "n_params_encoder": sum(p.numel() for p in enc.parameters()), "history": history}
    (out_dir / "train_log.json").write_text(json.dumps(summary, indent=1))
    return summary


def load_temporal(cfg: dict, dataset: str) -> ThresholdAwareEncoder:
    tc_t = cfg["temporal"]
    vcfg = variant_config(cfg, tc_t["init_variant"])
    setup = build_setup(vcfg)
    enc = ThresholdAwareEncoder(setup.context(calibrated=False), width=vcfg["train"]["width"], temporal=True)
    enc.load_state_dict(torch.load(run_dir(cfg, dataset, "temporal", tc_t["seed"]) / "encoder.pt",
                                   map_location=setup.device))
    return enc.to(setup.device).eval()


@torch.no_grad()
def evaluate_temporal(cfg: dict, dataset: str, log=print) -> dict:
    """Per-frame fidelity of each encoder on static and moving sequences, five held-out patients."""
    from .evaluate import baseline_factories

    tc_t = cfg["temporal"]
    setup = build_setup(cfg)
    seed = tc_t["seed"]
    gen_enc, _, _ = load_encoder(cfg, "main", run_dir(cfg, dataset, "main", seed))
    ta, _, _ = load_encoder(cfg, tc_t["init_variant"], run_dir(cfg, dataset, tc_t["init_variant"], seed))
    tmp = load_temporal(cfg, dataset)
    n, frames = tc_t["n_eval_per_patient"], tc_t["frames_eval"]
    _, test_set = load(dataset, cfg["data_root"], setup.params.resolution, n_train=1,
                       n_test=n * len(cfg["patients"]["test_seeds"]), seed=1)
    x_all = test_set.all()[0].to(setup.device)
    imax = setup.params.max_current_ua
    pw_nc = setup.params.pulse_width_s * 1e3
    out = {"dataset": dataset, "frames": frames, "frame_s": setup.params.train_s, "n_sequences": len(x_all),
           "speed_px_per_frame": tc_t["speed_px"], "modes": {}}
    for mode in ("static", "moving"):
        gen = torch.Generator().manual_seed(123)
        seq_all = make_sequences(x_all, frames, mode, tc_t["speed_px"], gen)
        res = {}
        for name, kind in (("intensity", "frame"), ("learned", "frame"), ("learned_threshold_aware", "threshold_aware"),
                           ("threshold_aware_compensated", "compensated"), ("learned_temporal", "temporal")):
            curves = {"ssim": [], "ssim_lowpass": [], "charge_nc": [], "active": [], "wasted_frac": [], "mean_brightness": []}
            per = {k: torch.zeros(frames) for k in curves}
            for pi, sd in enumerate(cfg["patients"]["test_seeds"]):
                patient = setup.patient(sd)
                cond = Condition(patient.threshold_current_ua(setup.params), patient.alive.clone())
                enc = {"intensity": baseline_factories(setup)["intensity"](patient), "learned": gen_enc,
                       "learned_threshold_aware": ta, "threshold_aware_compensated": ta, "learned_temporal": tmp}[name]
                seq = seq_all[pi * n:(pi + 1) * n]
                p, a, wst, _ = rollout(kind, enc, setup, patient, seq, cond, train=False)
                for t in range(frames):
                    per["ssim"][t] += float(ssim(p[:, t], seq[:, t]).mean()) / len(cfg["patients"]["test_seeds"])
                    per["ssim_lowpass"][t] += float(ssim_lowpass(p[:, t], seq[:, t]).mean()) / len(cfg["patients"]["test_seeds"])
                    per["charge_nc"][t] += float(a[:, t].sum(1).mean() * pw_nc) / len(cfg["patients"]["test_seeds"])
                    per["active"][t] += float((a[:, t] > 0).float().sum(1).mean()) / len(cfg["patients"]["test_seeds"])
                    act = (a[:, t] > 0).float().sum()
                    per["wasted_frac"][t] += float(wst[:, t].float().sum() / act.clamp_min(1)) / len(cfg["patients"]["test_seeds"])
                    per["mean_brightness"][t] += float(p[:, t].mean()) / len(cfg["patients"]["test_seeds"])
            r = {k: [round(float(v), 5) for v in per[k]] for k in curves}
            r["mean"] = {k: float(per[k].mean()) for k in curves}
            r["last"] = {k: float(per[k][-1]) for k in curves}
            res[name] = r
            log(f"temporal {dataset}/{mode} {name:28s} ssim mean={r['mean']['ssim']:.3f} first={r['ssim'][0]:.3f} "
                f"last={r['last']['ssim']:.3f} lp mean={r['mean']['ssim_lowpass']:.3f} charge={r['mean']['charge_nc']:.0f}nC "
                f"wasted={r['mean']['wasted_frac']:.3f}")
        out["modes"][mode] = res
    return out
