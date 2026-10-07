"""End-to-end training of the learned stimulus encoder through the differentiable simulator.

Loss (per batch, x = input image, a = SafetyLayer(E(x)), P = S(a; patient), x_hat = D(P)):

    L = w_recon * MSE(x_hat, x)                         # information preserved in the percept
      + w_percept * [MSE(P, x) + w_ssim * (1 - SSIM(P, x))]   # percept itself resembles the scene
      + w_charge * mean_n(a_n / I_max)                  # stimulation-charge (safety/power) cost

Patients are drawn from a training pool (domain randomisation over dead electrodes,
thresholds and positional jitter) and switched every ``switch_every`` steps.
"""
from __future__ import annotations

import contextlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .config import Setup, build_setup, variant_config
from .data import load
from .encoders import Decoder, LearnedEncoder
from .metrics import ssim
from .simulator import Patient


def percept_loss(p: torch.Tensor, x: torch.Tensor, w_ssim: float) -> torch.Tensor:
    return F.mse_loss(p, x) + w_ssim * (1 - ssim(p, x).mean())


def _autocast(device: torch.device):
    """bfloat16 autocast for the CNNs on CUDA; the simulator always runs in float32."""
    if device.type == "cuda":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def end_to_end_loss(enc, dec, setup: Setup, patient: Patient, x: torch.Tensor, w: dict, train: bool = True):
    """Compute the total loss and its parts for one batch."""
    with _autocast(x.device):
        u = enc(x)
    amp = setup.safety(u.float())
    p = setup.sim(amp, patient, hard=not train, fast=train)
    if w["w_recon"] > 0:
        with _autocast(x.device):
            rec = dec(p)
        recon = F.mse_loss(rec.float(), x)
    else:
        recon = torch.zeros((), device=x.device)
    parts = {
        "recon": recon,
        "percept": percept_loss(p, x, w["w_ssim"]),
        "charge": (amp / setup.params.max_current_ua).mean(),
    }
    total = w["w_recon"] * parts["recon"] + w["w_percept"] * parts["percept"] + w["w_charge"] * parts["charge"]
    return total, parts


def train_encoder(cfg: dict, dataset: str, seed: int, variant: str, out_dir: Path, log=print) -> dict:
    """Train one encoder (+ decoder); saves ``encoder.pt``, ``decoder.pt`` and ``train_log.json``."""
    cfg = variant_config(cfg, variant)
    torch.manual_seed(seed)
    np.random.seed(seed)
    setup = build_setup(cfg)
    tc, pc = cfg["train"], cfg["patients"]
    train_set, _ = load(dataset, cfg["data_root"], setup.params.resolution, n_train=tc["n_train"], n_test=1, seed=0)
    pool = [setup.patient(pc["train_seed0"] + i) for i in range(pc["train_pool"])]
    enc = LearnedEncoder(setup.context(calibrated=False), width=tc["width"]).to(setup.device)
    dec = Decoder(width=tc["width"]).to(setup.device)
    params = list(enc.parameters()) + list(dec.parameters())
    opt = torch.optim.Adam(params, lr=tc["lr"])
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 0.5 * (1 + math.cos(math.pi * s / tc["steps"])))
    rng = np.random.default_rng(seed)
    history, t0 = [], time.time()
    patient = pool[0]
    for step in range(tc["steps"]):
        if step % pc["switch_every"] == 0:
            patient = pool[rng.integers(len(pool))]
        x, _ = train_set.batch(rng.integers(len(train_set), size=tc["batch"]))
        x = x.to(setup.device)
        loss, parts = end_to_end_loss(enc, dec, setup, patient, x, tc, train=True)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        if step % tc["log_every"] == 0 or step == tc["steps"] - 1:
            rec = {"step": step, "loss": float(loss), **{k: float(v) for k, v in parts.items()},
                   "elapsed_s": round(time.time() - t0, 1)}
            history.append(rec)
            log(f"[{dataset}/{variant}/s{seed}] " + " ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in rec.items()))
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(enc.state_dict(), out_dir / "encoder.pt")
    torch.save(dec.state_dict(), out_dir / "decoder.pt")
    summary = {"dataset": dataset, "seed": seed, "variant": variant, "steps": tc["steps"],
               "train_time_s": round(time.time() - t0, 1), "n_params_encoder": sum(p.numel() for p in enc.parameters()),
               "history": history}
    (out_dir / "train_log.json").write_text(json.dumps(summary, indent=1))
    return summary


def load_encoder(cfg: dict, variant: str, run_dir: Path) -> tuple[LearnedEncoder, Decoder, Setup]:
    """Rebuild a trained encoder/decoder pair and the setup it was trained for."""
    vcfg = variant_config(cfg, variant)
    setup = build_setup(vcfg)
    enc = LearnedEncoder(setup.context(calibrated=False), width=vcfg["train"]["width"]).to(setup.device)
    dec = Decoder(width=vcfg["train"]["width"]).to(setup.device)
    enc.load_state_dict(torch.load(run_dir / "encoder.pt", map_location=setup.device))
    dec.load_state_dict(torch.load(run_dir / "decoder.pt", map_location=setup.device))
    return enc.eval(), dec.eval(), setup
