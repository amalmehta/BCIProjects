"""Evaluation of stimulus encoders on held-out simulated patients.

For each encoder we render percepts with the exact simulator for every test patient and
report:

* ``ssim`` / ``psnr`` between percept and input image (both 64x64 in [0, 1]);
* ``acc``: accuracy of a CNN trained on that encoder's percepts (train-split images)
  and tested on held-out images - a proxy for what a user could learn to recognise;
* stimulation cost: active electrodes per frame and total charge per phase per frame
  (sum over electrodes of I * pulse width, in nC);
* robustness: SSIM and accuracy after extra, unannounced electrode failures (the
  classifier and encoder are not retrained).
"""
from __future__ import annotations

import copy
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F

from .config import Setup, get_device
from .data import N_CLASSES, ImageSet
from .encoders import (EdgeEncoder, IntensityEncoder, LearnedEncoder, PerceptClassifier,
                       SalientEdgeEncoder)
from .metrics import psnr, ssim, ssim_lowpass
from .simulator import Patient
from .train import end_to_end_loss, percept_loss

EncoderFactory = Callable[[Patient], torch.nn.Module]


def baseline_factories(setup: Setup) -> dict[str, EncoderFactory]:
    """The non-learned encoders, constructed per patient (calibrated ones read its thresholds)."""
    return {
        "intensity_uncalibrated": lambda p: IntensityEncoder(setup.context(p, calibrated=False)),
        "intensity": lambda p: IntensityEncoder(setup.context(p)),
        "edge": lambda p: EdgeEncoder(setup.context(p)),
        "salient_edge": lambda p: SalientEdgeEncoder(setup.context(p)),
    }


@torch.no_grad()
def render_all(factory: EncoderFactory, setup: Setup, patients: list[Patient], x: torch.Tensor,
               assign: np.ndarray, batch: int = 256) -> tuple[torch.Tensor, torch.Tensor]:
    """Percepts (and amplitudes) for images ``x`` where image i is shown to patient ``assign[i]``."""
    percepts = torch.zeros_like(x)
    amps = torch.zeros(len(x), setup.n, device=x.device)
    for pi, patient in enumerate(patients):
        enc = factory(patient)
        sel = np.where(assign == pi)[0]
        for i in range(0, len(sel), batch):
            idx = torch.as_tensor(sel[i:i + batch], device=x.device)
            a = setup.safety(enc(x[idx])) * patient.alive
            amps[idx] = a
            percepts[idx] = setup.sim(a, patient)
    return percepts, amps


def train_classifier(p_train: torch.Tensor, y_train: torch.Tensor, n_classes: int, epochs: int, batch: int,
                     seed: int = 0) -> PerceptClassifier:
    """Fit the percept classifier (Adam, light shift augmentation)."""
    torch.manual_seed(seed)
    clf = PerceptClassifier(n_classes).to(p_train.device)
    opt = torch.optim.Adam(clf.parameters(), lr=2e-3)
    steps = epochs * (len(y_train) // batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=steps)
    g = torch.Generator().manual_seed(seed)
    clf.train()
    for _ in range(epochs):
        perm = torch.randperm(len(y_train), generator=g)
        for i in range(0, len(perm) - batch + 1, batch):
            idx = perm[i:i + batch]
            xb = p_train[idx.to(p_train.device)]
            dx, dy = torch.randint(-2, 3, (2,), generator=g).tolist()
            xb = torch.roll(xb, shifts=(dy, dx), dims=(2, 3))
            loss = F.cross_entropy(clf(xb), y_train[idx.to(y_train.device)])
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
    return clf.eval()


@torch.no_grad()
def accuracy(clf: PerceptClassifier, p: torch.Tensor, y: torch.Tensor) -> float:
    pred = torch.cat([clf(p[i:i + 512]).argmax(1) for i in range(0, len(p), 512)])
    return float((pred == y).float().mean())


def cost_stats(amps: torch.Tensor, setup: Setup) -> dict:
    active = (amps > 0).float().sum(1)
    charge_nc = amps.sum(1) * setup.params.pulse_width_s * 1e3  # uA * s -> nC per phase, summed
    return {"active_electrodes": float(active.mean()), "charge_nc_per_phase": float(charge_nc.mean()),
            "max_amp_ua": float(amps.max())}


def evaluate_factory(name: str, factory: EncoderFactory, setup: Setup, dataset: str, train_set: ImageSet,
                     test_set: ImageSet, cfg: dict, log=print, dropout_levels: list[float] | None = None,
                     decoder: torch.nn.Module | None = None, keep_examples: int = 0) -> dict:
    """Full evaluation of one encoder over the test patients."""
    ec = cfg["eval"]
    seeds = cfg["patients"]["test_seeds"]
    patients = [setup.patient(s) for s in seeds]
    dev = setup.device
    x_te, y_te = (t.to(dev) for t in test_set.all())
    x_tr, y_tr = (t.to(dev) for t in train_set.all())
    a_te = np.arange(len(x_te)) % len(patients)
    a_tr = np.arange(len(x_tr)) % len(patients)
    p_te, amp_te = render_all(factory, setup, patients, x_te, a_te)
    p_tr, _ = render_all(factory, setup, patients, x_tr, a_tr)
    s, q = ssim(p_te, x_te), psnr(p_te, x_te)
    per_patient = [float(s[torch.as_tensor(a_te == i, device=dev)].mean()) for i in range(len(patients))]
    clf = train_classifier(p_tr, y_tr, N_CLASSES[dataset], ec["clf_epochs"], ec["clf_batch"])
    res = {"encoder": name, "ssim": float(s.mean()), "ssim_patient_std": float(np.std(per_patient)),
           "ssim_lowpass": float(ssim_lowpass(p_te, x_te).mean()), "psnr": float(q.mean()), "acc": accuracy(clf, p_te, y_te), **cost_stats(amp_te, setup)}
    if decoder is not None:
        with torch.no_grad():
            rec = torch.cat([decoder(p_te[i:i + 512]) for i in range(0, len(p_te), 512)])
        res["decoder_ssim"] = float(ssim(rec, x_te).mean())
    if dropout_levels:
        res["dropout"] = {}
        for lvl in dropout_levels:
            dp = [setup.patient(sd, extra_dropout=lvl) for sd in seeds]
            # Encoders keep the calibration of the intact implant: failures are unannounced.
            pd, _ = render_all(lambda p, _m={id(a): b for a, b in zip(dp, patients)}: factory(_m[id(p)]),
                               setup, dp, x_te, a_te)
            res["dropout"][str(lvl)] = {"ssim": float(ssim(pd, x_te).mean()),
                                        "ssim_lowpass": float(ssim_lowpass(pd, x_te).mean()),
                                        "acc": accuracy(clf, pd, y_te)}
    if keep_examples:
        res["_examples"] = p_te[:keep_examples].cpu().clone()
    log(f"  {dataset:8s} {name:24s} ssim={res['ssim']:.3f} lp={res['ssim_lowpass']:.3f} psnr={res['psnr']:.2f} acc={res['acc']:.3f} "
        f"active={res['active_electrodes']:.0f} charge={res['charge_nc_per_phase']:.0f}nC")
    return res


def image_ceiling(dataset: str, train_set: ImageSet, test_set: ImageSet, cfg: dict) -> float:
    """Accuracy of the same classifier on the original images (upper reference)."""
    dev = get_device(cfg)
    x_tr, y_tr = (t.to(dev) for t in train_set.all())
    x_te, y_te = (t.to(dev) for t in test_set.all())
    clf = train_classifier(x_tr, y_tr, N_CLASSES[dataset], cfg["eval"]["clf_epochs"], cfg["eval"]["clf_batch"])
    return accuracy(clf, x_te, y_te)


# --- patient-specific adaptation --------------------------------------------------------

@torch.enable_grad()
def adapt_encoder(enc: LearnedEncoder, dec: torch.nn.Module, setup: Setup, patient: Patient, images: torch.Tensor,
                  cfg: dict, mode: str = "adapter", seed: int = 0) -> LearnedEncoder:
    """Fine-tune a copy of a generic encoder to one patient's calibrated forward model.

    Uses only unlabeled images (``images``, the "shots"). ``mode='adapter'`` trains the
    per-electrode gain and bias only (2N parameters); ``mode='full'`` trains everything.
    The decoder is frozen.
    """
    ac, tc = cfg["adapt"], cfg["train"]
    enc = copy.deepcopy(enc).train()
    for p in enc.parameters():
        p.requires_grad_(mode == "full")
    enc.gain.requires_grad_(True)
    enc.bias.requires_grad_(True)
    for p in dec.parameters():
        p.requires_grad_(False)
    params = [p for p in enc.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=ac["lr"] if mode == "adapter" else ac["lr"] * 0.1)
    g = torch.Generator().manual_seed(seed)
    for _ in range(ac["steps"]):
        idx = torch.randint(len(images), (min(ac["batch"], len(images)),), generator=g)
        loss, _ = end_to_end_loss(enc, dec, setup, patient, images[idx], tc, train=True)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return enc.eval()


# --- per-image optimisation (non-causal reference) --------------------------------------

@torch.enable_grad()
def oracle_amplitudes(setup: Setup, patient: Patient, x: torch.Tensor, cfg: dict) -> torch.Tensor:
    """Optimise amplitudes for each image directly through the patient's simulator (slow, non-causal)."""
    tc, steps = cfg["train"], cfg["eval"]["oracle_steps"]
    logits = torch.zeros(len(x), setup.n, device=x.device, requires_grad=True)
    opt = torch.optim.Adam([logits], lr=0.1)
    imax = setup.params.max_current_ua
    for _ in range(steps):
        amp = setup.safety(imax * torch.sigmoid(logits)) * patient.alive
        p = setup.sim(amp, patient, hard=False, fast=True)
        loss = percept_loss(p, x, tc["w_ssim"]) + tc["w_charge"] * (amp / imax).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        return setup.safety(imax * torch.sigmoid(logits)) * patient.alive


def evaluate_oracle(setup: Setup, test_set: ImageSet, cfg: dict, log=print) -> dict:
    """SSIM/PSNR of per-image optimised stimulation on the first ``oracle_images`` test images per patient."""
    n = cfg["eval"]["oracle_images"]
    ss, qs, ls, amps = [], [], [], []
    for sd in cfg["patients"]["test_seeds"]:
        patient = setup.patient(sd)
        x = test_set.batch(torch.arange(n))[0].to(setup.device)
        a = oracle_amplitudes(setup, patient, x, cfg)
        with torch.no_grad():
            p = setup.sim(a, patient)
        ss.append(ssim(p, x)); qs.append(psnr(p, x)); ls.append(ssim_lowpass(p, x)); amps.append(a)
    s = torch.cat(ss)
    res = {"encoder": "per_image_optimisation", "ssim": float(s.mean()),
           "ssim_patient_std": float(np.std([float(v.mean()) for v in ss])),
           "ssim_lowpass": float(torch.cat(ls).mean()), "psnr": float(torch.cat(qs).mean()),
           "acc": None, "n_images": n * len(ss), **cost_stats(torch.cat(amps), setup)}
    log(f"  oracle ssim={res['ssim']:.3f} psnr={res['psnr']:.2f}")
    return res
