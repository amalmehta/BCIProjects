"""Configuration loading and the shared experiment setup (simulator, layout, patients)."""
from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path

import torch
import yaml

from .encoders import EncoderContext, SafetyLayer
from .layouts import Layout, make_layout
from .simulator import Patient, PhospheneSimulator, SimParams, nominal_patient, sample_patient

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def get_device(cfg: dict) -> torch.device:
    """``device: auto`` picks CUDA when available, else CPU."""
    want = cfg.get("device", "auto")
    if want == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(want)


def load_config(path: str | Path) -> dict:
    """Read a YAML config; relative paths inside it are resolved against the project root."""
    with open(path) as f:
        cfg = yaml.safe_load(f)
    for key in ("data_root", "results_dir"):
        p = Path(cfg[key])
        cfg[key] = str(p if p.is_absolute() else (PROJECT_ROOT / p).resolve())
    return cfg


def deep_update(base: dict, override: dict) -> dict:
    """Recursively merge ``override`` into a copy of ``base``."""
    out = copy.deepcopy(base)
    for k, v in override.items():
        out[k] = deep_update(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def variant_config(cfg: dict, variant: str) -> dict:
    """Config for a named training variant (``main`` or an ablation listed under ``variants``)."""
    return cfg if variant == "main" else deep_update(cfg, cfg["variants"][variant])


@dataclass
class Setup:
    """Everything that defines one simulated implant: constants, layout, renderer, safety rules."""

    params: SimParams
    layout: Layout
    sim: PhospheneSimulator
    safety: SafetyLayer
    cfg: dict
    device: torch.device = torch.device("cpu")

    @property
    def n(self) -> int:
        return self.layout.n

    def nominal(self) -> Patient:
        return nominal_patient(self.layout, self.params).to(self.device)

    def patient(self, seed: int, extra_dropout: float = 0.0) -> Patient:
        pc = self.cfg["patients"]
        p = sample_patient(self.layout, self.params, seed, dropout=pc["dropout"], jitter_mm=pc["jitter_mm"])
        if extra_dropout > 0:
            g = torch.Generator().manual_seed(10_000 + seed)
            p.alive = p.alive * (torch.rand(self.n, generator=g) >= extra_dropout).float()
            p.name += f"_drop{extra_dropout}"
        return p.to(self.device)

    def context(self, patient: Patient | None = None, calibrated: bool = True) -> EncoderContext:
        """Encoder knowledge: nominal positions always; thresholds/dead electrodes only if calibrated."""
        nom = self.nominal()
        if patient is not None and calibrated:
            thr, alive = patient.threshold_current_ua(self.params), patient.alive.clone()
        else:
            thr, alive = nom.threshold_current_ua(self.params), torch.ones(self.n, device=self.device)
        return EncoderContext(nom.x_deg, nom.y_deg, self.params.fov_deg, self.params.resolution, thr, alive,
                              self.params.max_current_ua)


def build_setup(cfg: dict) -> Setup:
    """Instantiate simulator constants, electrode layout and safety layer from a config dict."""
    s = cfg["sim"]
    params = SimParams(fov_deg=s["fov_deg"], resolution=s["resolution"], max_current_ua=s["max_current_ua"],
                       sigma_floor_deg=s["sigma_floor_deg"])
    lc = cfg["layout"]
    layout = make_layout(lc["name"], lc["ecc_min"], lc["ecc_max"], lc.get("seed", 0))
    frac = cfg["safety"]["max_active_frac"]
    max_active = None if frac is None else max(1, int(round(frac * layout.n)))
    device = get_device(cfg)
    return Setup(params, layout, PhospheneSimulator(params).to(device), SafetyLayer(params.max_current_ua, max_active),
                 cfg, device)
