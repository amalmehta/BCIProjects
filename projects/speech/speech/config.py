"""YAML config loading with per-run overrides."""

from __future__ import annotations

import copy
from pathlib import Path

import torch
import yaml

PROJECT_DIR = Path(__file__).resolve().parents[1]


def deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge ``override`` into a copy of ``base``."""
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(path: str | Path) -> dict:
    """Load a YAML config (optionally inheriting from ``base:``) and resolve relative paths."""
    with open(path) as f:
        cfg = yaml.safe_load(f)
    if "base" in cfg:  # inherit from another config file, overriding selected keys
        with open(Path(path).parent / cfg.pop("base")) as f:
            cfg = deep_merge(yaml.safe_load(f), cfg)
    for key in ("data_dir", "results_dir"):
        p = Path(cfg[key])
        cfg[key] = str(p if p.is_absolute() else (PROJECT_DIR / p).resolve())
    return cfg


def run_config(cfg: dict, run: str) -> dict:
    """Config for a named run: the global defaults merged with the run's overrides.

    The run's ``model`` key names an entry of ``cfg['models']``.
    """
    spec = cfg["runs"][run]
    merged = deep_merge({k: cfg[k] for k in ("data", "train", "aug")}, {k: v for k, v in spec.items() if k != "model"})
    merged["model"] = deep_merge(cfg["models"][spec["model"]], spec.get("model_overrides", {}))
    merged["model_name"] = spec["model"]
    merged.pop("model_overrides", None)
    merged["name"] = run
    return merged


def lm_paths(cfg: dict) -> tuple[Path, Path]:
    """(resource directory with CMUdict/corpora, directory of the built LM for this config)."""
    lm_dir = Path(cfg["data_dir"]) / "lm"
    return lm_dir, lm_dir / cfg.get("lm", {}).get("build_name", "built")


def get_device(pref: str = "auto") -> torch.device:
    """CUDA when available (or requested), else CPU."""
    if pref == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(pref)
