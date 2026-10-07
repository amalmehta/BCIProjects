import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
torch.set_num_threads(2)
torch.set_flush_denormal(True)

from blindsight.config import build_setup, deep_update, load_config  # noqa: E402


@pytest.fixture(scope="session")
def cfg():
    """Small configuration: 256 thread sites, 32 px renders."""
    base = load_config(ROOT / "configs" / "cpu.yaml")
    return deep_update(base, {"layout": {"name": "threads_256"}, "sim": {"resolution": 32}})


@pytest.fixture(scope="session")
def setup(cfg):
    return build_setup(cfg)


@pytest.fixture(scope="session")
def setup64(cfg):
    """Same as ``setup`` at the 64 px reference resolution (export tests)."""
    return build_setup(deep_update(cfg, {"sim": {"resolution": 64}}))
