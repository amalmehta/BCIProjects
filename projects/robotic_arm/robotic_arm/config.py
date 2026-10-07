"""Tiny YAML config loader with dotted access."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


class Config(dict):
    """A dict whose keys are also attributes (recursively)."""

    def __getattr__(self, key: str) -> Any:
        try:
            value = self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc
        return Config(value) if isinstance(value, dict) else value


def load_config(path: str | Path) -> Config:
    """Load a YAML config file into a :class:`Config`."""
    with open(path) as fh:
        return Config(yaml.safe_load(fh))
