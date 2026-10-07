"""Image datasets used to evaluate stimulus encoders.

All images are converted to single-channel uint8 tensors at native resolution and
cached as ``<data_root>/<name>.pt``; batches are resized to the simulator's field of
view on the fly (the full image spans the rendered field of view).

* ``mnist``   - handwritten digits (10 classes)
* ``letters`` - EMNIST letters (26 classes; Cohen et al. 2017)
* ``cifar``   - CIFAR-10 converted to grayscale, a small natural-image proxy (10 classes)
"""
from __future__ import annotations

import os
import ssl
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

N_CLASSES = {"mnist": 10, "letters": 26, "cifar": 10}


def _use_certifi() -> None:
    """Some Python builds lack system CA certificates; point urllib at certifi's bundle."""
    try:
        import certifi

        os.environ.setdefault("SSL_CERT_FILE", certifi.where())
        ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=certifi.where())
    except ImportError:  # pragma: no cover - certifi ships with pip/requests
        pass


def prepare(name: str, data_root: str | Path) -> Path:
    """Download (if needed) and cache dataset ``name`` as uint8 tensors. Returns the cache path."""
    import torchvision as tv

    root = Path(data_root)
    out = root / f"{name}.pt"
    if out.exists():
        return out
    _use_certifi()
    root.mkdir(parents=True, exist_ok=True)
    splits = {}
    for train in (True, False):
        if name == "mnist":
            ds = tv.datasets.MNIST(root, train=train, download=True)
            x, y = ds.data.clone(), ds.targets.clone()
        elif name == "letters":
            ds = tv.datasets.EMNIST(root, split="letters", train=train, download=True)
            x, y = ds.data.transpose(1, 2).clone(), ds.targets.clone() - 1  # EMNIST is stored transposed; labels 1..26
        elif name == "cifar":
            ds = tv.datasets.CIFAR10(root, train=train, download=True)
            rgb = torch.as_tensor(ds.data).float()
            x = (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]).round().clamp(0, 255).to(torch.uint8)
            y = torch.as_tensor(ds.targets)
        else:
            raise ValueError(name)
        splits["train" if train else "test"] = (x.to(torch.uint8), y.long())
    torch.save(splits, out)
    return out


@dataclass
class ImageSet:
    """In-memory uint8 images with labels; ``batch`` returns float images resized to ``res``."""

    images: torch.Tensor  # (N, h, w) uint8
    labels: torch.Tensor
    res: int

    def __len__(self) -> int:
        return len(self.labels)

    def batch(self, idx: torch.Tensor | np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        idx = torch.as_tensor(idx)
        x = self.images[idx].float().div(255.0).unsqueeze(1)
        x = F.interpolate(x, size=(self.res, self.res), mode="bilinear", align_corners=False)
        return x.clamp(0, 1), self.labels[idx]

    def all(self, chunk: int = 4096) -> tuple[torch.Tensor, torch.Tensor]:
        xs = [self.batch(torch.arange(i, min(i + chunk, len(self)))) [0] for i in range(0, len(self), chunk)]
        return torch.cat(xs), self.labels.clone()


def load(name: str, data_root: str | Path, res: int, n_train: int | None = None, n_test: int | None = None,
         seed: int = 0) -> tuple[ImageSet, ImageSet]:
    """Load a cached dataset, optionally taking fixed random subsets of each split."""
    splits = torch.load(prepare(name, data_root))
    g = torch.Generator().manual_seed(seed)
    out = []
    for split, n in (("train", n_train), ("test", n_test)):
        x, y = splits[split]
        if n is not None and n < len(y):
            sel = torch.randperm(len(y), generator=g)[:n]
            x, y = x[sel], y[sel]
        out.append(ImageSet(x, y, res))
    return out[0], out[1]
