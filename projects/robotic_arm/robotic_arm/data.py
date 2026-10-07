"""FALCON H1 data loading, binning, caching and splits.

FALCON H1 (DANDI 000954) contains open-loop calibration blocks from one person with
tetraplegia in the Pitt BCI study. Each block has sorted/thresholded spike times on 176
channels and 7 kinematic degrees of freedom (``tx, ty, tz`` translation, ``rx`` wrist
rotation, ``g1..g3`` grasp shape) at 20 ms resolution. The participant *attempts* to
follow a virtual arm, so the covariates are the cued (intended) velocities.

Splits used here (FALCON's public files):

* ``held_in_calib``  - 13 blocks over 6 days: decoder training data.
* ``held_in_minival`` - FALCON's local "minival" split of the same 13 blocks, used as the
  within-session test set. **The public calibration files contain the minival bins**
  (minival = the first ``len(minival)`` bins of the matching calibration file, verified
  bin-for-bin), so :func:`load_blocks` strips them from the calibration blocks to avoid
  train/test leakage.
* ``held_out_calib`` - 14 blocks over 7 later days (6-20 days after the last training
  day). FALCON keeps the held-out *evaluation* blocks private, so we build our own
  cross-session protocol: per day, the earlier block is the (optionally labelled)
  recalibration block and the later block is the test block.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import numpy as np

from . import DATA_ROOT

BIN_S = 0.02
DANDISET = "000954"
SPLITS = ("held_in_calib", "held_in_minival", "held_out_calib")


@dataclass
class Block:
    """One recording block (a single NWB file) binned at 20 ms."""

    split: str
    day: date
    time: str
    spikes: np.ndarray  # (T, C) int counts
    vel: np.ndarray  # (T, 7) float cued velocity
    pos: np.ndarray  # (T, 7) float cued position
    eval_mask: np.ndarray  # (T,) bool, True where covariates are valid
    trial: np.ndarray  # (T,) trial number
    meta: dict = field(default_factory=dict)

    @property
    def key(self) -> str:
        return f"{self.day.isoformat()}T{self.time}"

    @property
    def n_bins(self) -> int:
        return self.spikes.shape[0]

    def slice(self, start: int, stop: int) -> "Block":
        """Return a time slice of the block (bins ``start:stop``)."""
        s = slice(start, stop)
        return Block(self.split, self.day, self.time, self.spikes[s], self.vel[s],
                     self.pos[s], self.eval_mask[s], self.trial[s], dict(self.meta))


def bin_spike_times(spike_times: list[np.ndarray], bin_end_times: np.ndarray,
                    bin_s: float = BIN_S) -> np.ndarray:
    """Bin spike times into counts; bin ``i`` covers ``(end_i - bin_s, end_i]``.

    Mirrors ``falcon_challenge.dataloaders.bin_units`` for evenly spaced timestamps.
    """
    edges = np.concatenate([[bin_end_times[0] - bin_s], bin_end_times])
    out = np.zeros((len(bin_end_times), len(spike_times)), dtype=np.uint8)
    for i, st in enumerate(spike_times):
        out[:, i] = np.histogram(st, bins=edges)[0]
    return out


def _parse_name(path: Path) -> tuple[str, date, str]:
    m = re.search(r"sub-HumanPitt-(held-in-calib|held-in-minival|held-out-calib)_ses-(\d{8})T(\d{6})",
                  path.name)
    if m is None:
        raise ValueError(f"Unexpected FALCON H1 filename: {path.name}")
    split = m.group(1).replace("-", "_")
    d = m.group(2)
    return split, date(int(d[:4]), int(d[4:6]), int(d[6:])), m.group(3)


def load_nwb_block(path: Path) -> Block:
    """Read one FALCON H1 NWB file and bin it (FALCON's own binning convention)."""
    from pynwb import NWBHDF5IO

    split, day, time = _parse_name(path)
    with NWBHDF5IO(str(path), "r") as io:
        nwb = io.read()
        kin = nwb.acquisition["OpenLoopKinematics"]
        vel = np.asarray(nwb.acquisition["OpenLoopKinematicsVelocity"].data[:], dtype=np.float64)
        pos = np.asarray(kin.data[:], dtype=np.float64)
        eval_mask = np.asarray(nwb.acquisition["eval_mask"].data[:]).astype(bool)
        trial = np.asarray(nwb.acquisition["TrialNum"].data[:])
        # FALCON: timestamps = offset + arange(T) * rate, where "rate" stores the bin width.
        ts = kin.offset + np.arange(vel.shape[0]) * kin.rate
        units = nwb.units.to_dataframe()
        spikes = bin_spike_times([np.asarray(s) for s in units.spike_times], ts)
    return Block(split, day, time, spikes, vel, pos, eval_mask, trial)


def cache_path() -> Path:
    return DATA_ROOT / "h1_binned.npz"


def build_cache(raw_dir: Path | None = None, out: Path | None = None) -> Path:
    """Bin every NWB file once and store all blocks in a single compressed npz."""
    raw_dir = raw_dir or DATA_ROOT / DANDISET
    out = out or cache_path()
    files = sorted(raw_dir.rglob("*.nwb"))
    if not files:
        raise FileNotFoundError(f"No NWB files under {raw_dir}; run scripts/download_data.sh")
    arrays: dict[str, np.ndarray] = {}
    for i, f in enumerate(files):
        b = load_nwb_block(f)
        tag = f"{b.split}|{b.key}"
        for name in ("spikes", "vel", "pos", "eval_mask", "trial"):
            arrays[f"{tag}|{name}"] = getattr(b, name)
    np.savez_compressed(out, **arrays)
    return out


def load_blocks(path: Path | None = None, strip_minival: bool = True) -> dict[str, list[Block]]:
    """Load cached blocks grouped by split, each list sorted by time.

    With ``strip_minival`` (default) the minival bins are removed from the held-in
    calibration blocks, so that minival is a clean within-session test set.
    """
    path = path or cache_path()
    if not path.exists():
        build_cache(out=path)
    data = np.load(path)
    grouped: dict[tuple[str, str], dict[str, np.ndarray]] = {}
    for k in data.files:
        split, key, name = k.split("|")
        grouped.setdefault((split, key), {})[name] = data[k]
    out: dict[str, list[Block]] = {s: [] for s in SPLITS}
    for (split, key), arr in grouped.items():
        d, t = key.split("T")
        y, m, dd = (int(x) for x in d.split("-"))
        out[split].append(Block(split, date(y, m, dd), t, arr["spikes"].astype(np.float32),
                                arr["vel"], arr["pos"], arr["eval_mask"].astype(bool), arr["trial"]))
    for s in out:
        out[s].sort(key=lambda b: b.key)
    if strip_minival:
        out["held_in_calib"] = _strip_minival(out["held_in_calib"], out["held_in_minival"])
    return out


def _strip_minival(calib: list[Block], minival: list[Block]) -> list[Block]:
    """Remove the minival bins (the leading segment of each calibration file)."""
    mv = {b.key: b for b in minival}
    out = []
    for c in calib:
        m = mv.get(c.key)
        if m is None:
            out.append(c)
            continue
        n = m.n_bins
        if not (np.array_equal(c.spikes[:n], m.spikes) and np.allclose(c.vel[:n], m.vel)):
            raise ValueError(f"minival for {c.key} is not the leading segment of its calibration file")
        out.append(c.slice(n, c.n_bins))
    return out


def held_out_pairs(blocks: list[Block]) -> list[tuple[Block, Block]]:
    """Group held-out blocks by day into (recalibration block, test block) pairs."""
    by_day: dict[date, list[Block]] = {}
    for b in blocks:
        by_day.setdefault(b.day, []).append(b)
    pairs = []
    for d in sorted(by_day):
        bs = sorted(by_day[d], key=lambda b: b.time)
        if len(bs) >= 2:
            pairs.append((bs[0], bs[-1]))
    return pairs


def causal_smooth(x: np.ndarray, tau_bins: float) -> np.ndarray:
    """Causal exponential smoothing along time (axis 0)."""
    if tau_bins <= 0:
        return x.astype(np.float64)
    a = np.exp(-1.0 / tau_bins)
    from scipy.signal import lfilter

    return lfilter([1 - a], [1, -a], x, axis=0)


def lag_features(x: np.ndarray, n_lags: int, stride: int = 1) -> np.ndarray:
    """Stack ``n_lags`` causal copies of ``x`` (lags 0, stride, 2*stride, ...) along features."""
    T, C = x.shape
    out = np.zeros((T, C * n_lags), dtype=x.dtype)
    for k in range(n_lags):
        lag = k * stride
        out[lag:, k * C:(k + 1) * C] = x[: T - lag]
    return out


@dataclass
class Normalizer:
    """Per-feature z-scoring with statistics from a reference set."""

    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, x: np.ndarray, eps: float = 1e-6) -> "Normalizer":
        return cls(x.mean(0), x.std(0) + eps)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / self.std

    def inverse(self, z: np.ndarray) -> np.ndarray:
        return z * self.std + self.mean
