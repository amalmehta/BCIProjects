"""Data download, preprocessing and loading for the O'Doherty et al. (2017) random-target
cursor reaching dataset (Zenodo record 583331, monkey Indy).

Each raw session is a MATLAB v7.3 (HDF5) file holding sorted spike times per channel,
cursor and target positions sampled at 250 Hz. We convert it into a compact ``.npz``:

* ``counts``  (T, C) uint8   - multi-unit threshold-crossing counts per M1 channel per bin
                               (all sorted units and the "hash" unit of a channel are merged,
                               mimicking a sorting-free, per-electrode decoder as used clinically)
* ``pos``     (T, 2) float32 - cursor position (mm) at the end of each bin
* ``vel``     (T, 2) float32 - cursor velocity (mm/s), central difference of smoothed position
* ``target``  (T, 2) float32 - current target position (mm)
* ``bin_ms``  scalar

After conversion the raw ``.mat`` (which also stores large waveform snippets) is deleted so
that the on-disk footprint stays small (a few MB per session).
"""
from __future__ import annotations

import dataclasses
import json
import os
import shutil
import subprocess
import urllib.request
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

ZENODO_URL = "https://zenodo.org/records/583331/files/{name}.mat?download=1"

# Sessions from monkey Indy spanning ~4 months (Sep 2016 - Jan 2017), all with the
# same 96-channel M1 Utah array. Chosen because their raw files are small (<210 MB each).
DEFAULT_SESSIONS: tuple[str, ...] = (
    "indy_20160915_01", "indy_20160916_01", "indy_20160921_01", "indy_20160927_04",
    "indy_20160930_02", "indy_20161005_06", "indy_20161006_02", "indy_20161007_02",
    "indy_20161011_03", "indy_20161013_03", "indy_20161014_04", "indy_20161017_02",
    "indy_20161024_03", "indy_20161025_04", "indy_20161026_03", "indy_20161027_03",
    "indy_20161206_02", "indy_20161207_02", "indy_20161212_02", "indy_20161220_02",
    "indy_20170123_02", "indy_20170124_01", "indy_20170127_03", "indy_20170131_02",
)


def session_day(name: str) -> np.datetime64:
    """Return the recording date of a session name like ``indy_20161005_06``."""
    d = name.split("_")[1]
    return np.datetime64(f"{d[:4]}-{d[4:6]}-{d[6:]}")


def _read_str(f, ref) -> str:
    return "".join(chr(int(c)) for c in f[ref][:].ravel())


def convert_mat(mat_path: Path, out_path: Path, bin_ms: float = 20.0, smooth_ms: float = 40.0) -> dict:
    """Convert one raw ``.mat`` session to a binned ``.npz`` (see module docstring)."""
    import h5py
    from scipy.ndimage import gaussian_filter1d

    with h5py.File(mat_path, "r") as f:
        t = f["t"][:].ravel()
        cursor = f["cursor_pos"][:].T  # (k, 2)
        target = f["target_pos"][:].T
        names = [_read_str(f, f["chan_names"][0, i]) for i in range(f["chan_names"].shape[1])]
        m1 = [i for i, n in enumerate(names) if n.startswith("M1")]
        spikes = f["spikes"]
        bw = bin_ms / 1000.0
        edges = np.arange(t[0], t[-1] + 1e-9, bw)
        counts = np.zeros((len(edges) - 1, len(m1)), dtype=np.int32)
        for j, ch in enumerate(m1):
            for u in range(spikes.shape[0]):
                arr = f[spikes[u, ch]][:]
                if arr.ndim != 2 or arr.size == 0 or arr.shape[0] != 1:
                    continue
                st = arr.ravel()
                counts[:, j] += np.histogram(st, edges)[0]
    # Behaviour at bin ends.
    tb = edges[1:]
    fs = 1.0 / np.median(np.diff(t))
    cs = gaussian_filter1d(cursor, sigma=smooth_ms / 1000.0 * fs, axis=0)
    vel_full = np.gradient(cs, t, axis=0)
    pos = np.stack([np.interp(tb, t, cursor[:, k]) for k in range(2)], 1)
    vel = np.stack([np.interp(tb, t, vel_full[:, k]) for k in range(2)], 1)
    tgt = np.stack([np.interp(tb, t, target[:, k]) for k in range(2)], 1)
    np.savez_compressed(
        out_path,
        counts=np.clip(counts, 0, 255).astype(np.uint8),
        pos=pos.astype(np.float32),
        vel=vel.astype(np.float32),
        target=tgt.astype(np.float32),
        bin_ms=np.float32(bin_ms),
    )
    return {"bins": int(len(tb)), "channels": len(m1), "minutes": float(len(tb) * bw / 60)}


def _fetch(url: str, dest: Path) -> None:
    """Download ``url`` to ``dest`` with curl (robust to local CA-bundle issues), else urllib."""
    if shutil.which("curl"):
        subprocess.run(["curl", "-sSL", "--retry", "5", "-o", str(dest), url], check=True)
    else:
        urllib.request.urlretrieve(url, dest)


def download_sessions(
    data_dir: Path, sessions: Sequence[str] = DEFAULT_SESSIONS, bin_ms: float = 20.0, keep_raw: bool = False
) -> None:
    """Download each session from Zenodo, convert it, and delete the raw file.

    Already converted sessions are skipped, so the call is resumable.
    """
    data_dir = Path(data_dir)
    (data_dir / "raw").mkdir(parents=True, exist_ok=True)
    (data_dir / "binned").mkdir(parents=True, exist_ok=True)
    manifest_path = data_dir / "binned" / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    for name in sessions:
        out = data_dir / "binned" / f"{name}.npz"
        if out.exists():
            continue
        raw = data_dir / "raw" / f"{name}.mat"
        if not raw.exists():
            print(f"downloading {name} ...", flush=True)
            tmp = raw.with_suffix(".part")
            _fetch(ZENODO_URL.format(name=name), tmp)
            os.replace(tmp, raw)
        info = convert_mat(raw, out, bin_ms=bin_ms)
        manifest[name] = info
        manifest_path.write_text(json.dumps(manifest, indent=1))
        print(f"converted {name}: {info}", flush=True)
        if not keep_raw:
            raw.unlink()


@dataclasses.dataclass
class Session:
    """One binned recording day."""

    name: str
    counts: np.ndarray  # (T, C) float32 spike counts
    pos: np.ndarray  # (T, 2)
    vel: np.ndarray  # (T, 2)
    target: np.ndarray  # (T, 2)
    bin_ms: float

    @property
    def day(self) -> np.datetime64:
        return session_day(self.name)

    def target_onsets(self) -> np.ndarray:
        """Bin indices where a new target appears (i.e. trial starts)."""
        change = np.any(np.abs(np.diff(self.target, axis=0)) > 1e-6, axis=1)
        return np.concatenate([[0], np.nonzero(change)[0] + 1])

    def trials(self, min_len: int = 10) -> list[tuple[int, int]]:
        """(start, end) bin ranges of individual reaches, one per target."""
        on = list(self.target_onsets()) + [len(self.counts)]
        return [(a, b) for a, b in zip(on[:-1], on[1:]) if b - a >= min_len]


def load_session(path: Path) -> Session:
    """Load a binned ``.npz`` produced by :func:`convert_mat`."""
    z = np.load(path)
    return Session(
        name=Path(path).stem,
        counts=z["counts"].astype(np.float32),
        pos=z["pos"],
        vel=z["vel"],
        target=z["target"],
        bin_ms=float(z["bin_ms"]),
    )


def load_sessions(data_dir: Path, names: Iterable[str] | None = None) -> list[Session]:
    """Load all (or the named) binned sessions, sorted by date."""
    d = Path(data_dir) / "binned"
    paths = sorted(d.glob("*.npz")) if names is None else [d / f"{n}.npz" for n in names]
    return sorted((load_session(p) for p in paths), key=lambda s: (s.day, s.name))


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Download and bin the O'Doherty RTT sessions.")
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--sessions", nargs="*", default=list(DEFAULT_SESSIONS))
    ap.add_argument("--bin-ms", type=float, default=20.0)
    a = ap.parse_args()
    download_sessions(Path(a.data_dir), a.sessions, a.bin_ms)
