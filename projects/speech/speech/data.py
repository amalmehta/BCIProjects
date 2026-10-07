"""Download, compact and load the Brain-to-Text '25 intracortical speech data.

Source: Card et al. (2024), "An accurate and rapidly calibrating speech
neuroprosthesis", NEJM; Dryad doi:10.5061/dryad.dncjsxm85. Dryad's API now
requires a bearer token for file downloads, so we read the per-session HDF5
files from a public Hugging Face mirror of the unzipped release
(``Paulownia/Brain-to-text_25_data``). Each trial holds 512 features per 20 ms
bin (threshold crossings + spike-band power on 256 electrodes in ventral
precentral gyrus), already z-scored by the authors over the preceding trials,
plus the cue sentence and its phoneme sequence.

To respect the disk budget, every HDF5 file is streamed to disk, converted to a
compressed float16 ``.npz`` (about half the size), and deleted.
"""

from __future__ import annotations

import os
import subprocess
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

MIRROR = (
    "https://huggingface.co/datasets/Paulownia/Brain-to-text_25_data/resolve/main/"
    "neural_data_unzipped/hdf5_data_final/{session}/data_{split}.hdf5"
)
N_FEATURES = 512


def _hdf5_to_npz(h5_path: Path, out_path: Path) -> int:
    """Convert one session/split HDF5 file into a compact float16 npz.

    Returns the number of trials written.
    """
    import h5py

    feats, f_off, phones, p_off, sentences, blocks, trials = [], [0], [], [0], [], [], []
    with h5py.File(h5_path, "r") as f:
        for key in sorted(f.keys()):
            g = f[key]
            x = g["input_features"][()].astype(np.float16)
            feats.append(x)
            f_off.append(f_off[-1] + x.shape[0])
            if "seq_class_ids" in g:
                n = int(g.attrs["seq_len"])
                ph = g["seq_class_ids"][()][:n].astype(np.int8)
                sentences.append(str(g.attrs["sentence_label"]))
            else:  # competition test split: no labels
                ph = np.zeros(0, np.int8)
                sentences.append("")
            phones.append(ph)
            p_off.append(p_off[-1] + len(ph))
            blocks.append(int(g.attrs["block_num"]))
            trials.append(int(g.attrs["trial_num"]))
    np.savez_compressed(
        out_path,
        features=np.concatenate(feats),
        feat_offsets=np.asarray(f_off, np.int64),
        phones=np.concatenate(phones),
        phone_offsets=np.asarray(p_off, np.int64),
        sentences=np.asarray(sentences),
        blocks=np.asarray(blocks, np.int32),
        trials=np.asarray(trials, np.int32),
    )
    return len(feats)


def download_sessions(sessions: list[str], data_dir: Path, splits=("train", "val")) -> None:
    """Fetch and compact the requested sessions into ``data_dir/sessions``.

    Already-converted files are skipped, so the function is resumable.
    """
    out_dir = Path(data_dir) / "sessions"
    out_dir.mkdir(parents=True, exist_ok=True)
    for session in sessions:
        for split in splits:
            out = out_dir / f"{session}_{split}.npz"
            if out.exists():
                continue
            tmp = out_dir / f"{session}_{split}.hdf5.part"
            url = MIRROR.format(session=session, split=split)
            print(f"[download] {session} {split}", flush=True)
            # curl handles redirects/retries more robustly than urllib for large files.
            subprocess.run(
                ["curl", "-sSfL", "--retry", "5", "-o", str(tmp), url], check=True
            )
            n = _hdf5_to_npz(tmp, out_dir / f"{session}_{split}.tmp.npz")
            os.replace(out_dir / f"{session}_{split}.tmp.npz", out)
            tmp.unlink()
            print(f"[download] {session} {split}: {n} trials -> {out.stat().st_size/1e6:.0f} MB", flush=True)


@dataclass
class Trial:
    """One attempted-speech sentence."""

    features: np.ndarray  # (T, 512) float16
    phones: np.ndarray  # (L,) int class ids (0 = blank never appears)
    sentence: str
    day: int  # index of the session in the config's session list
    session: str
    block: int


def load_split(data_dir: Path, sessions: list[str], split: str) -> list[Trial]:
    """Load every trial of ``split`` for the given sessions (day index = list position)."""
    out: list[Trial] = []
    for day, session in enumerate(sessions):
        path = Path(data_dir) / "sessions" / f"{session}_{split}.npz"
        if not path.exists():
            continue
        z = np.load(path)
        feats, fo = z["features"], z["feat_offsets"]
        phones, po = z["phones"], z["phone_offsets"]
        for i in range(len(fo) - 1):
            out.append(
                Trial(
                    features=feats[fo[i] : fo[i + 1]],
                    phones=phones[po[i] : po[i + 1]].astype(np.int64),
                    sentence=str(z["sentences"][i]),
                    day=day,
                    session=session,
                    block=int(z["blocks"][i]),
                )
            )
    return out


class SpeechDataset(Dataset):
    """Torch dataset over :class:`Trial` objects."""

    def __init__(self, trials: list[Trial]):
        self.trials = trials

    def __len__(self) -> int:
        return len(self.trials)

    def __getitem__(self, i: int):
        t = self.trials[i]
        return (
            torch.from_numpy(t.features.astype(np.float32)),
            torch.from_numpy(t.phones),
            t.day,
            i,
        )


def collate(batch):
    """Pad a list of (features, phones, day, index) into batch tensors."""
    feats, phones, days, idx = zip(*batch)
    x_len = torch.tensor([f.shape[0] for f in feats])
    y_len = torch.tensor([p.shape[0] for p in phones])
    x = torch.nn.utils.rnn.pad_sequence(feats, batch_first=True)
    y = torch.nn.utils.rnn.pad_sequence(phones, batch_first=True)
    return x, x_len, y, y_len, torch.tensor(days), torch.tensor(idx)


def fetch_text(url: str, dest: Path) -> Path:
    """Download a small text resource (CMUdict, corpora) if not present."""
    dest = Path(dest)
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, dest)
    return dest
