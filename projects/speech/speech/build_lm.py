"""Build the word trigram LM and vocabulary used by the beam-search decoder.

Usage::

    python -m speech.build_lm --config configs/cpu.yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import lm_paths, load_config
from .data import fetch_text
from .lm import build_language_model
from .train import prepare_data

CMUDICT_URL = "https://raw.githubusercontent.com/cmusphinx/cmudict/master/cmudict.dict"
OWT10K_URL = "https://cdn-datasets.huggingface.co/nlp/datasets/openwebtext/openwebtext-10k.tar.xz"
DAILYDIALOG_URL = "https://huggingface.co/datasets/roskoN/dailydialog/resolve/main/train.zip"


def fetch_lm_resources(lm_dir: Path) -> None:
    """Download CMUdict, OpenWebText-10k and DailyDialog (train) if missing."""
    import tarfile

    fetch_text(CMUDICT_URL, lm_dir / "cmudict.dict")
    fetch_text(DAILYDIALOG_URL, lm_dir / "dd_train.zip")
    if not (lm_dir / "openwebtext-10k" / "10k.xz").exists():
        tar_path = fetch_text(OWT10K_URL, lm_dir / "owt10k.tar.xz")
        with tarfile.open(tar_path) as t:
            t.extractall(lm_dir)
        tar_path.unlink()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    lm_dir, built_dir = lm_paths(cfg)
    if not args.force and (built_dir / "kn3.npz").exists():
        print(f"[build_lm] {built_dir} exists (use --force to rebuild)")
        return
    fetch_lm_resources(lm_dir)
    data = prepare_data(cfg["data"], cfg["data_dir"])
    train_sents = [t.sentence for t in data["train"]] + [t.sentence for t in data["heldout_train"]]
    eval_sents = [t.sentence for k in ("dev", "test", "heldout_test") for t in data[k]]
    lmcfg = cfg.get("lm", {})
    _, vocab, stats = build_language_model(
        lm_dir, train_sents, eval_sents,
        min_count=lmcfg.get("min_count", 2), max_owt_docs=lmcfg.get("max_owt_docs"),
        out_dir=built_dir,
    )
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
