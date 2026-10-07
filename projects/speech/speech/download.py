"""Fetch everything a config needs: neural sessions, block metadata, and GPT-2 small weights.

Usage::

    python -m speech.download --config configs/cpu.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .config import load_config
from .data import download_sessions, fetch_text

BLOCK_CSV_URL = ("https://raw.githubusercontent.com/Neuroprosthetics-Lab/nejm-brain-to-text/main/"
                 "data/t15_copyTaskData_description.csv")


def fetch_gpt2(dest: Path) -> None:
    """Download GPT-2 small (124M, ~0.5 GB) for n-best rescoring, if missing."""
    if (dest / "model.safetensors").exists():
        return
    from huggingface_hub import snapshot_download

    snapshot_download("openai-community/gpt2", local_dir=dest, allow_patterns=[
        "config.json", "model.safetensors", "tokenizer.json", "vocab.json", "merges.txt",
        "tokenizer_config.json", "generation_config.json"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--skip-gpt2", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    data_dir = Path(cfg["data_dir"])
    sessions = list(cfg["data"]["sessions"]) + ([cfg["data"]["heldout_session"]] if cfg["data"].get("heldout_session") else [])
    download_sessions(sessions, data_dir)
    fetch_text(BLOCK_CSV_URL, data_dir / "t15_copyTaskData_description.csv")
    if not args.skip_gpt2:
        fetch_gpt2(data_dir / "gpt2")


if __name__ == "__main__":
    main()
