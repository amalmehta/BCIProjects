"""Train every (run, seed) of the config's ``schedule`` with a small process pool.

Runs whose ``summary.json`` already exists are skipped, so the queue is resumable.

Usage::

    python -m speech.run_queue --config configs/cpu.yaml --parallel 2
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

from .config import PROJECT_DIR, load_config


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--parallel", type=int, default=2)
    ap.add_argument("--results-dir", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.results_dir:
        cfg["results_dir"] = str(Path(args.results_dir).resolve())
    runs_dir = Path(cfg["results_dir"]) / "runs"
    log_dir = Path(cfg["results_dir"]) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    todo = [(r, int(s)) for r, s in cfg["schedule"] if not (runs_dir / f"{r}_s{s}" / "summary.json").exists()]
    active: list[tuple[subprocess.Popen, str]] = []
    failed = []
    while todo or active:
        while todo and len(active) < args.parallel:
            run, seed = todo.pop(0)
            name = f"{run}_s{seed}"
            log = open(log_dir / f"{name}.log", "w")
            cmd = [sys.executable, "-m", "speech.train", "--config", args.config, "--run", run, "--seed", str(seed),
                   "--results-dir", cfg["results_dir"]]
            active.append((subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=PROJECT_DIR), name))
            print(f"[queue] started {name}", flush=True)
        time.sleep(5)
        for p, name in list(active):
            if p.poll() is not None:
                active.remove((p, name))
                print(f"[queue] finished {name} (exit {p.returncode})", flush=True)
                if p.returncode != 0:
                    failed.append(name)
    if failed:
        raise SystemExit(f"failed runs: {failed}")


if __name__ == "__main__":
    main()
