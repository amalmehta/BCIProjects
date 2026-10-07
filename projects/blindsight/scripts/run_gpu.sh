#!/usr/bin/env bash
# Full-scale Blindsight pipeline on a fresh Linux CUDA machine (configs/gpu.yaml).
# Usage: bash scripts/run_gpu.sh          (from anywhere; creates .venv-gpu next to the project)
set -euo pipefail
cd "$(dirname "$0")/.."
if [ ! -x .venv-gpu/bin/python ]; then
  python3 -m venv .venv-gpu
  .venv-gpu/bin/pip install --upgrade pip
  .venv-gpu/bin/pip install -r requirements-gpu.txt
fi
PY=.venv-gpu/bin/python
CFG="${CONFIG:-configs/gpu.yaml}"
run() { "$PY" -m blindsight.run --config "$CFG" "$@"; }

run --stage data          # downloads MNIST, EMNIST letters, CIFAR-10 (~1 GB incl. raw files) into data_root
run --stage validate
run --stage train         # one process; the GPU is the bottleneck
for d in mnist letters cifar; do run --stage evaluate --dataset "$d"; done
run --stage ablate
run --stage adapt
run --stage diagnostics
run --stage summarize
run --stage figures
run --stage export
"$PY" -m pytest -q tests
