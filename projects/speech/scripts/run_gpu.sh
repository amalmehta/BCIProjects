#!/usr/bin/env bash
# Full-scale Speech pipeline on a fresh Linux + CUDA machine.
#   git clone <repo> && cd projects/speech && bash scripts/run_gpu.sh
# Disk: ~11 GB for all 45 sessions (compressed float16), ~0.6 GB for GPT-2 and LM resources.
set -euo pipefail
cd "$(dirname "$0")/.."
CONFIG="${CONFIG:-configs/gpu.yaml}"
PARALLEL="${PARALLEL:-1}"   # one training job per GPU

if [ ! -x .venv/bin/python ]; then
  python3 -m venv .venv
  .venv/bin/pip install --upgrade pip
  .venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cu121
  .venv/bin/pip install -r requirements-gpu.txt
fi
PY=.venv/bin/python
"$PY" -c "import torch; print('CUDA available:', torch.cuda.is_available())"

# data lives in ../../data/speech (see configs/cpu.yaml: data_dir)
"$PY" -m speech.download --config "$CONFIG"
"$PY" -m speech.build_lm --config "$CONFIG"
"$PY" -m speech.run_queue --config "$CONFIG" --parallel "$PARALLEL"
"$PY" -m speech.evaluate --config "$CONFIG"
"$PY" -m speech.calibrate --config "$CONFIG"
"$PY" -m speech.robustness --config "$CONFIG"
"$PY" -m speech.latency --config "$CONFIG"
"$PY" -m speech.report --config "$CONFIG"
"$PY" -m pytest -q tests
echo "done: results in results_gpu/"
