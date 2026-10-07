#!/usr/bin/env bash
# Full-scale pipeline for a fresh Linux CUDA machine.
#   git clone <repo> && cd projects/robotic_arm && bash scripts/run_gpu.sh
# Creates a virtualenv, installs requirements-gpu.txt, downloads FALCON H1 (~102 MB),
# then runs the same pipeline with configs/gpu.yaml (device auto-detected, AMP on CUDA).
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
cd "$HERE"
VENV="${VENV:-$HERE/.venv-gpu}"
if [ ! -x "$VENV/bin/python" ]; then
  python3 -m venv "$VENV"
  "$VENV/bin/pip" install --upgrade pip
  "$VENV/bin/pip" install -r requirements-gpu.txt
fi
export PYTHON="$VENV/bin/python"
export RA_DATA="${RA_DATA:-$HERE/data}"
"$PYTHON" -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
bash scripts/run_cpu.sh configs/gpu.yaml
