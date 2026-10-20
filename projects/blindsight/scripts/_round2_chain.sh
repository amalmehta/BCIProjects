#!/usr/bin/env bash
# Round-2 CPU chain (one process at a time): waits for the training shards, then runs the remaining stages.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=../../.venv/bin/python
run() { echo "=== $(date +%H:%M:%S) $*"; "$PY" -m blindsight.run --config configs/cpu.yaml "$@" || echo "FAILED: $*"; }
while kill -0 54047 2>/dev/null || kill -0 54048 2>/dev/null; do sleep 60; done
echo "=== $(date +%H:%M:%S) training shards finished"
run --stage train --no-ablations   # picks up any job the shards did not finish
run --stage validate
run --stage temporal
for d in mnist letters cifar; do run --stage evaluate --dataset "$d"; done
run --stage ablate
run --stage summarize
run --stage figures
run --stage export
node tests/js/render_check.mjs results/site_export || echo "FAILED: node check"
run --stage adapt
run --stage summarize
run --stage figures
run --stage export
echo "=== $(date +%H:%M:%S) chain done"
