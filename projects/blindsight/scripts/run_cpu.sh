#!/usr/bin/env bash
# End-to-end Blindsight pipeline at CPU scale (configs/cpu.yaml):
# data -> simulator validation -> training -> evaluation -> ablations -> adaptation -> summary -> figures -> site export.
# Training runs in two parallel shards (2 threads each). Finished runs are skipped on re-run.
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-../../.venv/bin/python}"
CFG="${CONFIG:-configs/cpu.yaml}"
run() { "$PY" -m blindsight.run --config "$CFG" "$@"; }

run --stage data
run --stage validate
run --stage train --shard 0/2 & P0=$!
run --stage train --shard 1/2 & P1=$!
wait $P0; wait $P1
for d in mnist letters cifar; do run --stage evaluate --dataset "$d"; done
run --stage ablate
run --stage adapt
run --stage diagnostics
run --stage summarize
run --stage figures
run --stage export
"$PY" -m pytest -q tests
