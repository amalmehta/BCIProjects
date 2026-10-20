#!/usr/bin/env bash
# End-to-end Blindsight pipeline at CPU scale (configs/cpu.yaml):
# data -> simulator validation -> training (main, threshold-aware, failure-augmented, ablations) ->
# temporal encoder (habituation) -> evaluation -> ablations -> adaptation curve -> summary -> figures -> site export.
# One process at a time with 2 threads (shared machine). Finished runs/evaluations are skipped on re-run.
# The first-round results under the original dynaphos threshold fit are archived in results/v1_dynaphos_thresholds/;
# to reproduce them set sim.threshold_model: dynaphos and results_dir to a fresh directory.
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-../../.venv/bin/python}"
CFG="${CONFIG:-configs/cpu.yaml}"
run() { "$PY" -m blindsight.run --config "$CFG" "$@"; }

run --stage data
run --stage validate
run --stage train            # main encoders + encoder variants (seed-major), then the MNIST ablations
run --stage temporal         # fine-tune the temporal encoder from threshold_aware seed 0, evaluate sequences
for d in mnist letters cifar; do run --stage evaluate --dataset "$d"; done
run --stage ablate
run --stage adapt
run --stage summarize
run --stage figures
run --stage export
"$PY" -m pytest -q tests
node tests/js/render_check.mjs results/site_export
