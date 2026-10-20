#!/usr/bin/env bash
# Full-scale pipeline on a Linux CUDA box with configs/gpu.yaml (same code; GPU auto-detected, AMP on).
#   data download -> prep -> train -> eval -> kalman -> remap -> few-shot + stress -> closed-loop sim -> clda -> export.
# Stages are cached: finished checkpoints are skipped, so the script can be re-run after an interruption.
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-python}"
CFG="${CFG:-configs/gpu.yaml}"
DATA="$(grep '^data_dir:' "$CFG" | awk '{print $2}')"
LOGS="$(grep '^results_dir:' "$CFG" | awk '{print $2}')/logs"
mkdir -p "$LOGS"

# Fresh machine:  python -m venv .venv && source .venv/bin/activate && pip install -r requirements-gpu.txt
echo "== data (downloads ~3 GB of raw .mat files from Zenodo, keeps ~20 MB binned) =="
"$PY" -m telepathy.data --data-dir "$DATA"

run() { "$PY" -m telepathy.experiments --config "$CFG" "$@"; }

echo "== prep =="
run --stage prep

echo "== train (3 parallel streams sharing the GPU) =="
run --stage train --models sld:0,gru:0,sld_no_aug:0,sld_transformer:0 > "$LOGS"/train_stream0.out 2>&1 &
run --stage train --models gru:1,sld:1,sld_no_ssl:0,gru_aug:0 > "$LOGS"/train_stream1.out 2>&1 &
run --stage train --models sld:2,gru:2,sld_no_adapters:0 > "$LOGS"/train_stream2.out 2>&1 &
wait

echo "== eval =="
run --stage eval

echo "== strong Kalman (PVA) baseline, then label-free channel re-matching =="
run --stage kalman
run --stage remap

echo "== few-shot and stress tests (parallel) =="
run --stage fewshot > "$LOGS"/fewshot.out 2>&1 &
run --stage stress > "$LOGS"/stress.out 2>&1 &
wait

echo "== closed-loop WebGrid simulator =="
run --stage sim
run --stage simabl

echo "== closed-loop decoder adaptation (ReFIT-style) in the simulator =="
run --stage clda

echo "== export (metrics.json, figures, site_export) =="
run --stage export
echo "done: see results/metrics.json and results/site_export/"
