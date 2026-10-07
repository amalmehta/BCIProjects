#!/usr/bin/env bash
# End-to-end pipeline on CPU with configs/cpu.yaml:
#   data download -> prep -> train -> eval -> few-shot + stress -> closed-loop sim -> export.
# Stages are cached: finished checkpoints are skipped, so the script can be re-run after an interruption.
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-../../.venv/bin/python}"
CFG="${CFG:-configs/cpu.yaml}"
DATA="$(grep '^data_dir:' "$CFG" | awk '{print $2}')"
LOGS="$(grep '^results_dir:' "$CFG" | awk '{print $2}')/logs"
mkdir -p "$LOGS"

echo "== data =="
"$PY" -m telepathy.data --data-dir "$DATA"

run() { "$PY" -m telepathy.experiments --config "$CFG" "$@"; }

echo "== prep =="
run --stage prep

echo "== train (3 parallel streams, 2 torch threads each) =="
run --stage train --models sld:0,gru:0,sld_no_aug:0,sld_transformer:0 > "$LOGS"/train_stream0.out 2>&1 &
run --stage train --models gru:1,sld:1,sld_no_ssl:0,gru_aug:0 > "$LOGS"/train_stream1.out 2>&1 &
run --stage train --models sld:2,gru:2,sld_no_adapters:0 > "$LOGS"/train_stream2.out 2>&1 &
wait

echo "== eval =="
run --stage eval

echo "== few-shot and stress tests (parallel) =="
run --stage fewshot > "$LOGS"/fewshot.out 2>&1 &
run --stage stress > "$LOGS"/stress.out 2>&1 &
wait

echo "== closed-loop WebGrid simulator =="
run --stage sim
run --stage simabl

echo "== export (metrics.json, figures, site_export) =="
run --stage export
echo "done: see results/metrics.json and results/site_export/"
