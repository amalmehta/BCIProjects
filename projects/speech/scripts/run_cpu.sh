#!/usr/bin/env bash
# End-to-end Speech pipeline on CPU: data -> LM -> training -> decoding -> side studies -> figures/site export.
# Every stage skips work that is already done, so the script is resumable.
#   CONFIG=configs/smoke.yaml scripts/run_cpu.sh   # 10-minute smoke test of the whole pipeline
#   DIAG=1 scripts/run_cpu.sh                       # also run the CTC-instability probes (configs/diag.yaml)
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-../../.venv/bin/python}"
CONFIG="${CONFIG:-configs/cpu.yaml}"
PARALLEL="${PARALLEL:-1}"

echo "== 1/8 data (Brain-to-Text '25 sessions, block metadata, GPT-2 small)"
"$PY" -m speech.download --config "$CONFIG"
echo "== 2/8 word trigram LM + lexicon"
"$PY" -m speech.build_lm --config "$CONFIG"
echo "== 3/8 training (queue, $PARALLEL concurrent jobs)"
"$PY" -m speech.run_queue --config "$CONFIG" --parallel "$PARALLEL"
if [ "${DIAG:-0}" = "1" ]; then
  echo "== 3b diagnostic probes: previous vs stabilised recipe, seed 1, first 150 steps"
  "$PY" -m speech.run_queue --config configs/diag.yaml --parallel 1
fi
echo "== 3c training-dynamics summary (blank plateau, gradient norms)"
"$PY" -m speech.stability --config "$CONFIG"
echo "== 4/8 decoding (all decoder settings tuned on dev) + PER/WER + bootstrap CIs + paired tests"
"$PY" -m speech.evaluate --config "$CONFIG"
echo "== 5/8 new-day study: reuse, unsupervised adaptation (CORAL, self-training), supervised calibration"
"$PY" -m speech.calibrate --config "$CONFIG"
echo "== 6/8 electrode-loss robustness"
"$PY" -m speech.robustness --config "$CONFIG"
echo "== 7/8 latency"
"$PY" -m speech.latency --config "$CONFIG"
echo "== 8/8 metrics.json, figures, site export"
"$PY" -m speech.report --config "$CONFIG"
if [ "$CONFIG" = "configs/cpu.yaml" ]; then
  echo "== paper tables + PDF"
  "$PY" -m speech.paper_tables --config "$CONFIG"
  if command -v tectonic >/dev/null; then (cd paper && tectonic main.tex); fi
fi
echo "done: see results/ (or the config's results_dir)"
