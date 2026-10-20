#!/usr/bin/env bash
# End-to-end CPU pipeline: data -> decoders -> simulated closed loop -> figures -> site export.
# Usage: scripts/run_cpu.sh [config]   (default configs/cpu.yaml; configs/smoke.yaml runs in minutes)
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
cd "$HERE"
CONFIG="${1:-configs/cpu.yaml}"
export PYTHON="${PYTHON:-$HERE/../../.venv/bin/python}"
export PYTHONPATH="$HERE${PYTHONPATH:+:$PYTHONPATH}"
RES="${RA_RESULTS:-$HERE/results}"
mkdir -p "$RES/logs"

echo "== 1/5 data"
bash scripts/download_data.sh
"$PYTHON" -c "from robotic_arm.data import load_blocks; b=load_blocks(); print({k: len(v) for k, v in b.items()})"

echo "== 2/5 decoders: one process per (part, seed), MAX_PAR at a time (default 1 on a shared machine)"
rm -rf "$RES/decoding_parts"
JOBS=$("$PYTHON" -c "
from robotic_arm.config import load_config
d = load_config('$CONFIG').decoding
jobs = ['linear:']
for v in d.neural.variants:
    seeds = d.seeds if v in d.neural.multi_seed else d.seeds[:1]
    jobs += [f'{v}:{s}' for s in seeds]
print(' '.join(jobs))")
echo "$JOBS" | tr ' ' '\n' | xargs -P "${MAX_PAR:-1}" -I{} sh -c \
  'p="${2%%:*}"; s="${2#*:}"; "$PYTHON" -m robotic_arm.run_decoding --config "$0" --parts "$p" ${s:+--seeds "$s"} \
     > "$1/logs/decoding_${p}_s${s}.log" 2>&1 || { echo "decoding part $2 failed"; exit 255; }' "$CONFIG" "$RES" {}
"$PYTHON" -m robotic_arm.run_decoding --merge

echo "== 3/5 closed loop: synthetic noise model, then real decoder residuals (needs decoding_preds.npz)"
"$PYTHON" -m robotic_arm.run_closed_loop --config "$CONFIG" > "$RES/logs/closed_loop.log" 2>&1 \
  || { echo "closed loop failed, see $RES/logs/closed_loop.log"; exit 1; }
tail -n 2 "$RES/logs/closed_loop.log"
"$PYTHON" -m robotic_arm.run_closed_loop --config "$CONFIG" --real-residuals > "$RES/logs/closed_loop_real.log" 2>&1 \
  || { echo "real-residual closed loop failed, see $RES/logs/closed_loop_real.log"; exit 1; }
tail -n 2 "$RES/logs/closed_loop_real.log"

echo "== 4/5 figures + paper tables"
"$PYTHON" -m robotic_arm.figures
"$PYTHON" -m robotic_arm.tables
echo "== 5/5 site export"
"$PYTHON" -m robotic_arm.export
if command -v tectonic >/dev/null 2>&1 && [ -z "${RA_RESULTS:-}" ]; then
  (cd paper && tectonic main.tex > "$RES/logs/tectonic.log" 2>&1 && echo "paper/main.pdf compiled") || echo "paper compile failed (see logs)"
fi
echo "done: results in $RES"
