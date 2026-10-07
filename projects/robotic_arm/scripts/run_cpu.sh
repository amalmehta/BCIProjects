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

echo "== 2/5 decoders + 3/5 closed loop (in parallel)"
"$PYTHON" -m robotic_arm.run_closed_loop --config "$CONFIG" > "$RES/logs/closed_loop.log" 2>&1 &
P2=$!
# decoders: one process per part (linear filters, then each neural variant), MAX_PAR at a time
rm -rf "$RES/decoding_parts"
PARTS=$("$PYTHON" -c "from robotic_arm.config import load_config; print(' '.join(['linear'] + list(load_config('$CONFIG').decoding.neural.variants)))")
echo "$PARTS" | tr ' ' '\n' | xargs -P "${MAX_PAR:-3}" -I{} sh -c \
  '"$PYTHON" -m robotic_arm.run_decoding --config "$0" --parts {} > "$1/logs/decoding_{}.log" 2>&1 || { echo "decoding part {} failed"; exit 255; }' "$CONFIG" "$RES"
"$PYTHON" -m robotic_arm.run_decoding --merge
wait $P2 || { echo "closed loop failed, see $RES/logs/closed_loop.log"; exit 1; }
tail -n 2 "$RES/logs/closed_loop.log"

echo "== 4/5 figures + paper tables"
"$PYTHON" -m robotic_arm.figures
"$PYTHON" -m robotic_arm.tables
echo "== 5/5 site export"
"$PYTHON" -m robotic_arm.export
if command -v tectonic >/dev/null 2>&1 && [ -z "${RA_RESULTS:-}" ]; then
  (cd paper && tectonic main.tex > "$RES/logs/tectonic.log" 2>&1 && echo "paper/main.pdf compiled") || echo "paper compile failed (see logs)"
fi
echo "done: results in $RES"
