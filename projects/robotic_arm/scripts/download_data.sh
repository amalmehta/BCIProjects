#!/usr/bin/env bash
# Download FALCON H1 (DANDI 000954, ~102 MB) into $RA_DATA (default: <repo>/data/robotic_arm).
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
DATA="${RA_DATA:-$HERE/../../data/robotic_arm}"
PY="${PYTHON:-python}"
mkdir -p "$DATA"
if [ -d "$DATA/000954" ] && [ "$(find "$DATA/000954" -name '*.nwb' | wc -l)" -ge 40 ]; then
  echo "FALCON H1 already present in $DATA/000954"
else
  (cd "$DATA" && "$PY" -c "import sys; from dandi.cli.command import main; sys.exit(main())" download -o . DANDI:000954/draft)
fi
