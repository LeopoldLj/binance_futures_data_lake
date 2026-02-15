#!/usr/bin/env zsh
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 YYYY-MM [monthly|daily]"
  echo "Example: $0 2026-01 monthly"
  exit 1
fi

MONTH="$1"
PERIODICITY="${2:-monthly}"

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT_DIR"

PY="$ROOT_DIR/.venv/bin/python"
if [[ ! -x "$PY" ]]; then
  PY="python"
fi

echo "[RUN] month=$MONTH periodicity=$PERIODICITY"
"$PY" "$ROOT_DIR/scripts/research/download_aggtrades_binance_vision_v1.py" \
  --symbol BTCUSDT \
  --start-month "$MONTH" \
  --end-month "$MONTH" \
  --periodicity "$PERIODICITY" \
  --out-root "$ROOT_DIR/data/raw/binance_um/aggtrades" \
  --tmp-dir "$ROOT_DIR/data/tmp/binance_vision_aggtrades" \
  --chunk-rows 1000000
