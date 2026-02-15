#!/usr/bin/env zsh
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

export PYTHONPATH="$ROOT_DIR/src"

SYMBOL="${1:-BTCUSDT}"
LOCK_DIR="/tmp/bfdl_hourly_${SYMBOL}.lock"
LOG_DIR="$ROOT_DIR/logs/hourly/${SYMBOL}"
RUN_STAMP="$(date -u +"%Y%m%dT%H%M%SZ")"
LOG_FILE="${LOG_DIR}/hourly_${SYMBOL}_${RUN_STAMP}.log"

mkdir -p "$LOG_DIR"
exec > >(tee -a "$LOG_FILE") 2>&1

if ! mkdir "$LOCK_DIR" 2>/dev/null; then
  echo "[SKIP] hourly_job already running for ${SYMBOL} (lock: ${LOCK_DIR})"
  exit 0
fi
trap 'rmdir "$LOCK_DIR" 2>/dev/null || true' EXIT INT TERM

if [[ -x "$ROOT_DIR/.venv/bin/python" ]]; then
  PY="$ROOT_DIR/.venv/bin/python"
else
  PY="python"
fi

echo "=== HOURLY JOB START (zsh) ==="
echo "ts_utc:     $(date -u +"%Y-%m-%dT%H:%M:%SZ")"
echo "project:    $ROOT_DIR"
echo "python:     $PY"
echo "symbol:     $SYMBOL"
echo "PYTHONPATH: $PYTHONPATH"
echo "log_file:   $LOG_FILE"

"$PY" -m bfdl.cli.collect --symbol "$SYMBOL"
"$PY" -m bfdl.transforms.compact_staging --symbol "$SYMBOL"
"$PY" -m bfdl.transforms.gaps_report --symbol "$SYMBOL"
"$PY" -m bfdl.transforms.last_minute --symbol "$SYMBOL"

echo "=== HOURLY JOB END (zsh) ==="
