#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src"

echo "=== DAILY JOB START (macOS) ==="
echo "ProjectRoot: $PWD"
echo "PYTHONPATH:  $PYTHONPATH"

python -m bfdl.cli.collect
echo "=== DAILY JOB END (macOS) ==="
