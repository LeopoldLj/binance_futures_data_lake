#!/usr/bin/env zsh
set -euo pipefail

ROOT="/Users/lolo/PyCharmMiscProject/binance_futures_data_lake"
PY="$ROOT/.venv/bin/python"
PLOT="$ROOT/scripts/research/plot_detected_ranges_mtf_overlay_v1.py"
INPUT="$ROOT/data/research_debug/BTCUSDT/full_201910_202602/joined_full__enriched__router__structure_v1__mn_w_d_h4_h1_m15_context.parquet"
BOXES="$ROOT/data/research_debug/BTCUSDT/full_201910_202602/ichimoku_flat_levels_v1/range_boxes_from_flats_v2_with_w1_expanded.csv"
OUTDIR="$ROOT/data/research_debug/BTCUSDT/full_201910_202602/ichimoku_flat_levels_v1/charts"

"$PY" "$PLOT" \
  --input "$INPUT" \
  --boxes-csv "$BOXES" \
  --output-svg "$OUTDIR/ranges_overlay_global_h4.svg" \
  --base-tf H4 \
  --overlay-tfs W1,D1,H4,H1,M30 \
  --last-bars 900 \
  --max-boxes-per-tf 10 \
  --style-profile global_clean

"$PY" "$PLOT" \
  --input "$INPUT" \
  --boxes-csv "$BOXES" \
  --output-svg "$OUTDIR/ranges_overlay_micro_m30.svg" \
  --base-tf M30 \
  --overlay-tfs W1,D1,H4,H1,M30 \
  --last-bars 360 \
  --max-boxes-per-tf 8 \
  --style-profile micro_precise

echo "[OK] Generated:"
echo " - $OUTDIR/ranges_overlay_global_h4.svg"
echo " - $OUTDIR/ranges_overlay_micro_m30.svg"
