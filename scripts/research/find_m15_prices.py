# ============================================================
# FILE: scripts/research/find_m15_prices.py
# GOAL:
#   Search under ./data for a parquet containing M15-like data
#   with columns ['available_from','close'] (or common timestamp aliases).
# ============================================================

from __future__ import annotations

import argparse
import os
from pathlib import Path
import pandas as pd


TS_CANDIDATES = ["available_from", "dt", "dt_utc", "timestamp", "time", "open_time", "ts"]
CLOSE_CANDIDATES = ["close", "c"]


def sniff_parquet(path: Path) -> dict | None:
    try:
        df = pd.read_parquet(path, columns=None)
    except Exception:
        return None

    cols = set(df.columns)
    ts_col = next((c for c in TS_CANDIDATES if c in cols), None)
    close_col = next((c for c in CLOSE_CANDIDATES if c in cols), None)
    if not ts_col or not close_col:
        return None

    # Try to infer bar step in minutes on a small sample (if possible)
    step_min = None
    try:
        s = pd.to_datetime(df[ts_col], errors="coerce").dropna().sort_values()
        if len(s) >= 5:
            diffs = s.diff().dropna().dt.total_seconds().values
            # median step
            med = float(pd.Series(diffs).median())
            step_min = round(med / 60.0, 3)
    except Exception:
        step_min = None

    return {
        "path": str(path),
        "ts_col": ts_col,
        "close_col": close_col,
        "n_rows": int(len(df)),
        "step_min_est": step_min,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data", help="Root folder to search (default: data)")
    ap.add_argument("--limit", type=int, default=2000, help="Max number of parquet files to scan")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    if not root.exists():
        raise FileNotFoundError(f"Root does not exist: {root}")

    files = list(root.rglob("*.parquet"))
    files = files[: args.limit]

    hits = []
    for p in files:
        info = sniff_parquet(p)
        if info:
            hits.append(info)

    if not hits:
        print("No candidate parquet found with (timestamp + close) under:", str(root))
        return

    # Print best guesses: step ~15 minutes first
    hits_sorted = sorted(
        hits,
        key=lambda x: (abs((x["step_min_est"] or 1e9) - 15.0), -x["n_rows"]),
    )

    print("FOUND CANDIDATES (best first):")
    for h in hits_sorted[:50]:
        print(
            f"- {h['path']}\n"
            f"  ts_col={h['ts_col']} close_col={h['close_col']} rows={h['n_rows']} step_min_est={h['step_min_est']}\n"
        )


if __name__ == "__main__":
    main()
