# ============================================================
# FILE: scripts/research/build_m15_prices_from_m1.py
# GOAL:
#   Build M15 prices parquet (available_from + close + optional OHLCV)
#   from an M1 parquet/csv that contains dt_utc (or timestamp) + OHLCV.
# ============================================================

from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd


TS_CANDIDATES = ["dt_utc", "available_from", "timestamp", "time", "ts"]
REQUIRED_OHLCV = ["open", "high", "low", "close", "volume"]


def pick_ts_col(cols: list[str]) -> str:
    for c in TS_CANDIDATES:
        if c in cols:
            return c
    raise ValueError(f"No timestamp column found. Tried: {TS_CANDIDATES}. Available: {cols}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--m1_path", required=True, help="Path to M1 parquet/csv")
    ap.add_argument("--out", required=True, help="Output parquet path for M15")
    ap.add_argument("--tz", default=None, help="Optional timezone name if your timestamps are naive")
    args = ap.parse_args()

    in_path = Path(args.m1_path)
    if not in_path.exists():
        raise FileNotFoundError(str(in_path))

    if in_path.suffix.lower() == ".parquet":
        df = pd.read_parquet(in_path)
    else:
        df = pd.read_csv(in_path)

    ts_col = pick_ts_col(list(df.columns))
    df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")
    df = df.dropna(subset=[ts_col]).sort_values(ts_col)

    # Normalize to a canonical timestamp column
    df = df.rename(columns={ts_col: "available_from"})

    missing = [c for c in REQUIRED_OHLCV if c not in df.columns]
    if missing:
        raise ValueError(f"Missing OHLCV columns in M1 input: {missing}. Have: {list(df.columns)}")

    df = df.set_index("available_from")

    agg = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }

    m15 = df.resample("15min", label="left", closed="left").agg(agg).dropna()
    m15 = m15.reset_index()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    m15.to_parquet(out_path, index=False)

    print("OK wrote:", str(out_path))
    print("rows:", len(m15))
    print("cols:", list(m15.columns))


if __name__ == "__main__":
    main()
