# ============================================================
# FILE: scripts/research/build_m15_prices_from_raw_m1.py
# GOAL:
#   Build an M15 OHLCV parquet (available_from + open/high/low/close/volume_base)
#   from raw M1 parquet partitions:
#     data/raw/binance_um/klines_m1/symbol=BTCUSDT/year=YYYY/month=MM/part-000.parquet
#
# NOTES (based on your schema):
#   - ts is already a timestamp[ms, tz=UTC]
#   - volume columns are: volume_base, volume_quote (no "volume")
#
# OUTPUT:
#   - parquet with columns:
#       available_from, open, high, low, close, volume_base, volume_quote (optional)
# ============================================================

from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd


def _parse_dt_utc(x: str) -> pd.Timestamp:
    return pd.to_datetime(x, utc=True, errors="raise")


def _month_starts_utc(start_utc: pd.Timestamp, end_utc: pd.Timestamp) -> list[pd.Timestamp]:
    cur = start_utc.normalize().replace(day=1)
    end_month = end_utc.normalize().replace(day=1)
    months = []
    while cur <= end_month:
        months.append(cur)
        cur = cur + pd.offsets.MonthBegin(1)
    return months


def _load_month(raw_root: Path, symbol: str, year: int, month: int, cols: list[str]) -> pd.DataFrame:
    p = raw_root / f"symbol={symbol}" / f"year={year}" / f"month={month:02d}" / "part-000.parquet"
    if not p.exists():
        return pd.DataFrame(columns=cols)
    return pd.read_parquet(p, columns=cols).copy()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_root", required=True, help="Root folder, e.g. data/raw/binance_um/klines_m1")
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--start", required=True, help="Start UTC, e.g. 2026-02-03 00:00:00")
    ap.add_argument("--end", required=True, help="End UTC (exclusive), e.g. 2026-02-10 00:00:00")
    ap.add_argument("--out", required=True, help="Output parquet path")
    ap.add_argument("--keep_volume_quote", action="store_true", help="Also keep volume_quote aggregated")
    args = ap.parse_args()

    raw_root = Path(args.raw_root)
    if not raw_root.exists():
        raise FileNotFoundError(str(raw_root))

    start = _parse_dt_utc(args.start)
    end = _parse_dt_utc(args.end)
    if end <= start:
        raise ValueError("end must be > start")

    cols = ["ts", "open", "high", "low", "close", "volume_base"]
    if args.keep_volume_quote:
        cols.append("volume_quote")

    month_starts = _month_starts_utc(start, end)

    parts: list[pd.DataFrame] = []
    for ms in month_starts:
        dfm = _load_month(raw_root, args.symbol, int(ms.year), int(ms.month), cols)
        if not dfm.empty:
            parts.append(dfm)

    if not parts:
        raise FileNotFoundError("No raw M1 parquet partitions found for the requested window/months.")

    df = pd.concat(parts, ignore_index=True)

    df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    df = df.dropna(subset=["ts"]).sort_values("ts")

    df = df[(df["ts"] >= start) & (df["ts"] < end)].copy()
    if df.empty:
        raise ValueError("No rows in the requested time window after filtering.")

    df = df.set_index("ts")

    agg = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume_base": "sum",
    }
    if args.keep_volume_quote:
        agg["volume_quote"] = "sum"

    m15 = df.resample("15min", label="left", closed="left").agg(agg).dropna()
    m15 = m15.reset_index().rename(columns={"ts": "available_from"})

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    m15.to_parquet(out_path, index=False)

    print("OK wrote:", str(out_path))
    print("rows:", len(m15))
    print("start:", m15["available_from"].min())
    print("end:", m15["available_from"].max())
    print("cols:", list(m15.columns))


if __name__ == "__main__":
    main()
