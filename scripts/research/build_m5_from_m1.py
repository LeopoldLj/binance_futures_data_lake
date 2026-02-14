#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
import pandas as pd


def parse_utc(ts: str) -> pd.Timestamp:
    t = pd.Timestamp(ts)
    if t.tzinfo is None:
        t = t.tz_localize("UTC")
    else:
        t = t.tz_convert("UTC")
    return t


def list_month_parts(root: Path, symbol: str, year: int, month: int) -> list[Path]:
    base = root / f"symbol={symbol}" / f"year={year:04d}" / f"month={month:02d}"
    if not base.exists():
        raise FileNotFoundError(f"Month dir not found: {base}")
    parts = sorted(base.glob("part-*.parquet"))
    if not parts:
        raise FileNotFoundError(f"No parquet parts found in: {base}")
    return parts


def load_month(root: Path, symbol: str, year: int, month: int) -> pd.DataFrame:
    parts = list_month_parts(root, symbol, year, month)
    dfs: list[pd.DataFrame] = []
    for p in parts:
        d = pd.read_parquet(p)
        for col in ["symbol", "exchange", "market"]:
            if col in d.columns:
                d[col] = d[col].astype("string")
        dfs.append(d)
    return pd.concat(dfs, ignore_index=True)


def ensure_types(df: pd.DataFrame, ts_col: str) -> pd.DataFrame:
    if ts_col not in df.columns:
        raise KeyError(f"Missing required timestamp column: {ts_col}")

    df = df.copy()
    df[ts_col] = pd.to_datetime(df[ts_col], utc=True, errors="raise")

    for c in ["open", "high", "low", "close", "volume_base", "volume_quote", "taker_buy_base", "taker_buy_quote"]:
        if c in df.columns:
            df[c] = df[c].astype("float64")

    if "n_trades" in df.columns:
        df["n_trades"] = df["n_trades"].astype("int64")

    return df


def build_m5(df_m1: pd.DataFrame, ts_col: str, lag_minutes: int) -> pd.DataFrame:
    df = df_m1.sort_values(ts_col).set_index(ts_col)

    agg = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume_base": "sum",
        "volume_quote": "sum",
        "n_trades": "sum",
        "taker_buy_base": "sum",
        "taker_buy_quote": "sum",
    }
    agg = {k: v for k, v in agg.items() if k in df.columns}

    count_m1 = df["close"].resample("5min", label="left", closed="left").size().rename("count_m1")

    m5 = df.resample("5min", label="left", closed="left").agg(agg)
    m5 = m5.join(count_m1)

    need_ohlc = [c for c in ["open", "high", "low", "close"] if c in m5.columns]
    if need_ohlc:
        m5 = m5.dropna(subset=need_ohlc)

    m5 = m5.reset_index().rename(columns={ts_col: "ts"})

    if all(c in m5.columns for c in ["high", "low"]):
        m5["range"] = m5["high"] - m5["low"]

    m5["bucket_start"] = m5["ts"]
    m5["bucket_end"] = m5["ts"] + pd.Timedelta(minutes=5)
    m5["available_from"] = m5["bucket_end"] + pd.Timedelta(minutes=int(lag_minutes))

    return m5


def main() -> int:
    p = argparse.ArgumentParser(description="Build M5 candles from M1 (closed buckets, anti-lookahead).")
    p.add_argument("--root", required=True, help=r"Root folder like ...\data\raw\binance_um\klines_m1")
    p.add_argument("--symbol", required=True)
    p.add_argument("--year", type=int, required=True)
    p.add_argument("--month", type=int, required=True)
    p.add_argument("--start", default=None, help='UTC, e.g. "2026-02-10 06:00:00"')
    p.add_argument("--end", default=None, help='UTC, e.g. "2026-02-10 07:00:00"')
    p.add_argument("--ts-col", default="ts", help="Timestamp column name in parquet (default: ts)")
    p.add_argument("--lag-minutes", type=int, default=2, help="Safe lag in minutes for availability (default: 2)")
    p.add_argument("--out", default=None, help="Optional output parquet path")
    p.add_argument("--check-buckets", action="store_true", help="Print buckets where count_m1 != 5 (excluding edges)")
    args = p.parse_args()

    root = Path(args.root)
    df = load_month(root, args.symbol, args.year, args.month)
    df = ensure_types(df, args.ts_col)

    if args.start:
        t0 = parse_utc(args.start)
        df = df[df[args.ts_col] >= t0]
    if args.end:
        t1 = parse_utc(args.end)
        df = df[df[args.ts_col] < t1]

    m5 = build_m5(df, args.ts_col, args.lag_minutes)

    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 120)
    print(m5.tail(10).to_string(index=False))

    if args.check_buckets and len(m5) > 0:
        core = m5.iloc[1:-1] if len(m5) >= 3 else m5.iloc[0:0]
        bad = core[core["count_m1"] != 5]
        if len(bad) > 0:
            print("\n[WARN] Buckets with count_m1 != 5 (excluding edges):")
            print(bad[["ts", "count_m1", "bucket_start", "bucket_end", "available_from"]].to_string(index=False))
        else:
            print("\n[OK] All core buckets have count_m1 == 5")

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        m5.to_parquet(out_path, index=False)
        print(f"\n[OK] wrote: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
