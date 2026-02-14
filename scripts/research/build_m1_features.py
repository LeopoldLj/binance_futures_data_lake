#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
import pandas as pd
import numpy as np

EPS = 1e-12


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
    # Read parts one by one to avoid PyArrow schema merge errors
    parts = list_month_parts(root, symbol, year, month)

    dfs: list[pd.DataFrame] = []
    for p in parts:
        d = pd.read_parquet(p)

        # Harmonise common string columns that can be dictionary-encoded in some files
        for col in ["symbol", "exchange", "market"]:
            if col in d.columns:
                d[col] = d[col].astype("string")

        dfs.append(d)

    df = pd.concat(dfs, ignore_index=True)
    return df


def ensure_types(df: pd.DataFrame) -> pd.DataFrame:
    if "ts" not in df.columns:
        raise ValueError("Column 'ts' not found")
    df["ts"] = pd.to_datetime(df["ts"], utc=True)

    needed = [
        "open", "high", "low", "close",
        "volume_base", "taker_buy_base",
        "volume_quote", "n_trades",
    ]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    for c in ["open", "high", "low", "close", "volume_base", "taker_buy_base", "volume_quote"]:
        df[c] = df[c].astype("float64")
    df["n_trades"] = df["n_trades"].astype("int64")
    return df


def compute_atr(df: pd.DataFrame, atr_len: int) -> pd.DataFrame:
    h = df["high"]
    l = df["low"]
    c = df["close"]
    prev_c = c.shift(1)

    tr1 = (h - l)
    tr2 = (h - prev_c).abs()
    tr3 = (l - prev_c).abs()
    tr = np.maximum(tr1, np.maximum(tr2, tr3))

    atr = tr.rolling(atr_len, min_periods=atr_len).mean()

    out = df.copy()
    out["tr"] = tr
    out[f"atr{atr_len}"] = atr
    return out


def compute_features(df: pd.DataFrame, atr_len: int) -> pd.DataFrame:
    o = df["open"]
    h = df["high"]
    l = df["low"]
    c = df["close"]

    rng = (h - l)
    body = (c - o).abs()
    upper_wick = h - np.maximum(o, c)
    lower_wick = np.minimum(o, c) - l

    rng_safe = rng.where(rng > EPS, np.nan)
    body_pct = body / rng_safe
    upper_wick_pct = upper_wick / rng_safe
    lower_wick_pct = lower_wick / rng_safe
    close_pos = (c - l) / rng_safe

    market_buys = df["taker_buy_base"]
    market_sells = df["volume_base"] - df["taker_buy_base"]
    delta = market_buys - market_sells
    vol_safe = df["volume_base"].where(df["volume_base"] > EPS, np.nan)
    delta_norm = delta / vol_safe

    atr_col = f"atr{atr_len}"
    atr_vals = df[atr_col] if atr_col in df.columns else np.nan

    out = df[["ts", "open", "high", "low", "close", "volume_base", "volume_quote", "n_trades"]].copy()

    # price structure
    out["range"] = rng
    out["body"] = body
    out["upper_wick"] = upper_wick
    out["lower_wick"] = lower_wick
    out["body_pct"] = body_pct
    out["upper_wick_pct"] = upper_wick_pct
    out["lower_wick_pct"] = lower_wick_pct
    out["close_pos"] = close_pos

    # microstructure (market language)
    out["market_buys"] = market_buys
    out["market_sells"] = market_sells
    out["delta"] = delta
    out["delta_norm"] = delta_norm

    # volatility short-term
    out["tr"] = df["tr"] if "tr" in df.columns else np.nan
    out[atr_col] = atr_vals
    out["range_rel"] = out["range"] / np.maximum(out[atr_col], EPS)

    return out


def main() -> int:
    p = argparse.ArgumentParser(description="Build M1 features (price structure + market buy/sell + delta + ATR).")
    p.add_argument("--root", required=True, help="Root path to klines_m1 (e.g. ...\\data\\raw\\binance_um\\klines_m1)")
    p.add_argument("--symbol", required=True)
    p.add_argument("--year", type=int, required=True)
    p.add_argument("--month", type=int, required=True)
    p.add_argument("--start", required=False, default=None, help="UTC start timestamp (e.g. '2026-02-10 00:00:00')")
    p.add_argument("--end", required=False, default=None, help="UTC end timestamp (exclusive)")
    p.add_argument("--atr-len", dest="atr_len", type=int, default=14)
    p.add_argument("--out", required=False, default=None, help="Optional output parquet path")
    args = p.parse_args()

    root = Path(args.root)
    df = load_month(root, args.symbol, args.year, args.month)
    df = ensure_types(df)

    if args.start:
        t0 = parse_utc(args.start)
        df = df[df["ts"] >= t0]
    if args.end:
        t1 = parse_utc(args.end)
        df = df[df["ts"] < t1]

    df = df.sort_values("ts").reset_index(drop=True)

    # TR/ATR computed on the filtered window (OK for analysis; later we can extend with a warmup)
    df = compute_atr(df, args.atr_len)

    feat = compute_features(df, args.atr_len)

    pd.set_option("display.width", 240)
    pd.set_option("display.max_columns", 120)
    print(feat.tail(10).to_string(index=False))

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        feat.to_parquet(out_path, index=False)
        print(f"\n[OK] wrote: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
