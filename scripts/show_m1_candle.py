# scripts/show_m1_candle.py
# Usage:
#   python scripts/show_m1_candle.py --root "C:\Users\lolo_\PycharmProjects\binance_futures_data_lake\data\raw\binance_um\klines_m1" --symbol BTCUSDT --ts "2026-02-10 06:43:00"

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _parse_ts(ts_str: str) -> pd.Timestamp:
    ts = pd.Timestamp(ts_str)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts.floor("min")


def _month_path(root: Path, symbol: str, ts_utc: pd.Timestamp) -> Path:
    y = ts_utc.year
    m = ts_utc.month
    return root / f"symbol={symbol}" / f"year={y:04d}" / f"month={m:02d}" / "part-000.parquet"


def _load_month_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Parquet introuvable: {path}")
    df = pd.read_parquet(path)
    if "ts" not in df.columns:
        raise ValueError("Colonne 'ts' introuvable dans le parquet.")
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df = df.sort_values("ts").reset_index(drop=True)
    return df


def _fmt_float(x) -> str:
    if pd.isna(x):
        return "na"
    try:
        return f"{float(x):,.8f}".replace(",", " ")
    except Exception:
        return str(x)


def _fmt_int(x) -> str:
    if pd.isna(x):
        return "na"
    try:
        return f"{int(x)}"
    except Exception:
        return str(x)


def _get_float(row: pd.Series, key: str) -> float | pd._libs.missing.NAType:
    v = row.get(key, pd.NA)
    if pd.isna(v):
        return pd.NA
    return float(v)


def show_one_candle(df: pd.DataFrame, ts_utc: pd.Timestamp) -> None:
    idx = df.index[df["ts"] == ts_utc]
    if len(idx) == 0:
        prev_ts = df.loc[df["ts"] < ts_utc, "ts"].max()
        next_ts = df.loc[df["ts"] > ts_utc, "ts"].min()
        raise ValueError(
            f"Bougie introuvable pour ts={ts_utc}.\n"
            f"Plus proche avant: {prev_ts}\n"
            f"Plus proche après: {next_ts}"
        )

    i = int(idx[0])
    row = df.iloc[i].copy()

    prev_close = df.iloc[i - 1]["close"] if i > 0 else pd.NA
    prev_ts = df.iloc[i - 1]["ts"] if i > 0 else pd.NA

    # Natives
    o = float(row["open"])
    h = float(row["high"])
    l = float(row["low"])
    c = float(row["close"])

    # volume_base fallback: parfois les datasets ont "volume" au lieu de "volume_base"
    vol_base = row.get("volume_base", row.get("volume", pd.NA))
    vol_base = float(vol_base) if not pd.isna(vol_base) else pd.NA

    vol_quote = _get_float(row, "volume_quote")
    n_trades = row.get("n_trades", pd.NA)
    taker_buy_base = _get_float(row, "taker_buy_base")
    taker_buy_quote = _get_float(row, "taker_buy_quote")

    # Derived: price structure
    rng = h - l
    body = abs(c - o)
    upper_wick = h - max(o, c)
    lower_wick = min(o, c) - l

    body_pct = (body / rng) if rng != 0 else pd.NA
    upper_wick_pct = (upper_wick / rng) if rng != 0 else pd.NA
    lower_wick_pct = (lower_wick / rng) if rng != 0 else pd.NA

    # Returns
    if pd.isna(prev_close) or float(prev_close) == 0.0:
        ret = pd.NA
        logret = pd.NA
        gap = pd.NA
    else:
        prev_close_f = float(prev_close)
        ret = (c / prev_close_f) - 1.0
        logret = float(np.log(c / prev_close_f))
        gap = o - prev_close_f

    # Microstructure (taker flow)
    if pd.isna(vol_base) or vol_base == 0:
        buy_ratio = pd.NA
        taker_sell_base = pd.NA
        delta_base = pd.NA
        avg_trade_size = pd.NA
    else:
        taker_sell_base = vol_base - taker_buy_base if not pd.isna(taker_buy_base) else pd.NA
        buy_ratio = (taker_buy_base / vol_base) if not pd.isna(taker_buy_base) else pd.NA
        delta_base = (taker_buy_base - taker_sell_base) if not pd.isna(taker_sell_base) else pd.NA
        if pd.isna(n_trades) or int(n_trades) == 0:
            avg_trade_size = pd.NA
        else:
            avg_trade_size = vol_base / float(n_trades)

    if pd.isna(vol_quote) or vol_quote == 0:
        taker_sell_quote = pd.NA
        delta_quote = pd.NA
    else:
        taker_sell_quote = vol_quote - taker_buy_quote if not pd.isna(taker_buy_quote) else pd.NA
        delta_quote = (taker_buy_quote - taker_sell_quote) if not pd.isna(taker_sell_quote) else pd.NA

    direction = "UP" if c > o else "DOWN" if c < o else "FLAT"

    open_time_ms = row.get("open_time_ms", pd.NA)
    close_time_ms = row.get("close_time_ms", pd.NA)

    symbol = row.get("symbol", "na")
    exchange = row.get("exchange", "na")
    market = row.get("market", "na")

    print("\n=== M1 Candle ===")
    print(f"symbol      : {symbol}")
    print(f"exchange    : {exchange}")
    print(f"market      : {market}")
    print(f"ts (UTC)    : {row['ts']}")
    print(f"direction   : {direction}")
    print()

    print("— Natives (Binance / Parquet) —")
    print(f"open/high/low/close : {_fmt_float(o)} / {_fmt_float(h)} / {_fmt_float(l)} / {_fmt_float(c)}")
    print(f"volume_base         : {_fmt_float(vol_base)}")
    print(f"volume_quote        : {_fmt_float(vol_quote)}")
    print(f"n_trades            : {_fmt_int(n_trades)}")
    print(f"taker_buy_base      : {_fmt_float(taker_buy_base)}")
    print(f"taker_buy_quote     : {_fmt_float(taker_buy_quote)}")
    print(f"open_time_ms        : {_fmt_int(open_time_ms)}")
    print(f"close_time_ms       : {_fmt_int(close_time_ms)}")
    print()

    print("— Dérivées (structure prix) —")
    print(f"range               : {_fmt_float(rng)}")
    print(f"body                : {_fmt_float(body)}")
    print(f"upper_wick          : {_fmt_float(upper_wick)}")
    print(f"lower_wick          : {_fmt_float(lower_wick)}")
    print(f"body_pct            : {_fmt_float(body_pct)}")
    print(f"upper_wick_pct      : {_fmt_float(upper_wick_pct)}")
    print(f"lower_wick_pct      : {_fmt_float(lower_wick_pct)}")
    print()

    print("— Dérivées (retours) —")
    print(f"prev_ts             : {prev_ts}")
    print(f"prev_close          : {_fmt_float(prev_close)}")
    print(f"gap (open-prevC)    : {_fmt_float(gap)}")
    print(f"ret                 : {_fmt_float(ret)}")
    print(f"logret              : {_fmt_float(logret)}")
    print()

    print("— Dérivées (taker flow / microstructure) —")
    print(f"taker_sell_base     : {_fmt_float(taker_sell_base)}")
    print(f"delta_base          : {_fmt_float(delta_base)}")
    print(f"buy_ratio           : {_fmt_float(buy_ratio)}")
    print(f"taker_sell_quote    : {_fmt_float(taker_sell_quote)}")
    print(f"delta_quote         : {_fmt_float(delta_quote)}")
    print(f"avg_trade_size      : {_fmt_float(avg_trade_size)}")
    print()


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True, help="Chemin racine klines_m1 (ex: .../data/raw/binance_um/klines_m1)")
    p.add_argument("--symbol", required=True, help="Ex: BTCUSDT ou ETHUSDT")
    p.add_argument("--ts", required=True, help="Timestamp UTC open_time (ex: '2026-02-10 06:43:00')")
    args = p.parse_args()

    root = Path(args.root)
    symbol = args.symbol.strip().upper()
    ts_utc = _parse_ts(args.ts)

    month_file = _month_path(root, symbol, ts_utc)
    df = _load_month_parquet(month_file)
    show_one_candle(df, ts_utc)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
