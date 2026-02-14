from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

# ======================================================================================
# BF Data Lake — Research — Volatility Regime M5 — build_m5_vol_regime.py
#
# But :
# - Construire un dataset M5 enrichi "volatility regime" à partir du RAW M1.
# - Calculer un percentile glissant empirique du range M5 (range_pctl) sans fuite future.
# - Produire des états LOW/MID/HIGH et un filtre market_ready exploitable par le router.
#
# Anti-lookahead (strict) :
# - Buckets M5 resample("5min", label="left", closed="left") => ts = bucket_start (UTC)
# - Donnée M5 utilisable uniquement après clôture + safe lag :
#     * bucket_end       = ts + 5min
#     * available_from   = bucket_end + lag_minutes (default: 2)
#
# Entrées :
# - RAW M1 partitionné :
#   data/raw/binance_um/klines_m1/symbol=XXX/year=YYYY/month=MM/part-000.parquet
#
# Sorties :
# - Optionnel via --out (parquet) : un fichier M5 avec colonnes régime ajoutées
#
# Notes perf :
# - Percentile EXACT (rolling.apply) => OK research / backtest.
# - Pour prod long historique, on optimisera (histogramme glissant / t-digest).
#
# Compat :
# - Évite astype("string") (certaines versions pandas ne supportent pas).
# ======================================================================================


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build M5 vol regime (percentile range + states), anti-lookahead.")
    p.add_argument("--root", required=True, help=r"Root folder like ...\data\raw\binance_um\klines_m1")
    p.add_argument("--symbol", required=True)
    p.add_argument("--year", type=int, required=True)
    p.add_argument("--month", type=int, required=True)
    p.add_argument("--start", default=None, help='UTC, e.g. "2026-02-10 06:00:00"')
    p.add_argument("--end", default=None, help='UTC, e.g. "2026-02-10 07:00:00"')
    p.add_argument("--ts-col", default="ts", help="Timestamp column name in parquet (default: ts)")
    p.add_argument("--lag-minutes", type=int, default=2, help="Safe lag minutes for availability (default: 2)")

    p.add_argument("--lookback", type=int, default=288, help="Rolling lookback in M5 bars (default: 288 = 1 day)")
    p.add_argument("--min-lookback", type=int, default=3, help="Minimum allowed lookback for tests (default: 3)")
    p.add_argument("--p-low", type=float, default=0.25, help="LOW threshold (default: 0.25)")
    p.add_argument("--p-high", type=float, default=0.70, help="HIGH threshold (default: 0.70)")
    p.add_argument("--min-range", type=float, default=0.0, help="Min absolute range (default: 0.0)")
    p.add_argument("--min-trades", type=int, default=0, help="Min trades per M5 (default: 0)")

    p.add_argument("--out", default=None, help="Optional output parquet path")
    return p.parse_args()


def parse_utc(ts: str) -> pd.Timestamp:
    t = pd.Timestamp(ts)
    if t.tzinfo is None:
        t = t.tz_localize("UTC")
    else:
        t = t.tz_convert("UTC")
    return t


def list_month_parts(root: Path, symbol: str, year: int, month: int) -> List[Path]:
    base = root / f"symbol={symbol}" / f"year={year:04d}" / f"month={month:02d}"
    if not base.exists():
        raise FileNotFoundError(f"Month dir not found: {base}")
    parts = sorted(base.glob("part-*.parquet"))
    if not parts:
        raise FileNotFoundError(f"No parquet parts found in: {base}")
    return parts


def load_month(root: Path, symbol: str, year: int, month: int) -> pd.DataFrame:
    parts = list_month_parts(root, symbol, year, month)
    dfs: List[pd.DataFrame] = []
    for p in parts:
        d = pd.read_parquet(p)
        for col in ["symbol", "exchange", "market"]:
            if col in d.columns:
                d[col] = d[col].astype("object")
        dfs.append(d)
    return pd.concat(dfs, ignore_index=True)


def ensure_types_m1(df: pd.DataFrame, ts_col: str) -> pd.DataFrame:
    if ts_col not in df.columns:
        raise KeyError(f"Missing required timestamp column: {ts_col}")

    out = df.copy()
    out[ts_col] = pd.to_datetime(out[ts_col], utc=True, errors="raise")

    for c in ["open", "high", "low", "close", "volume_base", "volume_quote", "taker_buy_base", "taker_buy_quote"]:
        if c in out.columns:
            out[c] = out[c].astype("float64")

    if "n_trades" in out.columns:
        out["n_trades"] = out["n_trades"].astype("int64")

    return out


def build_m5_from_m1(df_m1: pd.DataFrame, ts_col: str, lag_minutes: int) -> pd.DataFrame:
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
    m5 = df.resample("5min", label="left", closed="left").agg(agg).join(count_m1)

    need_ohlc = [c for c in ["open", "high", "low", "close"] if c in m5.columns]
    if need_ohlc:
        m5 = m5.dropna(subset=need_ohlc)

    m5 = m5.reset_index().rename(columns={ts_col: "ts"})
    m5["range"] = m5["high"] - m5["low"]

    m5["bucket_start"] = m5["ts"]
    m5["bucket_end"] = m5["ts"] + pd.Timedelta(minutes=5)
    m5["available_from"] = m5["bucket_end"] + pd.Timedelta(minutes=int(lag_minutes))

    return m5


def _rolling_percentile_last(window: np.ndarray) -> float:
    last = window[-1]
    return float(np.mean(window <= last))


def add_vol_regime(
    m5: pd.DataFrame,
    lookback: int,
    min_lookback: int,
    p_low: float,
    p_high: float,
    min_range: float,
    min_trades: int,
) -> pd.DataFrame:
    out = m5.sort_values("ts").copy()

    if lookback < int(min_lookback):
        raise ValueError(f"lookback too small; use >= {min_lookback}")

    if lookback < 20:
        print(f"[WARN] lookback={lookback} is very small (test-only). For research use >= 288 when possible.")

    out["range_pctl"] = (
        out["range"]
        .rolling(window=lookback, min_periods=lookback)
        .apply(lambda x: _rolling_percentile_last(np.asarray(x, dtype="float64")), raw=False)
        .astype("float64")
    )

    vol_state = np.where(
        out["range_pctl"].isna(),
        "NA",
        np.where(out["range_pctl"] < p_low, "LOW", np.where(out["range_pctl"] > p_high, "HIGH", "MID")),
    )
    out["vol_state"] = pd.Series(vol_state, index=out.index).astype("object")

    trades = out["n_trades"] if "n_trades" in out.columns else pd.Series(index=out.index, data=np.nan)
    out["market_ready"] = (
        (~out["range_pctl"].isna())
        & (out["vol_state"] != "LOW")
        & (out["range"] >= float(min_range))
        & (trades.fillna(0).astype("float64") >= float(min_trades))
    )

    return out


def main() -> int:
    args = _parse_args()

    root = Path(args.root)
    df = load_month(root, args.symbol, args.year, args.month)
    df = ensure_types_m1(df, args.ts_col)

    if args.start:
        t0 = parse_utc(args.start)
        df = df[df[args.ts_col] >= t0]
    if args.end:
        t1 = parse_utc(args.end)
        df = df[df[args.ts_col] < t1]

    m5 = build_m5_from_m1(df, args.ts_col, int(args.lag_minutes))
    print(f"[INFO] M5 bars available after slicing: {len(m5)}")

    m5v = add_vol_regime(
        m5=m5,
        lookback=int(args.lookback),
        min_lookback=int(args.min_lookback),
        p_low=float(args.p_low),
        p_high=float(args.p_high),
        min_range=float(args.min_range),
        min_trades=int(args.min_trades),
    )

    pd.set_option("display.width", 240)
    pd.set_option("display.max_columns", 200)
    cols_show = [
        "ts",
        "range",
        "range_pctl",
        "vol_state",
        "market_ready",
        "n_trades",
        "count_m1",
        "bucket_end",
        "available_from",
    ]
    cols_show = [c for c in cols_show if c in m5v.columns]
    print(m5v.tail(20)[cols_show].to_string(index=False))

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        m5v.to_parquet(out_path, index=False)
        print(f"\n[OK] wrote: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
