from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


VERSION = "2026-02-14-plot-weekly-candles-v1"


def auto_detect_ts_col(df: pd.DataFrame) -> str:
    cols = df.columns.tolist()
    for c in ["t_x", "ts", "timestamp", "time", "datetime", "open_time", "open_ts", "time_utc", "ts_utc", "t"]:
        if c in cols:
            return c
    dt_cols = [c for c in cols if str(df[c].dtype).startswith("datetime64")]
    if dt_cols:
        return dt_cols[0]
    raise RuntimeError("Cannot auto-detect timestamp column.")


def normalize_ts_series(s: pd.Series) -> pd.Series:
    if str(s.dtype).startswith("datetime64"):
        return s
    if np.issubdtype(s.dtype, np.integer):
        return pd.to_datetime(s, unit="ms", utc=True).dt.tz_convert(None)
    return pd.to_datetime(s, utc=True, errors="coerce").dt.tz_convert(None)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot weekly candles from base parquet.")
    p.add_argument("--parquet-in", required=True, help="Input parquet with at least ts/open/high/low/close.")
    p.add_argument("--output", required=True, help="Output PNG path.")
    p.add_argument("--start", default=None, help="Optional UTC start timestamp.")
    p.add_argument("--end", default=None, help="Optional UTC end timestamp.")
    p.add_argument("--max-weeks", type=int, default=400, help="Display last N weekly candles.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    in_path = Path(args.parquet_in)
    out_path = Path(args.output)
    if not in_path.exists():
        raise FileNotFoundError(f"Parquet not found: {in_path}")

    df0 = pd.read_parquet(in_path, engine="pyarrow")
    ts_col = auto_detect_ts_col(df0)
    need = [ts_col, "open", "high", "low", "close"]
    missing = [c for c in need if c not in df0.columns]
    if missing:
        raise RuntimeError(f"Missing columns: {missing}")
    df = df0[need].copy()
    df[ts_col] = normalize_ts_series(df[ts_col])
    df = df.sort_values(ts_col).dropna(subset=[ts_col])

    if args.start:
        t0 = pd.Timestamp(args.start)
        t0 = t0.tz_localize("UTC") if t0.tzinfo is None else t0.tz_convert("UTC")
        df = df[df[ts_col] >= t0].copy()
    if args.end:
        t1 = pd.Timestamp(args.end)
        t1 = t1.tz_localize("UTC") if t1.tzinfo is None else t1.tz_convert("UTC")
        df = df[df[ts_col] < t1].copy()

    w = (
        df.set_index(ts_col)
        .resample("W-MON", label="left", closed="left")
        .agg(open=("open", "first"), high=("high", "max"), low=("low", "min"), close=("close", "last"))
        .dropna()
        .reset_index()
        .rename(columns={ts_col: "week_ts"})
    )
    if len(w) == 0:
        raise RuntimeError("No weekly candles after filtering.")

    if args.max_weeks > 0 and len(w) > args.max_weeks:
        w = w.tail(args.max_weeks).reset_index(drop=True)

    x = mdates.date2num(pd.to_datetime(w["week_ts"]))
    o = w["open"].to_numpy(dtype=float)
    h = w["high"].to_numpy(dtype=float)
    l = w["low"].to_numpy(dtype=float)
    c = w["close"].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(16, 8))
    width = 3.5  # days

    for i in range(len(w)):
        color = "#1f9d55" if c[i] >= o[i] else "#d64545"
        ax.vlines(x[i], l[i], h[i], color=color, linewidth=1.0, alpha=0.9)
        bottom = min(o[i], c[i])
        height = max(abs(c[i] - o[i]), 1e-9)
        rect = plt.Rectangle((x[i] - width / 2.0, bottom), width, height, facecolor=color, edgecolor=color, alpha=0.8)
        ax.add_patch(rect)

    ax.set_title(f"BTCUSDT Weekly Candles ({len(w)} bars) | {VERSION}")
    ax.set_ylabel("Price")
    ax.grid(True, alpha=0.2)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.autofmt_xdate()
    plt.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[OK] ts_col={ts_col} weekly_bars={len(w)} wrote={out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

