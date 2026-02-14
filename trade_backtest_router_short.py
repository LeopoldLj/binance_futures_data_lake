from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


# ======================================================================================
# BF Data Lake — SHORT Backtest with / without H1 Router
# File: trade_backtest_router_short.py
#
# - Entry = open(i+1)
# - SL = entry + sl_k * ATR
# - TP = entry - tp_r * risk
# - TIME exit at H
# - Intrabar: SL priority if both-touch
# ======================================================================================


def max_drawdown(equity: pd.Series) -> float:
    roll_max = equity.cummax()
    dd = equity - roll_max
    return float(dd.min())


def profit_factor(R: pd.Series) -> float:
    pos = R[R > 0].sum()
    neg = -R[R < 0].sum()
    return float(pos / neg) if neg > 0 else float("inf")


def backtest(df: pd.DataFrame, H: int, sl_k: float, tp_r: float, use_router: bool) -> Dict:

    sig = (df["is_add"] == True)

    if use_router:
        sig = sig & (df["router_allow_short"] == True) & (df["router_veto"] == False)

    sig_idx = np.flatnonzero(sig.values)
    sig_idx = [i for i in sig_idx if (i + 1) < len(df) and (i + H) < len(df)]

    Rs: List[float] = []

    for i in sig_idx:

        entry_i = i + 1
        entry = df.loc[entry_i, "open"]
        atr = df.loc[entry_i, "atr"]

        if pd.isna(atr) or atr <= 0:
            continue

        sl = entry + sl_k * atr
        risk = sl - entry
        tp = entry - tp_r * risk

        exit_px = None

        for j in range(entry_i, entry_i + H):
            hi = df.loc[j, "high"]
            lo = df.loc[j, "low"]

            hit_sl = hi >= sl
            hit_tp = lo <= tp

            if hit_sl and hit_tp:
                exit_px = sl
                break
            if hit_sl:
                exit_px = sl
                break
            if hit_tp:
                exit_px = tp
                break

        if exit_px is None:
            exit_px = df.loc[entry_i + H - 1, "close"]

        pnl = entry - exit_px
        R = pnl / risk
        Rs.append(R)

    R_series = pd.Series(Rs, dtype="float64")

    equity = R_series.cumsum()

    return {
        "n": len(R_series),
        "mean_R": float(R_series.mean()),
        "wr": float((R_series > 0).mean()),
        "pf": profit_factor(R_series),
        "max_dd": max_drawdown(equity),
        "equity": equity,
        "R_series": R_series,
    }


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", required=True)
    parser.add_argument("--H", type=int, default=16)
    parser.add_argument("--sl-k", type=float, default=1.0)
    parser.add_argument("--tp-r", type=float, default=1.75)
    parser.add_argument("--out-dir", required=True)

    args = parser.parse_args()

    df = pd.read_parquet(args.parquet)

    # --- ATR (simple SMA TR)
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        (df["high"] - df["low"]).abs(),
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs()
    ], axis=1).max(axis=1)

    df["atr"] = tr.rolling(14, min_periods=14).mean()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    res_no_router = backtest(df, args.H, args.sl_k, args.tp_r, use_router=False)
    res_router = backtest(df, args.H, args.sl_k, args.tp_r, use_router=True)

    summary = pd.DataFrame([
        {"version": "no_router", **{k: v for k, v in res_no_router.items() if k not in ["equity", "R_series"]}},
        {"version": "with_router", **{k: v for k, v in res_router.items() if k not in ["equity", "R_series"]}},
    ])

    summary.to_csv(out_dir / "summary_router_compare.csv", index=False)
    res_no_router["equity"].to_csv(out_dir / "equity_no_router.csv", index=False)
    res_router["equity"].to_csv(out_dir / "equity_with_router.csv", index=False)

    print("\n=== SUMMARY ===")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
