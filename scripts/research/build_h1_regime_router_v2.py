# build_h1_regime_router_v2.py
# VERSION=2026-02-14b
#
# FIX: pandas de ton env refuse "1H" et demande "1h".
# Anti-lookahead strict: shift(1) des colonnes H1 décisionnelles avant merge_asof(backward)

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd


def _ensure_utc_datetime(s: pd.Series) -> pd.Series:
    x = pd.to_datetime(s, utc=True, errors="coerce")
    if x.isna().any():
        bad = int(x.isna().sum())
        raise ValueError(f"[KO] {bad} timestamps invalides après to_datetime(utc=True, errors='coerce').")
    return x


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False, min_periods=span).mean()


def true_range(h: pd.Series, l: pd.Series, c_prev: pd.Series) -> pd.Series:
    tr1 = h - l
    tr2 = (h - c_prev).abs()
    tr3 = (l - c_prev).abs()
    return pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)


def atr(df: pd.DataFrame, length: int) -> pd.Series:
    c_prev = df["close"].shift(1)
    tr = true_range(df["high"], df["low"], c_prev)
    return tr.ewm(alpha=1 / length, adjust=False, min_periods=length).mean()


def rolling_percentile(series: pd.Series, window: int) -> pd.Series:
    def _pctl(x: np.ndarray) -> float:
        last = x[-1]
        return float((x <= last).sum() / len(x) * 100.0)

    return series.rolling(window=window, min_periods=window).apply(_pctl, raw=True)


def efficiency_ratio(close: pd.Series, window: int) -> pd.Series:
    net = (close - close.shift(window)).abs()
    dif = close.diff().abs()
    den = dif.rolling(window=window, min_periods=window).sum()
    return net / den.replace(0.0, np.nan)


def slope_per_bar(series: pd.Series, window: int) -> pd.Series:
    return (series - series.shift(window)) / float(window)


@dataclass(frozen=True)
class RouterParams:
    atr_len: int
    ema20: int
    ema50: int
    ema200: int
    er_lb: int
    atrpctl_lb: int
    p_hi: float
    er_low: float
    er_high: float
    slope_lb: int
    slope_hi: float


def build_h1_bars(m1: pd.DataFrame, ts_col: str) -> pd.DataFrame:
    # IMPORTANT: freq en "1h" (ton pandas refuse "1H")
    m1 = m1.sort_values(ts_col).copy()
    m1 = m1.set_index(ts_col)

    agg = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume_base": "sum",
        "volume_quote": "sum",
        "n_trades": "sum",
    }
    agg = {k: v for k, v in agg.items() if k in m1.columns}

    h1 = m1.resample("1h", label="right", closed="right").agg(agg)
    h1 = h1.dropna(subset=["open", "high", "low", "close"], how="any")
    h1 = h1.reset_index().rename(columns={ts_col: "t_h1"})
    return h1


def compute_router_features(h1: pd.DataFrame, p: RouterParams) -> pd.DataFrame:
    h1 = h1.sort_values("t_h1").copy()

    h1["atr_h1"] = atr(h1.rename(columns={"t_h1": "t"}), p.atr_len)
    h1["atr_pct_h1"] = h1["atr_h1"] / h1["close"].replace(0.0, np.nan)

    h1["ema20_h1"] = ema(h1["close"], p.ema20)
    h1["ema50_h1"] = ema(h1["close"], p.ema50)
    h1["ema200_h1"] = ema(h1["close"], p.ema200)

    h1["aligned_bull_h1"] = (h1["ema20_h1"] > h1["ema50_h1"]) & (h1["ema50_h1"] > h1["ema200_h1"])
    h1["aligned_bear_h1"] = (h1["ema20_h1"] < h1["ema50_h1"]) & (h1["ema50_h1"] < h1["ema200_h1"])

    h1["er_h1"] = efficiency_ratio(h1["close"], p.er_lb)

    h1["slope50_h1"] = slope_per_bar(h1["ema50_h1"], p.slope_lb)
    h1["slope50_norm_h1"] = h1["slope50_h1"] / h1["atr_h1"].replace(0.0, np.nan)

    h1["atr_pct_pctl_h1"] = rolling_percentile(h1["atr_pct_h1"], p.atrpctl_lb)

    h1["trend_score_h1"] = (
        0.50 * h1["er_h1"].clip(0, 1)
        + 0.30 * (h1["slope50_norm_h1"].abs().clip(0, 5) / 5.0)
        + 0.20 * (h1["aligned_bull_h1"].astype(float) + h1["aligned_bear_h1"].astype(float))
    )

    is_chaos = (h1["atr_pct_pctl_h1"] >= p.p_hi) & (h1["er_h1"] <= p.er_low)
    is_trend = (h1["er_h1"] >= p.er_high) & (h1["aligned_bull_h1"] | h1["aligned_bear_h1"] | (h1["slope50_norm_h1"].abs() >= p.slope_hi))
    regime = np.where(is_chaos, "CHAOS", np.where(is_trend, "TREND", "RANGE"))

    h1["regime_h1_raw"] = regime
    h1["is_chaos_h1_raw"] = is_chaos
    h1["is_trend_h1_raw"] = is_trend
    h1["router_mode_h1_raw"] = np.where(h1["regime_h1_raw"] == "CHAOS", "OFF", np.where(h1["regime_h1_raw"] == "TREND", "TREND", "RANGE"))

    return h1


def shift_h1_decision_cols(h1: pd.DataFrame) -> pd.DataFrame:
    h1 = h1.sort_values("t_h1").copy()

    decision_cols = [
        "regime_h1_raw",
        "is_chaos_h1_raw",
        "is_trend_h1_raw",
        "router_mode_h1_raw",
        "trend_score_h1",
        "atr_h1",
        "atr_pct_h1",
        "atr_pct_pctl_h1",
        "ema20_h1",
        "ema50_h1",
        "ema200_h1",
        "er_h1",
        "slope50_norm_h1",
        "aligned_bull_h1",
        "aligned_bear_h1",
    ]
    existing = [c for c in decision_cols if c in h1.columns]
    for c in existing:
        h1[c] = h1[c].shift(1)

    h1 = h1.rename(
        columns={
            "regime_h1_raw": "regime_h1",
            "is_chaos_h1_raw": "is_chaos_h1",
            "is_trend_h1_raw": "is_trend_h1",
            "router_mode_h1_raw": "router_mode_h1",
        }
    )
    return h1


def merge_router_to_m1(m1: pd.DataFrame, h1: pd.DataFrame, ts_col: str) -> pd.DataFrame:
    m1 = m1.sort_values(ts_col).copy()
    h1 = h1.sort_values("t_h1").copy()

    m1["hour_utc"] = pd.to_datetime(m1[ts_col], utc=True).dt.hour.astype(int)

    out = pd.merge_asof(
        m1,
        h1,
        left_on=ts_col,
        right_on="t_h1",
        direction="backward",
        allow_exact_matches=True,
    )
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--ts-col", default="t")
    ap.add_argument("--atr-len", type=int, default=14)
    ap.add_argument("--ema20", type=int, default=20)
    ap.add_argument("--ema50", type=int, default=50)
    ap.add_argument("--ema200", type=int, default=200)
    ap.add_argument("--h1-er-lb", type=int, default=24)
    ap.add_argument("--atrpctl-lb", type=int, default=48)
    ap.add_argument("--p-hi", type=float, default=80.0)
    ap.add_argument("--er-low", type=float, default=0.20)
    ap.add_argument("--er-high", type=float, default=0.55)
    ap.add_argument("--slope-lb", type=int, default=6)
    ap.add_argument("--slope-hi", type=float, default=0.30)
    args = ap.parse_args()

    print("[build_h1_regime_router_v2] VERSION=2026-02-14b")

    m1 = pd.read_parquet(args.input)
    if args.ts_col not in m1.columns:
        raise ValueError(f"[KO] ts-col '{args.ts_col}' introuvable.")

    m1 = m1.copy()
    m1[args.ts_col] = _ensure_utc_datetime(m1[args.ts_col])

    p = RouterParams(
        atr_len=args.atr_len,
        ema20=args.ema20,
        ema50=args.ema50,
        ema200=args.ema200,
        er_lb=args.h1_er_lb,
        atrpctl_lb=args.atrpctl_lb,
        p_hi=args.p_hi,
        er_low=args.er_low,
        er_high=args.er_high,
        slope_lb=args.slope_lb,
        slope_hi=args.slope_hi,
    )

    h1 = build_h1_bars(m1, ts_col=args.ts_col)
    print(f"[INFO] H1 bars: {len(h1):,}")

    h1f = compute_router_features(h1, p)
    h1s = shift_h1_decision_cols(h1f)

    keep_cols = [
        "t_h1",
        "regime_h1",
        "is_chaos_h1",
        "is_trend_h1",
        "router_mode_h1",
        "trend_score_h1",
        "atr_h1",
        "atr_pct_h1",
        "atr_pct_pctl_h1",
        "er_h1",
        "slope50_norm_h1",
        "ema20_h1",
        "ema50_h1",
        "ema200_h1",
        "aligned_bull_h1",
        "aligned_bear_h1",
    ]
    keep_cols = [c for c in keep_cols if c in h1s.columns]
    h1s = h1s[keep_cols]

    out = merge_router_to_m1(m1, h1s, ts_col=args.ts_col)

    if "t_h1" in out.columns:
        bad = int((out["t_h1"] > out[args.ts_col]).sum())
        if bad > 0:
            raise RuntimeError(f"[KO] merge_asof incohérent: {bad} lignes avec t_h1 > t_m1")

    out.to_parquet(args.output, index=False)
    print(f"[OK] wrote: {args.output}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as e:
        print(f"[ERROR] {type(e).__name__}: {e}", file=sys.stderr)
        raise
