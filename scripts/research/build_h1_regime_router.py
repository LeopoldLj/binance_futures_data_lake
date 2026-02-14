from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Set

import numpy as np
import pandas as pd

# ======================================================================================
# BF Data Lake — Research — H1 Regime Router Builder
# File: scripts/research/build_h1_regime_router.py
#
# But :
# - Charger un parquet M1 enrichi (joined_...__enriched.parquet)
# - Resampler en H1 (anti-lookahead) :
#     * buckets H1 complets
#     * label="left", closed="left"
#     * timestamp H1 = start of bucket (UTC)
# - Calculer features H1:
#     * ATR (SMA TR)
#     * EMA fast/mid/slow sur close H1
#     * slope sur EMA mid (diff sur slope_bars)
#     * volatilité rolling sur returns H1 (std) + percentile chaos
# - Déduire un régime H1:
#     * TREND si trend_score >= trend_th et vol_state != CHAOS
#     * CHAOS si vol_std >= rolling_quantile(vol_chaos_pctl)
#     * sinon RANGE
# - Joindre ces features H1 sur la table M1 par merge_asof (backward) + ffill
#
# Notes :
# - Fix pandas freq : utiliser "1h" (minuscule), pas "1H"
# ======================================================================================

VERSION = "2026-02-14a"


def _parse_set_int(csv: str) -> Set[int]:
    if csv is None or csv.strip() == "":
        return set()
    out: Set[int] = set()
    for part in csv.split(","):
        part = part.strip()
        if part == "":
            continue
        out.add(int(part))
    return out


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False, min_periods=span).mean()


def _compute_atr_sma(df_ohlc: pd.DataFrame, atr_len: int) -> pd.Series:
    prev_close = df_ohlc["close"].shift(1)
    tr1 = (df_ohlc["high"] - df_ohlc["low"]).abs()
    tr2 = (df_ohlc["high"] - prev_close).abs()
    tr3 = (df_ohlc["low"] - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(atr_len, min_periods=atr_len).mean()
    return atr


def _trend_score(close: pd.Series, ema_fast: pd.Series, ema_mid: pd.Series, ema_slow: pd.Series) -> pd.Series:
    # score simple dans [0,1] basé sur l’alignement des EMA + position du close
    s1 = (ema_fast > ema_mid).astype(float)
    s2 = (ema_mid > ema_slow).astype(float)
    s3 = (close > ema_mid).astype(float)
    return (s1 + s2 + s3) / 3.0


def _compute_h1(
    df_m1: pd.DataFrame,
    atr_len: int,
    ema_fast: int,
    ema_mid: int,
    ema_slow: int,
    slope_bars: int,
    vol_window: int,
    vol_chaos_pctl: float,
) -> pd.DataFrame:
    if "t" not in df_m1.columns:
        raise RuntimeError("Missing column 't' (timestamp).")

    tt = pd.to_datetime(df_m1["t"], utc=True, errors="coerce")
    if tt.isna().all():
        raise RuntimeError("Column 't' could not be parsed as UTC datetimes.")

    df = df_m1.copy()
    df["_t"] = tt
    df = df.sort_values("_t").reset_index(drop=True)
    df = df.set_index("_t", drop=False)

    need_cols = ["open", "high", "low", "close", "volume_base", "volume_quote", "n_trades"]
    missing = [c for c in need_cols if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing required columns for resample: {missing}")

    # IMPORTANT: freq "1h" (minuscule) pour compat pandas récente
    ohlc = (
        df[need_cols]
        .resample("1h", label="left", closed="left")
        .agg(
            {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume_base": "sum",
                "volume_quote": "sum",
                "n_trades": "sum",
            }
        )
    )

    # buckets complets uniquement (si open/close NA => bucket incomplet)
    ohlc = ohlc.dropna(subset=["open", "high", "low", "close"]).copy()

    ohlc["atr_h1"] = _compute_atr_sma(ohlc, atr_len=atr_len)

    ohlc["ema_fast_h1"] = _ema(ohlc["close"], span=ema_fast)
    ohlc["ema_mid_h1"] = _ema(ohlc["close"], span=ema_mid)
    ohlc["ema_slow_h1"] = _ema(ohlc["close"], span=ema_slow)

    ohlc["slope_mid_h1"] = (ohlc["ema_mid_h1"] - ohlc["ema_mid_h1"].shift(slope_bars)) / float(max(slope_bars, 1))

    ret = ohlc["close"].pct_change()
    ohlc["vol_std_h1"] = ret.rolling(vol_window, min_periods=vol_window).std(ddof=1)

    # seuil chaos: quantile rolling sur vol_std
    ohlc["vol_chaos_th_h1"] = ohlc["vol_std_h1"].rolling(vol_window, min_periods=vol_window).quantile(vol_chaos_pctl)
    ohlc["is_chaos_h1"] = (ohlc["vol_std_h1"] >= ohlc["vol_chaos_th_h1"]).fillna(False)

    ohlc["trend_score_h1"] = _trend_score(
        close=ohlc["close"],
        ema_fast=ohlc["ema_fast_h1"],
        ema_mid=ohlc["ema_mid_h1"],
        ema_slow=ohlc["ema_slow_h1"],
    )

    # regime label
    ohlc["regime_h1"] = np.where(ohlc["is_chaos_h1"], "CHAOS", "RANGE")

    return ohlc.reset_index(drop=False).rename(columns={"_t": "t_h1"})


def _join_h1_to_m1(df_m1: pd.DataFrame, h1: pd.DataFrame) -> pd.DataFrame:
    out = df_m1.copy()

    out["_t"] = pd.to_datetime(out["t"], utc=True, errors="coerce")
    if out["_t"].isna().all():
        raise RuntimeError("M1 column 't' could not be parsed as UTC datetimes.")

    h1 = h1.copy()
    h1["t_h1"] = pd.to_datetime(h1["t_h1"], utc=True, errors="coerce")

    out = out.sort_values("_t").reset_index(drop=True)
    h1 = h1.sort_values("t_h1").reset_index(drop=True)

    # merge_asof backward: chaque M1 prend la dernière barre H1 close <= t
    joined = pd.merge_asof(out, h1, left_on="_t", right_on="t_h1", direction="backward")

    # ffill des colonnes H1 pour les trous initiaux (optionnel)
    h1_cols = [c for c in h1.columns if c not in ["t_h1"]]
    for c in h1_cols:
        if c in joined.columns:
            joined[c] = joined[c].ffill()

    return joined.drop(columns=["_t"])


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build H1 regime router features and join to M1 parquet.")
    p.add_argument("--parquet", required=True, help="Input M1 enriched parquet (joined_...__enriched.parquet)")
    p.add_argument("--out-parquet", required=True, help="Output M1 parquet with router features (..__router.parquet)")
    p.add_argument("--out-h1-parquet", required=True, help="Output H1 features parquet")
    p.add_argument("--block-hours", default="0,1,14,17,19,23", help="CSV hours UTC to block (stored as metadata cols)")
    p.add_argument("--atr-len", type=int, default=14, help="ATR SMA(TR) length on H1")
    p.add_argument("--ema-fast", type=int, default=20, help="Fast EMA length on H1 close")
    p.add_argument("--ema-mid", type=int, default=50, help="Mid EMA length on H1 close")
    p.add_argument("--ema-slow", type=int, default=200, help="Slow EMA length on H1 close")
    p.add_argument("--slope-bars", type=int, default=8, help="Slope bars on ema_mid_h1")
    p.add_argument("--vol-window", type=int, default=720, help="Rolling window for vol features (H1 bars)")
    p.add_argument("--vol-chaos-pctl", type=float, default=0.80, help="Chaos threshold quantile on vol_std_h1")
    p.add_argument("--trend-th", type=float, default=0.60, help="Trend threshold on trend_score_h1")
    return p.parse_args()


def main() -> int:
    args = _parse_args()

    in_path = Path(args.parquet)
    if not in_path.exists():
        raise FileNotFoundError(f"Input parquet not found: {in_path}")

    out_path = Path(args.out_parquet)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    out_h1_path = Path(args.out_h1_parquet)
    out_h1_path.parent.mkdir(parents=True, exist_ok=True)

    block_hours = sorted(list(_parse_set_int(args.block_hours)))

    atr_len = int(args.atr_len)
    ema_fast = int(args.ema_fast)
    ema_mid = int(args.ema_mid)
    ema_slow = int(args.ema_slow)
    slope_bars = int(args.slope_bars)
    vol_window = int(args.vol_window)
    vol_chaos_pctl = float(args.vol_chaos_pctl)
    trend_th = float(args.trend_th)

    df = pd.read_parquet(in_path)

    # Build H1
    h1 = _compute_h1(
        df_m1=df,
        atr_len=atr_len,
        ema_fast=ema_fast,
        ema_mid=ema_mid,
        ema_slow=ema_slow,
        slope_bars=slope_bars,
        vol_window=vol_window,
        vol_chaos_pctl=vol_chaos_pctl,
    )

    # Router decision at H1 level
    h1["is_trend_h1"] = (h1["trend_score_h1"] >= trend_th) & (~h1["is_chaos_h1"])
    h1["router_mode_h1"] = np.where(h1["is_chaos_h1"], "OFF", np.where(h1["is_trend_h1"], "TREND", "RANGE"))

    # Keep block hours as informational
    h1["block_hours_utc"] = ",".join(str(x) for x in block_hours)

    # Join to M1
    joined = _join_h1_to_m1(df_m1=df, h1=h1)

    # Add convenience columns on M1
    t_m1 = pd.to_datetime(joined["t"], utc=True, errors="coerce")
    joined["hour_utc"] = t_m1.dt.hour
    joined["is_block_hour"] = joined["hour_utc"].isin(block_hours)

    # Write outputs
    h1.to_parquet(out_h1_path, index=False)
    joined.to_parquet(out_path, index=False)

    print(f"[build_h1_regime_router] VERSION={VERSION}")
    print("Config:")
    print(f"  in_parquet={in_path}")
    print(f"  out_parquet={out_path}")
    print(f"  out_h1_parquet={out_h1_path}")
    print(f"  block_hours_utc={block_hours}")
    print(f"  atr_len={atr_len} ema_fast={ema_fast} ema_mid={ema_mid} ema_slow={ema_slow} slope_bars={slope_bars}")
    print(f"  vol_window={vol_window} vol_chaos_pctl={vol_chaos_pctl} trend_th={trend_th}")
    print(f"H1 rows={len(h1)}  M1 rows={len(joined)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
