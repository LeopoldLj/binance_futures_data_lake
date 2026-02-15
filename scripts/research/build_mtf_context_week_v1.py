from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd


VERSION = "2026-02-14-mtf-week-v1"
EPS = 1e-12


def auto_detect_ts_col(df: pd.DataFrame) -> str:
    cols = df.columns.tolist()
    for c in ["t_x", "ts", "timestamp", "time", "datetime", "open_time", "open_ts", "time_utc", "ts_utc", "t"]:
        if c in cols:
            return c
    dt_cols = [c for c in cols if str(df[c].dtype).startswith("datetime64")]
    if dt_cols:
        return dt_cols[0]
    raise RuntimeError("Cannot auto-detect timestamp column. Use --ts-col.")


def normalize_ts_series(s: pd.Series) -> pd.Series:
    if str(s.dtype).startswith("datetime64"):
        return s
    if np.issubdtype(s.dtype, np.integer):
        return pd.to_datetime(s, unit="ms", utc=True).dt.tz_convert(None)
    return pd.to_datetime(s, utc=True, errors="coerce").dt.tz_convert(None)


def detect_volume_col(df: pd.DataFrame) -> str:
    for c in ["volume", "volume_base_x", "volume_base", "base_volume", "volume_base_y"]:
        if c in df.columns:
            return c
    raise RuntimeError("No volume column found.")


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    return pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)


def build_weekly_ohlcv(df: pd.DataFrame, ts_col: str, vol_col: str) -> pd.DataFrame:
    x = df.sort_values(ts_col).set_index(ts_col)
    agg = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        vol_col: "sum",
    }
    if "market_buys" in x.columns:
        agg["market_buys"] = "sum"
    if "market_sells" in x.columns:
        agg["market_sells"] = "sum"
    if "taker_buy_base" in x.columns:
        agg["taker_buy_base"] = "sum"
    if "delta_norm" in x.columns:
        agg["delta_norm"] = "mean"
    m = x.resample("W-MON", label="left", closed="left").agg(agg)
    m = m.dropna(subset=["open", "high", "low", "close"]).reset_index().rename(columns={ts_col: "bar_ts", vol_col: "volume"})
    return m


def add_week_features(m: pd.DataFrame, swing: int, bb_len: int, kc_len: int) -> pd.DataFrame:
    out = m.copy()

    # Candle geometry
    rng = (out["high"] - out["low"]).replace(0.0, np.nan)
    out["ret1"] = out["close"].pct_change()
    out["range_pct"] = (out["high"] - out["low"]) / out["close"].replace(0.0, np.nan)
    out["body_pct"] = (out["close"] - out["open"]).abs() / rng
    out["upper_wick_pct"] = (out["high"] - out[["open", "close"]].max(axis=1)) / rng
    out["lower_wick_pct"] = (out[["open", "close"]].min(axis=1) - out["low"]) / rng
    out["close_pos"] = (out["close"] - out["low"]) / rng

    # Volatility
    tr = true_range(out["high"], out["low"], out["close"])
    out["atr14"] = tr.ewm(alpha=1.0 / 14.0, adjust=False, min_periods=14).mean()
    out["atr_pct"] = out["atr14"] / out["close"].replace(0.0, np.nan)

    # EMA structure
    out["ema20"] = out["close"].ewm(span=20, adjust=False, min_periods=20).mean()
    out["ema50"] = out["close"].ewm(span=50, adjust=False, min_periods=50).mean()
    out["ema200"] = out["close"].ewm(span=200, adjust=False, min_periods=200).mean()
    out["ema20_50_spread"] = out["ema20"] - out["ema50"]
    out["ema50_200_spread"] = out["ema50"] - out["ema200"]
    out["ema20_slope"] = out["ema20"].pct_change()
    out["ema50_slope"] = out["ema50"].pct_change()

    # Anchored VWAP (cumulative)
    tp = (out["high"] + out["low"] + out["close"]) / 3.0
    pv = tp * out["volume"].fillna(0.0)
    cum_pv = pv.cumsum()
    cum_v = out["volume"].fillna(0.0).cumsum()
    out["vwap"] = cum_pv / cum_v.replace(0.0, np.nan)
    out["vwap_dist"] = (out["close"] - out["vwap"]) / out["vwap"].replace(0.0, np.nan)
    mu = out["vwap_dist"].rolling(24, min_periods=12).mean()
    sd = out["vwap_dist"].rolling(24, min_periods=12).std(ddof=0)
    out["vwap_dist_z"] = (out["vwap_dist"] - mu) / (sd + EPS)

    # BB/KC compression
    bb_mid = out["close"].rolling(bb_len, min_periods=bb_len).mean()
    bb_std = out["close"].rolling(bb_len, min_periods=bb_len).std(ddof=0)
    bb_u = bb_mid + 2.0 * bb_std
    bb_l = bb_mid - 2.0 * bb_std
    out["bb_width"] = (bb_u - bb_l) / bb_mid.replace(0.0, np.nan)

    kc_mid = out["close"].ewm(span=kc_len, adjust=False, min_periods=kc_len).mean()
    kc_atr = tr.ewm(alpha=1.0 / float(kc_len), adjust=False, min_periods=kc_len).mean()
    kc_u = kc_mid + 1.5 * kc_atr
    kc_l = kc_mid - 1.5 * kc_atr
    out["kc_width"] = (kc_u - kc_l) / kc_mid.replace(0.0, np.nan)
    out["squeeze_ratio"] = out["bb_width"] / out["kc_width"].replace(0.0, np.nan)

    # Structure and breaks
    prev_high_max = out["high"].shift(1).rolling(swing, min_periods=swing).max()
    prev_low_min = out["low"].shift(1).rolling(swing, min_periods=swing).min()
    out["is_hh"] = (out["high"] > prev_high_max).astype(int)
    out["is_ll"] = (out["low"] < prev_low_min).astype(int)
    out["hh_hl_state"] = np.where(out["is_hh"] == 1, 1, np.where(out["is_ll"] == 1, -1, 0)).astype(int)

    bos_up_thr = prev_high_max + 0.2 * out["atr14"]
    bos_dn_thr = prev_low_min - 0.2 * out["atr14"]
    out["bos_up"] = (out["close"] > bos_up_thr).astype(int)
    out["bos_down"] = (out["close"] < bos_dn_thr).astype(int)

    prior_state = out["hh_hl_state"].replace(0, np.nan).ffill().shift(1).fillna(0)
    out["choch_up"] = ((out["bos_up"] == 1) & (prior_state < 0)).astype(int)
    out["choch_down"] = ((out["bos_down"] == 1) & (prior_state > 0)).astype(int)

    # Flow features (weekly proxy)
    if "delta_norm" not in out.columns:
        out["delta_norm"] = 0.0
    if "market_buys" in out.columns and "market_sells" in out.columns:
        buys = out["market_buys"].astype(float)
        sells = out["market_sells"].astype(float)
    elif "taker_buy_base" in out.columns:
        buys = out["taker_buy_base"].astype(float)
        sells = (out["volume"].astype(float) - buys).clip(lower=0.0)
    else:
        buys = pd.Series(0.0, index=out.index)
        sells = pd.Series(0.0, index=out.index)
    out["aggr_buy_ratio"] = buys / (buys + sells).replace(0.0, np.nan)
    delta_aggr = buys - sells
    cvd = delta_aggr.cumsum()
    out["cvd_slope_3"] = cvd - cvd.shift(3)
    out["cvd_slope_8"] = cvd - cvd.shift(8)
    vol_mu = out["volume"].rolling(24, min_periods=12).mean()
    vol_sd = out["volume"].rolling(24, min_periods=12).std(ddof=0)
    out["volume_z"] = (out["volume"] - vol_mu) / (vol_sd + EPS)

    # Composite trend score
    n1 = out["ema20_50_spread"] / out["close"].replace(0.0, np.nan)
    n2 = out["ema50_200_spread"] / out["close"].replace(0.0, np.nan)
    n3 = out["ema20_slope"]
    out["trend_score"] = np.tanh(8.0 * (n1.fillna(0.0) + n2.fillna(0.0)) + 4.0 * n3.fillna(0.0))

    # Keep numeric safe
    out = out.replace([np.inf, -np.inf], np.nan)
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build Weekly (W) MTF context features and project to base timeframe.")
    p.add_argument("--input", required=True, help="Input parquet path.")
    p.add_argument("--output", required=True, help="Output parquet path.")
    p.add_argument("--ts-col", default="", help="Timestamp column (auto if empty).")
    p.add_argument("--swing", type=int, default=6, help="Weekly swing lookback for HH/LL.")
    p.add_argument("--bb-len", type=int, default=20)
    p.add_argument("--kc-len", type=int, default=20)
    p.add_argument("--shift-periods", type=int, default=1, help="Shift weekly features by N periods (anti-lookahead).")
    p.add_argument("--keep-all-cols", action="store_true", help="Keep original columns (default behavior).")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    in_path = Path(args.input)
    out_path = Path(args.output)

    print(f"[build_mtf_context_week_v1] VERSION={VERSION}")
    print(f"[INFO] input={in_path}")
    if not in_path.exists():
        raise FileNotFoundError(f"Input parquet not found: {in_path}")

    schema = pd.read_parquet(in_path, engine="pyarrow")
    ts_col = args.ts_col if args.ts_col else auto_detect_ts_col(schema)
    vol_col = detect_volume_col(schema)

    base_cols = [
        ts_col, "open", "high", "low", "close", vol_col,
        "market_buys", "market_sells", "taker_buy_base", "delta_norm",
    ]
    read_cols = [c for c in base_cols if c in schema.columns]
    df = pd.read_parquet(in_path, columns=read_cols, engine="pyarrow")
    df[ts_col] = normalize_ts_series(df[ts_col])
    df = df.sort_values(ts_col, kind="mergesort").reset_index(drop=True)

    wk = build_weekly_ohlcv(df, ts_col=ts_col, vol_col=vol_col)
    wk = add_week_features(wk, swing=int(args.swing), bb_len=int(args.bb_len), kc_len=int(args.kc_len))

    feature_cols = [
        "ret1", "range_pct", "body_pct", "upper_wick_pct", "lower_wick_pct", "close_pos", "atr14", "atr_pct",
        "ema20_50_spread", "ema50_200_spread", "ema20_slope", "ema50_slope", "vwap_dist", "vwap_dist_z",
        "bb_width", "kc_width", "squeeze_ratio", "is_hh", "is_ll", "hh_hl_state", "bos_up", "bos_down",
        "choch_up", "choch_down", "delta_norm", "aggr_buy_ratio", "cvd_slope_3", "cvd_slope_8", "volume_z",
        "trend_score",
    ]

    # Anti-lookahead: project only completed weekly bar context.
    shift_n = int(args.shift_periods)
    if shift_n > 0:
        wk[feature_cols] = wk[feature_cols].shift(shift_n)

    mn_cols = ["bar_ts"] + feature_cols
    mn_map = wk[mn_cols].copy()
    rename_map = {c: f"{c}_w" for c in feature_cols}
    mn_map = mn_map.rename(columns=rename_map)
    mn_map = mn_map.sort_values("bar_ts").reset_index(drop=True)

    mapped = pd.merge_asof(
        df[[ts_col]].sort_values(ts_col),
        mn_map,
        left_on=ts_col,
        right_on="bar_ts",
        direction="backward",
    )
    mapped = mapped.drop(columns=["bar_ts"])

    full = pd.read_parquet(in_path, engine="pyarrow")
    full[ts_col] = normalize_ts_series(full[ts_col])
    full = full.sort_values(ts_col, kind="mergesort").reset_index(drop=True)
    out = pd.concat([full, mapped.drop(columns=[ts_col])], axis=1)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False)

    print(f"[INFO] ts_col={ts_col} vol_col={vol_col} rows={len(out):,} cols={len(out.columns)}")
    print("[INFO] weekly_features=" + ", ".join([f"{c}_w" for c in feature_cols]))
    print(f"[OK] wrote: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
