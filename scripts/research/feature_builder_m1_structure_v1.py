from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd


VERSION = "2026-02-14-structure-v1"
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


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    return pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)


def detect_volume_col(df: pd.DataFrame) -> str:
    candidates = [
        "volume_base",
        "volume",
        "base_volume",
        "volume_base_x",
        "volume_base_y",
    ]
    for c in candidates:
        if c in df.columns:
            return c
    raise RuntimeError("No volume column found (expected one of volume_base/volume/base_volume).")


def add_daily_vwap(df: pd.DataFrame, ts_col: str, vol_col: str) -> pd.DataFrame:
    out = df.copy()
    px = (out["high"] + out["low"] + out["close"]) / 3.0
    vol = out[vol_col].astype(float).fillna(0.0)
    day_key = out[ts_col].dt.floor("D")
    pv = px * vol
    cum_pv = pv.groupby(day_key).cumsum()
    cum_v = vol.groupby(day_key).cumsum()
    out["vwap_d"] = cum_pv / cum_v.replace(0.0, np.nan)
    out["vwap_dist"] = (out["close"] - out["vwap_d"]) / out["vwap_d"].replace(0.0, np.nan)
    return out


def add_hh_hl_structure(df: pd.DataFrame, swing: int) -> pd.DataFrame:
    out = df.copy()
    prev_high_max = out["high"].shift(1).rolling(swing, min_periods=swing).max()
    prev_low_min = out["low"].shift(1).rolling(swing, min_periods=swing).min()
    hh = out["high"] > prev_high_max
    ll = out["low"] < prev_low_min
    out["is_hh"] = hh.astype(int)
    out["is_ll"] = ll.astype(int)
    # +1 up-break, -1 down-break, 0 otherwise
    out["hh_hl_state"] = np.where(hh, 1, np.where(ll, -1, 0)).astype(int)
    return out


def add_ema_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["ema20"] = out["close"].ewm(span=20, adjust=False, min_periods=20).mean()
    out["ema50"] = out["close"].ewm(span=50, adjust=False, min_periods=50).mean()
    out["ema200"] = out["close"].ewm(span=200, adjust=False, min_periods=200).mean()
    out["ema20_50_spread_m1"] = out["ema20"] - out["ema50"]
    out["ema50_200_spread_m1"] = out["ema50"] - out["ema200"]
    return out


def add_bb_kc(df: pd.DataFrame, bb_len: int, bb_k: float, kc_len: int, kc_k: float) -> pd.DataFrame:
    out = df.copy()
    mid = out["close"].rolling(bb_len, min_periods=bb_len).mean()
    std = out["close"].rolling(bb_len, min_periods=bb_len).std(ddof=0)
    bb_upper = mid + (bb_k * std)
    bb_lower = mid - (bb_k * std)
    bb_width = (bb_upper - bb_lower) / mid.replace(0.0, np.nan)
    bb_pos = (out["close"] - bb_lower) / (bb_upper - bb_lower).replace(0.0, np.nan)

    tr = true_range(out["high"], out["low"], out["close"])
    atr_kc = tr.ewm(alpha=1.0 / float(kc_len), adjust=False, min_periods=kc_len).mean()
    kc_mid = out["close"].ewm(span=kc_len, adjust=False, min_periods=kc_len).mean()
    kc_upper = kc_mid + (kc_k * atr_kc)
    kc_lower = kc_mid - (kc_k * atr_kc)
    kc_width = (kc_upper - kc_lower) / kc_mid.replace(0.0, np.nan)

    out["bb_mid"] = mid
    out["bb_upper"] = bb_upper
    out["bb_lower"] = bb_lower
    out["bb_width"] = bb_width
    out["bb_pos"] = bb_pos
    out["kc_mid"] = kc_mid
    out["kc_upper"] = kc_upper
    out["kc_lower"] = kc_lower
    out["kc_width"] = kc_width
    out["squeeze_ratio"] = bb_width / kc_width.replace(0.0, np.nan)
    return out


def add_vwap_zscore(df: pd.DataFrame, z_win: int) -> pd.DataFrame:
    out = df.copy()
    x = out["vwap_dist"]
    mu = x.rolling(z_win, min_periods=z_win).mean()
    sd = x.rolling(z_win, min_periods=z_win).std(ddof=0).replace(0.0, np.nan)
    out["vwap_dist_z"] = (x - mu) / (sd + EPS)
    return out


def shift_feature_cols(df: pd.DataFrame, cols: List[str], n: int) -> pd.DataFrame:
    if n <= 0:
        return df
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = out[c].shift(n)
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build M1 structure features: VWAP, HH/HL, EMA, BB/KC, squeeze.")
    p.add_argument("--input", required=True, help="Input parquet path (M1-level joined/enriched dataset).")
    p.add_argument("--output", required=True, help="Output parquet path.")
    p.add_argument("--ts-col", default="", help="Timestamp column. Empty => auto-detect.")
    p.add_argument("--swing", type=int, default=20, help="Lookback for HH/LL structure.")
    p.add_argument("--bb-len", type=int, default=20)
    p.add_argument("--bb-k", type=float, default=2.0)
    p.add_argument("--kc-len", type=int, default=20)
    p.add_argument("--kc-k", type=float, default=1.5)
    p.add_argument("--vwap-z-win", type=int, default=120, help="Rolling window for vwap_dist_z.")
    p.add_argument("--shift-features", type=int, default=1, help="Shift generated feature columns by N bars.")
    p.add_argument("--keep-all-cols", action="store_true", help="Keep all original columns (default).")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    in_path = Path(args.input)
    out_path = Path(args.output)

    print(f"[feature_builder_m1_structure_v1] VERSION={VERSION}")
    print(f"[INFO] input={in_path}")
    if not in_path.exists():
        raise FileNotFoundError(f"Input parquet not found: {in_path}")

    df = pd.read_parquet(in_path, engine="pyarrow")
    ts_col = args.ts_col if args.ts_col else auto_detect_ts_col(df)
    if ts_col not in df.columns:
        raise RuntimeError(f"Timestamp column not found: {ts_col}")

    required = ["open", "high", "low", "close"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing required columns: {missing}")
    vol_col = detect_volume_col(df)

    df[ts_col] = normalize_ts_series(df[ts_col])
    df = df.sort_values(ts_col, kind="mergesort").reset_index(drop=True)

    out = add_daily_vwap(df, ts_col=ts_col, vol_col=vol_col)
    out = add_hh_hl_structure(out, swing=int(args.swing))
    out = add_ema_features(out)
    out = add_bb_kc(
        out,
        bb_len=int(args.bb_len),
        bb_k=float(args.bb_k),
        kc_len=int(args.kc_len),
        kc_k=float(args.kc_k),
    )
    out = add_vwap_zscore(out, z_win=int(args.vwap_z_win))

    feat_cols = [
        "vwap_d",
        "vwap_dist",
        "vwap_dist_z",
        "is_hh",
        "is_ll",
        "hh_hl_state",
        "ema20",
        "ema50",
        "ema200",
        "ema20_50_spread_m1",
        "ema50_200_spread_m1",
        "bb_mid",
        "bb_upper",
        "bb_lower",
        "bb_width",
        "bb_pos",
        "kc_mid",
        "kc_upper",
        "kc_lower",
        "kc_width",
        "squeeze_ratio",
    ]

    out = shift_feature_cols(out, feat_cols, n=int(args.shift_features))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False)
    print(f"[INFO] ts_col={ts_col} vol_col={vol_col} rows={len(out):,} cols={len(out.columns)}")
    print(f"[OK] wrote: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
