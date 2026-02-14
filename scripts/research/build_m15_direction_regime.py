#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =============================================================================
# build_m15_direction_regime.py
#
# ETAPE B — Direction Regime M15 (V4)
#
# Objectifs V4 :
# - Sticky hysteresis + entry/exit persistence + min-hold (V3)
# - + Flip direct BULL<->BEAR (réduit le churn "BULL->NEUTRAL->BEAR")
# - + Colonnes ML-ready : dir_regime_id, dir_state_age
#
# Contrainte anti-lookahead (STRICT) :
# - Timestamp = open_time UTC (ts) pour M1 et M15
# - Buckets fermés uniquement (M15 complet = 15 bars M1)
# - available_from = bucket_end + lag_minutes
# - Aucune feature ne doit utiliser de données postérieures au bucket_end
#
# Style / contraintes projet :
# - Code Python structuré avec bandeau en tête
# - Toujours fournir le fichier Python complet corrigé
# - Compatible Pandas standard (pas de astype("string"))
# - Architecture modulaire (agrégation séparée du calcul de régime)
# =============================================================================

import argparse
import os
from dataclasses import dataclass
from typing import Optional, List, Tuple

import numpy as np
import pandas as pd
from pandas.api.types import is_datetime64_any_dtype, is_numeric_dtype


@dataclass(frozen=True)
class DirectionParams:
    lookback: int
    ema_fast: int
    ema_slow: int
    atr_len: int
    score_smooth_span: int
    w_impulse: float
    w_trend: float
    enter_th: float
    exit_th: float
    enter_persist: int
    exit_persist: int
    min_hold_bars: int
    allow_direct_flip: bool
    eps: float


def _to_utc_datetime(series: pd.Series) -> pd.Series:
    if is_datetime64_any_dtype(series):
        return pd.to_datetime(series, utc=True, errors="coerce")
    if is_numeric_dtype(series):
        vals = series.astype("int64")
        if len(vals) > 0 and vals.max() > 10_000_000_000:
            return pd.to_datetime(vals, unit="ms", utc=True, errors="coerce")
        return pd.to_datetime(vals, unit="s", utc=True, errors="coerce")
    return pd.to_datetime(series, utc=True, errors="coerce")


def _parse_utc(s: str) -> Optional[pd.Timestamp]:
    s2 = str(s).strip() if s is not None else ""
    if s2 == "":
        return None
    t = pd.Timestamp(s2)
    return t.tz_convert("UTC") if t.tzinfo else pd.Timestamp(s2, tz="UTC")


def _list_files(root: str, ext: str) -> List[str]:
    out = []
    if not os.path.exists(root):
        return out
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if fn.lower().endswith(ext):
                out.append(os.path.join(dirpath, fn))
    out.sort()
    return out


def _infer_ts_col(cols: List[str]) -> str:
    candidates = ["ts", "open_time", "open_time_ms", "open_time_s", "timestamp", "time"]
    for c in candidates:
        if c in cols:
            return c
    raise ValueError(f"Impossible d'inférer la colonne timestamp. Colonnes: {cols}")


def _filter_symbol_path(files: List[str], symbol: str) -> List[str]:
    key = f"symbol={symbol}"
    filtered = [f for f in files if key in f.replace("/", "\\")]
    return filtered if filtered else files


def _read_m1_parquet(
    root: str,
    symbol: str,
    start_utc: Optional[pd.Timestamp],
    end_utc: Optional[pd.Timestamp],
    ts_col_hint: str,
) -> pd.DataFrame:
    files = _list_files(root, ".parquet")
    if not files:
        return pd.DataFrame()

    files = _filter_symbol_path(files, symbol)

    dfs = []
    for fp in files:
        df = pd.read_parquet(fp)
        if df.empty:
            continue

        if "symbol" in df.columns:
            df = df[df["symbol"].astype(str) == symbol]
            if df.empty:
                continue

        ts_col = ts_col_hint if ts_col_hint and ts_col_hint in df.columns else _infer_ts_col(list(df.columns))
        df[ts_col] = _to_utc_datetime(df[ts_col])

        if start_utc is not None:
            df = df[df[ts_col] >= start_utc]
        if end_utc is not None:
            df = df[df[ts_col] < end_utc]
        if df.empty:
            continue

        if ts_col != "ts":
            df = df.rename(columns={ts_col: "ts"})

        if "volume_base" not in df.columns:
            if "volume" in df.columns:
                df = df.rename(columns={"volume": "volume_base"})
            elif "base_volume" in df.columns:
                df = df.rename(columns={"base_volume": "volume_base"})

        dfs.append(df)

    if not dfs:
        return pd.DataFrame()

    out = pd.concat(dfs, axis=0, ignore_index=True).sort_values("ts").reset_index(drop=True)
    return out


def _read_m1_csv_binance(
    root: str,
    symbol: str,
    start_utc: Optional[pd.Timestamp],
    end_utc: Optional[pd.Timestamp],
) -> pd.DataFrame:
    files = _list_files(root, ".csv")
    if not files:
        return pd.DataFrame()

    files_sym = [f for f in files if symbol in os.path.basename(f)]
    files = files_sym if files_sym else files

    dfs = []
    for fp in files:
        try:
            df = pd.read_csv(fp)
        except Exception:
            continue
        if df.empty:
            continue

        cols = list(df.columns)

        if "ts" in cols:
            df["ts"] = _to_utc_datetime(df["ts"])
        elif "open_time" in cols:
            df["ts"] = _to_utc_datetime(df["open_time"])
        elif "open_time_ms" in cols:
            df["ts"] = _to_utc_datetime(df["open_time_ms"])
        elif "timestamp" in cols:
            df["ts"] = _to_utc_datetime(df["timestamp"])
        else:
            first = cols[0]
            if is_numeric_dtype(df[first]):
                df["ts"] = _to_utc_datetime(df[first])
            else:
                continue

        if "volume_base" not in df.columns:
            if "volume" in df.columns:
                df = df.rename(columns={"volume": "volume_base"})
            elif "Volume" in df.columns:
                df = df.rename(columns={"Volume": "volume_base"})
            elif "base_volume" in df.columns:
                df = df.rename(columns={"base_volume": "volume_base"})

        for k in ["open", "high", "low", "close", "volume_base"]:
            if k in df.columns:
                df[k] = pd.to_numeric(df[k], errors="coerce")

        df = df.dropna(subset=["ts", "open", "high", "low", "close"])

        if start_utc is not None:
            df = df[df["ts"] >= start_utc]
        if end_utc is not None:
            df = df[df["ts"] < end_utc]
        if df.empty:
            continue

        keep = ["ts", "open", "high", "low", "close"]
        if "volume_base" in df.columns:
            keep.append("volume_base")
        df = df[keep].copy()
        if "volume_base" not in df.columns:
            df["volume_base"] = 0.0

        dfs.append(df)

    if not dfs:
        return pd.DataFrame()

    out = pd.concat(dfs, axis=0, ignore_index=True).sort_values("ts").reset_index(drop=True)
    if "volume_base" not in out.columns:
        out["volume_base"] = 0.0
    return out


def aggregate_m1_to_m15_closed(m1: pd.DataFrame, lag_minutes: int) -> pd.DataFrame:
    required = ["ts", "open", "high", "low", "close", "volume_base"]
    for c in required:
        if c not in m1.columns:
            raise RuntimeError(f"Colonne manquante M1: {c}. Colonnes: {list(m1.columns)}")

    df = m1.copy()
    df["ts"] = _to_utc_datetime(df["ts"])
    df = df.sort_values("ts").reset_index(drop=True)

    df["_bucket_start"] = df["ts"].dt.floor("15min")
    agg = df.groupby("_bucket_start", sort=True).agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume_base", "sum"),
        n_m1=("close", "size"),
    ).reset_index().rename(columns={"_bucket_start": "ts"})

    agg = agg[agg["n_m1"] == 15].copy()
    agg = agg.sort_values("ts").reset_index(drop=True)
    agg["bucket_end"] = agg["ts"] + pd.Timedelta(minutes=15)
    agg["available_from"] = agg["bucket_end"] + pd.Timedelta(minutes=int(lag_minutes))
    return agg


def _true_range(high: pd.Series, low: pd.Series, prev_close: pd.Series) -> pd.Series:
    a = (high - low).abs()
    b = (high - prev_close).abs()
    c = (low - prev_close).abs()
    return pd.concat([a, b, c], axis=1).max(axis=1)


def _rma(x: pd.Series, length: int) -> pd.Series:
    return x.ewm(alpha=1.0 / float(length), adjust=False).mean()


def _efficiency_ratio(close: pd.Series, lookback: int, eps: float) -> pd.Series:
    change = (close - close.shift(lookback)).abs()
    volatility = (close - close.shift(1)).abs().rolling(lookback, min_periods=lookback).sum()
    return change / (volatility + eps)


def _sticky_state_v4(
    score: pd.Series,
    enter_th: float,
    exit_th: float,
    enter_persist: int,
    exit_persist: int,
    min_hold_bars: int,
    allow_direct_flip: bool,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Returns:
      - state series: NEUTRAL/BULL/BEAR
      - regime_id: increments on any state change
      - state_age: bars since regime start (1..)
    """
    xs = score.values
    states = []
    regime_ids = []
    ages = []

    state = "NEUTRAL"
    hold = 0
    regime_id = 0
    age = 0

    bull_enter_run = 0
    bear_enter_run = 0
    bull_exit_run = 0
    bear_exit_run = 0
    bull_rev_run = 0
    bear_rev_run = 0

    def _bump_regime(new_state: str):
        nonlocal state, regime_id, age, hold, bull_enter_run, bear_enter_run, bull_exit_run, bear_exit_run, bull_rev_run, bear_rev_run
        if new_state != state:
            regime_id += 1
            state = new_state
            age = 0
            hold = 0
            bull_enter_run = 0
            bear_enter_run = 0
            bull_exit_run = 0
            bear_exit_run = 0
            bull_rev_run = 0
            bear_rev_run = 0

    for x in xs:
        if np.isnan(x):
            _bump_regime("NEUTRAL")
            age = 0
            hold = 0
            states.append("NEUTRAL")
            regime_ids.append(regime_id)
            ages.append(age)
            continue

        bull_enter_run = bull_enter_run + 1 if x >= enter_th else 0
        bear_enter_run = bear_enter_run + 1 if x <= -enter_th else 0

        if state == "NEUTRAL":
            if bull_enter_run >= enter_persist:
                _bump_regime("BULL")
            elif bear_enter_run >= enter_persist:
                _bump_regime("BEAR")

        elif state == "BULL":
            hold += 1
            # Exit BULL if x <= exit_th for exit_persist bars, after min_hold
            bull_exit_run = bull_exit_run + 1 if x <= exit_th else 0

            if allow_direct_flip:
                bear_rev_run = bear_rev_run + 1 if x <= -enter_th else 0
            else:
                bear_rev_run = 0

            if hold >= max(min_hold_bars, 1):
                if allow_direct_flip and bear_rev_run >= enter_persist:
                    _bump_regime("BEAR")
                elif bull_exit_run >= exit_persist:
                    _bump_regime("NEUTRAL")

        elif state == "BEAR":
            hold += 1
            # Exit BEAR if x >= -exit_th for exit_persist bars, after min_hold
            bear_exit_run = bear_exit_run + 1 if x >= -exit_th else 0

            if allow_direct_flip:
                bull_rev_run = bull_rev_run + 1 if x >= enter_th else 0
            else:
                bull_rev_run = 0

            if hold >= max(min_hold_bars, 1):
                if allow_direct_flip and bull_rev_run >= enter_persist:
                    _bump_regime("BULL")
                elif bear_exit_run >= exit_persist:
                    _bump_regime("NEUTRAL")

        age += 1
        states.append(state)
        regime_ids.append(regime_id)
        ages.append(age)

    return (
        pd.Series(states, index=score.index, dtype="object"),
        pd.Series(regime_ids, index=score.index, dtype="int64"),
        pd.Series(ages, index=score.index, dtype="int64"),
    )


def _compute_scores(df: pd.DataFrame, params: DirectionParams) -> pd.Series:
    close = df["close"].astype("float64")
    high = df["high"].astype("float64")
    low = df["low"].astype("float64")

    ret1 = np.log(close / close.shift(1))
    retL = np.log(close / close.shift(params.lookback))

    vol1 = ret1.rolling(params.lookback, min_periods=params.lookback).std(ddof=0)
    volL = vol1 * np.sqrt(float(params.lookback))
    impulse = retL / (volL + params.eps)

    prev_close = close.shift(1)
    tr = _true_range(high, low, prev_close)
    atr = _rma(tr, params.atr_len)

    ema_fast = close.ewm(span=params.ema_fast, adjust=False).mean()
    ema_slow = close.ewm(span=params.ema_slow, adjust=False).mean()
    trend = (ema_fast - ema_slow) / (atr + params.eps)

    er = _efficiency_ratio(close, params.lookback, params.eps)
    quality = 0.5 + 0.5 * er

    score_raw = (params.w_impulse * impulse) + (params.w_trend * trend)
    score_mod = score_raw * quality
    score_smooth = score_mod.ewm(span=params.score_smooth_span, adjust=False).mean()
    return np.tanh(score_smooth.astype("float64"))


def compute_direction_regime_m15(m15: pd.DataFrame, params: DirectionParams) -> pd.DataFrame:
    df = m15.copy().sort_values("ts").reset_index(drop=True)

    df["dir_score"] = _compute_scores(df, params)

    state, regime_id, age = _sticky_state_v4(
        df["dir_score"],
        params.enter_th,
        params.exit_th,
        params.enter_persist,
        params.exit_persist,
        params.min_hold_bars,
        params.allow_direct_flip,
    )
    df["dir_state"] = state
    df["dir_regime_id"] = regime_id
    df["dir_state_age"] = age

    df["dir_ready"] = True
    df["dir_ready"] = df["dir_ready"] & (df["n_m1"] == 15)
    df["dir_ready"] = df["dir_ready"] & (~pd.isna(df["dir_score"]))
    df["dir_ready"] = df["dir_ready"] & (~pd.isna(df["available_from"])) & (~pd.isna(df["bucket_end"]))

    return df


def write_parquet(df: pd.DataFrame, out_path: str) -> None:
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    df.to_parquet(out_path, index=False)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="ETAPE B — Build M15 Direction Regime (no lookahead, V4).")
    p.add_argument("--m1-root", required=True, help="Root folder containing M1 parquet OR csv files.")
    p.add_argument("--symbol", required=True, help="Symbol (e.g., BTCUSDT).")
    p.add_argument("--start", default="")
    p.add_argument("--end", default="")
    p.add_argument("--ts-col", default="", help="Optional parquet timestamp column hint.")
    p.add_argument("--lag-minutes", type=int, default=2)
    p.add_argument("--out", required=True)

    p.add_argument("--lookback", type=int, default=16)
    p.add_argument("--ema-fast", type=int, default=8)
    p.add_argument("--ema-slow", type=int, default=21)
    p.add_argument("--atr-len", type=int, default=14)
    p.add_argument("--score-smooth-span", type=int, default=5)
    p.add_argument("--w-impulse", type=float, default=1.0)
    p.add_argument("--w-trend", type=float, default=0.8)

    p.add_argument("--enter-th", type=float, default=0.40)
    p.add_argument("--exit-th", type=float, default=0.22)
    p.add_argument("--enter-persist", type=int, default=3)
    p.add_argument("--exit-persist", type=int, default=2)
    p.add_argument("--min-hold-bars", type=int, default=4)
    p.add_argument("--allow-direct-flip", action="store_true", help="Enable direct flip BULL<->BEAR (default off unless specified).")
    p.add_argument("--no-direct-flip", action="store_true", help="Force disable direct flip (overrides allow-direct-flip).")

    return p


def main() -> int:
    args = build_parser().parse_args()

    if float(args.exit_th) >= float(args.enter_th):
        raise RuntimeError("--exit-th must be strictly lower than --enter-th.")
    if int(args.enter_persist) < 1 or int(args.exit_persist) < 1:
        raise RuntimeError("--enter-persist and --exit-persist must be >= 1.")
    if int(args.min_hold_bars) < 1:
        raise RuntimeError("--min-hold-bars must be >= 1.")

    allow_direct = bool(args.allow_direct_flip) and (not bool(args.no_direct_flip))

    start_utc = _parse_utc(args.start) if args.start else None
    end_utc = _parse_utc(args.end) if args.end else None

    m1 = _read_m1_parquet(args.m1_root, args.symbol, start_utc, end_utc, args.ts_col.strip())
    src = "parquet"
    if m1.empty:
        m1 = _read_m1_csv_binance(args.m1_root, args.symbol, start_utc, end_utc)
        src = "csv"
    if m1.empty:
        raise RuntimeError(f"Aucune donnée M1 trouvée dans {args.m1_root} (parquet/csv).")

    for c in ["ts", "open", "high", "low", "close", "volume_base"]:
        if c not in m1.columns:
            if c == "volume_base":
                m1["volume_base"] = 0.0
            else:
                raise RuntimeError(f"Colonne manquante M1: {c}. Colonnes: {list(m1.columns)}")

    m15 = aggregate_m1_to_m15_closed(m1, lag_minutes=int(args.lag_minutes))
    if m15.empty:
        raise RuntimeError("Aucun bucket M15 complet (15 bars M1) sur la fenêtre demandée.")

    params = DirectionParams(
        lookback=int(args.lookback),
        ema_fast=int(args.ema_fast),
        ema_slow=int(args.ema_slow),
        atr_len=int(args.atr_len),
        score_smooth_span=int(args.score_smooth_span),
        w_impulse=float(args.w_impulse),
        w_trend=float(args.w_trend),
        enter_th=float(args.enter_th),
        exit_th=float(args.exit_th),
        enter_persist=int(args.enter_persist),
        exit_persist=int(args.exit_persist),
        min_hold_bars=int(args.min_hold_bars),
        allow_direct_flip=allow_direct,
        eps=1e-12,
    )

    out = compute_direction_regime_m15(m15, params)

    if (out["available_from"] < out["bucket_end"]).any():
        raise RuntimeError("Violation anti-lookahead: available_from < bucket_end détecté.")

    write_parquet(out, args.out)

    last_bucket_end = out["bucket_end"].iloc[-1]
    ready = int(out["dir_ready"].sum())
    print(f"[OK] src={src} symbol={args.symbol} rows={len(out)} dir_ready={ready} last_bucket_end={last_bucket_end} direct_flip={allow_direct}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
