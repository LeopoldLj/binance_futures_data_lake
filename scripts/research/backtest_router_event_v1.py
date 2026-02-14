# C:\Users\lolo_\PycharmProjects\binance_futures_data_lake\scripts\backtest\backtest_router_event_v1.py
# backtest_router_event_v1.py
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


VERSION = "2026-02-14a"


# =========================
# CONFIG
# =========================
PARQUET_IN = r"C:\Users\lolo_\PycharmProjects\binance_futures_data_lake\data\research_debug\BTCUSDT\long_20260101_20260210\joined_20260101_20260210__enriched__router.parquet"

OUTDIR = r"C:\Users\lolo_\PycharmProjects\binance_futures_data_lake\data\research_debug\BTCUSDT\long_20260101_20260210\backtest_v1"
SYMBOL = "BTCUSDT"

INITIAL_EQUITY = 1.0  # equity in "R-units" (we track everything in R; you can later map to $)
ALLOW_LONGS = True
ALLOW_SHORTS = True
ONE_POSITION_MAX = True

# --- Regime definitions ---
VOL_EXCLUDE_NA = True  # exclude vol_state == "NA" for both engines
TREND_VOL_OK = {"MID", "HIGH"}

# --- Trend impulse thresholds (M1) ---
TREND_D = 0.20
TREND_P = 0.65
TREND_RR = 0.80

# --- Range MR thresholds (M1) ---
MR_D = 0.20
MR_RR = 0.80
MR_CP_LOW = 0.20
MR_CP_HIGH = 0.80

# --- Risk (ATR-based) ---
TREND_SL_ATR = 1.00
TREND_TP_R = 2.00
TREND_TIME_STOP_BARS = 60

MR_SL_ATR = 0.80
MR_TP_R = 1.20
MR_TIME_STOP_BARS = 30
MR_BREAK_EVEN_AT_R = 0.60  # set to None to disable
MR_BE_OFFSET_R = 0.00      # 0 => strict BE, can set 0.1 to lock small profit

# --- Costs (optional) ---
USE_COSTS = False
COST_R_PER_TRADE = 0.00  # in R-units, subtracted at entry (commission+slippage approximation)

# --- Cooldown (optional) ---
USE_COOLDOWN = True
COOLDOWN_AFTER_LOSS_BARS = 5


# =========================
# DATA STRUCTURES
# =========================
@dataclass
class Trade:
    symbol: str
    engine: str  # "TREND" or "RANGE"
    side: str    # "LONG" or "SHORT"
    entry_ts: str
    exit_ts: str
    entry_price: float
    exit_price: float
    sl_price: float
    tp_price: float
    atr_entry: float
    r_mult: float
    pnl_r: float
    bars_in_trade: int
    exit_reason: str
    router_mode_entry: str
    router_mode_exit: str
    dir_state_entry: str
    vol_state_entry: str


@dataclass
class Position:
    engine: str
    side: str
    entry_i: int
    entry_ts: str
    entry_price: float
    atr_entry: float
    sl_price: float
    tp_price: float
    risk_per_unit: float  # |entry - SL|
    router_mode_entry: str
    dir_state_entry: str
    vol_state_entry: str
    be_armed: bool = False


# =========================
# HELPERS
# =========================
def _find_ts_col(columns: List[str]) -> Optional[str]:
    candidates = ["ts", "timestamp", "time", "datetime", "open_time", "open_ts"]
    for c in candidates:
        if c in columns:
            return c
    for c in columns:
        cl = c.lower()
        if cl in candidates:
            return c
    return None


def _safe_str(x) -> str:
    if pd.isna(x):
        return "NA"
    return str(x)


def _to_iso(ts_val) -> str:
    if pd.isna(ts_val):
        return "NA"
    if isinstance(ts_val, (pd.Timestamp, np.datetime64)):
        return pd.Timestamp(ts_val).tz_localize(None).isoformat()
    return str(ts_val)


def _r_mult_for_exit(side: str, entry: float, exit_p: float, risk: float) -> float:
    if risk <= 0 or not np.isfinite(risk):
        return 0.0
    if side == "LONG":
        return (exit_p - entry) / risk
    return (entry - exit_p) / risk


def _max_drawdown(equity_curve: np.ndarray) -> float:
    if equity_curve.size == 0:
        return 0.0
    peak = -np.inf
    max_dd = 0.0
    for x in equity_curve:
        peak = max(peak, x)
        dd = peak - x
        max_dd = max(max_dd, dd)
    return max_dd


def _profit_factor(r_list: List[float]) -> float:
    gains = sum([r for r in r_list if r > 0])
    losses = -sum([r for r in r_list if r < 0])
    if losses <= 0:
        return float("inf") if gains > 0 else 0.0
    return gains / losses


# =========================
# SIGNALS
# =========================
def compute_regimes_and_signals(df: pd.DataFrame) -> pd.DataFrame:
    # Normalize string-like columns
    df["router_mode_h1"] = df["router_mode_h1"].astype("string")
    df["dir_state"] = df["dir_state"].astype("string")
    df["vol_state"] = df["vol_state"].astype("string")

    # Base gates
    tradable = df["tradable_final"] == True
    dir_ready = df["dir_ready"] == True  # already bool in your dataset
    vol_ok = df["vol_state"] != "NA" if VOL_EXCLUDE_NA else pd.Series(True, index=df.index)

    # Regimes
    df["is_trend_regime"] = tradable & dir_ready & vol_ok & (df["router_mode_h1"] == "TREND") & df["vol_state"].isin(list(TREND_VOL_OK))
    df["is_range_regime"] = tradable & dir_ready & vol_ok & (df["router_mode_h1"] == "RANGE")

    # Trend impulse triggers
    impulse_long = (df["delta_norm"] > TREND_D) & (df["close_pos"] > TREND_P) & (df["range_rel"] > TREND_RR)
    impulse_short = (df["delta_norm"] < -TREND_D) & (df["close_pos"] < (1.0 - TREND_P)) & (df["range_rel"] > TREND_RR)

    df["trend_long_signal"] = df["is_trend_regime"] & (df["dir_state"] == "BULL") & impulse_long
    df["trend_short_signal"] = df["is_trend_regime"] & (df["dir_state"] == "BEAR") & impulse_short

    # Range MR setups (aligned with dir_state to reduce knives)
    mr_long_setup = (df["close_pos"] <= MR_CP_LOW) & (df["delta_norm"] <= -MR_D) & (df["range_rel"] >= MR_RR) & (df["dir_state"] == "BULL")
    mr_short_setup = (df["close_pos"] >= MR_CP_HIGH) & (df["delta_norm"] >= MR_D) & (df["range_rel"] >= MR_RR) & (df["dir_state"] == "BEAR")

    df["mr_long_setup"] = df["is_range_regime"] & mr_long_setup
    df["mr_short_setup"] = df["is_range_regime"] & mr_short_setup

    # MR confirmation at t+1 (entry happens on next bar)
    # Long entry at i if setup at i-1 and confirmation at i
    # Confirmation: delta_norm > 0 OR close_pos > 0.50 (long), symm for short
    conf_long = (df["delta_norm"] > 0) | (df["close_pos"] > 0.50)
    conf_short = (df["delta_norm"] < 0) | (df["close_pos"] < 0.50)

    df["mr_long_signal"] = df["mr_long_setup"].shift(1, fill_value=False) & df["is_range_regime"] & conf_long
    df["mr_short_signal"] = df["mr_short_setup"].shift(1, fill_value=False) & df["is_range_regime"] & conf_short

    return df


# =========================
# BACKTEST CORE (event-driven)
# =========================
def run_backtest(df: pd.DataFrame, ts_col: str) -> Tuple[List[Trade], pd.DataFrame, Dict]:
    equity = INITIAL_EQUITY
    equity_curve = np.empty(len(df), dtype=float)
    equity_curve[:] = np.nan

    trades: List[Trade] = []
    pos: Optional[Position] = None
    cooldown_until_i = -1

    def can_enter(i: int) -> bool:
        nonlocal pos, cooldown_until_i
        if ONE_POSITION_MAX and pos is not None:
            return False
        if USE_COOLDOWN and i <= cooldown_until_i:
            return False
        return True

    def apply_costs_on_entry() -> None:
        nonlocal equity
        if USE_COSTS and COST_R_PER_TRADE > 0:
            equity -= COST_R_PER_TRADE

    for i in range(len(df)):
        row = df.iloc[i]
        ts = _to_iso(row[ts_col])

        # mark equity each bar (step function)
        equity_curve[i] = equity

        # 1) Manage existing position exits (SL/TP/time/flip)
        if pos is not None:
            high = float(row["high"])
            low = float(row["low"])
            close = float(row["close"])
            router_mode_now = _safe_str(row["router_mode_h1"])

            bars_in_trade = i - pos.entry_i

            # Flip regime exit
            flip_exit = False
            if pos.engine == "TREND":
                if router_mode_now != "TREND":
                    flip_exit = True
            else:
                if router_mode_now != "RANGE":
                    flip_exit = True

            # Time stop
            time_exit = False
            if pos.engine == "TREND" and bars_in_trade >= TREND_TIME_STOP_BARS:
                time_exit = True
            if pos.engine == "RANGE" and bars_in_trade >= MR_TIME_STOP_BARS:
                time_exit = True

            # Break-even arm for MR
            if pos.engine == "RANGE" and MR_BREAK_EVEN_AT_R is not None and not pos.be_armed:
                # unrealized R at close
                unreal_r = _r_mult_for_exit(pos.side, pos.entry_price, close, pos.risk_per_unit)
                if unreal_r >= float(MR_BREAK_EVEN_AT_R):
                    # move SL to entry + offset in R
                    if pos.side == "LONG":
                        pos.sl_price = pos.entry_price + (pos.risk_per_unit * float(MR_BE_OFFSET_R))
                    else:
                        pos.sl_price = pos.entry_price - (pos.risk_per_unit * float(MR_BE_OFFSET_R))
                    pos.be_armed = True

            # SL/TP checks using intrabar extremes (worst-case handling)
            sl_hit = False
            tp_hit = False

            if pos.side == "LONG":
                sl_hit = low <= pos.sl_price
                tp_hit = high >= pos.tp_price
            else:
                sl_hit = high >= pos.sl_price
                tp_hit = low <= pos.tp_price

            exit_reason = None
            exit_price = None

            # If both SL and TP hit same bar, choose conservative (SL first)
            if sl_hit and tp_hit:
                exit_reason = "SL_and_TP_same_bar"
                exit_price = pos.sl_price
            elif sl_hit:
                exit_reason = "SL"
                exit_price = pos.sl_price
            elif tp_hit:
                exit_reason = "TP"
                exit_price = pos.tp_price
            elif flip_exit:
                exit_reason = "FLIP_ROUTER"
                exit_price = close
            elif time_exit:
                exit_reason = "TIME_STOP"
                exit_price = close

            if exit_reason is not None:
                r_mult = _r_mult_for_exit(pos.side, pos.entry_price, float(exit_price), pos.risk_per_unit)
                pnl_r = r_mult
                equity += pnl_r

                tr = Trade(
                    symbol=SYMBOL,
                    engine=pos.engine,
                    side=pos.side,
                    entry_ts=pos.entry_ts,
                    exit_ts=ts,
                    entry_price=float(pos.entry_price),
                    exit_price=float(exit_price),
                    sl_price=float(pos.sl_price),
                    tp_price=float(pos.tp_price),
                    atr_entry=float(pos.atr_entry),
                    r_mult=float(r_mult),
                    pnl_r=float(pnl_r),
                    bars_in_trade=int(bars_in_trade),
                    exit_reason=str(exit_reason),
                    router_mode_entry=str(pos.router_mode_entry),
                    router_mode_exit=str(router_mode_now),
                    dir_state_entry=str(pos.dir_state_entry),
                    vol_state_entry=str(pos.vol_state_entry),
                )
                trades.append(tr)
                # cooldown after loss
                if USE_COOLDOWN and pnl_r < 0:
                    cooldown_until_i = i + COOLDOWN_AFTER_LOSS_BARS
                pos = None

        # 2) Entries (if no position)
        if can_enter(i):
            # Evaluate signals in priority: TREND first, then RANGE
            # (You can flip priority later if needed)
            trend_long = bool(row["trend_long_signal"]) if "trend_long_signal" in df.columns else False
            trend_short = bool(row["trend_short_signal"]) if "trend_short_signal" in df.columns else False
            mr_long = bool(row["mr_long_signal"]) if "mr_long_signal" in df.columns else False
            mr_short = bool(row["mr_short_signal"]) if "mr_short_signal" in df.columns else False

            # Choose entry
            entry_engine = None
            entry_side = None

            if (ALLOW_LONGS and trend_long) or (ALLOW_SHORTS and trend_short):
                # trend priority
                if ALLOW_LONGS and trend_long:
                    entry_engine, entry_side = "TREND", "LONG"
                elif ALLOW_SHORTS and trend_short:
                    entry_engine, entry_side = "TREND", "SHORT"
            elif (ALLOW_LONGS and mr_long) or (ALLOW_SHORTS and mr_short):
                if ALLOW_LONGS and mr_long:
                    entry_engine, entry_side = "RANGE", "LONG"
                elif ALLOW_SHORTS and mr_short:
                    entry_engine, entry_side = "RANGE", "SHORT"

            if entry_engine is not None:
                entry_price = float(row["close"])  # event-driven, enter at close of signal bar
                atr = float(row["atr14"])
                if not np.isfinite(atr) or atr <= 0:
                    # skip entry if ATR invalid
                    continue

                if entry_engine == "TREND":
                    sl_dist = TREND_SL_ATR * atr
                    tp_r = TREND_TP_R
                else:
                    sl_dist = MR_SL_ATR * atr
                    tp_r = MR_TP_R

                if entry_side == "LONG":
                    sl_price = entry_price - sl_dist
                    risk = entry_price - sl_price
                    tp_price = entry_price + tp_r * risk
                else:
                    sl_price = entry_price + sl_dist
                    risk = sl_price - entry_price
                    tp_price = entry_price - tp_r * risk

                if not np.isfinite(risk) or risk <= 0:
                    continue

                apply_costs_on_entry()

                pos = Position(
                    engine=entry_engine,
                    side=entry_side,
                    entry_i=i,
                    entry_ts=ts,
                    entry_price=entry_price,
                    atr_entry=atr,
                    sl_price=float(sl_price),
                    tp_price=float(tp_price),
                    risk_per_unit=float(risk),
                    router_mode_entry=_safe_str(row["router_mode_h1"]),
                    dir_state_entry=_safe_str(row["dir_state"]),
                    vol_state_entry=_safe_str(row["vol_state"]),
                    be_armed=False,
                )

    # finalize equity curve forward-fill
    # (already step-filled each bar; ensure last bars are set)
    equity_curve = np.where(np.isfinite(equity_curve), equity_curve, np.nan)
    equity_df = pd.DataFrame({ts_col: df[ts_col].astype("string"), "equity_r": equity_curve})

    # metrics
    r_list = [t.pnl_r for t in trades]
    wins = [r for r in r_list if r > 0]
    losses = [r for r in r_list if r < 0]
    metrics = {
        "version": VERSION,
        "symbol": SYMBOL,
        "n_bars": int(len(df)),
        "n_trades": int(len(trades)),
        "winrate": float(len(wins) / max(len(trades), 1)),
        "avg_r": float(np.mean(r_list)) if len(r_list) else 0.0,
        "median_r": float(np.median(r_list)) if len(r_list) else 0.0,
        "profit_factor": float(_profit_factor(r_list)) if len(r_list) else 0.0,
        "max_drawdown_r": float(_max_drawdown(equity_curve[np.isfinite(equity_curve)])),
        "sum_r": float(np.sum(r_list)) if len(r_list) else 0.0,
        "by_engine": {},
        "by_side": {},
    }

    # breakdowns
    for engine in ["TREND", "RANGE"]:
        rr = [t.pnl_r for t in trades if t.engine == engine]
        metrics["by_engine"][engine] = {
            "n_trades": int(len(rr)),
            "winrate": float(sum([1 for x in rr if x > 0]) / max(len(rr), 1)),
            "avg_r": float(np.mean(rr)) if len(rr) else 0.0,
            "profit_factor": float(_profit_factor(rr)) if len(rr) else 0.0,
            "sum_r": float(np.sum(rr)) if len(rr) else 0.0,
        }
    for side in ["LONG", "SHORT"]:
        rr = [t.pnl_r for t in trades if t.side == side]
        metrics["by_side"][side] = {
            "n_trades": int(len(rr)),
            "winrate": float(sum([1 for x in rr if x > 0]) / max(len(rr), 1)),
            "avg_r": float(np.mean(rr)) if len(rr) else 0.0,
            "profit_factor": float(_profit_factor(rr)) if len(rr) else 0.0,
            "sum_r": float(np.sum(rr)) if len(rr) else 0.0,
        }

    return trades, equity_df, metrics


# =========================
# MAIN
# =========================
def main() -> int:
    print(f"[backtest_router_event] VERSION={VERSION}")
    print(f"[INFO] parquet_in={PARQUET_IN}")

    outdir = Path(OUTDIR)
    outdir.mkdir(parents=True, exist_ok=True)

    # Read minimal columns (plus OHLC to simulate SL/TP)
    # We need high/low/close for execution + ATR
    df_schema = pd.read_parquet(PARQUET_IN, engine="pyarrow")
    all_cols = list(df_schema.columns)

    ts_col = _find_ts_col(all_cols)
    if ts_col is None:
        raise RuntimeError("No timestamp column found. Expected one of: ts, timestamp, time, datetime, open_time, open_ts.")

    required = [
        ts_col,
        "open", "high", "low", "close",
        "router_mode_h1",
        "tradable_final", "dir_ready", "dir_state", "vol_state",
        "delta_norm", "close_pos", "range_rel", "atr14",
    ]
    missing = [c for c in required if c not in all_cols]
    if missing:
        raise RuntimeError(f"Missing required columns: {missing}")

    df = pd.read_parquet(PARQUET_IN, columns=required, engine="pyarrow")

    # Ensure sorted by time
    try:
        df = df.sort_values(ts_col, kind="mergesort").reset_index(drop=True)
    except Exception:
        df = df.reset_index(drop=True)

    # Compute regimes + signals
    df = compute_regimes_and_signals(df)

    # Run backtest
    trades, equity_df, metrics = run_backtest(df, ts_col=ts_col)

    # Save outputs
    trades_df = pd.DataFrame([asdict(t) for t in trades])
    trades_path = outdir / "trades.parquet"
    equity_path = outdir / "equity_curve.parquet"
    metrics_path = outdir / "metrics.json"

    trades_df.to_parquet(trades_path, index=False)
    equity_df.to_parquet(equity_path, index=False)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print(f"[OK] trades={len(trades)} -> {trades_path}")
    print(f"[OK] equity_curve -> {equity_path}")
    print(f"[OK] metrics -> {metrics_path}")

    # Console summary
    print("\n=== METRICS (summary) ===")
    print(f"n_trades     : {metrics['n_trades']}")
    print(f"winrate      : {metrics['winrate']:.3f}")
    print(f"avg_r        : {metrics['avg_r']:.3f}")
    print(f"profit_factor: {metrics['profit_factor']:.3f}")
    print(f"max_dd_r     : {metrics['max_drawdown_r']:.3f}")
    print(f"sum_r        : {metrics['sum_r']:.3f}")
    print("\nby_engine:", metrics["by_engine"])
    print("by_side  :", metrics["by_side"])

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
