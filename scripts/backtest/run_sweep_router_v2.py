from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from pandas.api.types import is_integer_dtype


VERSION = "2026-02-14-mr-v2"

# ============================================================================
# CONFIG (easy to edit)
# ============================================================================
PARQUET_IN = Path(
    "/Users/lolo/PyCharmMiscProject/binance_futures_data_lake/data/research_debug/"
    "BTCUSDT/long_20260101_20260210/joined_20260101_20260210__enriched__router.parquet"
)

TS_COL: Optional[str] = None  # None => auto-detect

ALLOW_LONGS = True
ALLOW_SHORTS = True
ONE_POSITION_MAX = True

VOL_EXCLUDE_NA = True
TREND_VOL_OK = {"MID", "HIGH"}

# Trend signal default thresholds (overridden by cfg sweep)
TREND_D = 0.20
TREND_P = 0.65
TREND_RR = 0.80

# Range setup thresholds (kept as v1/v1_1)
MR_D = 0.20
MR_RR = 0.80
MR_CP_LOW = 0.20
MR_CP_HIGH = 0.80

# Trend risk/exit (unchanged)
TREND_SL_ATR = 1.00
TREND_TP_R = 2.00
TREND_TIME_STOP = 60
MR_TP1_CP = 0.50
MR_TP1_FRACTION = 0.50
MR_BE_OFFSET_R = 0.00

# Sweep space (mini-sweep ciblé):
# - base figée sur le best full-history
# - on teste seulement: session + cap ATR percentile + time stop
SWEEP_TP_CP = [0.60]  # TP2 for MR runner; TP1 fixed at MR_TP1_CP
SWEEP_TP1_FRACTION = [0.60]
SWEEP_MR_SL_ATR = [1.8]
SWEEP_VOL_FILTER_HIGH = [True]
SWEEP_MR_TIME_STOP = [20]
SWEEP_PRIORITY = ["TREND_FIRST"]
SWEEP_MR_D = [0.40]
SWEEP_MR_RR = [1.10]
SWEEP_MR_CP_BANDS = [(0.20, 0.80)]
SWEEP_MR_MEAN_DIST = [0.20]
SWEEP_MR_ATR_PCTL_MAX = [0.90]
SWEEP_MINUTE_GUARD = [5, 10, 15]

TREND_PRESETS = [
    (False, 0.20, 0.65, 0.80),  # trend OFF (MR only session study)
]

SESSION_PRESETS = [
    ("US_15_16", (15, 16)),
    ("US_16_18", (16, 18)),
    ("US_15_16_18", (15, 16, 18)),
]

SHOW_TOP_N = 15
SHOW_DIAGNOSTIC_TOP_N = 5
CSV_OUT = Path(
    "/Users/lolo/PyCharmMiscProject/binance_futures_data_lake/data/research_debug/"
    "BTCUSDT/long_20260101_20260210/backtest_v2_session_15_20_tp_sweep.csv"
)

# Best current fixed config (from latest sweep)
BEST_SINGLE_CFG = {
    "mr_tp1_cp": 0.50,
    "mr_be_offset_r": 0.00,
    "tp_cp": 0.60,
    "tp1_fraction": 0.60,
    "mr_sl_atr": 1.8,
    "vol_filter_high": True,
    "mr_time_stop": 20,
    "priority": "TREND_FIRST",
    "mr_d": 0.40,
    "mr_rr": 1.10,
    "mr_cp_low": 0.20,
    "mr_cp_high": 0.80,
    "mr_mean_dist": 0.20,
    "mr_atr_pctl_max": 0.90,
    "trend_enabled": False,
    "trend_d": 0.20,
    "trend_p": 0.65,
    "trend_rr": 0.80,
    "session_name": "US_15_16",
    "session_hours": "15,16",
    "minute_guard": 15,
}


@dataclass(frozen=True)
class Cfg:
    name: str
    mr_tp1_cp: float
    mr_be_offset_r: float
    tp_cp: float
    tp1_fraction: float
    mr_sl_atr: float
    vol_filter_high: bool
    mr_time_stop: int
    priority: str
    mr_d: float
    mr_rr: float
    mr_cp_low: float
    mr_cp_high: float
    mr_mean_dist: float
    mr_atr_pctl_max: float
    trend_enabled: bool
    trend_d: float
    trend_p: float
    trend_rr: float
    session_name: str
    session_hours: str
    minute_guard: int


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run MR-only sweep on router parquet.")
    p.add_argument("--parquet-in", default=None, help="Override input parquet path")
    p.add_argument("--csv-out", default=None, help="Override output CSV path")
    p.add_argument("--single-config", action="store_true", help="Run only one fixed config (no sweep)")
    p.add_argument("--trades-out", default=None, help="Output CSV path for per-trade results (single-config mode)")
    p.add_argument("--start", default=None, help='UTC start filter inclusive, e.g. "2024-01-01 00:00:00+00:00"')
    p.add_argument("--end", default=None, help='UTC end filter exclusive, e.g. "2026-02-11 00:00:00+00:00"')
    p.add_argument("--mr-v3-experiment", action="store_true", help="Use focused MR v3 mini-grid (8 configs)")
    p.add_argument("--gate-csv", default=None, help="Optional gate CSV with ts + allow_long/allow_short (+size_mult).")
    p.add_argument(
        "--gate-mode",
        default="hard",
        choices=["hard", "size_only", "off"],
        help="Gate behavior: hard=filter directions + size, size_only=ignore direction filter and keep size multiplier only, off=disable gate.",
    )
    return p.parse_args()


def auto_detect_ts_col(df_schema: pd.DataFrame) -> str:
    cols = df_schema.columns.tolist()

    for c in ["t_x", "ts", "timestamp", "time", "datetime", "open_time", "open_ts", "time_utc", "ts_utc"]:
        if c in cols:
            return c

    dt_cols = [c for c in cols if str(df_schema[c].dtype).startswith("datetime64")]
    if len(dt_cols) == 1:
        return dt_cols[0]
    if len(dt_cols) > 1:
        for key in ["t_x", "ts", "time", "open", "close", "date"]:
            for c in dt_cols:
                if key in c.lower():
                    return c
        return dt_cols[0]

    for c in cols:
        cl = c.lower()
        if ("time" in cl or "ts" in cl) and ("ms" in cl or "millis" in cl):
            return c

    raise RuntimeError("Cannot auto-detect timestamp column. Set TS_COL explicitly.")


def normalize_ts_series(s: pd.Series) -> pd.Series:
    if str(s.dtype).startswith("datetime64"):
        return pd.to_datetime(s, utc=True, errors="coerce").dt.tz_convert(None)
    if is_integer_dtype(s.dtype):
        return pd.to_datetime(s, unit="ms", utc=True).dt.tz_convert(None)
    return pd.to_datetime(s, utc=True, errors="coerce").dt.tz_convert(None)


def load_gate_df(gate_csv: Path) -> pd.DataFrame:
    if not gate_csv.exists():
        raise FileNotFoundError(f"Gate CSV not found: {gate_csv}")
    g = pd.read_csv(gate_csv)
    if "ts" not in g.columns:
        raise RuntimeError("Gate CSV must contain 'ts' column.")
    g["ts"] = normalize_ts_series(g["ts"])
    g["ts"] = pd.to_datetime(g["ts"]).astype("datetime64[ns]")
    g = g.dropna(subset=["ts"]).sort_values("ts").reset_index(drop=True)
    if "allow_long" not in g.columns:
        g["allow_long"] = 1
    if "allow_short" not in g.columns:
        g["allow_short"] = 1
    if "size_mult_long" not in g.columns and "size_mult" in g.columns:
        g["size_mult_long"] = g["size_mult"]
    if "size_mult_short" not in g.columns and "size_mult" in g.columns:
        g["size_mult_short"] = g["size_mult"]
    if "size_mult_base" not in g.columns and "size_mult" in g.columns:
        g["size_mult_base"] = g["size_mult"]
    if "size_mult_long" not in g.columns:
        g["size_mult_long"] = 1.0
    if "size_mult_short" not in g.columns:
        g["size_mult_short"] = 1.0
    if "size_mult_base" not in g.columns:
        g["size_mult_base"] = np.maximum(g["size_mult_long"], g["size_mult_short"])

    keep = ["ts", "allow_long", "allow_short", "size_mult_base", "size_mult_long", "size_mult_short"]
    return g[keep].copy()


def build_base_arrays(df: pd.DataFrame, ts_col: str) -> Dict[str, np.ndarray]:
    hour_utc = df[ts_col].dt.hour.to_numpy(dtype=int)
    minute_utc = df[ts_col].dt.minute.to_numpy(dtype=int)
    return {
        "open": df["open"].to_numpy(dtype=float),
        "high": df["high"].to_numpy(dtype=float),
        "low": df["low"].to_numpy(dtype=float),
        "close": df["close"].to_numpy(dtype=float),
        "atr14": df["atr14"].to_numpy(dtype=float),
        "close_pos": df["close_pos"].to_numpy(dtype=float),
        "delta_norm": df["delta_norm"].to_numpy(dtype=float),
        "range_rel": df["range_rel"].to_numpy(dtype=float),
        "atr_pct_pctl_h1": df["atr_pct_pctl_h1"].to_numpy(dtype=float),
        "router_mode_h1": df["router_mode_h1"].astype("string").fillna("NA").to_numpy(dtype=object),
        "dir_state": df["dir_state"].astype("string").fillna("NA").to_numpy(dtype=object),
        "vol_state": df["vol_state"].astype("string").fillna("NA").to_numpy(dtype=object),
        "tradable_final": (df["tradable_final"] == True).to_numpy(dtype=bool),
        "dir_ready": (df["dir_ready"] == True).to_numpy(dtype=bool),
        "gate_allow_long": (df["gate_allow_long"] == True).to_numpy(dtype=bool),
        "gate_allow_short": (df["gate_allow_short"] == True).to_numpy(dtype=bool),
        "gate_size_mult_long": df["gate_size_mult_long"].to_numpy(dtype=float),
        "gate_size_mult_short": df["gate_size_mult_short"].to_numpy(dtype=float),
        "ts": df[ts_col].to_numpy(),
        "hour_utc": hour_utc,
        "minute_utc": minute_utc,
    }


def compute_signals(base: Dict[str, np.ndarray], cfg: Cfg) -> Dict[str, np.ndarray]:
    router = base["router_mode_h1"]
    dstate = base["dir_state"]
    vstate = base["vol_state"]
    tradable = base["tradable_final"]
    dir_ready = base["dir_ready"]
    delta_norm = base["delta_norm"]
    close_pos = base["close_pos"]
    range_rel = base["range_rel"]
    atr_pct_pctl = base["atr_pct_pctl_h1"]
    hour_utc = base["hour_utc"]
    minute_utc = base["minute_utc"]

    vol_ok = (vstate != "NA") if VOL_EXCLUDE_NA else np.ones_like(tradable, dtype=bool)
    if cfg.trend_enabled:
        is_trend = tradable & dir_ready & vol_ok & (router == "TREND") & np.isin(vstate, list(TREND_VOL_OK))
    else:
        is_trend = np.zeros_like(tradable, dtype=bool)
    is_range = tradable & dir_ready & vol_ok & (router == "RANGE")
    session_hours = np.array([int(x) for x in cfg.session_hours.split(",") if x != ""], dtype=int)
    session_mask = np.isin(hour_utc, session_hours)
    if cfg.minute_guard <= 0:
        minute_mask = np.ones_like(session_mask, dtype=bool)
    else:
        minute_mask = (minute_utc >= cfg.minute_guard) & (minute_utc <= (59 - cfg.minute_guard))
    is_range = is_range & session_mask & minute_mask
    if cfg.vol_filter_high:
        is_range = is_range & np.isin(vstate, ["LOW", "MID"])

    impulse_long = (delta_norm > cfg.trend_d) & (close_pos > cfg.trend_p) & (range_rel > cfg.trend_rr)
    impulse_short = (delta_norm < -cfg.trend_d) & (close_pos < (1.0 - cfg.trend_p)) & (range_rel > cfg.trend_rr)

    trend_long_signal = is_trend & (dstate == "BULL") & impulse_long
    trend_short_signal = is_trend & (dstate == "BEAR") & impulse_short

    mean_dist_ok = np.abs(close_pos - 0.5) >= cfg.mr_mean_dist

    # Support both scales for atr percentile:
    # - normalized [0,1]
    # - percentage [0,100]
    finite_atr = atr_pct_pctl[np.isfinite(atr_pct_pctl)]
    if finite_atr.size == 0:
        atr_cap = cfg.mr_atr_pctl_max
    else:
        atr_cap = cfg.mr_atr_pctl_max * 100.0 if float(np.nanmax(finite_atr)) > 1.5 else cfg.mr_atr_pctl_max

    # Missing ATR percentile should not hard-block entries.
    atr_pctl_ok = (~np.isfinite(atr_pct_pctl)) | (atr_pct_pctl <= atr_cap)

    mr_long_setup = (
        is_range
        & mean_dist_ok
        & atr_pctl_ok
        & (close_pos <= cfg.mr_cp_low)
        & (delta_norm <= -cfg.mr_d)
        & (range_rel >= cfg.mr_rr)
        & (dstate == "BULL")
    )
    mr_short_setup = (
        is_range
        & mean_dist_ok
        & atr_pctl_ok
        & (close_pos >= cfg.mr_cp_high)
        & (delta_norm >= cfg.mr_d)
        & (range_rel >= cfg.mr_rr)
        & (dstate == "BEAR")
    )

    conf_long = (delta_norm > 0) | (close_pos > 0.50)
    conf_short = (delta_norm < 0) | (close_pos < 0.50)

    mr_long_setup_prev = np.zeros_like(mr_long_setup, dtype=bool)
    mr_short_setup_prev = np.zeros_like(mr_short_setup, dtype=bool)
    mr_long_setup_prev[1:] = mr_long_setup[:-1]
    mr_short_setup_prev[1:] = mr_short_setup[:-1]

    mr_long_signal = mr_long_setup_prev & is_range & conf_long
    mr_short_signal = mr_short_setup_prev & is_range & conf_short

    return {
        "trend_long_signal": trend_long_signal,
        "trend_short_signal": trend_short_signal,
        "mr_long_signal": mr_long_signal,
        "mr_short_signal": mr_short_signal,
    }


def r_mult(side: str, entry: float, exit_p: float, risk: float) -> float:
    if risk <= 0 or not np.isfinite(risk):
        return 0.0
    return (exit_p - entry) / risk if side == "LONG" else (entry - exit_p) / risk


def profit_factor(rs: List[float]) -> float:
    gains = float(sum(x for x in rs if x > 0))
    losses = float(-sum(x for x in rs if x < 0))
    if losses <= 0:
        return float("inf") if gains > 0 else 0.0
    return gains / losses


def _diag_exit_rates(exit_reasons: List[str]) -> Dict[str, float]:
    n = max(len(exit_reasons), 1)
    sl_like = sum(1 for x in exit_reasons if x in {"SL", "SL_and_TP_same_bar"})
    tp = sum(1 for x in exit_reasons if x in {"TP_CP", "TP2_CP"})
    flip = sum(1 for x in exit_reasons if x in {"FLIP_ROUTER", "VOL_HIGH_KILL"})
    time_stop = sum(1 for x in exit_reasons if x == "TIME_STOP")
    return {
        "SL_rate": sl_like / n,
        "TP_rate": tp / n,
        "FLIP_rate": flip / n,
        "TIME_rate": time_stop / n,
    }


def run_bt(base: Dict[str, np.ndarray], signals: Dict[str, np.ndarray], cfg: Cfg) -> tuple[Dict, pd.DataFrame]:
    pos = None
    all_rs: List[float] = []
    trend_rs: List[float] = []
    range_rs: List[float] = []
    engines: List[str] = []
    mr_exit_reasons: List[str] = []
    trade_rows: List[Dict] = []

    high_arr = base["high"]
    low_arr = base["low"]
    close_arr = base["close"]
    atr_arr = base["atr14"]
    close_pos_arr = base["close_pos"]
    router_arr = base["router_mode_h1"]
    vol_arr = base["vol_state"]
    ts_arr = base["ts"]

    tl_arr = signals["trend_long_signal"]
    tshort_arr = signals["trend_short_signal"]
    ml_arr = signals["mr_long_signal"]
    ms_arr = signals["mr_short_signal"]
    gate_allow_long_arr = base["gate_allow_long"]
    gate_allow_short_arr = base["gate_allow_short"]
    gate_size_mult_long_arr = base["gate_size_mult_long"]
    gate_size_mult_short_arr = base["gate_size_mult_short"]

    def record_trade(exit_i: int, rr_value: float, reason: str) -> None:
        trade_rows.append(
            {
                "engine": pos["engine"],
                "side": pos["side"],
                "entry_i": int(pos["entry_i"]),
                "exit_i": int(exit_i),
                "entry_ts": str(pd.Timestamp(ts_arr[pos["entry_i"]])),
                "exit_ts": str(pd.Timestamp(ts_arr[exit_i])),
                "entry_price": float(pos["entry"]),
                "exit_price": float(close_arr[exit_i]),
                "r_mult": float(rr_value),
                "exit_reason": reason,
            }
        )

    for i in range(len(close_arr)):

        if pos is not None:
            high = float(high_arr[i])
            low = float(low_arr[i])
            close = float(close_arr[i])
            close_pos = float(close_pos_arr[i])
            bars = i - pos["entry_i"]

            router_now = str(router_arr[i])
            vol_now = str(vol_arr[i])

            flip = router_now != pos["router_need"]
            vol_kill = pos["engine"] == "RANGE" and cfg.vol_filter_high and vol_now == "HIGH"
            time_exit = bars >= (TREND_TIME_STOP if pos["engine"] == "TREND" else cfg.mr_time_stop)

            if pos["side"] == "LONG":
                sl_hit = low <= pos["sl"]
                trend_tp_hit = high >= pos["tp"]
                mr_tp1_hit = close_pos >= pos["mr_tp1_cp"]
                mr_tp2_hit = close_pos >= cfg.tp_cp
            else:
                sl_hit = high >= pos["sl"]
                trend_tp_hit = low <= pos["tp"]
                mr_tp1_hit = close_pos <= (1.0 - pos["mr_tp1_cp"])
                mr_tp2_hit = close_pos <= (1.0 - cfg.tp_cp)

            # MR partial management: TP1 (50%) on close_pos mean-revert, then runner to TP2 with BE stop.
            if pos["engine"] == "RANGE":
                if sl_hit:
                    rr_total = pos["rr_accum"] + (pos["qty"] * r_mult(pos["side"], pos["entry"], float(pos["sl"]), pos["risk"]))
                    all_rs.append(rr_total)
                    engines.append(pos["engine"])
                    range_rs.append(rr_total)
                    mr_exit_reasons.append("SL")
                    record_trade(i, rr_total, "SL")
                    pos = None
                    continue

                if not pos["tp1_done"] and mr_tp1_hit:
                    qty_exit = min(float(cfg.tp1_fraction), pos["qty"])
                    rr_piece = qty_exit * r_mult(pos["side"], pos["entry"], close, pos["risk"])
                    pos["rr_accum"] += rr_piece
                    pos["qty"] -= qty_exit
                    pos["tp1_done"] = True
                    if pos["side"] == "LONG":
                        pos["sl"] = pos["entry"] + (pos["risk"] * pos["mr_be_offset_r"])
                    else:
                        pos["sl"] = pos["entry"] - (pos["risk"] * pos["mr_be_offset_r"])
                    if pos["qty"] <= 1e-12:
                        all_rs.append(pos["rr_accum"])
                        engines.append(pos["engine"])
                        range_rs.append(pos["rr_accum"])
                        mr_exit_reasons.append("TP_CP")
                        pos = None
                        continue

                if pos is not None and mr_tp2_hit:
                    rr_total = pos["rr_accum"] + (pos["qty"] * r_mult(pos["side"], pos["entry"], close, pos["risk"]))
                    all_rs.append(rr_total)
                    engines.append(pos["engine"])
                    range_rs.append(rr_total)
                    mr_exit_reasons.append("TP2_CP")
                    record_trade(i, rr_total, "TP2_CP")
                    pos = None
                    continue

                if pos is not None and flip:
                    rr_total = pos["rr_accum"] + (pos["qty"] * r_mult(pos["side"], pos["entry"], close, pos["risk"]))
                    all_rs.append(rr_total)
                    engines.append(pos["engine"])
                    range_rs.append(rr_total)
                    mr_exit_reasons.append("FLIP_ROUTER")
                    record_trade(i, rr_total, "FLIP_ROUTER")
                    pos = None
                    continue

                if pos is not None and vol_kill:
                    rr_total = pos["rr_accum"] + (pos["qty"] * r_mult(pos["side"], pos["entry"], close, pos["risk"]))
                    all_rs.append(rr_total)
                    engines.append(pos["engine"])
                    range_rs.append(rr_total)
                    mr_exit_reasons.append("VOL_HIGH_KILL")
                    record_trade(i, rr_total, "VOL_HIGH_KILL")
                    pos = None
                    continue

                if pos is not None and time_exit:
                    rr_total = pos["rr_accum"] + (pos["qty"] * r_mult(pos["side"], pos["entry"], close, pos["risk"]))
                    all_rs.append(rr_total)
                    engines.append(pos["engine"])
                    range_rs.append(rr_total)
                    mr_exit_reasons.append("TIME_STOP")
                    record_trade(i, rr_total, "TIME_STOP")
                    pos = None
                    continue
            else:
                tp_hit = trend_tp_hit
                exit_reason = None
                exit_price = None

                if sl_hit and tp_hit:
                    exit_reason = "SL_and_TP_same_bar"
                    exit_price = pos["sl"]
                elif sl_hit:
                    exit_reason = "SL"
                    exit_price = pos["sl"]
                elif tp_hit:
                    exit_reason = "TP_TREND"
                    exit_price = pos["tp"]
                elif flip:
                    exit_reason = "FLIP_ROUTER"
                    exit_price = close
                elif time_exit:
                    exit_reason = "TIME_STOP"
                    exit_price = close

                if exit_price is not None:
                    rr = r_mult(pos["side"], pos["entry"], float(exit_price), pos["risk"])
                    all_rs.append(rr)
                    engines.append(pos["engine"])
                    trend_rs.append(rr)
                    record_trade(i, rr, str(exit_reason))
                    pos = None

        if ONE_POSITION_MAX and pos is not None:
            continue

        tl = bool(tl_arr[i]) if ALLOW_LONGS else False
        ts = bool(tshort_arr[i]) if ALLOW_SHORTS else False
        ml = bool(ml_arr[i]) if ALLOW_LONGS else False
        ms = bool(ms_arr[i]) if ALLOW_SHORTS else False

        if not bool(gate_allow_long_arr[i]):
            tl = False
            ml = False
        if not bool(gate_allow_short_arr[i]):
            ts = False
            ms = False

        if cfg.priority == "MR_FIRST":
            picks = [("RANGE", "LONG", ml), ("RANGE", "SHORT", ms), ("TREND", "LONG", tl), ("TREND", "SHORT", ts)]
        else:
            picks = [("TREND", "LONG", tl), ("TREND", "SHORT", ts), ("RANGE", "LONG", ml), ("RANGE", "SHORT", ms)]

        chosen = None
        for eng, side, ok in picks:
            if ok:
                chosen = (eng, side)
                break
        if chosen is None:
            continue

        eng, side = chosen
        size_mult = float(gate_size_mult_long_arr[i]) if side == "LONG" else float(gate_size_mult_short_arr[i])
        if not np.isfinite(size_mult) or size_mult <= 0:
            continue
        entry = float(close_arr[i])
        atr = float(atr_arr[i])
        if not np.isfinite(atr) or atr <= 0:
            continue

        if eng == "TREND":
            sl_dist = TREND_SL_ATR * atr
            tp_r = TREND_TP_R
            router_need = "TREND"
        else:
            sl_dist = float(cfg.mr_sl_atr) * atr
            tp_r = 0.0  # not used for MR v2 TP; still keep a finite placeholder
            router_need = "RANGE"

        if side == "LONG":
            sl = entry - sl_dist
            risk = entry - sl
            tp = entry + tp_r * risk
        else:
            sl = entry + sl_dist
            risk = sl - entry
            tp = entry - tp_r * risk

        if risk <= 0 or not np.isfinite(risk):
            continue

        pos = {
            "engine": eng,
            "side": side,
            "entry_i": i,
            "entry": float(entry),
            "sl": float(sl),
            "tp": float(tp),
            "risk": float(risk),
            "router_need": router_need,
            "qty": float(size_mult),
            "rr_accum": 0.0,
            "tp1_done": False,
            "mr_tp1_cp": float(cfg.mr_tp1_cp),
            "mr_be_offset_r": float(cfg.mr_be_offset_r),
        }

    out_row = {
        "cfg": cfg.name,
        "mr_tp1_cp": cfg.mr_tp1_cp,
        "mr_be_offset_r": cfg.mr_be_offset_r,
        "tp_cp": cfg.tp_cp,
        "tp1_fraction": cfg.tp1_fraction,
        "mr_sl_atr": cfg.mr_sl_atr,
        "vol_filter_high": cfg.vol_filter_high,
        "mr_time_stop": cfg.mr_time_stop,
        "priority": cfg.priority,
        "mr_d": cfg.mr_d,
        "mr_rr": cfg.mr_rr,
        "mr_cp_low": cfg.mr_cp_low,
        "mr_cp_high": cfg.mr_cp_high,
        "mr_mean_dist": cfg.mr_mean_dist,
        "mr_atr_pctl_max": cfg.mr_atr_pctl_max,
        "trend_enabled": cfg.trend_enabled,
        "trend_d": cfg.trend_d,
        "trend_p": cfg.trend_p,
        "trend_rr": cfg.trend_rr,
        "session_name": cfg.session_name,
        "session_hours": cfg.session_hours,
        "minute_guard": cfg.minute_guard,
        "n_trades": int(len(all_rs)),
        "n_trend": int(sum(1 for e in engines if e == "TREND")),
        "n_range": int(sum(1 for e in engines if e == "RANGE")),
        "winrate": float(sum(1 for x in all_rs if x > 0) / max(len(all_rs), 1)),
        "avg_r": float(np.mean(all_rs)) if all_rs else 0.0,
        "pf": float(profit_factor(all_rs)) if all_rs else 0.0,
        "sum_r": float(np.sum(all_rs)) if all_rs else 0.0,
        "trend_avg_r": float(np.mean(trend_rs)) if trend_rs else 0.0,
        "trend_pf": float(profit_factor(trend_rs)) if trend_rs else 0.0,
        "range_avg_r": float(np.mean(range_rs)) if range_rs else 0.0,
        "range_pf": float(profit_factor(range_rs)) if range_rs else 0.0,
        "r_p10": float(np.percentile(all_rs, 10)) if all_rs else 0.0,
        "r_p50": float(np.percentile(all_rs, 50)) if all_rs else 0.0,
        "r_p90": float(np.percentile(all_rs, 90)) if all_rs else 0.0,
    }
    out_row.update(_diag_exit_rates(mr_exit_reasons))
    trades_df = pd.DataFrame(trade_rows)
    return out_row, trades_df


def build_cfgs(mr_v3_experiment: bool = False) -> List[Cfg]:
    if mr_v3_experiment:
        # Focused MR v3 mini-grid (8 configs):
        # - stricter entry (mr_d/mr_rr)
        # - asymmetrical exit via earlier TP1 + BE offset after TP1
        trend_presets = [(False, 0.20, 0.65, 0.80)]
        session_presets = [("US_15_16", (15, 16))]
        sweep_tp_cp = [0.58, 0.60]
        sweep_tp1_fraction = [0.50]
        sweep_mr_sl_atr = [1.8]
        sweep_vol_filter_high = [True]
        sweep_mr_time_stop = [20]
        sweep_priority = ["TREND_FIRST"]
        sweep_mr_d = [0.45, 0.50]
        sweep_mr_rr = [1.20]
        sweep_mr_cp_bands = [(0.20, 0.80)]
        sweep_mr_mean_dist = [0.20]
        sweep_mr_atr_pctl_max = [0.90]
        sweep_minute_guard = [15]
        sweep_mr_tp1_cp = [0.45]
        sweep_mr_be_offset_r = [0.10, 0.20]
    else:
        trend_presets = TREND_PRESETS
        session_presets = SESSION_PRESETS
        sweep_tp_cp = SWEEP_TP_CP
        sweep_tp1_fraction = SWEEP_TP1_FRACTION
        sweep_mr_sl_atr = SWEEP_MR_SL_ATR
        sweep_vol_filter_high = SWEEP_VOL_FILTER_HIGH
        sweep_mr_time_stop = SWEEP_MR_TIME_STOP
        sweep_priority = SWEEP_PRIORITY
        sweep_mr_d = SWEEP_MR_D
        sweep_mr_rr = SWEEP_MR_RR
        sweep_mr_cp_bands = SWEEP_MR_CP_BANDS
        sweep_mr_mean_dist = SWEEP_MR_MEAN_DIST
        sweep_mr_atr_pctl_max = SWEEP_MR_ATR_PCTL_MAX
        sweep_minute_guard = SWEEP_MINUTE_GUARD
        sweep_mr_tp1_cp = [MR_TP1_CP]
        sweep_mr_be_offset_r = [MR_BE_OFFSET_R]

    cfgs: List[Cfg] = []
    i = 0
    for trend_enabled, trend_d, trend_p, trend_rr in trend_presets:
        for session_name, session_hours in session_presets:
            session_hours_csv = ",".join(str(h) for h in session_hours)
            for tp_cp in sweep_tp_cp:
                for tp1_fraction in sweep_tp1_fraction:
                    for mr_sl in sweep_mr_sl_atr:
                        for vol_filter in sweep_vol_filter_high:
                            for mr_time in sweep_mr_time_stop:
                                for priority in sweep_priority:
                                    for mr_d in sweep_mr_d:
                                        for mr_rr in sweep_mr_rr:
                                            for mr_cp_low, mr_cp_high in sweep_mr_cp_bands:
                                                for mr_mean_dist in sweep_mr_mean_dist:
                                                    for mr_atr_pctl_max in sweep_mr_atr_pctl_max:
                                                        for minute_guard in sweep_minute_guard:
                                                            for mr_tp1_cp in sweep_mr_tp1_cp:
                                                                for mr_be_offset_r in sweep_mr_be_offset_r:
                                                                    i += 1
                                                                    cfgs.append(
                                                                        Cfg(
                                                                            name=(
                                                                                f"C{i:05d}_{session_name}_tp2{tp_cp:.2f}_tp1f{tp1_fraction:.2f}_sl{mr_sl:.1f}_high{int(vol_filter)}_t{mr_time}_"
                                                                                f"d{mr_d:.2f}_rr{mr_rr:.2f}_cp{mr_cp_low:.2f}_{mr_cp_high:.2f}_"
                                                                                f"md{mr_mean_dist:.2f}_ap{mr_atr_pctl_max:.2f}_mg{minute_guard}_tp1cp{mr_tp1_cp:.2f}_be{mr_be_offset_r:.2f}_"
                                                                                f"tr{int(trend_enabled)}_td{trend_d:.2f}_tp{trend_p:.2f}_trr{trend_rr:.2f}_{priority}"
                                                                            ),
                                                                            mr_tp1_cp=float(mr_tp1_cp),
                                                                            mr_be_offset_r=float(mr_be_offset_r),
                                                                            tp_cp=float(tp_cp),
                                                                            tp1_fraction=float(tp1_fraction),
                                                                            mr_sl_atr=float(mr_sl),
                                                                            vol_filter_high=bool(vol_filter),
                                                                            mr_time_stop=int(mr_time),
                                                                            priority=priority,
                                                                            mr_d=float(mr_d),
                                                                            mr_rr=float(mr_rr),
                                                                            mr_cp_low=float(mr_cp_low),
                                                                            mr_cp_high=float(mr_cp_high),
                                                                            mr_mean_dist=float(mr_mean_dist),
                                                                            mr_atr_pctl_max=float(mr_atr_pctl_max),
                                                                            trend_enabled=bool(trend_enabled),
                                                                            trend_d=float(trend_d),
                                                                            trend_p=float(trend_p),
                                                                            trend_rr=float(trend_rr),
                                                                            session_name=session_name,
                                                                            session_hours=session_hours_csv,
                                                                            minute_guard=int(minute_guard),
                                                                        )
                                                                    )
    return cfgs


def print_diagnostic(df_results: pd.DataFrame, top_n: int) -> None:
    print("\n=== DIAGNOSTIC (top configs) ===")
    top = df_results.head(top_n).copy()
    for _, r in top.iterrows():
        print(f"\n[{r['cfg']}]")
        print(
            "n_trades={n} n_range={nr} n_trend={nt} avg_r={avg:.4f} pf={pf:.4f}".format(
                n=int(r["n_trades"]),
                nr=int(r["n_range"]),
                nt=int(r["n_trend"]),
                avg=float(r["avg_r"]),
                pf=float(r["pf"]),
            )
        )
        print(
            "TREND(avg_r={tavg:.4f}, pf={tpf:.4f}) | RANGE(avg_r={ravg:.4f}, pf={rpf:.4f})".format(
                tavg=float(r["trend_avg_r"]),
                tpf=float(r["trend_pf"]),
                ravg=float(r["range_avg_r"]),
                rpf=float(r["range_pf"]),
            )
        )
        print(
            "R-dist p10={p10:.4f} p50={p50:.4f} p90={p90:.4f}".format(
                p10=float(r["r_p10"]),
                p50=float(r["r_p50"]),
                p90=float(r["r_p90"]),
            )
        )
        print(
            "MR exits SL={sl:.2%} TP={tp:.2%} FLIP={fl:.2%} TIME={tm:.2%}".format(
                sl=float(r["SL_rate"]),
                tp=float(r["TP_rate"]),
                fl=float(r["FLIP_rate"]),
                tm=float(r["TIME_rate"]),
            )
        )


def main() -> int:
    args = _parse_args()
    parquet_in = Path(args.parquet_in) if args.parquet_in else PARQUET_IN
    csv_out = Path(args.csv_out) if args.csv_out else CSV_OUT

    print(f"[run_sweep_router_v2] VERSION={VERSION}")
    print(f"[INFO] parquet_in={parquet_in}")

    if not parquet_in.exists():
        raise FileNotFoundError(f"Parquet not found: {parquet_in}")

    df_schema = pd.read_parquet(parquet_in, engine="pyarrow")
    ts_col = TS_COL if TS_COL else auto_detect_ts_col(df_schema)
    print(f"[INFO] ts_col={ts_col}")

    required = [
        ts_col,
        "open",
        "high",
        "low",
        "close",
        "router_mode_h1",
        "tradable_final",
        "dir_ready",
        "dir_state",
        "vol_state",
        "delta_norm",
        "close_pos",
        "range_rel",
        "atr14",
        "atr_pct_pctl_h1",
    ]
    missing = [c for c in required if c not in df_schema.columns]
    if missing:
        raise RuntimeError(f"Missing required columns: {missing}")

    df = pd.read_parquet(parquet_in, columns=required, engine="pyarrow")
    df[ts_col] = normalize_ts_series(df[ts_col])
    df[ts_col] = pd.to_datetime(df[ts_col]).astype("datetime64[ns]")
    df = df.sort_values(ts_col, kind="mergesort").reset_index(drop=True)
    if args.start:
        t0 = pd.to_datetime(args.start, utc=True, errors="coerce")
        if pd.isna(t0):
            raise RuntimeError(f"Invalid --start timestamp: {args.start}")
        t0 = t0.tz_convert(None)
        df = df[df[ts_col] >= t0].copy()
    if args.end:
        t1 = pd.to_datetime(args.end, utc=True, errors="coerce")
        if pd.isna(t1):
            raise RuntimeError(f"Invalid --end timestamp: {args.end}")
        t1 = t1.tz_convert(None)
        df = df[df[ts_col] < t1].copy()
    if df.empty:
        raise RuntimeError("No rows after --start/--end filtering.")
    if args.start or args.end:
        print(
            "[INFO] date_filter start={s} end={e} rows={n}".format(
                s=args.start if args.start else "-",
                e=args.end if args.end else "-",
                n=len(df),
            )
        )

    if args.gate_csv and args.gate_mode != "off":
        gate_csv = Path(args.gate_csv)
        gate = load_gate_df(gate_csv)
        df = pd.merge_asof(
            df.sort_values(ts_col),
            gate.sort_values("ts"),
            left_on=ts_col,
            right_on="ts",
            direction="backward",
        )
        df["allow_long"] = df["allow_long"].fillna(0).astype(int)
        df["allow_short"] = df["allow_short"].fillna(0).astype(int)
        df["size_mult_base"] = df["size_mult_base"].fillna(1.0).astype(float)
        df["size_mult_long"] = df["size_mult_long"].fillna(0.0).astype(float)
        df["size_mult_short"] = df["size_mult_short"].fillna(0.0).astype(float)

        if args.gate_mode == "size_only":
            df["gate_allow_long"] = True
            df["gate_allow_short"] = True
            df["gate_size_mult_long"] = df["size_mult_base"]
            df["gate_size_mult_short"] = df["size_mult_base"]
        else:
            df["gate_allow_long"] = df["allow_long"] == 1
            df["gate_allow_short"] = df["allow_short"] == 1
            df["gate_size_mult_long"] = df["size_mult_long"]
            df["gate_size_mult_short"] = df["size_mult_short"]

        print(
            "[INFO] gate_csv={g} gate_mode={m} allow_long_rate={lr:.4f} allow_short_rate={sr:.4f}".format(
                g=gate_csv,
                m=args.gate_mode,
                lr=float(df["gate_allow_long"].mean()),
                sr=float(df["gate_allow_short"].mean()),
            )
        )
    else:
        df["gate_allow_long"] = True
        df["gate_allow_short"] = True
        df["gate_size_mult_long"] = 1.0
        df["gate_size_mult_short"] = 1.0

    base = build_base_arrays(df, ts_col=ts_col)
    if args.single_config:
        single = Cfg(
            name="BEST_SINGLE_CFG",
            mr_tp1_cp=float(BEST_SINGLE_CFG.get("mr_tp1_cp", MR_TP1_CP)),
            mr_be_offset_r=float(BEST_SINGLE_CFG.get("mr_be_offset_r", MR_BE_OFFSET_R)),
            tp_cp=float(BEST_SINGLE_CFG["tp_cp"]),
            tp1_fraction=float(BEST_SINGLE_CFG["tp1_fraction"]),
            mr_sl_atr=float(BEST_SINGLE_CFG["mr_sl_atr"]),
            vol_filter_high=bool(BEST_SINGLE_CFG["vol_filter_high"]),
            mr_time_stop=int(BEST_SINGLE_CFG["mr_time_stop"]),
            priority=str(BEST_SINGLE_CFG["priority"]),
            mr_d=float(BEST_SINGLE_CFG["mr_d"]),
            mr_rr=float(BEST_SINGLE_CFG["mr_rr"]),
            mr_cp_low=float(BEST_SINGLE_CFG["mr_cp_low"]),
            mr_cp_high=float(BEST_SINGLE_CFG["mr_cp_high"]),
            mr_mean_dist=float(BEST_SINGLE_CFG["mr_mean_dist"]),
            mr_atr_pctl_max=float(BEST_SINGLE_CFG["mr_atr_pctl_max"]),
            trend_enabled=bool(BEST_SINGLE_CFG["trend_enabled"]),
            trend_d=float(BEST_SINGLE_CFG["trend_d"]),
            trend_p=float(BEST_SINGLE_CFG["trend_p"]),
            trend_rr=float(BEST_SINGLE_CFG["trend_rr"]),
            session_name=str(BEST_SINGLE_CFG["session_name"]),
            session_hours=str(BEST_SINGLE_CFG["session_hours"]),
            minute_guard=int(BEST_SINGLE_CFG["minute_guard"]),
        )
        signals = compute_signals(base, single)
        row, trades_df = run_bt(base, signals, single)
        out = pd.DataFrame([row])
        print("[INFO] mode=single-config")
        print(out[["cfg", "pf", "avg_r", "sum_r", "n_trades", "n_range", "n_trend", "winrate"]].to_string(index=False))

        trades_out = Path(args.trades_out) if args.trades_out else csv_out.with_name(csv_out.stem + "_trades.csv")
        csv_out.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(csv_out, index=False)
        trades_out.parent.mkdir(parents=True, exist_ok=True)
        trades_df.to_csv(trades_out, index=False)
        print(f"[OK] CSV exported: {csv_out}")
        print(f"[OK] Trades exported: {trades_out}")
        return 0

    cfgs = build_cfgs(mr_v3_experiment=bool(args.mr_v3_experiment))
    if args.mr_v3_experiment:
        print("[INFO] mode=mr-v3-experiment")
    print(f"[INFO] sweep configs={len(cfgs)}")

    rows: List[Dict] = []
    for cfg in cfgs:
        signals = compute_signals(base, cfg)
        row, _ = run_bt(base, signals, cfg)
        rows.append(row)

    out = pd.DataFrame(rows).sort_values(["pf", "avg_r"], ascending=[False, False]).reset_index(drop=True)

    cols_show = [
        "cfg",
        "mr_tp1_cp",
        "mr_be_offset_r",
        "pf",
        "avg_r",
        "sum_r",
        "n_trades",
        "n_range",
        "n_trend",
        "tp_cp",
        "tp1_fraction",
        "mr_sl_atr",
        "vol_filter_high",
        "mr_time_stop",
        "priority",
        "mr_d",
        "mr_rr",
        "mr_cp_low",
        "mr_cp_high",
        "mr_mean_dist",
        "mr_atr_pctl_max",
        "trend_enabled",
        "trend_d",
        "trend_p",
        "trend_rr",
        "session_name",
        "session_hours",
        "minute_guard",
    ]
    print("\n=== SWEEP RESULTS (sorted by pf, avg_r) ===")
    print(out[cols_show].head(SHOW_TOP_N).to_string(index=False))

    out_with_range = out[out["n_range"] > 0].copy()
    if not out_with_range.empty:
        print("\n=== SWEEP RESULTS (n_range > 0) ===")
        print(out_with_range[cols_show].head(SHOW_TOP_N).to_string(index=False))
    else:
        print("\n=== SWEEP RESULTS (n_range > 0) ===")
        print("[WARN] No configurations produced RANGE trades.")

    csv_out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(csv_out, index=False)
    print(f"\n[OK] CSV exported: {csv_out}")

    print_diagnostic(out, top_n=SHOW_DIAGNOSTIC_TOP_N)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
