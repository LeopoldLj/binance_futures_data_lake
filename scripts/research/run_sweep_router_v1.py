@'
# C:\Users\lolo_\PycharmProjects\binance_futures_data_lake\scripts\backtest\run_sweep_router_v1.py
# run_sweep_router_v1.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

VERSION = "2026-02-14a"

PARQUET_IN = r"C:\Users\lolo_\PycharmProjects\binance_futures_data_lake\data\research_debug\BTCUSDT\long_20260101_20260210\joined_20260101_20260210__enriched__router.parquet"


@dataclass
class Cfg:
    name: str
    range_allow_high: bool
    mr_sl_atr: float
    mr_tp_r: float
    mr_be_at_r: Optional[float]
    priority: str  # "TREND_FIRST" or "MR_FIRST"


def auto_detect_ts_col(df_schema: pd.DataFrame) -> str:
    cols = df_schema.columns.tolist()
    dt_cols = [c for c in cols if str(df_schema[c].dtype).startswith("datetime64")]
    if len(dt_cols) == 1:
        return dt_cols[0]
    for c in ["t_x", "ts", "timestamp", "time", "datetime", "open_time", "open_ts"]:
        if c in cols:
            return c
    raise RuntimeError("Cannot detect ts column.")


def normalize_ts_series(s: pd.Series) -> pd.Series:
    if str(s.dtype).startswith("datetime64"):
        return s
    if np.issubdtype(s.dtype, np.integer):
        return pd.to_datetime(s, unit="ms", utc=True).dt.tz_convert(None)
    return pd.to_datetime(s, utc=True, errors="coerce").dt.tz_convert(None)


def compute_signals(df: pd.DataFrame, cfg: Cfg) -> pd.DataFrame:
    TREND_VOL_OK = {"MID", "HIGH"}
    TREND_D = 0.20
    TREND_P = 0.65
    TREND_RR = 0.80

    MR_D = 0.20
    MR_RR = 0.80
    MR_CP_LOW = 0.20
    MR_CP_HIGH = 0.80

    df["router_mode_h1"] = df["router_mode_h1"].astype("string")
    df["dir_state"] = df["dir_state"].astype("string")
    df["vol_state"] = df["vol_state"].astype("string")

    vol_ok = df["vol_state"] != "NA"
    tradable = df["tradable_final"] == True
    dir_ready = df["dir_ready"] == True

    is_trend = tradable & dir_ready & vol_ok & (df["router_mode_h1"] == "TREND") & df["vol_state"].isin(list(TREND_VOL_OK))

    if cfg.range_allow_high:
        is_range = tradable & dir_ready & vol_ok & (df["router_mode_h1"] == "RANGE")
    else:
        is_range = tradable & dir_ready & vol_ok & (df["router_mode_h1"] == "RANGE") & df["vol_state"].isin(["LOW", "MID"])

    impulse_long = (df["delta_norm"] > TREND_D) & (df["close_pos"] > TREND_P) & (df["range_rel"] > TREND_RR)
    impulse_short = (df["delta_norm"] < -TREND_D) & (df["close_pos"] < (1.0 - TREND_P)) & (df["range_rel"] > TREND_RR)

    df["trend_long_signal"] = is_trend & (df["dir_state"] == "BULL") & impulse_long
    df["trend_short_signal"] = is_trend & (df["dir_state"] == "BEAR") & impulse_short

    mr_long_setup = (df["close_pos"] <= MR_CP_LOW) & (df["delta_norm"] <= -MR_D) & (df["range_rel"] >= MR_RR) & (df["dir_state"] == "BULL")
    mr_short_setup = (df["close_pos"] >= MR_CP_HIGH) & (df["delta_norm"] >= MR_D) & (df["range_rel"] >= MR_RR) & (df["dir_state"] == "BEAR")

    df["mr_long_setup"] = is_range & mr_long_setup
    df["mr_short_setup"] = is_range & mr_short_setup

    conf_long = (df["delta_norm"] > 0) | (df["close_pos"] > 0.50)
    conf_short = (df["delta_norm"] < 0) | (df["close_pos"] < 0.50)

    df["mr_long_signal"] = df["mr_long_setup"].shift(1, fill_value=False) & is_range & conf_long
    df["mr_short_signal"] = df["mr_short_setup"].shift(1, fill_value=False) & is_range & conf_short

    return df


def r_mult(side: str, entry: float, exit_p: float, risk: float) -> float:
    if risk <= 0 or not np.isfinite(risk):
        return 0.0
    return (exit_p - entry) / risk if side == "LONG" else (entry - exit_p) / risk


def profit_factor(rs: List[float]) -> float:
    g = sum([x for x in rs if x > 0])
    l = -sum([x for x in rs if x < 0])
    if l <= 0:
        return float("inf") if g > 0 else 0.0
    return g / l


def run_bt(df: pd.DataFrame, cfg: Cfg) -> Dict:
    TREND_SL_ATR = 1.00
    TREND_TP_R = 2.00
    TREND_TIME_STOP = 60

    MR_SL_ATR = float(cfg.mr_sl_atr)
    MR_TP_R = float(cfg.mr_tp_r)
    MR_TIME_STOP = 30
    MR_BE_AT = cfg.mr_be_at_r
    MR_BE_OFFSET_R = 0.0

    pos = None
    rs: List[float] = []
    engines: List[str] = []

    for i in range(len(df)):
        row = df.iloc[i]

        if pos is not None:
            high = float(row["high"])
            low = float(row["low"])
            close = float(row["close"])
            bars = i - pos["entry_i"]

            router_now = str(row["router_mode_h1"])
            flip = router_now != pos["router_need"]

            if pos["engine"] == "RANGE" and MR_BE_AT is not None and not pos["be"]:
                ur = r_mult(pos["side"], pos["entry"], close, pos["risk"])
                if ur >= float(MR_BE_AT):
                    if pos["side"] == "LONG":
                        pos["sl"] = pos["entry"] + pos["risk"] * MR_BE_OFFSET_R
                    else:
                        pos["sl"] = pos["entry"] - pos["risk"] * MR_BE_OFFSET_R
                    pos["be"] = True

            sl_hit = (low <= pos["sl"]) if pos["side"] == "LONG" else (high >= pos["sl"])
            tp_hit = (high >= pos["tp"]) if pos["side"] == "LONG" else (low <= pos["tp"])

            time_exit = bars >= (TREND_TIME_STOP if pos["engine"] == "TREND" else MR_TIME_STOP)

            exit_p = None
            if sl_hit and tp_hit:
                exit_p = pos["sl"]
            elif sl_hit:
                exit_p = pos["sl"]
            elif tp_hit:
                exit_p = pos["tp"]
            elif flip or time_exit:
                exit_p = close

            if exit_p is not None:
                rr = r_mult(pos["side"], pos["entry"], float(exit_p), pos["risk"])
                rs.append(rr)
                engines.append(pos["engine"])
                pos = None

        if pos is None:
            tl = bool(row["trend_long_signal"])
            ts = bool(row["trend_short_signal"])
            ml = bool(row["mr_long_signal"])
            ms = bool(row["mr_short_signal"])

            if cfg.priority == "MR_FIRST":
                picks = [("RANGE","LONG",ml),("RANGE","SHORT",ms),("TREND","LONG",tl),("TREND","SHORT",ts)]
            else:
                picks = [("TREND","LONG",tl),("TREND","SHORT",ts),("RANGE","LONG",ml),("RANGE","SHORT",ms)]

            chosen = None
            for eng, side, ok in picks:
                if ok:
                    chosen = (eng, side)
                    break

            if chosen is not None:
                eng, side = chosen
                entry = float(row["close"])
                atr = float(row["atr14"])
                if not np.isfinite(atr) or atr <= 0:
                    continue

                if eng == "TREND":
                    sl_dist = TREND_SL_ATR * atr
                    tp_r = TREND_TP_R
                    router_need = "TREND"
                else:
                    sl_dist = MR_SL_ATR * atr
                    tp_r = MR_TP_R
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

                pos = {"engine": eng, "side": side, "entry_i": i, "entry": entry, "sl": float(sl), "tp": float(tp), "risk": float(risk), "router_need": router_need, "be": False}

    return {
        "cfg": cfg.name,
        "n_trades": int(len(rs)),
        "winrate": float(sum([1 for x in rs if x > 0]) / max(len(rs), 1)),
        "avg_r": float(np.mean(rs)) if len(rs) else 0.0,
        "pf": float(profit_factor(rs)) if len(rs) else 0.0,
        "sum_r": float(np.sum(rs)) if len(rs) else 0.0,
        "n_trend": int(sum([1 for e in engines if e == "TREND"])),
        "n_range": int(sum([1 for e in engines if e == "RANGE"])),
    }


def main() -> int:
    print(f"[run_sweep_router] VERSION={VERSION}")

    df_schema = pd.read_parquet(PARQUET_IN, engine="pyarrow")
    ts_col = auto_detect_ts_col(df_schema)
    print("[INFO] ts_col=", ts_col)

    cols = [
        ts_col, "open","high","low","close",
        "router_mode_h1","tradable_final","dir_ready","dir_state","vol_state",
        "delta_norm","close_pos","range_rel","atr14",
    ]
    df = pd.read_parquet(PARQUET_IN, columns=cols, engine="pyarrow")
    df[ts_col] = normalize_ts_series(df[ts_col])
    df = df.sort_values(ts_col, kind="mergesort").reset_index(drop=True)

    cfgs = [
        Cfg("BASE_TREND_FIRST", True, 0.8, 1.2, 0.6, "TREND_FIRST"),
        Cfg("A1_noHIGH_TREND_FIRST", False, 0.8, 1.2, 0.6, "TREND_FIRST"),
        Cfg("B1_MR_widerSL_closeTP_TREND_FIRST", True, 1.2, 0.8, 0.8, "TREND_FIRST"),
        Cfg("A1B1_TREND_FIRST", False, 1.2, 0.8, 0.8, "TREND_FIRST"),
        Cfg("A1B1_MR_FIRST", False, 1.2, 0.8, 0.8, "MR_FIRST"),
        Cfg("A1_MR_FIRST", False, 0.8, 1.2, 0.6, "MR_FIRST"),
    ]

    rows = []
    for cfg in cfgs:
        dfx = df.copy()
        dfx = compute_signals(dfx, cfg)
        rows.append(run_bt(dfx, cfg))

    out = pd.DataFrame(rows).sort_values(["pf","avg_r"], ascending=[False, False]).reset_index(drop=True)
    print("\n=== SWEEP RESULTS (sorted by pf, avg_r) ===")
    print(out.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'@ | Set-Content -Encoding UTF8 .\scripts\backtest\run_sweep_router_v1.py
