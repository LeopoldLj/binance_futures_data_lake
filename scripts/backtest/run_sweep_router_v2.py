from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


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

# Trend signal thresholds (kept as v1/v1_1)
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

# Sweep space (MR v2)
SWEEP_TP_CP = [0.50, 0.55]
SWEEP_MR_SL_ATR = [0.8, 1.0, 1.2]
SWEEP_VOL_FILTER_HIGH = [False, True]
SWEEP_MR_TIME_STOP = [20, 30]
SWEEP_PRIORITY = ["TREND_FIRST", "MR_FIRST"]  # or MR_FIRST
SWEEP_MR_D = [0.20, 0.25, 0.30]
SWEEP_MR_RR = [0.80, 0.90, 1.00]
SWEEP_MR_CP_BANDS = [(0.20, 0.80), (0.15, 0.85)]
SWEEP_MR_MEAN_DIST = [0.20, 0.25, 0.30]
SWEEP_MR_ATR_PCTL_MAX = [0.80, 0.85, 0.90]

SHOW_TOP_N = 15
SHOW_DIAGNOSTIC_TOP_N = 5
CSV_OUT = Path(
    "/Users/lolo/PyCharmMiscProject/binance_futures_data_lake/data/research_debug/"
    "BTCUSDT/long_20260101_20260210/backtest_v2_entry_sweep.csv"
)


@dataclass(frozen=True)
class Cfg:
    name: str
    tp_cp: float
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
        return s
    if np.issubdtype(s.dtype, np.integer):
        return pd.to_datetime(s, unit="ms", utc=True).dt.tz_convert(None)
    return pd.to_datetime(s, utc=True, errors="coerce").dt.tz_convert(None)


def build_base_arrays(df: pd.DataFrame, ts_col: str) -> Dict[str, np.ndarray]:
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
        "ts": df[ts_col].to_numpy(),
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

    vol_ok = (vstate != "NA") if VOL_EXCLUDE_NA else np.ones_like(tradable, dtype=bool)
    is_trend = tradable & dir_ready & vol_ok & (router == "TREND") & np.isin(vstate, list(TREND_VOL_OK))
    is_range = tradable & dir_ready & vol_ok & (router == "RANGE")
    if cfg.vol_filter_high:
        is_range = is_range & np.isin(vstate, ["LOW", "MID"])

    impulse_long = (delta_norm > TREND_D) & (close_pos > TREND_P) & (range_rel > TREND_RR)
    impulse_short = (delta_norm < -TREND_D) & (close_pos < (1.0 - TREND_P)) & (range_rel > TREND_RR)

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
    tp = sum(1 for x in exit_reasons if x == "TP_CP")
    flip = sum(1 for x in exit_reasons if x in {"FLIP_ROUTER", "VOL_HIGH_KILL"})
    time_stop = sum(1 for x in exit_reasons if x == "TIME_STOP")
    return {
        "SL_rate": sl_like / n,
        "TP_rate": tp / n,
        "FLIP_rate": flip / n,
        "TIME_rate": time_stop / n,
    }


def run_bt(base: Dict[str, np.ndarray], signals: Dict[str, np.ndarray], cfg: Cfg) -> Dict:
    pos = None
    all_rs: List[float] = []
    trend_rs: List[float] = []
    range_rs: List[float] = []
    engines: List[str] = []
    mr_exit_reasons: List[str] = []

    high_arr = base["high"]
    low_arr = base["low"]
    close_arr = base["close"]
    atr_arr = base["atr14"]
    close_pos_arr = base["close_pos"]
    router_arr = base["router_mode_h1"]
    vol_arr = base["vol_state"]

    tl_arr = signals["trend_long_signal"]
    ts_arr = signals["trend_short_signal"]
    ml_arr = signals["mr_long_signal"]
    ms_arr = signals["mr_short_signal"]

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
                mr_tp_hit = close_pos >= cfg.tp_cp
            else:
                sl_hit = high >= pos["sl"]
                trend_tp_hit = low <= pos["tp"]
                mr_tp_hit = close_pos <= (1.0 - cfg.tp_cp)

            tp_hit = trend_tp_hit if pos["engine"] == "TREND" else mr_tp_hit

            exit_reason = None
            exit_price = None

            if sl_hit and tp_hit:
                exit_reason = "SL_and_TP_same_bar"
                exit_price = pos["sl"]
            elif sl_hit:
                exit_reason = "SL"
                exit_price = pos["sl"]
            elif tp_hit:
                exit_reason = "TP_TREND" if pos["engine"] == "TREND" else "TP_CP"
                exit_price = pos["tp"] if pos["engine"] == "TREND" else close
            elif flip:
                exit_reason = "FLIP_ROUTER"
                exit_price = close
            elif vol_kill:
                exit_reason = "VOL_HIGH_KILL"
                exit_price = close
            elif time_exit:
                exit_reason = "TIME_STOP"
                exit_price = close

            if exit_price is not None:
                rr = r_mult(pos["side"], pos["entry"], float(exit_price), pos["risk"])
                all_rs.append(rr)
                engines.append(pos["engine"])
                if pos["engine"] == "TREND":
                    trend_rs.append(rr)
                else:
                    range_rs.append(rr)
                    mr_exit_reasons.append(str(exit_reason))
                pos = None

        if ONE_POSITION_MAX and pos is not None:
            continue

        tl = bool(tl_arr[i]) if ALLOW_LONGS else False
        ts = bool(ts_arr[i]) if ALLOW_SHORTS else False
        ml = bool(ml_arr[i]) if ALLOW_LONGS else False
        ms = bool(ms_arr[i]) if ALLOW_SHORTS else False

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
        }

    out_row = {
        "cfg": cfg.name,
        "tp_cp": cfg.tp_cp,
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
    return out_row


def build_cfgs() -> List[Cfg]:
    cfgs: List[Cfg] = []
    i = 0
    for tp_cp in SWEEP_TP_CP:
        for mr_sl in SWEEP_MR_SL_ATR:
            for vol_filter in SWEEP_VOL_FILTER_HIGH:
                for mr_time in SWEEP_MR_TIME_STOP:
                    for priority in SWEEP_PRIORITY:
                        for mr_d in SWEEP_MR_D:
                            for mr_rr in SWEEP_MR_RR:
                                for mr_cp_low, mr_cp_high in SWEEP_MR_CP_BANDS:
                                    for mr_mean_dist in SWEEP_MR_MEAN_DIST:
                                        for mr_atr_pctl_max in SWEEP_MR_ATR_PCTL_MAX:
                                            i += 1
                                            cfgs.append(
                                                Cfg(
                                                    name=(
                                                        f"C{i:05d}_tp{tp_cp:.2f}_sl{mr_sl:.1f}_high{int(vol_filter)}_t{mr_time}_"
                                                        f"d{mr_d:.2f}_rr{mr_rr:.2f}_cp{mr_cp_low:.2f}_{mr_cp_high:.2f}_"
                                                        f"md{mr_mean_dist:.2f}_ap{mr_atr_pctl_max:.2f}_{priority}"
                                                    ),
                                                    tp_cp=float(tp_cp),
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
    print(f"[run_sweep_router_v2] VERSION={VERSION}")
    print(f"[INFO] parquet_in={PARQUET_IN}")

    if not PARQUET_IN.exists():
        raise FileNotFoundError(f"Parquet not found: {PARQUET_IN}")

    df_schema = pd.read_parquet(PARQUET_IN, engine="pyarrow")
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

    df = pd.read_parquet(PARQUET_IN, columns=required, engine="pyarrow")
    df[ts_col] = normalize_ts_series(df[ts_col])
    df = df.sort_values(ts_col, kind="mergesort").reset_index(drop=True)

    cfgs = build_cfgs()
    print(f"[INFO] sweep configs={len(cfgs)}")

    base = build_base_arrays(df, ts_col=ts_col)

    rows: List[Dict] = []
    for cfg in cfgs:
        signals = compute_signals(base, cfg)
        rows.append(run_bt(base, signals, cfg))

    out = pd.DataFrame(rows).sort_values(["pf", "avg_r"], ascending=[False, False]).reset_index(drop=True)

    cols_show = [
        "cfg",
        "pf",
        "avg_r",
        "sum_r",
        "n_trades",
        "n_range",
        "n_trend",
        "tp_cp",
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

    CSV_OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(CSV_OUT, index=False)
    print(f"\n[OK] CSV exported: {CSV_OUT}")

    print_diagnostic(out, top_n=SHOW_DIAGNOSTIC_TOP_N)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
