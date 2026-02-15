from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


VERSION = "2026-02-14-v3-breakout"

PARQUET_IN = Path(
    "/Users/lolo/PyCharmMiscProject/binance_futures_data_lake/data/research_debug/"
    "BTCUSDT/full_201910_202602/joined_full__enriched__router.parquet"
)
CSV_OUT = Path(
    "/Users/lolo/PyCharmMiscProject/binance_futures_data_lake/data/research_debug/"
    "BTCUSDT/full_201910_202602/breakout_v3_sweep.csv"
)

TS_COL: Optional[str] = None
ALLOW_LONGS = True
ALLOW_SHORTS = True
ONE_POSITION_MAX = True
TREND_VOL_OK = {"MID", "HIGH"}
VOL_EXCLUDE_NA = True

SWEEP_SESSION = [("US_15_16", (15, 16))]
SWEEP_MINUTE_GUARD = [10, 15]
SWEEP_TREND_D = [0.25, 0.30]
SWEEP_TREND_P = [0.60, 0.65]
SWEEP_TREND_RR = [0.90, 1.10]
SWEEP_SL_ATR = [1.00]
SWEEP_TP_R = [2.00, 2.50]
SWEEP_TIME_STOP = [45, 60]

SHOW_TOP_N = 15
SHOW_DIAGNOSTIC_TOP_N = 5

BEST_SINGLE_CFG = {
    "trend_d": 0.30,
    "trend_p": 0.65,
    "trend_rr": 1.10,
    "trend_sl_atr": 1.00,
    "trend_tp_r": 2.50,
    "trend_time_stop": 60,
    "session_name": "US_15_16",
    "session_hours": "15,16",
    "minute_guard": 15,
}


@dataclass(frozen=True)
class Cfg:
    name: str
    trend_d: float
    trend_p: float
    trend_rr: float
    trend_sl_atr: float
    trend_tp_r: float
    trend_time_stop: int
    session_name: str
    session_hours: str
    minute_guard: int


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Trend-only breakout sweep on router parquet.")
    p.add_argument("--parquet-in", default=None)
    p.add_argument("--csv-out", default=None)
    p.add_argument("--single-config", action="store_true")
    p.add_argument("--trades-out", default=None)
    p.add_argument("--start", default=None, help='UTC inclusive start, e.g. "2024-01-01 00:00:00+00:00"')
    p.add_argument("--end", default=None, help='UTC exclusive end, e.g. "2026-02-11 00:00:00+00:00"')
    return p.parse_args()


def auto_detect_ts_col(df_schema: pd.DataFrame) -> str:
    cols = df_schema.columns.tolist()
    for c in ["t_x", "ts", "timestamp", "time", "datetime", "open_time", "open_ts", "time_utc", "ts_utc"]:
        if c in cols:
            return c
    dt_cols = [c for c in cols if str(df_schema[c].dtype).startswith("datetime64")]
    if dt_cols:
        return dt_cols[0]
    raise RuntimeError("Cannot auto-detect timestamp column")


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
        "router_mode_h1": df["router_mode_h1"].astype("string").fillna("NA").to_numpy(dtype=object),
        "dir_state": df["dir_state"].astype("string").fillna("NA").to_numpy(dtype=object),
        "vol_state": df["vol_state"].astype("string").fillna("NA").to_numpy(dtype=object),
        "tradable_final": (df["tradable_final"] == True).to_numpy(dtype=bool),
        "dir_ready": (df["dir_ready"] == True).to_numpy(dtype=bool),
        "ts": df[ts_col].to_numpy(),
        "hour_utc": df[ts_col].dt.hour.to_numpy(dtype=int),
        "minute_utc": df[ts_col].dt.minute.to_numpy(dtype=int),
    }


def profit_factor(rs: List[float]) -> float:
    gains = float(sum(x for x in rs if x > 0))
    losses = float(-sum(x for x in rs if x < 0))
    if losses <= 0:
        return float("inf") if gains > 0 else 0.0
    return gains / losses


def r_mult(side: str, entry: float, exit_p: float, risk: float) -> float:
    if risk <= 0 or not np.isfinite(risk):
        return 0.0
    return (exit_p - entry) / risk if side == "LONG" else (entry - exit_p) / risk


def compute_signals(base: Dict[str, np.ndarray], cfg: Cfg) -> Dict[str, np.ndarray]:
    router = base["router_mode_h1"]
    dstate = base["dir_state"]
    vstate = base["vol_state"]
    tradable = base["tradable_final"]
    dir_ready = base["dir_ready"]
    delta_norm = base["delta_norm"]
    close_pos = base["close_pos"]
    range_rel = base["range_rel"]
    hour_utc = base["hour_utc"]
    minute_utc = base["minute_utc"]

    vol_ok = (vstate != "NA") if VOL_EXCLUDE_NA else np.ones_like(tradable, dtype=bool)
    session_hours = np.array([int(x) for x in cfg.session_hours.split(",") if x != ""], dtype=int)
    session_mask = np.isin(hour_utc, session_hours)
    if cfg.minute_guard <= 0:
        minute_mask = np.ones_like(session_mask, dtype=bool)
    else:
        minute_mask = (minute_utc >= cfg.minute_guard) & (minute_utc <= (59 - cfg.minute_guard))

    is_trend = (
        tradable
        & dir_ready
        & vol_ok
        & np.isin(vstate, list(TREND_VOL_OK))
        & (router == "TREND")
        & session_mask
        & minute_mask
    )
    impulse_long = (delta_norm > cfg.trend_d) & (close_pos > cfg.trend_p) & (range_rel > cfg.trend_rr)
    impulse_short = (delta_norm < -cfg.trend_d) & (close_pos < (1.0 - cfg.trend_p)) & (range_rel > cfg.trend_rr)
    return {
        "trend_long_signal": is_trend & (dstate == "BULL") & impulse_long,
        "trend_short_signal": is_trend & (dstate == "BEAR") & impulse_short,
    }


def run_bt(base: Dict[str, np.ndarray], signals: Dict[str, np.ndarray], cfg: Cfg) -> tuple[Dict, pd.DataFrame]:
    pos = None
    rs: List[float] = []
    exit_reasons: List[str] = []
    trades: List[Dict] = []

    for i in range(len(base["close"])):
        if pos is not None:
            high = float(base["high"][i])
            low = float(base["low"][i])
            close = float(base["close"][i])
            bars = i - pos["entry_i"]

            flip = str(base["router_mode_h1"][i]) != "TREND"
            time_exit = bars >= cfg.trend_time_stop

            if pos["side"] == "LONG":
                sl_hit = low <= pos["sl"]
                tp_hit = high >= pos["tp"]
            else:
                sl_hit = high >= pos["sl"]
                tp_hit = low <= pos["tp"]

            reason = None
            exit_price = None
            if sl_hit and tp_hit:
                reason = "SL_and_TP_same_bar"
                exit_price = pos["sl"]
            elif sl_hit:
                reason = "SL"
                exit_price = pos["sl"]
            elif tp_hit:
                reason = "TP_TREND"
                exit_price = pos["tp"]
            elif flip:
                reason = "FLIP_ROUTER"
                exit_price = close
            elif time_exit:
                reason = "TIME_STOP"
                exit_price = close

            if exit_price is not None:
                rr = r_mult(pos["side"], pos["entry"], float(exit_price), pos["risk"])
                rs.append(rr)
                exit_reasons.append(str(reason))
                trades.append(
                    {
                        "engine": "TREND",
                        "side": pos["side"],
                        "entry_i": int(pos["entry_i"]),
                        "exit_i": int(i),
                        "entry_ts": str(pd.Timestamp(base["ts"][pos["entry_i"]])),
                        "exit_ts": str(pd.Timestamp(base["ts"][i])),
                        "entry_price": float(pos["entry"]),
                        "exit_price": float(exit_price),
                        "r_mult": float(rr),
                        "exit_reason": str(reason),
                    }
                )
                pos = None

        if ONE_POSITION_MAX and pos is not None:
            continue

        tl = bool(signals["trend_long_signal"][i]) if ALLOW_LONGS else False
        ts = bool(signals["trend_short_signal"][i]) if ALLOW_SHORTS else False
        if not (tl or ts):
            continue

        side = "LONG" if tl else "SHORT"
        entry = float(base["close"][i])
        atr = float(base["atr14"][i])
        if not np.isfinite(atr) or atr <= 0:
            continue
        sl_dist = cfg.trend_sl_atr * atr
        if side == "LONG":
            sl = entry - sl_dist
            risk = entry - sl
            tp = entry + (cfg.trend_tp_r * risk)
        else:
            sl = entry + sl_dist
            risk = sl - entry
            tp = entry - (cfg.trend_tp_r * risk)
        if risk <= 0:
            continue
        pos = {
            "entry_i": i,
            "side": side,
            "entry": float(entry),
            "sl": float(sl),
            "tp": float(tp),
            "risk": float(risk),
        }

    row = {
        "cfg": cfg.name,
        "trend_d": cfg.trend_d,
        "trend_p": cfg.trend_p,
        "trend_rr": cfg.trend_rr,
        "trend_sl_atr": cfg.trend_sl_atr,
        "trend_tp_r": cfg.trend_tp_r,
        "trend_time_stop": cfg.trend_time_stop,
        "session_name": cfg.session_name,
        "session_hours": cfg.session_hours,
        "minute_guard": cfg.minute_guard,
        "n_trades": int(len(rs)),
        "n_trend": int(len(rs)),
        "n_range": 0,
        "winrate": float(sum(1 for x in rs if x > 0) / max(len(rs), 1)),
        "avg_r": float(np.mean(rs)) if rs else 0.0,
        "pf": float(profit_factor(rs)) if rs else 0.0,
        "sum_r": float(np.sum(rs)) if rs else 0.0,
        "r_p10": float(np.percentile(rs, 10)) if rs else 0.0,
        "r_p50": float(np.percentile(rs, 50)) if rs else 0.0,
        "r_p90": float(np.percentile(rs, 90)) if rs else 0.0,
        "SL_rate": float(sum(1 for x in exit_reasons if x in {"SL", "SL_and_TP_same_bar"}) / max(len(exit_reasons), 1)),
        "TP_rate": float(sum(1 for x in exit_reasons if x == "TP_TREND") / max(len(exit_reasons), 1)),
        "FLIP_rate": float(sum(1 for x in exit_reasons if x == "FLIP_ROUTER") / max(len(exit_reasons), 1)),
        "TIME_rate": float(sum(1 for x in exit_reasons if x == "TIME_STOP") / max(len(exit_reasons), 1)),
    }
    return row, pd.DataFrame(trades)


def build_cfgs() -> List[Cfg]:
    cfgs: List[Cfg] = []
    i = 0
    for sname, shours in SWEEP_SESSION:
        hours_csv = ",".join(str(h) for h in shours)
        for mg in SWEEP_MINUTE_GUARD:
            for d in SWEEP_TREND_D:
                for p in SWEEP_TREND_P:
                    for rr in SWEEP_TREND_RR:
                        for sl in SWEEP_SL_ATR:
                            for tp in SWEEP_TP_R:
                                for tstop in SWEEP_TIME_STOP:
                                    i += 1
                                    cfgs.append(
                                        Cfg(
                                            name=f"C{i:05d}_{sname}_mg{mg}_d{d:.2f}_p{p:.2f}_rr{rr:.2f}_sl{sl:.2f}_tp{tp:.2f}_t{tstop}",
                                            trend_d=float(d),
                                            trend_p=float(p),
                                            trend_rr=float(rr),
                                            trend_sl_atr=float(sl),
                                            trend_tp_r=float(tp),
                                            trend_time_stop=int(tstop),
                                            session_name=sname,
                                            session_hours=hours_csv,
                                            minute_guard=int(mg),
                                        )
                                    )
    return cfgs


def print_diagnostic(df_results: pd.DataFrame, top_n: int) -> None:
    print("\n=== DIAGNOSTIC (top configs) ===")
    top = df_results.head(top_n).copy()
    for _, r in top.iterrows():
        print(f"\n[{r['cfg']}]")
        print(
            "n_trades={n} avg_r={avg:.4f} pf={pf:.4f} winrate={w:.2%}".format(
                n=int(r["n_trades"]), avg=float(r["avg_r"]), pf=float(r["pf"]), w=float(r["winrate"])
            )
        )
        print(
            "R-dist p10={p10:.4f} p50={p50:.4f} p90={p90:.4f}".format(
                p10=float(r["r_p10"]), p50=float(r["r_p50"]), p90=float(r["r_p90"])
            )
        )
        print(
            "Exits SL={sl:.2%} TP={tp:.2%} FLIP={fl:.2%} TIME={tm:.2%}".format(
                sl=float(r["SL_rate"]), tp=float(r["TP_rate"]), fl=float(r["FLIP_rate"]), tm=float(r["TIME_rate"])
            )
        )


def main() -> int:
    args = _parse_args()
    parquet_in = Path(args.parquet_in) if args.parquet_in else PARQUET_IN
    csv_out = Path(args.csv_out) if args.csv_out else CSV_OUT

    print(f"[run_sweep_router_v3_breakout] VERSION={VERSION}")
    print(f"[INFO] parquet_in={parquet_in}")
    if not parquet_in.exists():
        raise FileNotFoundError(f"Parquet not found: {parquet_in}")

    df_schema = pd.read_parquet(parquet_in, engine="pyarrow")
    ts_col = TS_COL if TS_COL else auto_detect_ts_col(df_schema)
    print(f"[INFO] ts_col={ts_col}")

    required = [
        ts_col, "open", "high", "low", "close", "atr14", "router_mode_h1", "tradable_final", "dir_ready",
        "dir_state", "vol_state", "delta_norm", "close_pos", "range_rel",
    ]
    missing = [c for c in required if c not in df_schema.columns]
    if missing:
        raise RuntimeError(f"Missing required columns: {missing}")

    df = pd.read_parquet(parquet_in, columns=required, engine="pyarrow")
    df[ts_col] = normalize_ts_series(df[ts_col])
    df = df.sort_values(ts_col, kind="mergesort").reset_index(drop=True)
    if args.start:
        t0 = pd.Timestamp(args.start)
        t0 = t0.tz_localize("UTC") if t0.tzinfo is None else t0.tz_convert("UTC")
        df = df[df[ts_col] >= t0].copy()
    if args.end:
        t1 = pd.Timestamp(args.end)
        t1 = t1.tz_localize("UTC") if t1.tzinfo is None else t1.tz_convert("UTC")
        df = df[df[ts_col] < t1].copy()
    if df.empty:
        raise RuntimeError("No rows after --start/--end filtering.")
    if args.start or args.end:
        print(f"[INFO] date_filter start={args.start or '-'} end={args.end or '-'} rows={len(df)}")

    base = build_base_arrays(df, ts_col=ts_col)

    if args.single_config:
        cfg = Cfg(
            name="BEST_SINGLE_CFG",
            trend_d=float(BEST_SINGLE_CFG["trend_d"]),
            trend_p=float(BEST_SINGLE_CFG["trend_p"]),
            trend_rr=float(BEST_SINGLE_CFG["trend_rr"]),
            trend_sl_atr=float(BEST_SINGLE_CFG["trend_sl_atr"]),
            trend_tp_r=float(BEST_SINGLE_CFG["trend_tp_r"]),
            trend_time_stop=int(BEST_SINGLE_CFG["trend_time_stop"]),
            session_name=str(BEST_SINGLE_CFG["session_name"]),
            session_hours=str(BEST_SINGLE_CFG["session_hours"]),
            minute_guard=int(BEST_SINGLE_CFG["minute_guard"]),
        )
        row, trades_df = run_bt(base, compute_signals(base, cfg), cfg)
        out = pd.DataFrame([row])
        print("[INFO] mode=single-config")
        print(out[["cfg", "pf", "avg_r", "sum_r", "n_trades", "winrate"]].to_string(index=False))
        trades_out = Path(args.trades_out) if args.trades_out else csv_out.with_name(csv_out.stem + "_trades.csv")
        csv_out.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(csv_out, index=False)
        trades_out.parent.mkdir(parents=True, exist_ok=True)
        trades_df.to_csv(trades_out, index=False)
        print(f"[OK] CSV exported: {csv_out}")
        print(f"[OK] Trades exported: {trades_out}")
        return 0

    cfgs = build_cfgs()
    print(f"[INFO] sweep configs={len(cfgs)}")
    rows: List[Dict] = []
    for cfg in cfgs:
        row, _ = run_bt(base, compute_signals(base, cfg), cfg)
        rows.append(row)

    out = pd.DataFrame(rows).sort_values(["pf", "avg_r"], ascending=[False, False]).reset_index(drop=True)
    cols_show = [
        "cfg", "pf", "avg_r", "sum_r", "n_trades", "winrate", "trend_d", "trend_p", "trend_rr",
        "trend_sl_atr", "trend_tp_r", "trend_time_stop", "session_name", "session_hours", "minute_guard",
    ]
    print("\n=== SWEEP RESULTS (sorted by pf, avg_r) ===")
    print(out[cols_show].head(SHOW_TOP_N).to_string(index=False))
    csv_out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(csv_out, index=False)
    print(f"\n[OK] CSV exported: {csv_out}")
    print_diagnostic(out, top_n=SHOW_DIAGNOSTIC_TOP_N)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
