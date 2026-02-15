from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


VERSION = "2026-02-14-ml-filter-v2"

PARQUET_IN = Path(
    "/Users/lolo/PyCharmMiscProject/binance_futures_data_lake/data/research_debug/"
    "BTCUSDT/full_201910_202602/joined_full__enriched__router.parquet"
)
CSV_OUT = Path(
    "/Users/lolo/PyCharmMiscProject/binance_futures_data_lake/data/research_debug/"
    "BTCUSDT/full_201910_202602/ml_filter_v2_summary.csv"
)
PRED_OUT = Path(
    "/Users/lolo/PyCharmMiscProject/binance_futures_data_lake/data/research_debug/"
    "BTCUSDT/full_201910_202602/ml_filter_v2_predictions.csv"
)


@dataclass(frozen=True)
class MrCfg:
    tp_cp: float = 0.60
    tp1_fraction: float = 0.60
    mr_tp1_cp: float = 0.50
    mr_be_offset_r: float = 0.00
    mr_sl_atr: float = 1.8
    mr_time_stop: int = 20
    vol_filter_high: bool = True
    mr_d: float = 0.40
    mr_rr: float = 1.10
    mr_cp_low: float = 0.20
    mr_cp_high: float = 0.80
    mr_mean_dist: float = 0.20
    mr_atr_pctl_max: float = 0.90
    session_hours: str = "15,16"
    minute_guard: int = 15


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ML filter v2: train/val/test + calibration + threshold selection.")
    p.add_argument("--parquet-in", default=str(PARQUET_IN))
    p.add_argument("--csv-out", default=str(CSV_OUT))
    p.add_argument("--pred-out", default=str(PRED_OUT))
    p.add_argument("--train-end", default="2023-01-01 00:00:00+00:00", help="Train end timestamp UTC (exclusive).")
    p.add_argument("--val-end", default="2024-01-01 00:00:00+00:00", help="Validation end timestamp UTC (exclusive).")
    p.add_argument("--min-trades-val", type=int, default=30, help="Minimum number of validation trades for threshold eligibility.")
    p.add_argument("--pos-weight", type=float, default=1.0, help="Positive class weight for logistic loss.")
    p.add_argument("--start", default=None)
    p.add_argument("--end", default=None)
    return p.parse_args()


def auto_detect_ts_col(df_schema: pd.DataFrame) -> str:
    cols = df_schema.columns.tolist()
    for c in ["t_x", "ts", "timestamp", "time", "datetime", "open_time", "open_ts", "time_utc", "ts_utc"]:
        if c in cols:
            return c
    dt_cols = [c for c in cols if str(df_schema[c].dtype).startswith("datetime64")]
    if dt_cols:
        return dt_cols[0]
    raise RuntimeError("Cannot auto-detect timestamp column.")


def normalize_ts_series(s: pd.Series) -> pd.Series:
    if str(s.dtype).startswith("datetime64"):
        return s
    if np.issubdtype(s.dtype, np.integer):
        return pd.to_datetime(s, unit="ms", utc=True).dt.tz_convert(None)
    return pd.to_datetime(s, utc=True, errors="coerce").dt.tz_convert(None)


def profit_factor(rs: np.ndarray) -> float:
    gains = float(rs[rs > 0].sum())
    losses = float(-rs[rs < 0].sum())
    if losses <= 0:
        return float("inf") if gains > 0 else 0.0
    return gains / losses


def sigmoid(z: np.ndarray) -> np.ndarray:
    zc = np.clip(z, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-zc))


def fit_logistic_gd(
    x: np.ndarray,
    y: np.ndarray,
    pos_weight: float = 1.0,
    l2: float = 1e-3,
    lr: float = 0.05,
    n_iter: int = 1500,
) -> np.ndarray:
    n, d = x.shape
    w = np.zeros(d, dtype=float)
    ww = np.where(y > 0.5, float(pos_weight), 1.0)
    for _ in range(n_iter):
        p = sigmoid(x @ w)
        grad = (x.T @ ((p - y) * ww)) / max(n, 1)
        grad += l2 * w
        w -= lr * grad
    return w


def fit_platt_scaler(logits: np.ndarray, y: np.ndarray) -> np.ndarray:
    x = np.c_[np.ones((len(logits), 1)), logits.astype(float)]
    return fit_logistic_gd(x, y.astype(float), pos_weight=1.0, l2=1e-3, lr=0.05, n_iter=1200)


def apply_platt_scaler(logits: np.ndarray, w_platt: np.ndarray) -> np.ndarray:
    x = np.c_[np.ones((len(logits), 1)), logits.astype(float)]
    return sigmoid(x @ w_platt)


def choose_threshold_on_val(
    probs: np.ndarray,
    r_mult: np.ndarray,
    min_trades_val: int,
) -> tuple[float, pd.DataFrame]:
    rows: List[Dict] = []
    if probs.size == 0:
        return 1.0, pd.DataFrame()

    quantiles = np.linspace(0.50, 0.95, 10)
    best_thr = float(np.quantile(probs, 0.70))
    best_score = -1e9
    for q in quantiles:
        thr = float(np.quantile(probs, q))
        keep = probs >= thr
        n = int(keep.sum())
        if n <= 0:
            continue
        rs = r_mult[keep]
        avg_r = float(rs.mean()) if rs.size else 0.0
        pf = float(profit_factor(rs)) if rs.size else 0.0
        win = float((rs > 0).mean()) if rs.size else 0.0
        rows.append(
            {
                "quantile": float(q),
                "threshold": thr,
                "n_trades": n,
                "avg_r": avg_r,
                "pf": pf,
                "winrate": win,
            }
        )
        # Objective: maximize avg_r with guardrails on sample size.
        if n >= min_trades_val:
            score = avg_r + (0.01 * np.log(max(pf, 1e-9)))
            if score > best_score:
                best_score = score
                best_thr = thr

    df = pd.DataFrame(rows)
    return best_thr, df.sort_values(["avg_r", "pf"], ascending=[False, False]).reset_index(drop=True)


def r_mult(side: str, entry: float, exit_p: float, risk: float) -> float:
    if risk <= 0 or not np.isfinite(risk):
        return 0.0
    return (exit_p - entry) / risk if side == "LONG" else (entry - exit_p) / risk


def encode_state(s: str, mapping: Dict[str, int], default: int = 0) -> int:
    return int(mapping.get(str(s), default))


def generate_mr_trades(df: pd.DataFrame, ts_col: str, cfg: MrCfg) -> pd.DataFrame:
    # Arrays
    close = df["close"].to_numpy(dtype=float)
    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    atr14 = df["atr14"].to_numpy(dtype=float)
    close_pos = df["close_pos"].to_numpy(dtype=float)
    delta_norm = df["delta_norm"].to_numpy(dtype=float)
    range_rel = df["range_rel"].to_numpy(dtype=float)
    atr_pctl = df["atr_pct_pctl_h1"].to_numpy(dtype=float)
    router = df["router_mode_h1"].astype("string").fillna("NA").to_numpy(dtype=object)
    dstate = df["dir_state"].astype("string").fillna("NA").to_numpy(dtype=object)
    vstate = df["vol_state"].astype("string").fillna("NA").to_numpy(dtype=object)
    tradable = (df["tradable_final"] == True).to_numpy(dtype=bool)
    dir_ready = (df["dir_ready"] == True).to_numpy(dtype=bool)
    ts = df[ts_col].to_numpy()
    h_utc = df[ts_col].dt.hour.to_numpy(dtype=int)
    m_utc = df[ts_col].dt.minute.to_numpy(dtype=int)

    # Optional features at entry
    slope50_norm = df["slope50_norm_h1"].to_numpy(dtype=float) if "slope50_norm_h1" in df.columns else np.full(len(df), np.nan)
    er_h1 = df["er_h1"].to_numpy(dtype=float) if "er_h1" in df.columns else np.full(len(df), np.nan)
    trend_score_h1 = df["trend_score_h1"].to_numpy(dtype=float) if "trend_score_h1" in df.columns else np.full(len(df), np.nan)
    ema20 = df["ema20_h1"].to_numpy(dtype=float) if "ema20_h1" in df.columns else np.full(len(df), np.nan)
    ema50 = df["ema50_h1"].to_numpy(dtype=float) if "ema50_h1" in df.columns else np.full(len(df), np.nan)
    ema200 = df["ema200_h1"].to_numpy(dtype=float) if "ema200_h1" in df.columns else np.full(len(df), np.nan)

    session_hours = np.array([int(x) for x in cfg.session_hours.split(",") if x != ""], dtype=int)
    session_mask = np.isin(h_utc, session_hours)
    minute_mask = (m_utc >= cfg.minute_guard) & (m_utc <= (59 - cfg.minute_guard)) if cfg.minute_guard > 0 else np.ones(len(df), dtype=bool)

    is_range = tradable & dir_ready & (router == "RANGE") & (vstate != "NA") & session_mask & minute_mask
    if cfg.vol_filter_high:
        is_range = is_range & np.isin(vstate, ["LOW", "MID"])

    mean_dist_ok = np.abs(close_pos - 0.5) >= cfg.mr_mean_dist
    finite_atr = atr_pctl[np.isfinite(atr_pctl)]
    if finite_atr.size == 0:
        atr_cap = cfg.mr_atr_pctl_max
    else:
        atr_cap = cfg.mr_atr_pctl_max * 100.0 if float(np.nanmax(finite_atr)) > 1.5 else cfg.mr_atr_pctl_max
    atr_ok = (~np.isfinite(atr_pctl)) | (atr_pctl <= atr_cap)

    long_setup = (
        is_range
        & mean_dist_ok
        & atr_ok
        & (close_pos <= cfg.mr_cp_low)
        & (delta_norm <= -cfg.mr_d)
        & (range_rel >= cfg.mr_rr)
        & (dstate == "BULL")
    )
    short_setup = (
        is_range
        & mean_dist_ok
        & atr_ok
        & (close_pos >= cfg.mr_cp_high)
        & (delta_norm >= cfg.mr_d)
        & (range_rel >= cfg.mr_rr)
        & (dstate == "BEAR")
    )

    conf_long = (delta_norm > 0) | (close_pos > 0.50)
    conf_short = (delta_norm < 0) | (close_pos < 0.50)
    long_prev = np.zeros_like(long_setup, dtype=bool)
    short_prev = np.zeros_like(short_setup, dtype=bool)
    long_prev[1:] = long_setup[:-1]
    short_prev[1:] = short_setup[:-1]
    long_signal = long_prev & is_range & conf_long
    short_signal = short_prev & is_range & conf_short

    rows: List[Dict] = []
    pos = None
    for i in range(len(df)):
        if pos is not None:
            bars = i - pos["entry_i"]
            c = float(close[i])
            cp = float(close_pos[i])
            h = float(high[i])
            l = float(low[i])
            router_flip = str(router[i]) != "RANGE"
            vol_kill = cfg.vol_filter_high and str(vstate[i]) == "HIGH"
            time_exit = bars >= cfg.mr_time_stop
            if pos["side"] == "LONG":
                sl_hit = l <= pos["sl"]
                tp1_hit = cp >= cfg.mr_tp1_cp
                tp2_hit = cp >= cfg.tp_cp
            else:
                sl_hit = h >= pos["sl"]
                tp1_hit = cp <= (1.0 - cfg.mr_tp1_cp)
                tp2_hit = cp <= (1.0 - cfg.tp_cp)

            if sl_hit:
                rr = pos["rr_acc"] + (pos["qty"] * r_mult(pos["side"], pos["entry"], float(pos["sl"]), pos["risk"]))
                rows.append({**pos["feat"], "entry_ts": pos["entry_ts"], "exit_ts": str(pd.Timestamp(ts[i])), "r_mult": rr, "exit_reason": "SL"})
                pos = None
                continue

            if not pos["tp1_done"] and tp1_hit:
                q = min(cfg.tp1_fraction, pos["qty"])
                pos["rr_acc"] += q * r_mult(pos["side"], pos["entry"], c, pos["risk"])
                pos["qty"] -= q
                pos["tp1_done"] = True
                if pos["side"] == "LONG":
                    pos["sl"] = pos["entry"] + (pos["risk"] * cfg.mr_be_offset_r)
                else:
                    pos["sl"] = pos["entry"] - (pos["risk"] * cfg.mr_be_offset_r)
                if pos["qty"] <= 1e-12:
                    rows.append({**pos["feat"], "entry_ts": pos["entry_ts"], "exit_ts": str(pd.Timestamp(ts[i])), "r_mult": pos["rr_acc"], "exit_reason": "TP1_FULL"})
                    pos = None
                    continue

            if pos is not None and tp2_hit:
                rr = pos["rr_acc"] + (pos["qty"] * r_mult(pos["side"], pos["entry"], c, pos["risk"]))
                rows.append({**pos["feat"], "entry_ts": pos["entry_ts"], "exit_ts": str(pd.Timestamp(ts[i])), "r_mult": rr, "exit_reason": "TP2_CP"})
                pos = None
                continue

            if pos is not None and (router_flip or vol_kill or time_exit):
                rr = pos["rr_acc"] + (pos["qty"] * r_mult(pos["side"], pos["entry"], c, pos["risk"]))
                reason = "FLIP_ROUTER" if router_flip else ("VOL_HIGH_KILL" if vol_kill else "TIME_STOP")
                rows.append({**pos["feat"], "entry_ts": pos["entry_ts"], "exit_ts": str(pd.Timestamp(ts[i])), "r_mult": rr, "exit_reason": reason})
                pos = None
                continue

        if pos is not None:
            continue
        go_long = bool(long_signal[i])
        go_short = bool(short_signal[i])
        if not (go_long or go_short):
            continue
        side = "LONG" if go_long else "SHORT"
        entry = float(close[i])
        atr = float(atr14[i])
        if not np.isfinite(atr) or atr <= 0:
            continue
        sl_dist = cfg.mr_sl_atr * atr
        if side == "LONG":
            sl = entry - sl_dist
            risk = entry - sl
        else:
            sl = entry + sl_dist
            risk = sl - entry
        if risk <= 0:
            continue

        feat = {
            "side": 1 if side == "LONG" else -1,
            "delta_norm": float(delta_norm[i]),
            "close_pos": float(close_pos[i]),
            "range_rel": float(range_rel[i]),
            "atr14": float(atr14[i]),
            "atr_pct_pctl_h1": float(atr_pctl[i]),
            "slope50_norm_h1": float(slope50_norm[i]),
            "er_h1": float(er_h1[i]),
            "trend_score_h1": float(trend_score_h1[i]),
            "ema20_50_spread": float(ema20[i] - ema50[i]) if np.isfinite(ema20[i]) and np.isfinite(ema50[i]) else np.nan,
            "ema50_200_spread": float(ema50[i] - ema200[i]) if np.isfinite(ema50[i]) and np.isfinite(ema200[i]) else np.nan,
            "dir_state_code": encode_state(str(dstate[i]), {"BEAR": -1, "NEUTRAL": 0, "BULL": 1}, 0),
            "vol_state_code": encode_state(str(vstate[i]), {"LOW": 0, "MID": 1, "HIGH": 2, "NA": -1}, -1),
            "router_mode_code": encode_state(str(router[i]), {"OFF": 0, "RANGE": 1, "TREND": 2}, 0),
            "hour_utc": int(h_utc[i]),
            "minute_utc": int(m_utc[i]),
        }
        pos = {
            "side": side,
            "entry_i": i,
            "entry": entry,
            "entry_ts": str(pd.Timestamp(ts[i])),
            "sl": float(sl),
            "risk": float(risk),
            "qty": 1.0,
            "rr_acc": 0.0,
            "tp1_done": False,
            "feat": feat,
        }
    return pd.DataFrame(rows)


def compute_metrics(df: pd.DataFrame, mask: np.ndarray) -> Dict[str, float]:
    sub = df.loc[mask].copy()
    rs = sub["r_mult"].to_numpy(dtype=float) if not sub.empty else np.array([], dtype=float)
    return {
        "n_trades": int(len(sub)),
        "avg_r": float(rs.mean()) if rs.size else 0.0,
        "sum_r": float(rs.sum()) if rs.size else 0.0,
        "pf": float(profit_factor(rs)) if rs.size else 0.0,
        "winrate": float((rs > 0).mean()) if rs.size else 0.0,
    }


def main() -> int:
    args = parse_args()
    parquet_in = Path(args.parquet_in)
    csv_out = Path(args.csv_out)
    pred_out = Path(args.pred_out)
    train_end = pd.Timestamp(args.train_end)
    train_end = train_end.tz_localize("UTC") if train_end.tzinfo is None else train_end.tz_convert("UTC")
    val_end = pd.Timestamp(args.val_end)
    val_end = val_end.tz_localize("UTC") if val_end.tzinfo is None else val_end.tz_convert("UTC")
    if val_end <= train_end:
        raise RuntimeError("--val-end must be strictly after --train-end.")

    print(f"[run_ml_filter_v2] VERSION={VERSION}")
    print(f"[INFO] parquet_in={parquet_in}")
    if not parquet_in.exists():
        raise FileNotFoundError(f"Parquet not found: {parquet_in}")

    schema = pd.read_parquet(parquet_in, engine="pyarrow")
    ts_col = auto_detect_ts_col(schema)
    print(f"[INFO] ts_col={ts_col}")

    req = [ts_col, "open", "high", "low", "close", "atr14", "router_mode_h1", "tradable_final", "dir_ready", "dir_state", "vol_state", "delta_norm", "close_pos", "range_rel"]
    opt = ["atr_pct_pctl_h1", "slope50_norm_h1", "er_h1", "trend_score_h1", "ema20_h1", "ema50_h1", "ema200_h1"]
    cols = req + [c for c in opt if c in schema.columns]
    df = pd.read_parquet(parquet_in, columns=cols, engine="pyarrow")
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
    print(f"[INFO] rows={len(df)} train_end={train_end} val_end={val_end}")

    trades = generate_mr_trades(df, ts_col=ts_col, cfg=MrCfg())
    if trades.empty:
        raise RuntimeError("No candidate trades generated.")
    trades["entry_ts"] = pd.to_datetime(trades["entry_ts"], utc=True, errors="coerce")
    trades["y"] = (trades["r_mult"] > 0).astype(int)

    feature_cols = [
        "side", "delta_norm", "close_pos", "range_rel", "atr14", "atr_pct_pctl_h1", "slope50_norm_h1", "er_h1",
        "trend_score_h1", "ema20_50_spread", "ema50_200_spread", "dir_state_code", "vol_state_code",
        "router_mode_code", "hour_utc", "minute_utc",
    ]
    for c in feature_cols:
        if c not in trades.columns:
            trades[c] = np.nan
    x_all = trades[feature_cols].copy()
    x_all = x_all.replace([np.inf, -np.inf], np.nan)

    train_mask = trades["entry_ts"] < train_end
    val_mask = (trades["entry_ts"] >= train_end) & (trades["entry_ts"] < val_end)
    test_mask = trades["entry_ts"] >= val_end
    if train_mask.sum() < 50 or val_mask.sum() < 40 or test_mask.sum() < 30:
        raise RuntimeError(
            "Not enough trades for split. train={tr} val={va} test={te}".format(
                tr=int(train_mask.sum()),
                va=int(val_mask.sum()),
                te=int(test_mask.sum()),
            )
        )

    med = x_all.loc[train_mask].median(numeric_only=True)
    x_all = x_all.fillna(med)

    x_train = x_all.loc[train_mask].to_numpy(dtype=float)
    y_train = trades.loc[train_mask, "y"].to_numpy(dtype=float)
    x_val = x_all.loc[val_mask].to_numpy(dtype=float)
    y_val = trades.loc[val_mask, "y"].to_numpy(dtype=float)
    x_test = x_all.loc[test_mask].to_numpy(dtype=float)

    mu = x_train.mean(axis=0)
    sd = x_train.std(axis=0)
    sd = np.where(sd < 1e-9, 1.0, sd)
    x_train_z = (x_train - mu) / sd
    x_val_z = (x_val - mu) / sd
    x_test_z = (x_test - mu) / sd

    x_train_b = np.c_[np.ones((x_train_z.shape[0], 1)), x_train_z]
    x_val_b = np.c_[np.ones((x_val_z.shape[0], 1)), x_val_z]
    x_test_b = np.c_[np.ones((x_test_z.shape[0], 1)), x_test_z]
    w = fit_logistic_gd(x_train_b, y_train, pos_weight=float(args.pos_weight), l2=1e-3, lr=0.05, n_iter=1800)

    z_train = x_train_b @ w
    z_val = x_val_b @ w
    z_test = x_test_b @ w

    # Calibrate probabilities on validation set (Platt scaling)
    w_platt = fit_platt_scaler(z_val, y_val)
    p_train = apply_platt_scaler(z_train, w_platt)
    p_val = apply_platt_scaler(z_val, w_platt)
    p_test = apply_platt_scaler(z_test, w_platt)
    trades.loc[train_mask, "p_ml"] = p_train
    trades.loc[val_mask, "p_ml"] = p_val
    trades.loc[test_mask, "p_ml"] = p_test

    summary_rows: List[Dict] = []
    train_base = compute_metrics(trades, train_mask.to_numpy())
    val_base = compute_metrics(trades, val_mask.to_numpy())
    test_base = compute_metrics(trades, test_mask.to_numpy())
    summary_rows.append({"set": "TRAIN_BASELINE", "keep_frac": 1.0, **train_base})
    summary_rows.append({"set": "VAL_BASELINE", "keep_frac": 1.0, **val_base})
    summary_rows.append({"set": "TEST_BASELINE", "keep_frac": 1.0, **test_base})

    thr, val_grid = choose_threshold_on_val(
        probs=p_val,
        r_mult=trades.loc[val_mask, "r_mult"].to_numpy(dtype=float),
        min_trades_val=int(args.min_trades_val),
    )
    val_keep = np.zeros(len(trades), dtype=bool)
    val_keep[np.where(val_mask.to_numpy())[0]] = p_val >= thr
    test_keep = np.zeros(len(trades), dtype=bool)
    test_keep[np.where(test_mask.to_numpy())[0]] = p_test >= thr

    val_filtered = compute_metrics(trades, val_keep)
    test_filtered = compute_metrics(trades, test_keep)
    summary_rows.append({"set": "VAL_FILTERED", "keep_frac": float(val_keep.sum() / max(val_mask.sum(), 1)), "p_threshold": thr, **val_filtered})
    summary_rows.append({"set": "TEST_FILTERED", "keep_frac": float(test_keep.sum() / max(test_mask.sum(), 1)), "p_threshold": thr, **test_filtered})

    out = pd.DataFrame(summary_rows)
    out = out.sort_values(["set", "keep_frac"], ascending=[True, False]).reset_index(drop=True)

    coef_rows = []
    coef_rows.append({"feature": "bias", "coef": float(w[0])})
    for i, c in enumerate(feature_cols, start=1):
        coef_rows.append({"feature": c, "coef": float(w[i])})
    coef_df = pd.DataFrame(coef_rows).sort_values("coef", ascending=False).reset_index(drop=True)

    csv_out.parent.mkdir(parents=True, exist_ok=True)
    pred_out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(csv_out, index=False)
    trades.to_csv(pred_out, index=False)
    coef_path = csv_out.with_name(csv_out.stem + "_coefs.csv")
    coef_df.to_csv(coef_path, index=False)
    val_grid_path = csv_out.with_name(csv_out.stem + "_val_grid.csv")
    val_grid.to_csv(val_grid_path, index=False)

    print("\n=== ML FILTER V2 SUMMARY ===")
    print(out.to_string(index=False))
    print("\n=== VAL THRESHOLD GRID (top) ===")
    if not val_grid.empty:
        print(val_grid.head(10).to_string(index=False))
    print(f"[INFO] selected_threshold={thr:.6f}")
    print(f"\n[OK] summary_csv={csv_out}")
    print(f"[OK] pred_csv={pred_out}")
    print(f"[OK] coef_csv={coef_path}")
    print(f"[OK] val_grid_csv={val_grid_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
