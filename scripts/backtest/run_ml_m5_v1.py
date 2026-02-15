from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


VERSION = "2026-02-14-ml-m5-v1"

PARQUET_IN = Path(
    "/Users/lolo/PyCharmMiscProject/binance_futures_data_lake/data/research_debug/"
    "BTCUSDT/full_201910_202602/joined_full__enriched__router.parquet"
)
CSV_OUT = Path(
    "/Users/lolo/PyCharmMiscProject/binance_futures_data_lake/data/research_debug/"
    "BTCUSDT/full_201910_202602/ml_m5_v1_summary.csv"
)
PRED_OUT = Path(
    "/Users/lolo/PyCharmMiscProject/binance_futures_data_lake/data/research_debug/"
    "BTCUSDT/full_201910_202602/ml_m5_v1_predictions.csv"
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="M5 ML v1 (numpy logistic) with temporal train/val/test.")
    p.add_argument("--parquet-in", default=str(PARQUET_IN))
    p.add_argument("--csv-out", default=str(CSV_OUT))
    p.add_argument("--pred-out", default=str(PRED_OUT))
    p.add_argument("--start", default=None)
    p.add_argument("--end", default=None)
    p.add_argument("--train-end", default="2023-01-01 00:00:00+00:00")
    p.add_argument("--val-end", default="2024-01-01 00:00:00+00:00")
    p.add_argument("--horizon-bars", type=int, default=6, help="Forward horizon in M5 bars (default 6=30min).")
    p.add_argument("--min-trades-val", type=int, default=150)
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


def sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-z))


def fit_logistic_gd(x: np.ndarray, y: np.ndarray, l2: float = 1e-3, lr: float = 0.05, n_iter: int = 2000) -> np.ndarray:
    n, d = x.shape
    w = np.zeros(d, dtype=float)
    for _ in range(n_iter):
        p = sigmoid(x @ w)
        grad = (x.T @ (p - y)) / max(n, 1)
        grad += l2 * w
        w -= lr * grad
    return w


def fit_platt(logits: np.ndarray, y: np.ndarray) -> np.ndarray:
    x = np.c_[np.ones((len(logits), 1)), logits.astype(float)]
    return fit_logistic_gd(x, y.astype(float), l2=1e-3, lr=0.05, n_iter=1200)


def apply_platt(logits: np.ndarray, w: np.ndarray) -> np.ndarray:
    x = np.c_[np.ones((len(logits), 1)), logits.astype(float)]
    return sigmoid(x @ w)


def profit_factor(rs: np.ndarray) -> float:
    gains = float(rs[rs > 0].sum())
    losses = float(-rs[rs < 0].sum())
    if losses <= 0:
        return float("inf") if gains > 0 else 0.0
    return gains / losses


def encode_state(s: str, mapping: Dict[str, int], default: int = 0) -> int:
    return int(mapping.get(str(s), default))


def to_m5_dataset(df: pd.DataFrame, ts_col: str, horizon: int) -> pd.DataFrame:
    d = df.copy()
    d = d.sort_values(ts_col).reset_index(drop=True)
    d = d.set_index(ts_col)

    agg = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "delta_norm": "mean",
        "close_pos": "mean",
        "range_rel": "mean",
        "atr14": "last",
        "atr_pct_pctl_h1": "last",
        "slope50_norm_h1": "last" if "slope50_norm_h1" in d.columns else "mean",
        "er_h1": "last" if "er_h1" in d.columns else "mean",
        "trend_score_h1": "last" if "trend_score_h1" in d.columns else "mean",
        "ema20_h1": "last" if "ema20_h1" in d.columns else "mean",
        "ema50_h1": "last" if "ema50_h1" in d.columns else "mean",
        "ema200_h1": "last" if "ema200_h1" in d.columns else "mean",
        "dir_state": "last",
        "vol_state": "last",
        "router_mode_h1": "last",
        "tradable_final": "last",
        "dir_ready": "last",
    }
    agg = {k: v for k, v in agg.items() if k in d.columns}

    cnt = d["close"].resample("5min", label="left", closed="left").size().rename("count_m1")
    m5 = d.resample("5min", label="left", closed="left").agg(agg).join(cnt)
    m5 = m5.dropna(subset=["open", "high", "low", "close"])
    m5 = m5[m5["count_m1"] == 5].copy()
    m5 = m5.reset_index().rename(columns={ts_col: "ts"})

    # Additional M5-only features
    m5["ret1"] = m5["close"].pct_change()
    m5["range_pct"] = (m5["high"] - m5["low"]) / m5["close"].replace(0.0, np.nan)
    m5["body_pct"] = (m5["close"] - m5["open"]).abs() / (m5["high"] - m5["low"]).replace(0.0, np.nan)
    m5["hour_utc"] = m5["ts"].dt.hour.astype(int)
    m5["minute_utc"] = m5["ts"].dt.minute.astype(int)
    if "ema20_h1" in m5.columns and "ema50_h1" in m5.columns:
        m5["ema20_50_spread"] = m5["ema20_h1"] - m5["ema50_h1"]
    else:
        m5["ema20_50_spread"] = np.nan
    if "ema50_h1" in m5.columns and "ema200_h1" in m5.columns:
        m5["ema50_200_spread"] = m5["ema50_h1"] - m5["ema200_h1"]
    else:
        m5["ema50_200_spread"] = np.nan

    m5["dir_state_code"] = m5["dir_state"].astype(str).map({"BEAR": -1, "NEUTRAL": 0, "BULL": 1}).fillna(0).astype(int)
    m5["vol_state_code"] = m5["vol_state"].astype(str).map({"LOW": 0, "MID": 1, "HIGH": 2, "NA": -1}).fillna(-1).astype(int)
    m5["router_mode_code"] = m5["router_mode_h1"].astype(str).map({"OFF": 0, "RANGE": 1, "TREND": 2}).fillna(0).astype(int)

    # Label: next-open to future-close return
    m5["entry_next_open"] = m5["open"].shift(-1)
    m5["future_close"] = m5["close"].shift(-int(horizon))
    m5["ret_fwd_long"] = (m5["future_close"] - m5["entry_next_open"]) / m5["entry_next_open"].replace(0.0, np.nan)
    m5["y"] = (m5["ret_fwd_long"] > 0).astype(int)

    m5 = m5.replace([np.inf, -np.inf], np.nan)
    m5 = m5.dropna(subset=["entry_next_open", "future_close", "ret_fwd_long"]).reset_index(drop=True)
    return m5


def evaluate_strategy(probs: np.ndarray, ret_fwd_long: np.ndarray, thr: float) -> Dict[str, float]:
    side = np.zeros(len(probs), dtype=float)
    side[probs >= thr] = 1.0
    side[probs <= (1.0 - thr)] = -1.0
    traded = side != 0.0
    rs = side[traded] * ret_fwd_long[traded]
    return {
        "n_trades": int(traded.sum()),
        "avg_r": float(rs.mean()) if rs.size else 0.0,
        "sum_r": float(rs.sum()) if rs.size else 0.0,
        "pf": float(profit_factor(rs)) if rs.size else 0.0,
        "winrate": float((rs > 0).mean()) if rs.size else 0.0,
        "keep_frac": float(traded.mean()),
        "threshold": float(thr),
    }


def choose_threshold_val(probs_val: np.ndarray, ret_val: np.ndarray, min_trades_val: int) -> tuple[float, pd.DataFrame]:
    rows: List[Dict] = []
    best_thr = 0.50
    best_score = -1e9
    found_eligible = False
    for thr in np.linspace(0.55, 0.80, 11):
        m = evaluate_strategy(probs_val, ret_val, float(thr))
        m["eligible"] = bool(m["n_trades"] >= min_trades_val)
        rows.append(m)
        if m["eligible"]:
            found_eligible = True
            score = m["avg_r"] + (0.01 * np.log(max(m["pf"], 1e-9)))
            if score > best_score:
                best_score = score
                best_thr = float(thr)
    if not found_eligible:
        # Fallback to baseline threshold if validation is too sparse.
        best_thr = 0.50
    grid = pd.DataFrame(rows).sort_values(["avg_r", "pf"], ascending=[False, False]).reset_index(drop=True)
    return best_thr, grid


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
        raise RuntimeError("--val-end must be > --train-end.")

    print(f"[run_ml_m5_v1] VERSION={VERSION}")
    print(f"[INFO] parquet_in={parquet_in}")
    if not parquet_in.exists():
        raise FileNotFoundError(f"Parquet not found: {parquet_in}")

    schema = pd.read_parquet(parquet_in, engine="pyarrow")
    ts_col = auto_detect_ts_col(schema)
    print(f"[INFO] ts_col={ts_col}")

    base_cols = [
        ts_col, "open", "high", "low", "close", "delta_norm", "close_pos", "range_rel", "atr14", "atr_pct_pctl_h1",
        "dir_state", "vol_state", "router_mode_h1", "tradable_final", "dir_ready",
        "slope50_norm_h1", "er_h1", "trend_score_h1", "ema20_h1", "ema50_h1", "ema200_h1",
    ]
    cols = [c for c in base_cols if c in schema.columns]
    df = pd.read_parquet(parquet_in, columns=cols, engine="pyarrow")
    df[ts_col] = normalize_ts_series(df[ts_col])
    df = df.sort_values(ts_col).reset_index(drop=True)
    if args.start:
        t0 = pd.Timestamp(args.start)
        t0 = t0.tz_localize("UTC") if t0.tzinfo is None else t0.tz_convert("UTC")
        df = df[df[ts_col] >= t0].copy()
    if args.end:
        t1 = pd.Timestamp(args.end)
        t1 = t1.tz_localize("UTC") if t1.tzinfo is None else t1.tz_convert("UTC")
        df = df[df[ts_col] < t1].copy()

    m5 = to_m5_dataset(df, ts_col=ts_col, horizon=int(args.horizon_bars))
    print(f"[INFO] m5_rows={len(m5)} train_end={train_end} val_end={val_end}")

    feature_cols = [
        "delta_norm", "close_pos", "range_rel", "atr14", "atr_pct_pctl_h1", "ret1", "range_pct", "body_pct",
        "slope50_norm_h1", "er_h1", "trend_score_h1", "ema20_50_spread", "ema50_200_spread",
        "dir_state_code", "vol_state_code", "router_mode_code", "hour_utc", "minute_utc",
    ]
    for c in feature_cols:
        if c not in m5.columns:
            m5[c] = np.nan

    m5["ts"] = pd.to_datetime(m5["ts"], utc=True, errors="coerce")
    train_mask = m5["ts"] < train_end
    val_mask = (m5["ts"] >= train_end) & (m5["ts"] < val_end)
    test_mask = m5["ts"] >= val_end
    if train_mask.sum() < 500 or val_mask.sum() < 200 or test_mask.sum() < 200:
        raise RuntimeError(f"Not enough M5 samples. train={int(train_mask.sum())} val={int(val_mask.sum())} test={int(test_mask.sum())}")

    x = m5[feature_cols].replace([np.inf, -np.inf], np.nan)
    med = x.loc[train_mask].median(numeric_only=True)
    x = x.fillna(med)
    y = m5["y"].to_numpy(dtype=float)

    x_train = x.loc[train_mask].to_numpy(dtype=float)
    x_val = x.loc[val_mask].to_numpy(dtype=float)
    x_test = x.loc[test_mask].to_numpy(dtype=float)
    y_train = y[train_mask.to_numpy()]
    y_val = y[val_mask.to_numpy()]
    y_test = y[test_mask.to_numpy()]

    mu = x_train.mean(axis=0)
    sd = x_train.std(axis=0)
    sd = np.where(sd < 1e-9, 1.0, sd)
    x_train_z = (x_train - mu) / sd
    x_val_z = (x_val - mu) / sd
    x_test_z = (x_test - mu) / sd

    x_train_b = np.c_[np.ones((x_train_z.shape[0], 1)), x_train_z]
    x_val_b = np.c_[np.ones((x_val_z.shape[0], 1)), x_val_z]
    x_test_b = np.c_[np.ones((x_test_z.shape[0], 1)), x_test_z]

    w = fit_logistic_gd(x_train_b, y_train, l2=1e-3, lr=0.05, n_iter=1800)
    z_train = x_train_b @ w
    z_val = x_val_b @ w
    z_test = x_test_b @ w
    w_cal = fit_platt(z_val, y_val)
    p_train = apply_platt(z_train, w_cal)
    p_val = apply_platt(z_val, w_cal)
    p_test = apply_platt(z_test, w_cal)

    m5.loc[train_mask, "p_ml"] = p_train
    m5.loc[val_mask, "p_ml"] = p_val
    m5.loc[test_mask, "p_ml"] = p_test

    ret_train = m5.loc[train_mask, "ret_fwd_long"].to_numpy(dtype=float)
    ret_val = m5.loc[val_mask, "ret_fwd_long"].to_numpy(dtype=float)
    ret_test = m5.loc[test_mask, "ret_fwd_long"].to_numpy(dtype=float)

    val_thr, val_grid = choose_threshold_val(p_val, ret_val, min_trades_val=int(args.min_trades_val))

    # Baseline directional model: threshold 0.50 (always long/short)
    test_baseline = evaluate_strategy(p_test, ret_test, thr=0.50)
    val_filtered = evaluate_strategy(p_val, ret_val, thr=val_thr)
    test_filtered = evaluate_strategy(p_test, ret_test, thr=val_thr)
    train_filtered = evaluate_strategy(p_train, ret_train, thr=val_thr)

    summary = pd.DataFrame(
        [
            {"set": "TRAIN_FILTERED", **train_filtered},
            {"set": "VAL_FILTERED", **val_filtered},
            {"set": "TEST_BASELINE", **test_baseline},
            {"set": "TEST_FILTERED", **test_filtered},
        ]
    )

    coef_rows = [{"feature": "bias", "coef": float(w[0])}]
    for i, c in enumerate(feature_cols, start=1):
        coef_rows.append({"feature": c, "coef": float(w[i])})
    coef_df = pd.DataFrame(coef_rows).sort_values("coef", ascending=False).reset_index(drop=True)

    csv_out.parent.mkdir(parents=True, exist_ok=True)
    pred_out.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(csv_out, index=False)
    m5.to_csv(pred_out, index=False)
    coef_path = csv_out.with_name(csv_out.stem + "_coefs.csv")
    val_grid_path = csv_out.with_name(csv_out.stem + "_val_grid.csv")
    coef_df.to_csv(coef_path, index=False)
    val_grid.to_csv(val_grid_path, index=False)

    print("\n=== ML M5 V1 SUMMARY ===")
    print(summary.to_string(index=False))
    print("\n=== VAL GRID (top) ===")
    print(val_grid.head(10).to_string(index=False))
    print(f"[INFO] selected_threshold={val_thr:.4f}")
    print(f"\n[OK] summary_csv={csv_out}")
    print(f"[OK] pred_csv={pred_out}")
    print(f"[OK] coef_csv={coef_path}")
    print(f"[OK] val_grid_csv={val_grid_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
