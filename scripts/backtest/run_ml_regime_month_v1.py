from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


VERSION = "2026-02-14-ml-regime-month-v1"

PARQUET_IN = Path(
    "/Users/lolo/PyCharmMiscProject/binance_futures_data_lake/data/research_debug/"
    "BTCUSDT/full_201910_202602/joined_full__enriched__router__structure_v1__mn_w_d_h4_h1_m15_context.parquet"
)
CSV_OUT = Path(
    "/Users/lolo/PyCharmMiscProject/binance_futures_data_lake/data/research_debug/"
    "BTCUSDT/full_201910_202602/ml_regime_month_v1_summary.csv"
)
PRED_OUT = Path(
    "/Users/lolo/PyCharmMiscProject/binance_futures_data_lake/data/research_debug/"
    "BTCUSDT/full_201910_202602/ml_regime_month_v1_predictions.csv"
)

LABEL_MAP = {0: "RANGE", 1: "TREND_UP", 2: "TREND_DOWN"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Monthly regime ML v1 (supervised, 3-class).")
    p.add_argument("--parquet-in", default=str(PARQUET_IN))
    p.add_argument("--csv-out", default=str(CSV_OUT))
    p.add_argument("--pred-out", default=str(PRED_OUT))
    p.add_argument("--train-end", default="2023-01-01 00:00:00+00:00")
    p.add_argument("--val-end", default="2024-01-01 00:00:00+00:00")
    p.add_argument("--horizon-months", type=int, default=3, help="Forward horizon in months for regime label.")
    p.add_argument("--trend-quantile", type=float, default=0.60, help="Train quantile on |future return| for trend threshold.")
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


def sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-z))


def fit_logistic_gd(x: np.ndarray, y: np.ndarray, l2: float = 1e-3, lr: float = 0.05, n_iter: int = 1800) -> np.ndarray:
    n, d = x.shape
    w = np.zeros(d, dtype=float)
    for _ in range(n_iter):
        p = sigmoid(x @ w)
        grad = (x.T @ (p - y)) / max(n, 1)
        grad += l2 * w
        w -= lr * grad
    return w


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray, labels: List[int]) -> float:
    f1s: List[float] = []
    for c in labels:
        tp = float(((y_true == c) & (y_pred == c)).sum())
        fp = float(((y_true != c) & (y_pred == c)).sum())
        fn = float(((y_true == c) & (y_pred != c)).sum())
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2.0 * p * r / (p + r)) if (p + r) > 0 else 0.0
        f1s.append(f1)
    return float(np.mean(f1s)) if f1s else 0.0


def class_dist(y: np.ndarray) -> str:
    if y.size == 0:
        return ""
    vals, cnts = np.unique(y, return_counts=True)
    parts = [f"{LABEL_MAP.get(int(v), str(v))}:{int(c)}" for v, c in zip(vals, cnts)]
    return ",".join(parts)


def build_monthly_frame(df: pd.DataFrame, ts_col: str, mn_cols: List[str]) -> pd.DataFrame:
    x = df.sort_values(ts_col).copy()
    x["month_ts"] = x[ts_col].dt.to_period("M").dt.to_timestamp()

    ohlc = x.groupby("month_ts", sort=True).agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
    )
    feat = x.groupby("month_ts", sort=True)[mn_cols].last()
    m = ohlc.join(feat, how="left").reset_index().rename(columns={"month_ts": "ts_mn"})
    return m


def main() -> int:
    args = parse_args()
    parquet_in = Path(args.parquet_in)
    csv_out = Path(args.csv_out)
    pred_out = Path(args.pred_out)
    if not parquet_in.exists():
        raise FileNotFoundError(f"Parquet not found: {parquet_in}")

    train_end = pd.Timestamp(args.train_end)
    train_end = train_end.tz_localize("UTC") if train_end.tzinfo is None else train_end.tz_convert("UTC")
    val_end = pd.Timestamp(args.val_end)
    val_end = val_end.tz_localize("UTC") if val_end.tzinfo is None else val_end.tz_convert("UTC")
    if val_end <= train_end:
        raise RuntimeError("--val-end must be > --train-end.")

    print(f"[run_ml_regime_month_v1] VERSION={VERSION}")
    print(f"[INFO] parquet_in={parquet_in}")

    schema = pd.read_parquet(parquet_in, engine="pyarrow")
    ts_col = auto_detect_ts_col(schema)
    mn_cols = sorted([c for c in schema.columns if c.endswith("_mn")])
    if not mn_cols:
        raise RuntimeError("No *_mn columns found. Build monthly context first.")

    read_cols = [c for c in [ts_col, "open", "high", "low", "close"] + mn_cols if c in schema.columns]
    df = pd.read_parquet(parquet_in, columns=read_cols, engine="pyarrow")
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

    m = build_monthly_frame(df, ts_col=ts_col, mn_cols=mn_cols)
    m["ts_mn"] = pd.to_datetime(m["ts_mn"], utc=True, errors="coerce")
    m["ret_fwd"] = m["close"].shift(-int(args.horizon_months)) / m["close"] - 1.0
    m = m.dropna(subset=["ret_fwd"]).reset_index(drop=True)

    train_mask = m["ts_mn"] < train_end
    val_mask = (m["ts_mn"] >= train_end) & (m["ts_mn"] < val_end)
    test_mask = m["ts_mn"] >= val_end
    if train_mask.sum() < 24 or val_mask.sum() < 6 or test_mask.sum() < 6:
        raise RuntimeError(
            f"Not enough monthly samples. train={int(train_mask.sum())} val={int(val_mask.sum())} test={int(test_mask.sum())}"
        )

    thr = float(np.quantile(np.abs(m.loc[train_mask, "ret_fwd"].to_numpy()), float(args.trend_quantile)))
    m["y"] = 0
    m.loc[m["ret_fwd"] >= thr, "y"] = 1
    m.loc[m["ret_fwd"] <= -thr, "y"] = 2

    feature_cols = mn_cols.copy()
    x = m[feature_cols].replace([np.inf, -np.inf], np.nan)
    med = x.loc[train_mask].median(numeric_only=True)
    x = x.fillna(med)

    x_train = x.loc[train_mask].to_numpy(dtype=float)
    x_val = x.loc[val_mask].to_numpy(dtype=float)
    x_test = x.loc[test_mask].to_numpy(dtype=float)
    y_train = m.loc[train_mask, "y"].to_numpy(dtype=int)
    y_val = m.loc[val_mask, "y"].to_numpy(dtype=int)
    y_test = m.loc[test_mask, "y"].to_numpy(dtype=int)

    mu = x_train.mean(axis=0)
    sd = x_train.std(axis=0)
    sd = np.where(sd < 1e-9, 1.0, sd)
    x_train = (x_train - mu) / sd
    x_val = (x_val - mu) / sd
    x_test = (x_test - mu) / sd
    x_train_b = np.c_[np.ones((x_train.shape[0], 1)), x_train]
    x_val_b = np.c_[np.ones((x_val.shape[0], 1)), x_val]
    x_test_b = np.c_[np.ones((x_test.shape[0], 1)), x_test]

    # One-vs-rest 3-class logistic
    ws: List[np.ndarray] = []
    for cls in [0, 1, 2]:
        y_bin = (y_train == cls).astype(float)
        ws.append(fit_logistic_gd(x_train_b, y_bin, l2=1e-3, lr=0.05, n_iter=2200))

    def predict_proba(xb: np.ndarray) -> np.ndarray:
        raw = np.column_stack([sigmoid(xb @ w) for w in ws])
        s = raw.sum(axis=1, keepdims=True)
        s = np.where(s <= 1e-12, 1.0, s)
        return raw / s

    p_train = predict_proba(x_train_b)
    p_val = predict_proba(x_val_b)
    p_test = predict_proba(x_test_b)

    pred_train = p_train.argmax(axis=1)
    pred_val = p_val.argmax(axis=1)
    pred_test = p_test.argmax(axis=1)

    summary = pd.DataFrame(
        [
            {
                "set": "TRAIN",
                "n_samples": int(len(y_train)),
                "accuracy": float((pred_train == y_train).mean()),
                "macro_f1": macro_f1(y_train, pred_train, [0, 1, 2]),
                "class_dist": class_dist(y_train),
            },
            {
                "set": "VAL",
                "n_samples": int(len(y_val)),
                "accuracy": float((pred_val == y_val).mean()),
                "macro_f1": macro_f1(y_val, pred_val, [0, 1, 2]),
                "class_dist": class_dist(y_val),
            },
            {
                "set": "TEST",
                "n_samples": int(len(y_test)),
                "accuracy": float((pred_test == y_test).mean()),
                "macro_f1": macro_f1(y_test, pred_test, [0, 1, 2]),
                "class_dist": class_dist(y_test),
            },
        ]
    )

    pred_df = m[["ts_mn", "close", "ret_fwd", "y"]].copy()
    pred_df["y_name"] = pred_df["y"].map(LABEL_MAP)
    probs_all = predict_proba(np.c_[np.ones((x.shape[0], 1)), ((x.to_numpy(dtype=float) - mu) / sd)])
    pred_df["p_range"] = probs_all[:, 0]
    pred_df["p_trend_up"] = probs_all[:, 1]
    pred_df["p_trend_down"] = probs_all[:, 2]
    pred_df["pred"] = probs_all.argmax(axis=1)
    pred_df["pred_name"] = pred_df["pred"].map(LABEL_MAP)

    coef_rows: List[Dict] = []
    for cls, w in enumerate(ws):
        coef_rows.append({"class": LABEL_MAP[cls], "feature": "bias", "coef": float(w[0])})
        for i, c in enumerate(feature_cols, start=1):
            coef_rows.append({"class": LABEL_MAP[cls], "feature": c, "coef": float(w[i])})
    coef_df = pd.DataFrame(coef_rows)

    csv_out.parent.mkdir(parents=True, exist_ok=True)
    pred_out.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(csv_out, index=False)
    pred_df.to_csv(pred_out, index=False)
    coef_path = csv_out.with_name(csv_out.stem + "_coefs.csv")
    coef_df.to_csv(coef_path, index=False)

    print(f"[INFO] ts_col={ts_col} monthly_rows={len(m)} trend_threshold={thr:.6f}")
    print("\n=== ML MONTHLY REGIME V1 SUMMARY ===")
    print(summary.to_string(index=False))
    print(f"\n[OK] summary_csv={csv_out}")
    print(f"[OK] pred_csv={pred_out}")
    print(f"[OK] coef_csv={coef_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

