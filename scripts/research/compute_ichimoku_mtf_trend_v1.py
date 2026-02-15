from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


VERSION = "2026-02-14-ichimoku-mtf-trend-v1"

TF_RULES = {
    "MN1": "ME",
    "W1": "W-MON",
    "D1": "1D",
    "H4": "4h",
    "H1": "1h",
    "M30": "30min",
}

TF_ORDER = ["MN1", "W1", "D1", "H4", "H1", "M30"]


@dataclass(frozen=True)
class IchimokuMTFParams:
    tenkan_len: int = 9
    kijun_len: int = 26
    senkou_b_len: int = 52
    disp: int = 25
    w_kumo: int = 1
    w_price: int = 1
    w_chikou: int = 1
    w_tk: int = 1
    bull_pass: int = 3
    bear_pass: int = 3
    neutral_delta: float = 1.0

    @property
    def max_score(self) -> int:
        return int(self.w_kumo + self.w_price + self.w_chikou + self.w_tk)

    @property
    def warmup_bars(self) -> int:
        return int(max(self.tenkan_len, self.kijun_len, self.senkou_b_len) + self.disp)


def auto_detect_ts_col(df: pd.DataFrame) -> str:
    for c in ["t_x", "ts", "timestamp", "time", "datetime", "open_time", "open_ts", "time_utc", "ts_utc", "t"]:
        if c in df.columns:
            return c
    dt_cols = [c for c in df.columns if str(df[c].dtype).startswith("datetime64")]
    if dt_cols:
        return dt_cols[0]
    raise RuntimeError("Cannot auto-detect timestamp column.")


def normalize_ts_series(s: pd.Series) -> pd.Series:
    if str(s.dtype).startswith("datetime64"):
        return s
    if np.issubdtype(s.dtype, np.integer):
        return pd.to_datetime(s, unit="ms", utc=True).dt.tz_convert(None)
    return pd.to_datetime(s, utc=True, errors="coerce").dt.tz_convert(None)


def ensure_ohlc_index(df: pd.DataFrame, ts_col: str = "") -> pd.DataFrame:
    x = df.copy()
    if isinstance(x.index, pd.DatetimeIndex):
        idx = x.index
        if idx.tz is not None:
            x.index = idx.tz_convert("UTC").tz_localize(None)
    else:
        c = ts_col if ts_col else auto_detect_ts_col(x)
        x[c] = normalize_ts_series(x[c])
        x = x.set_index(c)

    req = ["open", "high", "low", "close"]
    miss = [c for c in req if c not in x.columns]
    if miss:
        raise RuntimeError(f"Missing required OHLC columns: {miss}")

    x = x.sort_index()
    x = x[~x.index.duplicated(keep="last")]
    return x[req].copy()


def resample_ohlc(df_ohlc: pd.DataFrame, rule: str) -> pd.DataFrame:
    if rule == "1min":
        return df_ohlc.copy()
    out = (
        df_ohlc.resample(rule, label="left", closed="left")
        .agg({"open": "first", "high": "max", "low": "min", "close": "last"})
        .dropna()
    )
    return out


def compute_ichimoku_tf(df_tf: pd.DataFrame, p: IchimokuMTFParams, include_debug: bool) -> pd.DataFrame:
    out = df_tf.copy()
    h = out["high"]
    l = out["low"]
    c = out["close"]

    tenkan = (h.rolling(p.tenkan_len, min_periods=p.tenkan_len).max() + l.rolling(p.tenkan_len, min_periods=p.tenkan_len).min()) / 2.0
    kijun = (h.rolling(p.kijun_len, min_periods=p.kijun_len).max() + l.rolling(p.kijun_len, min_periods=p.kijun_len).min()) / 2.0
    ssa = (tenkan + kijun) / 2.0
    ssb = (h.rolling(p.senkou_b_len, min_periods=p.senkou_b_len).max() + l.rolling(p.senkou_b_len, min_periods=p.senkou_b_len).min()) / 2.0

    ssa_vis = ssa.shift(p.disp)
    ssb_vis = ssb.shift(p.disp)
    close_lag = c.shift(p.disp)

    bull_kumo = (ssa > ssb).astype(int)
    bull_price = (c > np.maximum(ssa_vis, ssb_vis)).astype(int)
    bull_chikou = (c > close_lag).astype(int)
    bull_tk = (tenkan > kijun).astype(int)

    bear_kumo = (ssa < ssb).astype(int)
    bear_price = (c < np.minimum(ssa_vis, ssb_vis)).astype(int)
    bear_chikou = (c < close_lag).astype(int)
    bear_tk = (tenkan < kijun).astype(int)

    bull_score = (
        bull_kumo * p.w_kumo
        + bull_price * p.w_price
        + bull_chikou * p.w_chikou
        + bull_tk * p.w_tk
    ).astype(float)
    bear_score = (
        bear_kumo * p.w_kumo
        + bear_price * p.w_price
        + bear_chikou * p.w_chikou
        + bear_tk * p.w_tk
    ).astype(float)

    ready = tenkan.notna() & kijun.notna() & ssa.notna() & ssb.notna() & ssa_vis.notna() & ssb_vis.notna() & close_lag.notna()
    ready &= (np.arange(len(out)) >= p.warmup_bars)

    neutral_forced = (np.abs(bull_score - bear_score) < float(p.neutral_delta))
    is_long = ready & (~neutral_forced) & (bull_score >= p.bull_pass) & (bull_score > bear_score)
    is_short = ready & (~neutral_forced) & (bear_score >= p.bear_pass) & (bear_score > bull_score)

    label = np.where(is_long, "LONG", np.where(is_short, "SHORT", "NEUTRE"))
    score_display = np.where(label == "LONG", bull_score, np.where(label == "SHORT", bear_score, np.maximum(bull_score, bear_score)))
    score_display = np.where(ready, score_display, 0.0)
    bull_score = np.where(ready, bull_score, 0.0)
    bear_score = np.where(ready, bear_score, 0.0)

    out_res = pd.DataFrame(
        {
            "bull_score": bull_score.astype(int),
            "bear_score": bear_score.astype(int),
            "label": label.astype(str),
            "score_display": score_display.astype(int),
        },
        index=out.index,
    )

    if include_debug:
        dbg = pd.DataFrame(
            {
                "tenkan": tenkan,
                "kijun": kijun,
                "ssa": ssa,
                "ssb": ssb,
                "ssa_vis": ssa_vis,
                "ssb_vis": ssb_vis,
                "close_lag": close_lag,
                "bull_kumo": bull_kumo,
                "bull_price": bull_price,
                "bull_chikou": bull_chikou,
                "bull_tk": bull_tk,
                "bear_kumo": bear_kumo,
                "bear_price": bear_price,
                "bear_chikou": bear_chikou,
                "bear_tk": bear_tk,
                "ready": ready.astype(int),
            },
            index=out.index,
        )
        out_res = pd.concat([out_res, dbg], axis=1)

    return out_res


def compute_ichimoku_mtf_trend(df_m1: pd.DataFrame, params: IchimokuMTFParams, include_debug: bool = False) -> Dict[str, pd.DataFrame]:
    base = ensure_ohlc_index(df_m1)
    out: Dict[str, pd.DataFrame] = {}
    for tf in TF_ORDER:
        tf_ohlc = resample_ohlc(base, TF_RULES[tf])
        out[tf] = compute_ichimoku_tf(tf_ohlc, params, include_debug=include_debug)
    return out


def snapshot_last(results: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: List[Dict] = []
    for tf in TF_ORDER:
        df = results.get(tf)
        if df is None or len(df) == 0:
            rows.append({"tf": tf, "arrow": "•", "score_display": 0, "label": "NEUTRE"})
            continue
        r = df.iloc[-1]
        label = str(r.get("label", "NEUTRE"))
        arrow = "↑" if label == "LONG" else ("↓" if label == "SHORT" else "•")
        rows.append(
            {
                "tf": tf,
                "arrow": arrow,
                "score_display": int(r.get("score_display", 0)),
                "bull_score": int(r.get("bull_score", 0)),
                "bear_score": int(r.get("bear_score", 0)),
                "label": label,
            }
        )
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute Ichimoku trend scores on MTF (MN1,W1,D1,H4,H1,M30).")
    p.add_argument("--input", required=True, help="Input parquet with M1 OHLC (and ts col or DatetimeIndex).")
    p.add_argument("--output-dir", required=True, help="Output folder for per-TF csv + snapshot.")
    p.add_argument("--ts-col", default="", help="Timestamp column (auto if empty).")
    p.add_argument("--tenkan-len", type=int, default=9)
    p.add_argument("--kijun-len", type=int, default=26)
    p.add_argument("--senkou-b-len", type=int, default=52)
    p.add_argument("--disp", type=int, default=25)
    p.add_argument("--w-kumo", type=int, default=1)
    p.add_argument("--w-price", type=int, default=1)
    p.add_argument("--w-chikou", type=int, default=1)
    p.add_argument("--w-tk", type=int, default=1)
    p.add_argument("--bull-pass", type=int, default=3)
    p.add_argument("--bear-pass", type=int, default=3)
    p.add_argument("--neutral-delta", type=float, default=1.0)
    p.add_argument("--debug-cols", action="store_true", help="Include Ichimoku internal columns in outputs.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    in_path = Path(args.input)
    out_dir = Path(args.output_dir)
    if not in_path.exists():
        raise FileNotFoundError(f"Input parquet not found: {in_path}")

    p = IchimokuMTFParams(
        tenkan_len=int(args.tenkan_len),
        kijun_len=int(args.kijun_len),
        senkou_b_len=int(args.senkou_b_len),
        disp=int(args.disp),
        w_kumo=int(args.w_kumo),
        w_price=int(args.w_price),
        w_chikou=int(args.w_chikou),
        w_tk=int(args.w_tk),
        bull_pass=int(args.bull_pass),
        bear_pass=int(args.bear_pass),
        neutral_delta=float(args.neutral_delta),
    )
    if p.bull_pass > p.max_score or p.bear_pass > p.max_score:
        raise RuntimeError(f"bull_pass/bear_pass must be <= max_score ({p.max_score}).")

    print(f"[compute_ichimoku_mtf_trend_v1] VERSION={VERSION}")
    print(f"[INFO] input={in_path}")
    print(f"[INFO] max_score={p.max_score} warmup_bars={p.warmup_bars}")

    df = pd.read_parquet(in_path, engine="pyarrow")
    if not isinstance(df.index, pd.DatetimeIndex):
        ts_col = args.ts_col if args.ts_col else auto_detect_ts_col(df)
        df[ts_col] = normalize_ts_series(df[ts_col])
        df = df.set_index(ts_col)
    else:
        if df.index.tz is not None:
            df.index = df.index.tz_convert("UTC").tz_localize(None)

    results = compute_ichimoku_mtf_trend(df, params=p, include_debug=bool(args.debug_cols))
    snap = snapshot_last(results)

    out_dir.mkdir(parents=True, exist_ok=True)
    for tf, tf_df in results.items():
        tf_path = out_dir / f"ichimoku_trend_{tf.lower()}.csv"
        tf_df.to_csv(tf_path, index=True)
    snap_path = out_dir / "ichimoku_trend_snapshot.csv"
    snap.to_csv(snap_path, index=False)

    print("\n=== SNAPSHOT ===")
    print(snap.to_string(index=False))
    print(f"\n[OK] snapshot_csv={snap_path}")
    print(f"[OK] tf_csv_dir={out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
