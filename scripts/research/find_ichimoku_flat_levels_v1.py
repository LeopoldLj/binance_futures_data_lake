from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


VERSION = "2026-02-15-find-ichimoku-flat-levels-v1"

TF_RULES = {
    "MN1": "ME",
    "W1": "W-MON",
    "D1": "1D",
    "H4": "4h",
    "H1": "1h",
    "M30": "30min",
}


@dataclass(frozen=True)
class Params:
    kijun_len: int = 26
    senkou_b_len: int = 52
    min_flat_bars: int = 3
    abs_tol: float = 1e-8
    rel_tol: float = 1e-8


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Find longest flat Kijun/SSB segments on multiple TF.")
    p.add_argument("--input", required=True, help="Input parquet with ts/open/high/low/close.")
    p.add_argument("--output-dir", required=True, help="Output directory for CSV reports.")
    p.add_argument("--tfs", default="MN1,W1,D1,H4,H1,M30", help="Comma-separated TF list.")
    p.add_argument("--kijun-len", type=int, default=26)
    p.add_argument("--senkou-b-len", type=int, default=52)
    p.add_argument("--min-flat-bars", type=int, default=3, help="Minimum plateau length in bars.")
    p.add_argument("--abs-tol", type=float, default=1e-8, help="Absolute tolerance for flat equality.")
    p.add_argument("--rel-tol", type=float, default=1e-8, help="Relative tolerance for flat equality.")
    p.add_argument("--top-n", type=int, default=20, help="Top N longest flats for console summary.")
    return p.parse_args()


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
        return pd.to_datetime(s, utc=True, errors="coerce").dt.tz_convert(None)
    if pd.api.types.is_integer_dtype(s.dtype):
        return pd.to_datetime(s, unit="ms", utc=True).dt.tz_convert(None)
    return pd.to_datetime(s, utc=True, errors="coerce").dt.tz_convert(None)


def ensure_ohlc_index(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    ts_col = auto_detect_ts_col(x)
    x[ts_col] = normalize_ts_series(x[ts_col])
    x = x.dropna(subset=[ts_col]).set_index(ts_col).sort_index()
    x = x[~x.index.duplicated(keep="last")]
    req = ["open", "high", "low", "close"]
    miss = [c for c in req if c not in x.columns]
    if miss:
        raise RuntimeError(f"Missing OHLC columns: {miss}")
    return x[req].copy()


def resample_ohlc(df_ohlc: pd.DataFrame, rule: str) -> pd.DataFrame:
    out = (
        df_ohlc.resample(rule, label="left", closed="left")
        .agg({"open": "first", "high": "max", "low": "min", "close": "last"})
        .dropna()
    )
    return out


def compute_lines(df_tf: pd.DataFrame, p: Params) -> pd.DataFrame:
    h = df_tf["high"]
    l = df_tf["low"]
    kijun = (h.rolling(p.kijun_len, min_periods=p.kijun_len).max() + l.rolling(p.kijun_len, min_periods=p.kijun_len).min()) / 2.0
    ssb = (h.rolling(p.senkou_b_len, min_periods=p.senkou_b_len).max() + l.rolling(p.senkou_b_len, min_periods=p.senkou_b_len).min()) / 2.0
    return pd.DataFrame({"kijun": kijun, "ssb": ssb}, index=df_tf.index)


def _flat_mask(v: pd.Series, abs_tol: float, rel_tol: float) -> np.ndarray:
    arr = v.to_numpy(dtype=float)
    prev = np.roll(arr, 1)
    prev[0] = np.nan
    tol = np.maximum(abs_tol, rel_tol * np.maximum(1.0, np.abs(prev)))
    flat_step = np.isfinite(arr) & np.isfinite(prev) & (np.abs(arr - prev) <= tol)
    return flat_step


def find_flat_segments(index: pd.DatetimeIndex, values: pd.Series, line_name: str, tf: str, p: Params) -> pd.DataFrame:
    step_flat = _flat_mask(values, abs_tol=p.abs_tol, rel_tol=p.rel_tol)
    segments: List[Dict] = []

    start_i = None
    for i in range(1, len(step_flat)):
        if step_flat[i]:
            if start_i is None:
                start_i = i - 1
        else:
            if start_i is not None:
                end_i = i - 1
                length = end_i - start_i + 1
                if length >= p.min_flat_bars:
                    segments.append(
                        {
                            "tf": tf,
                            "line": line_name,
                            "start_ts": index[start_i],
                            "end_ts": index[end_i],
                            "length_bars": int(length),
                            "value": float(values.iloc[start_i]),
                        }
                    )
                start_i = None

    if start_i is not None:
        end_i = len(step_flat) - 1
        length = end_i - start_i + 1
        if length >= p.min_flat_bars:
            segments.append(
                {
                    "tf": tf,
                    "line": line_name,
                    "start_ts": index[start_i],
                    "end_ts": index[end_i],
                    "length_bars": int(length),
                    "value": float(values.iloc[start_i]),
                }
            )

    out = pd.DataFrame(segments)
    if out.empty:
        return out
    out["duration_days"] = (pd.to_datetime(out["end_ts"]) - pd.to_datetime(out["start_ts"])).dt.total_seconds() / 86400.0
    out = out.sort_values(["length_bars", "duration_days"], ascending=[False, False]).reset_index(drop=True)
    return out


def main() -> int:
    args = parse_args()
    in_path = Path(args.input)
    out_dir = Path(args.output_dir)
    if not in_path.exists():
        raise FileNotFoundError(f"Input parquet not found: {in_path}")

    tfs = [x.strip().upper() for x in str(args.tfs).split(",") if x.strip()]
    bad = [tf for tf in tfs if tf not in TF_RULES]
    if bad:
        raise RuntimeError(f"Unsupported TF(s): {bad}. Allowed={list(TF_RULES.keys())}")

    p = Params(
        kijun_len=int(args.kijun_len),
        senkou_b_len=int(args.senkou_b_len),
        min_flat_bars=int(args.min_flat_bars),
        abs_tol=float(args.abs_tol),
        rel_tol=float(args.rel_tol),
    )

    print(f"[find_ichimoku_flat_levels_v1] VERSION={VERSION}")
    print(f"[INFO] input={in_path}")
    print(f"[INFO] tfs={','.join(tfs)} kijun_len={p.kijun_len} senkou_b_len={p.senkou_b_len} min_flat_bars={p.min_flat_bars}")

    df = pd.read_parquet(in_path, engine="pyarrow")
    ohlc = ensure_ohlc_index(df)

    all_rows: List[pd.DataFrame] = []
    out_dir.mkdir(parents=True, exist_ok=True)

    for tf in tfs:
        tf_ohlc = resample_ohlc(ohlc, TF_RULES[tf])
        lines = compute_lines(tf_ohlc, p)
        k = find_flat_segments(lines.index, lines["kijun"], "kijun", tf, p)
        s = find_flat_segments(lines.index, lines["ssb"], "ssb", tf, p)
        tf_rows = pd.concat([k, s], ignore_index=True) if (not k.empty or not s.empty) else pd.DataFrame()
        if not tf_rows.empty:
            tf_rows = tf_rows.sort_values(["line", "length_bars", "duration_days"], ascending=[True, False, False]).reset_index(drop=True)
        tf_out = out_dir / f"ichimoku_flat_{tf.lower()}.csv"
        tf_rows.to_csv(tf_out, index=False)
        all_rows.append(tf_rows if not tf_rows.empty else pd.DataFrame(columns=["tf", "line", "start_ts", "end_ts", "length_bars", "value", "duration_days"]))

    flat_all = pd.concat(all_rows, ignore_index=True)
    flat_all_out = out_dir / "ichimoku_flat_all_tfs.csv"
    flat_all.to_csv(flat_all_out, index=False)

    summary = (
        flat_all.groupby(["tf", "line"], dropna=False)
        .agg(
            n_segments=("line", "size"),
            max_len_bars=("length_bars", "max"),
            p95_len_bars=("length_bars", lambda x: float(np.percentile(x, 95)) if len(x) else 0.0),
            median_len_bars=("length_bars", "median"),
        )
        .reset_index()
        .sort_values(["tf", "line"])
    ) if not flat_all.empty else pd.DataFrame(columns=["tf", "line", "n_segments", "max_len_bars", "p95_len_bars", "median_len_bars"])
    summary_out = out_dir / "ichimoku_flat_summary.csv"
    summary.to_csv(summary_out, index=False)

    print("\n=== TOP FLATS ===")
    if flat_all.empty:
        print("[WARN] no flat segments found with current parameters.")
    else:
        print(flat_all.sort_values(["length_bars", "duration_days"], ascending=[False, False]).head(int(args.top_n)).to_string(index=False))

    print(f"\n[OK] flat_all_csv={flat_all_out}")
    print(f"[OK] summary_csv={summary_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
