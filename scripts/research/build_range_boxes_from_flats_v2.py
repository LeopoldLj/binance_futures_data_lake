from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd


VERSION = "2026-02-15-build-range-boxes-from-flats-v2"

TF_RULES = {
    "MN1": "ME",
    "W1": "W-MON",
    "D1": "1D",
    "H4": "4h",
    "H1": "1h",
    "M30": "30min",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build range boxes (low/high) from flat-line candidates.")
    p.add_argument("--input", required=True, help="Input parquet (M1 OHLC with ts).")
    p.add_argument("--candidates-csv", required=True, help="CSV from range_candidates_from_flats_v1.csv")
    p.add_argument("--output-csv", required=True, help="Output CSV with box metrics and breakout status.")
    p.add_argument("--summary-csv", default="", help="Optional summary CSV by TF/status.")
    p.add_argument("--confirm-bars", type=int, default=2, help="Consecutive closes needed to confirm breakout.")
    p.add_argument("--lookahead-bars", type=int, default=32, help="How many bars after end_ts to scan for breakout.")
    p.add_argument("--buffer-frac", type=float, default=0.10, help="Breakout buffer as fraction of range width.")
    p.add_argument(
        "--expand-window",
        action="store_true",
        help="Extend box window left/right while full candles (wick included) remain inside [range_low, range_high].",
    )
    p.add_argument("--max-expand-bars", type=int, default=200, help="Max bars for each side expansion.")
    p.add_argument("--contain-tol-frac", type=float, default=0.0, help="Containment tolerance as fraction of range width.")
    p.add_argument(
        "--expand-contain-mode",
        default="wick",
        choices=["wick", "body", "close"],
        help="Containment rule for extension: wick=low/high inside, body=open/close inside, close=close only.",
    )
    p.add_argument(
        "--expand-break-confirm-bars",
        type=int,
        default=2,
        help="Stop extension only after this many consecutive out-of-range bars.",
    )
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


def ensure_ohlc(df: pd.DataFrame) -> pd.DataFrame:
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
    return (
        df_ohlc.resample(rule, label="left", closed="left")
        .agg({"open": "first", "high": "max", "low": "min", "close": "last"})
        .dropna()
    )


def _first_confirmed_idx(cond: np.ndarray, n_confirm: int) -> int:
    if len(cond) == 0:
        return -1
    run = 0
    for i, ok in enumerate(cond):
        run = run + 1 if ok else 0
        if run >= n_confirm:
            return i - n_confirm + 1
    return -1


def _expand_box_window(
    tf_df: pd.DataFrame,
    st: pd.Timestamp,
    en: pd.Timestamp,
    range_low: float,
    range_high: float,
    max_expand_bars: int,
    contain_tol_frac: float,
    contain_mode: str,
    break_confirm_bars: int,
) -> tuple[pd.Timestamp, pd.Timestamp, int, int]:
    idx = tf_df.index
    if len(idx) == 0:
        return st, en, 0, 0

    i0 = int(idx.searchsorted(st, side="left"))
    i1 = int(idx.searchsorted(en, side="right")) - 1
    if i0 < 0:
        i0 = 0
    if i1 < i0:
        i1 = i0
    if i1 >= len(tf_df):
        i1 = len(tf_df) - 1

    width = max(0.0, float(range_high - range_low))
    tol = float(contain_tol_frac) * width

    def is_inside(r: pd.Series) -> bool:
        lo = range_low - tol
        hi = range_high + tol
        if contain_mode == "close":
            c = float(r["close"])
            return c >= lo and c <= hi
        if contain_mode == "body":
            o = float(r["open"])
            c = float(r["close"])
            return min(o, c) >= lo and max(o, c) <= hi
        # wick mode
        l = float(r["low"])
        h = float(r["high"])
        return l >= lo and h <= hi

    confirm = max(1, int(break_confirm_bars))

    left_n = 0
    outside_run = 0
    last_inside_i0 = i0
    steps = 0
    j = i0 - 1
    while j >= 0 and steps < max_expand_bars:
        steps += 1
        prev = tf_df.iloc[j]
        if is_inside(prev):
            last_inside_i0 = j
            outside_run = 0
            left_n += 1
        else:
            outside_run += 1
            if outside_run >= confirm:
                break
        j -= 1
    i0 = last_inside_i0

    right_n = 0
    outside_run = 0
    last_inside_i1 = i1
    steps = 0
    j = i1 + 1
    while j < len(tf_df) and steps < max_expand_bars:
        steps += 1
        nxt = tf_df.iloc[j]
        if is_inside(nxt):
            last_inside_i1 = j
            outside_run = 0
            right_n += 1
        else:
            outside_run += 1
            if outside_run >= confirm:
                break
        j += 1
    i1 = last_inside_i1

    return idx[i0], idx[i1], left_n, right_n


def main() -> int:
    args = parse_args()
    in_path = Path(args.input)
    cand_path = Path(args.candidates_csv)
    out_path = Path(args.output_csv)
    summary_path = Path(args.summary_csv) if args.summary_csv else None

    if not in_path.exists():
        raise FileNotFoundError(f"Input parquet not found: {in_path}")
    if not cand_path.exists():
        raise FileNotFoundError(f"Candidates CSV not found: {cand_path}")

    print(f"[build_range_boxes_from_flats_v2] VERSION={VERSION}")
    print(f"[INFO] input={in_path}")
    print(f"[INFO] candidates_csv={cand_path}")

    raw = pd.read_parquet(in_path, engine="pyarrow")
    base = ensure_ohlc(raw)

    c = pd.read_csv(cand_path)
    needed = ["tf", "start_ts", "end_ts"]
    miss = [x for x in needed if x not in c.columns]
    if miss:
        raise RuntimeError(f"Missing required candidate columns: {miss}")
    c["start_ts"] = pd.to_datetime(c["start_ts"], utc=True, errors="coerce").dt.tz_convert(None)
    c["end_ts"] = pd.to_datetime(c["end_ts"], utc=True, errors="coerce").dt.tz_convert(None)
    c = c.dropna(subset=["start_ts", "end_ts"]).copy()
    c = c[c["start_ts"] <= c["end_ts"]].copy()
    if c.empty:
        raise RuntimeError("No valid candidate rows after ts parsing.")

    tf_cache: Dict[str, pd.DataFrame] = {}
    for tf in sorted(c["tf"].unique()):
        if tf not in TF_RULES:
            continue
        tf_cache[tf] = resample_ohlc(base, TF_RULES[tf])

    rows = []
    n_confirm = max(1, int(args.confirm_bars))
    lookahead = max(1, int(args.lookahead_bars))
    buffer_frac = max(0.0, float(args.buffer_frac))

    for _, r in c.iterrows():
        tf = str(r["tf"])
        if tf not in tf_cache:
            continue
        tf_df = tf_cache[tf]

        st_orig = pd.Timestamp(r["start_ts"])
        en_orig = pd.Timestamp(r["end_ts"])
        box = tf_df[(tf_df.index >= st_orig) & (tf_df.index <= en_orig)]
        if box.empty:
            continue

        range_low = float(box["low"].min())
        range_high = float(box["high"].max())
        range_width = float(range_high - range_low)
        if not np.isfinite(range_width) or range_width <= 0:
            continue
        range_mid = (range_low + range_high) / 2.0
        buffer_abs = buffer_frac * range_width

        st = st_orig
        en = en_orig
        expanded_left_bars = 0
        expanded_right_bars = 0
        if args.expand_window:
            st, en, expanded_left_bars, expanded_right_bars = _expand_box_window(
                tf_df=tf_df,
                st=st_orig,
                en=en_orig,
                range_low=range_low,
                range_high=range_high,
                max_expand_bars=max(0, int(args.max_expand_bars)),
                contain_tol_frac=max(0.0, float(args.contain_tol_frac)),
                contain_mode=str(args.expand_contain_mode),
                break_confirm_bars=max(1, int(args.expand_break_confirm_bars)),
            )

        fwd = tf_df[tf_df.index > en].head(lookahead)
        status = "ACTIVE"
        breakout_ts = pd.NaT
        breakout_side = "NONE"
        bars_to_break = np.nan

        if len(fwd) > 0:
            up_cond = (fwd["close"].to_numpy(dtype=float) > (range_high + buffer_abs))
            dn_cond = (fwd["close"].to_numpy(dtype=float) < (range_low - buffer_abs))
            up_i = _first_confirmed_idx(up_cond, n_confirm=n_confirm)
            dn_i = _first_confirmed_idx(dn_cond, n_confirm=n_confirm)

            if up_i >= 0 and (dn_i < 0 or up_i <= dn_i):
                status = "BREAKOUT_UP"
                breakout_side = "UP"
                breakout_ts = fwd.index[up_i]
                bars_to_break = float(up_i + 1)
            elif dn_i >= 0 and (up_i < 0 or dn_i < up_i):
                status = "BREAKOUT_DOWN"
                breakout_side = "DOWN"
                breakout_ts = fwd.index[dn_i]
                bars_to_break = float(dn_i + 1)

        out = dict(r.to_dict())
        out.update(
            {
                "start_ts_orig": st_orig,
                "end_ts_orig": en_orig,
                "start_ts": st,
                "end_ts": en,
                "range_low": range_low,
                "range_high": range_high,
                "range_mid_box": range_mid,
                "range_width": range_width,
                "buffer_abs": float(buffer_abs),
                "expanded_left_bars": int(expanded_left_bars),
                "expanded_right_bars": int(expanded_right_bars),
                "expanded_total_bars": int(expanded_left_bars + expanded_right_bars),
                "status": status,
                "breakout_side": breakout_side,
                "breakout_ts": breakout_ts,
                "bars_to_break": bars_to_break,
            }
        )
        rows.append(out)

    out_df = pd.DataFrame(rows)
    out_df = out_df.sort_values(["tf", "strength"], ascending=[True, False]) if ("strength" in out_df.columns and not out_df.empty) else out_df

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)

    if summary_path is not None:
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        if out_df.empty:
            summary = pd.DataFrame(columns=["tf", "status", "n", "avg_width", "avg_strength"])
        else:
            summary = (
                out_df.groupby(["tf", "status"], dropna=False)
                .agg(
                    n=("status", "size"),
                    avg_width=("range_width", "mean"),
                    avg_strength=("strength", "mean") if "strength" in out_df.columns else ("range_width", "mean"),
                )
                .reset_index()
                .sort_values(["tf", "n"], ascending=[True, False])
            )
        summary.to_csv(summary_path, index=False)

    print("\n=== BOX SUMMARY ===")
    if out_df.empty:
        print("[WARN] no output rows.")
    else:
        show_cols = [c for c in ["tf", "start_ts", "end_ts", "range_low", "range_high", "range_mid_box", "range_width", "status", "breakout_side", "bars_to_break"] if c in out_df.columns]
        print(out_df.head(20)[show_cols].to_string(index=False))
    print(f"\n[OK] output_csv={out_path}")
    if summary_path is not None:
        print(f"[OK] summary_csv={summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
