from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


VERSION = "2026-02-14-audit-regime-cases-v1"
TF_ORDER = ["MN1", "W1", "D1", "H4", "H1", "M30"]
TF_FILE = {tf: f"ichimoku_trend_{tf.lower()}.csv" for tf in TF_ORDER}
DIR_MAP = {"LONG": 1, "SHORT": -1, "NEUTRE": 0}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Audit regime cases from Ichimoku MTF outputs.")
    p.add_argument("--tf-dir", required=True, help="Directory containing ichimoku_trend_<tf>.csv files.")
    p.add_argument("--out-csv", required=True, help="Detailed per-bar audit CSV (M30 timeline).")
    p.add_argument("--summary-csv", required=True, help="Summary CSV by regime_case.")
    p.add_argument("--neutral-thresh", type=float, default=3.0, help="Abs(score) threshold under which context is neutral-ish.")
    return p.parse_args()


def _read_tf(path: Path, tf: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing TF file: {path}")
    df = pd.read_csv(path)
    ts_col = df.columns[0]
    df = df.rename(columns={ts_col: "ts"})
    df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    df = df.dropna(subset=["ts"]).sort_values("ts").reset_index(drop=True)
    keep = [c for c in ["ts", "label", "bull_score", "bear_score", "score_display"] if c in df.columns]
    out = df[keep].copy()
    out[f"label_{tf.lower()}"] = out["label"].astype(str)
    out[f"dir_{tf.lower()}"] = out[f"label_{tf.lower()}"].map(DIR_MAP).fillna(0).astype(int)
    out[f"score_{tf.lower()}"] = out.get("score_display", 0)
    return out[["ts", f"label_{tf.lower()}", f"dir_{tf.lower()}", f"score_{tf.lower()}"]]


def _align_to_m30(m30: pd.DataFrame, tfs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    out = m30.copy()
    for tf in TF_ORDER:
        if tf == "M30":
            continue
        src = tfs[tf]
        out = pd.merge_asof(out.sort_values("ts"), src.sort_values("ts"), on="ts", direction="backward")
    return out


def _size_mult(score: float, regime_case: str) -> float:
    if regime_case in {"WARMUP_UNDEFINED", "DATA_GAP_OR_INVALID", "CHAOS_CONFLICT_TF"}:
        return 0.0
    a = abs(score)
    if a >= 10:
        return 1.00
    if a >= 7:
        return 0.75
    if a >= 4:
        return 0.50
    return 0.25


def _assign_case(row: pd.Series, prev_case: str, neutral_thresh: float) -> str:
    d_mn = int(row.get("dir_mn1", 0))
    d_w = int(row.get("dir_w1", 0))
    d_d = int(row.get("dir_d1", 0))
    d_h4 = int(row.get("dir_h4", 0))
    d_h1 = int(row.get("dir_h1", 0))
    d_m30 = int(row.get("dir_m30", 0))

    if row.get("score_m30", 0) == 0 and row.get("label_m30", "NEUTRE") == "NEUTRE":
        return "WARMUP_UNDEFINED"
    if pd.isna(d_w) or pd.isna(d_d) or pd.isna(d_h1) or pd.isna(d_m30):
        return "DATA_GAP_OR_INVALID"

    # Weighted confluence score (MN1 intentionally small weight due to warmup constraints).
    score = 1 * d_mn + 3 * d_w + 3 * d_d + 2 * d_h4 + 2 * d_h1 + 2 * d_m30

    macro_bull = (d_w == 1 and d_d == 1)
    macro_bear = (d_w == -1 and d_d == -1)
    micro_bull = (d_h1 == 1 and d_m30 == 1)
    micro_bear = (d_h1 == -1 and d_m30 == -1)
    conflict = ((d_w * d_d) == -1 and d_w != 0 and d_d != 0) or ((d_d * d_h4) == -1 and d_d != 0 and d_h4 != 0)

    if conflict:
        return "CHAOS_CONFLICT_TF"
    if macro_bull and micro_bear:
        return "MACRO_BULL_MICRO_BEAR"
    if macro_bear and micro_bull:
        return "MACRO_BEAR_MICRO_BULL"

    if macro_bull and micro_bull and d_h4 >= 0:
        if prev_case == "RANGE_NEUTRAL":
            return "BREAKOUT_UP_FROM_RANGE"
        return "TREND_CONTINUATION_LONG"
    if macro_bear and micro_bear and d_h4 <= 0:
        if prev_case == "RANGE_NEUTRAL":
            return "BREAKOUT_DOWN_FROM_RANGE"
        return "TREND_CONTINUATION_SHORT"

    if macro_bull and (d_h1 < 0 or d_m30 < 0):
        return "PULLBACK_LONG_IN_BULL_TREND"
    if macro_bear and (d_h1 > 0 or d_m30 > 0):
        return "PULLBACK_SHORT_IN_BEAR_TREND"

    if abs(score) < neutral_thresh:
        if d_w > 0:
            return "RANGE_BIASED_LONG"
        if d_w < 0:
            return "RANGE_BIASED_SHORT"
        return "RANGE_NEUTRAL"

    if prev_case == "BREAKOUT_UP_FROM_RANGE" and d_m30 <= 0:
        return "FAKE_BREAKOUT_UP"
    if prev_case == "BREAKOUT_DOWN_FROM_RANGE" and d_m30 >= 0:
        return "FAKE_BREAKOUT_DOWN"

    if macro_bull and d_h1 < 0 and d_m30 < 0 and score > 0:
        return "EXHAUSTION_TOP"
    if macro_bear and d_h1 > 0 and d_m30 > 0 and score < 0:
        return "EXHAUSTION_BOTTOM"

    return "LOW_CONFIDENCE_NEUTRAL"


def main() -> int:
    args = parse_args()
    tf_dir = Path(args.tf_dir)
    out_csv = Path(args.out_csv)
    summary_csv = Path(args.summary_csv)

    print(f"[audit_regime_cases_v1] VERSION={VERSION}")
    print(f"[INFO] tf_dir={tf_dir}")

    tfs: Dict[str, pd.DataFrame] = {}
    for tf in TF_ORDER:
        tfs[tf] = _read_tf(tf_dir / TF_FILE[tf], tf=tf)

    m30 = tfs["M30"].copy()
    aligned = _align_to_m30(m30, tfs)

    # Add score and primary case
    w = {"mn1": 1, "w1": 3, "d1": 3, "h4": 2, "h1": 2, "m30": 2}
    aligned["confluence_score"] = (
        w["mn1"] * aligned["dir_mn1"]
        + w["w1"] * aligned["dir_w1"]
        + w["d1"] * aligned["dir_d1"]
        + w["h4"] * aligned["dir_h4"]
        + w["h1"] * aligned["dir_h1"]
        + w["m30"] * aligned["dir_m30"]
    )

    cases: List[str] = []
    prev = "RANGE_NEUTRAL"
    for _, r in aligned.iterrows():
        c = _assign_case(r, prev_case=prev, neutral_thresh=float(args.neutral_thresh))
        cases.append(c)
        prev = c
    aligned["regime_case"] = cases
    aligned["size_mult"] = [
        _size_mult(s, c) for s, c in zip(aligned["confluence_score"].to_numpy(dtype=float), aligned["regime_case"].tolist())
    ]

    # Human-readable direction stack
    aligned["dir_stack"] = (
        aligned["label_mn1"].astype(str)
        + "|"
        + aligned["label_w1"].astype(str)
        + "|"
        + aligned["label_d1"].astype(str)
        + "|"
        + aligned["label_h4"].astype(str)
        + "|"
        + aligned["label_h1"].astype(str)
        + "|"
        + aligned["label_m30"].astype(str)
    )

    summary = (
        aligned.groupby("regime_case", dropna=False)
        .agg(
            n=("regime_case", "size"),
            avg_score=("confluence_score", "mean"),
            avg_size_mult=("size_mult", "mean"),
        )
        .sort_values("n", ascending=False)
        .reset_index()
    )

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    aligned.to_csv(out_csv, index=False)
    summary.to_csv(summary_csv, index=False)

    print("\n=== REGIME CASE SUMMARY ===")
    print(summary.to_string(index=False))
    print(f"\n[OK] audit_csv={out_csv}")
    print(f"[OK] summary_csv={summary_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
