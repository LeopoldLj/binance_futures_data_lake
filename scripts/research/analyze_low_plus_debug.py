# scripts/research/analyze_low_plus_debug.py
# VERSION=2026-02-13a
#
# Analyze router_low_plus_sweep debug parquet (ADD set diagnostics).
# Fix: accept files that do NOT contain 'tradable_base' but contain 'tradable_keep_ready'.

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import pandas as pd


VERSION = "2026-02-13a"


@dataclass(frozen=True)
class RequiredCols:
    # NOTE: tradable_base can be missing in debug outputs; we can rebuild from tradable_keep_ready.
    base: List[str] = None

    def __post_init__(self):
        object.__setattr__(
            self,
            "base",
            [
                "t",
                "dir_state",
                "dir_score",
                "range",
                "n_trades",
                "dir_state_age",
                "vol_state",
                "range_pctl",
                "market_ready",
                "low_plus",
                "tradable_keep_ready",
                "tradable_override",
                "is_add",
                "hour_utc",
                "dow_utc",
                "date_utc",
            ],
        )


REQ = RequiredCols()


def _ensure_outdir(outdir: str) -> Path:
    p = Path(outdir)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _th_to_tag(th: float) -> str:
    # 0.14 -> "0p14"
    s = f"{th:.4f}".rstrip("0").rstrip(".")
    return s.replace(".", "p")


def _safe_mean_bool(s: pd.Series) -> float:
    if s.empty:
        return float("nan")
    # bool -> mean works, but keep safe with numeric conversion
    return float(pd.to_numeric(s, errors="coerce").mean())


def _coerce_bool_col(df: pd.DataFrame, col: str) -> None:
    if col in df.columns:
        if df[col].dtype != bool:
            df[col] = df[col].astype(bool)


def _validate_and_normalize(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in REQ.base if c not in df.columns]
    if missing:
        raise ValueError(f"Joined file missing required cols: {missing}. cols={list(df.columns)}")

    # Optional columns
    # - tradable_base: if missing, rebuild from tradable_keep_ready
    if "tradable_base" not in df.columns:
        # Fallback that matches intended meaning: base tradable before override policy
        df["tradable_base"] = df["tradable_keep_ready"]

    # Normalize dtypes
    _coerce_bool_col(df, "market_ready")
    _coerce_bool_col(df, "low_plus")
    _coerce_bool_col(df, "tradable_keep_ready")
    _coerce_bool_col(df, "tradable_override")
    _coerce_bool_col(df, "tradable_base")
    _coerce_bool_col(df, "is_add")

    # Ensure timestamp is parsed
    if not pd.api.types.is_datetime64_any_dtype(df["t"]):
        df["t"] = pd.to_datetime(df["t"], utc=True, errors="coerce")

    return df


def _write_df(df: pd.DataFrame, outdir: Path, name: str) -> None:
    out_csv = outdir / f"{name}.csv"
    out_parq = outdir / f"{name}.parquet"
    df.to_csv(out_csv, index=False)
    df.to_parquet(out_parq, index=False)


def _value_counts_2d(df: pd.DataFrame, col1: str, col2: str) -> pd.DataFrame:
    vc = df[[col1, col2]].value_counts(dropna=False).reset_index()
    vc.columns = [col1, col2, "count"]
    vc = vc.sort_values("count", ascending=False).reset_index(drop=True)
    return vc


def build_report(joined_path: str, th: float, outdir: str) -> None:
    outdir_p = _ensure_outdir(outdir)
    p = Path(joined_path)
    stem = p.stem
    th_tag = _th_to_tag(th)

    j = pd.read_parquet(joined_path)
    j = _validate_and_normalize(j)

    # Work on ADD rows only (file is usually already ADD-only, but keep robust)
    add = j[j["is_add"] == True].copy()

    # Core summary
    summary = pd.DataFrame(
        [
            {
                "file": str(p),
                "rows_total": int(len(j)),
                "rows_add": int(len(add)),
                "th": float(th),
                "market_ready_rate": _safe_mean_bool(add["market_ready"]) if "market_ready" in add.columns else float("nan"),
                "low_plus_rate": _safe_mean_bool(add["low_plus"]) if "low_plus" in add.columns else float("nan"),
                "tradable_base_rate": _safe_mean_bool(add["tradable_base"]) if "tradable_base" in add.columns else float("nan"),
                "tradable_keep_ready_rate": _safe_mean_bool(add["tradable_keep_ready"]) if "tradable_keep_ready" in add.columns else float("nan"),
                "tradable_override_rate": _safe_mean_bool(add["tradable_override"]) if "tradable_override" in add.columns else float("nan"),
            }
        ]
    )

    # Breakdown tables
    by_side = add["dir_state"].value_counts(dropna=False).reset_index()
    by_side.columns = ["dir_state", "count"]
    by_side = by_side.sort_values("count", ascending=False).reset_index(drop=True)

    by_vol = add["vol_state"].value_counts(dropna=False).reset_index()
    by_vol.columns = ["vol_state", "count"]
    by_vol = by_vol.sort_values("count", ascending=False).reset_index(drop=True)

    by_side_vol = _value_counts_2d(add, "dir_state", "vol_state")

    by_hour = add["hour_utc"].value_counts(dropna=False).reset_index()
    by_hour.columns = ["hour_utc", "count"]
    by_hour = by_hour.sort_values("hour_utc").reset_index(drop=True)

    # If bins exist, keep them; else create coarse bins
    if "dir_score_bin" not in add.columns:
        add["dir_score_bin"] = pd.cut(add["dir_score"], bins=10)

    by_dirscore = add["dir_score_bin"].value_counts(dropna=False).reset_index()
    by_dirscore.columns = ["dir_score_bin", "count"]
    by_dirscore = by_dirscore.sort_values("count", ascending=False).reset_index(drop=True)

    if "rp_bucket" not in add.columns:
        add["rp_bucket"] = pd.cut(add["range_pctl"], bins=10)

    by_rangepctl_bucket = add["rp_bucket"].value_counts(dropna=False).reset_index()
    by_rangepctl_bucket.columns = ["rp_bucket", "count"]
    by_rangepctl_bucket = by_rangepctl_bucket.sort_values("count", ascending=False).reset_index(drop=True)

    # “Override reasons” style quick diagnostics: what is blocking market_ready?
    # In your samples, market_ready is always False. This helps quantify why.
    # We build a simple reasons table based on boolean gates that usually exist.
    reasons_rows = []
    if not add.empty:
        # Expected gate logic (generic):
        # - must be low_plus True
        # - must be market_ready True
        # - base tradable True
        # If market_ready always False, we still report rates of each gate.
        reasons_rows.append({"gate": "low_plus_true_rate", "rate": _safe_mean_bool(add["low_plus"])})
        reasons_rows.append({"gate": "market_ready_true_rate", "rate": _safe_mean_bool(add["market_ready"])})
        reasons_rows.append({"gate": "tradable_base_true_rate", "rate": _safe_mean_bool(add["tradable_base"])})
        reasons_rows.append({"gate": "tradable_override_true_rate", "rate": _safe_mean_bool(add["tradable_override"])})
    block_reasons = pd.DataFrame(reasons_rows)

    # Write outputs (keep naming close to your existing report patterns)
    prefix = f"{stem}__th{th_tag}__DEBUG"
    _write_df(summary, outdir_p, f"{prefix}")
    _write_df(by_side, outdir_p, f"{prefix}_by_side")
    _write_df(by_vol, outdir_p, f"{prefix}_by_vol")
    _write_df(by_side_vol, outdir_p, f"{prefix}_by_side_vol")
    _write_df(by_hour, outdir_p, f"{prefix}_by_hour")
    _write_df(by_dirscore, outdir_p, f"{prefix}_by_dirscore")
    _write_df(by_rangepctl_bucket, outdir_p, f"{prefix}_by_rangepctl_bucket")
    _write_df(block_reasons, outdir_p, f"{prefix}_block_reasons")

    print(f"[OK] wrote reports in: {outdir_p}")


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Analyze router_low_plus_sweep debug joined parquet (ADD set diagnostics).")
    p.add_argument("--joined", required=True, help="Path to joined debug parquet exported by router_low_plus_sweep.py")
    p.add_argument("--th", type=float, default=0.14, help="LOW+ threshold on range_pctl (default: 0.14)")
    p.add_argument("--outdir", required=True, help="Output folder for reports (csv/parquet)")
    return p.parse_args(argv)


def main() -> None:
    args = parse_args()
    print(f"[analyze_low_plus_debug] VERSION={VERSION}")
    print(f"[INFO] joined={args.joined}")
    print(f"[INFO] th={args.th}")
    print(f"[INFO] outdir={args.outdir}")
    if not os.path.exists(args.joined):
        raise FileNotFoundError(args.joined)
    build_report(joined_path=args.joined, th=args.th, outdir=args.outdir)


if __name__ == "__main__":
    main()
