import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Optional, List, Dict

import numpy as np
import pandas as pd


VERSION = "2026-02-13c"


@dataclass(frozen=True)
class LowPlusPolicy:
    allow_hours_utc: Optional[Tuple[int, ...]] = None
    block_hours_utc: Tuple[int, ...] = ()
    bull_only: bool = False
    bear_only: bool = False
    forbid_neutral: bool = True
    # Legacy band filter (kept, but default None for ADD analysis)
    dir_score_min: Optional[float] = None
    dir_score_max: Optional[float] = None
    # Robust filter (recommended)
    dir_score_abs_min: Optional[float] = 0.30
    dir_score_abs_max: Optional[float] = None
    allow_range_pctl: Tuple[Tuple[float, float], ...] = ()
    block_range_pctl: Tuple[Tuple[float, float], ...] = ()


def _require_cols(df: pd.DataFrame, cols: List[str], ctx: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"[{ctx}] missing required cols: {missing}. cols={list(df.columns)}")


def _detect_first(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _to_bool(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s
    if np.issubdtype(s.dtype, np.number):
        return s.fillna(0).astype(float).ne(0.0)
    return s.astype(str).str.lower().isin(["1", "true", "t", "yes", "y"])


def _bucket_range_pctl(x: float) -> str:
    if pd.isna(x):
        return "NA"
    b = np.floor(float(x) / 0.02) * 0.02
    lo = round(b, 2)
    hi = round(b + 0.02, 2)
    return f"[{lo:.2f}, {hi:.2f})"


def _synth_signed_returns(df: pd.DataFrame, price_col: str, horizons: List[int]) -> List[str]:
    cols = []
    px = df[price_col].astype(float)
    for h in horizons:
        col = f"sret_{h}"
        df[col] = (px.shift(-h) / px - 1.0).fillna(0.0)
        cols.append(col)
    return cols


def _apply_policy_filters(df: pd.DataFrame, policy: LowPlusPolicy) -> Tuple[pd.Series, pd.DataFrame]:
    _require_cols(df, ["hour_utc", "dir_state", "dir_score", "range_pctl"], "policy_filters")

    reasons: Dict[str, int] = {}
    allow = pd.Series(True, index=df.index)

    # hour allow/block
    if policy.allow_hours_utc is not None:
        m = df["hour_utc"].isin(list(policy.allow_hours_utc))
        allow &= m
        reasons["hour_not_allowed"] = int((~m).sum())

    if policy.block_hours_utc:
        m = ~df["hour_utc"].isin(list(policy.block_hours_utc))
        allow &= m
        reasons["hour_block"] = int((~m).sum())

    # dir_state constraints
    ds = df["dir_state"].astype(str).str.upper()

    if policy.forbid_neutral:
        m = ds.ne("NEUTRAL")
        allow &= m
        reasons["neutral_block"] = int((~m).sum())

    if policy.bull_only and policy.bear_only:
        raise ValueError("Invalid policy: bull_only and bear_only cannot both be True.")

    if policy.bull_only:
        m = ds.eq("BULL")
        allow &= m
        reasons["not_bull_block"] = int((~m).sum())

    if policy.bear_only:
        m = ds.eq("BEAR")
        allow &= m
        reasons["not_bear_block"] = int((~m).sum())

    # dir_score filters
    dsc = df["dir_score"].astype(float)

    # robust abs filter (recommended)
    if policy.dir_score_abs_min is not None:
        m = dsc.abs().ge(policy.dir_score_abs_min)
        allow &= m
        reasons["dir_score_abs_min_block"] = int((~m).sum())

    if policy.dir_score_abs_max is not None:
        m = dsc.abs().le(policy.dir_score_abs_max)
        allow &= m
        reasons["dir_score_abs_max_block"] = int((~m).sum())

    # legacy band filter (optional)
    if policy.dir_score_min is not None:
        m = dsc.ge(policy.dir_score_min)
        allow &= m
        reasons["dir_score_min_block"] = int((~m).sum())

    if policy.dir_score_max is not None:
        m = dsc.le(policy.dir_score_max)
        allow &= m
        reasons["dir_score_max_block"] = int((~m).sum())

    # range_pctl allow/block windows
    rp = df["range_pctl"].astype(float)

    if policy.allow_range_pctl:
        m_any = pd.Series(False, index=df.index)
        for lo, hi in policy.allow_range_pctl:
            m_any |= (rp.ge(lo) & rp.lt(hi))
        allow &= m_any
        reasons["range_pctl_not_allowed"] = int((~m_any).sum())

    if policy.block_range_pctl:
        m_block = pd.Series(False, index=df.index)
        for lo, hi in policy.block_range_pctl:
            m_block |= (rp.ge(lo) & rp.lt(hi))
        allow &= ~m_block
        reasons["range_pctl_blocked"] = int(m_block.sum())

    dbg = pd.DataFrame({"block_reason": list(reasons.keys()), "n": list(reasons.values())}).sort_values("n", ascending=False)
    return allow, dbg


def run(joined_path: str, outdir: str, policy: LowPlusPolicy, ret_cols: Optional[List[str]], horizons: List[int], price_col: str, add_col: str) -> None:
    joined_path = str(joined_path)
    outdir = str(outdir)

    df = pd.read_parquet(joined_path)

    print("\nSANITY\n------")
    print("rows:", len(df))

    time_col = _detect_first(df, ["t", "ts", "time", "datetime"])
    if time_col is None:
        raise ValueError(f"Cannot detect time column. cols={list(df.columns)}")

    low_plus_col = _detect_first(df, ["low_plus", "lowplus", "low_plus"])
    if low_plus_col is None:
        raise ValueError(f"Cannot detect low_plus column. cols={list(df.columns)}")

    # derived time columns
    if "hour_utc" not in df.columns or "dow_utc" not in df.columns or "date_utc" not in df.columns:
        dt = pd.to_datetime(df[time_col], utc=True)
        if "hour_utc" not in df.columns:
            df["hour_utc"] = dt.dt.hour.astype(int)
        if "dow_utc" not in df.columns:
            df["dow_utc"] = dt.dt.dayofweek.astype(int)
        if "date_utc" not in df.columns:
            df["date_utc"] = dt.dt.date.astype(str)

    if "rp_bucket" not in df.columns:
        df["rp_bucket"] = df["range_pctl"].apply(_bucket_range_pctl)

    # returns
    detected_ret_cols: List[str] = []
    if ret_cols:
        detected_ret_cols = [c for c in ret_cols if c in df.columns]
    else:
        detected_ret_cols = [c for c in ["sret_1", "sret_2", "sret_4", "sret_8"] if c in df.columns]

    if not detected_ret_cols:
        if price_col not in df.columns:
            raise ValueError(f"No returns cols found and price_col '{price_col}' missing.")
        detected_ret_cols = _synth_signed_returns(df, price_col=price_col, horizons=horizons)
        print(f"[WARN] No return columns found in joined; synthesized signed returns from '{price_col}': {detected_ret_cols}")

    # ADD selection
    add_mode = add_col.strip().lower()
    add_series_name: Optional[str] = None

    if add_mode == "auto":
        add_series_name = _detect_first(df, ["is_add", "add", "low_plus_add"])
        if add_series_name is None:
            if "tradable_override" in df.columns and "tradable_base" in df.columns:
                df["__is_add_derived__"] = _to_bool(df["tradable_override"]) & ~_to_bool(df["tradable_base"])
                add_series_name = "__is_add_derived__"
            else:
                df["__is_add_legacy__"] = _to_bool(df[low_plus_col])
                add_series_name = "__is_add_legacy__"
    else:
        if add_col not in df.columns:
            raise ValueError(f"--add-col '{add_col}' not found in joined. cols={list(df.columns)}")
        add_series_name = add_col

    if add_series_name is None:
        raise ValueError("Cannot determine ADD column (auto failed).")

    print("\nDETECTED COLUMNS\n----------------")
    print(f"low_plus_col: {low_plus_col}")
    print(f"time_col: {time_col}")
    print("return_cols:", detected_ret_cols)
    print(f"add_col_used: {add_series_name} (arg={add_col})")

    allow_mask, dbg_blocks = _apply_policy_filters(df, policy)

    # ADD population = is_add AND policy allows
    add_mask = _to_bool(df[add_series_name]) & allow_mask
    add_df = df.loc[add_mask].copy()

    print("\nADD SUMMARY\n-----------")
    n_add = int(len(add_df))
    print("n_add:", n_add)

    for c in detected_ret_cols:
        if n_add == 0:
            print(f"ADD {c}: mean=nan p05=nan win=nan n=0")
        else:
            x = add_df[c].astype(float)
            win = float((x > 0).mean())
            mean = float(x.mean())
            p05 = float(np.quantile(x, 0.05))
            print(f"ADD {c}: mean={mean:.6g} p05={p05:.6g} win={win:.3f} n={len(x)}")

    outdir_p = Path(outdir)
    outdir_p.mkdir(parents=True, exist_ok=True)

    out_add_parq = outdir_p / "__LOWP_POLICY__ADD.parquet"
    out_add_csv = outdir_p / "__LOWP_POLICY__ADD.csv"
    out_by_hour = outdir_p / "__LOWP_POLICY__ADD_by_hour.csv"
    out_by_side = outdir_p / "__LOWP_POLICY__ADD_by_side.csv"
    out_by_dirscore = outdir_p / "__LOWP_POLICY__ADD_by_dirscore.csv"
    out_by_rp = outdir_p / "__LOWP_POLICY__ADD_by_rangepctl_bucket.csv"
    out_blocks = outdir_p / "__LOWP_POLICY__LOWPLUS_block_reasons.csv"

    keep_cols = [time_col, "dir_state", "dir_score", "range_pctl", "range", "n_trades", "hour_utc", "dow_utc", "date_utc", "rp_bucket", low_plus_col, add_series_name] + detected_ret_cols
    keep_cols = [c for c in keep_cols if c in add_df.columns]

    add_df_out = add_df[keep_cols].copy()
    add_df_out.to_parquet(out_add_parq, index=False)
    add_df_out.to_csv(out_add_csv, index=False)

    if n_add > 0:
        add_df_out.groupby("hour_utc").size().reset_index(name="n").sort_values("n", ascending=False).to_csv(out_by_hour, index=False)
        add_df_out.groupby("dir_state").size().reset_index(name="n").sort_values("n", ascending=False).to_csv(out_by_side, index=False)
        if "dir_score" in add_df_out.columns:
            q = min(6, n_add)
            bins = pd.qcut(add_df_out["dir_score"].astype(float), q=q, duplicates="drop")
            add_df_out.assign(dir_score_bin=bins.astype(str)).groupby("dir_score_bin").size().reset_index(name="n").sort_values("n", ascending=False).to_csv(out_by_dirscore, index=False)
        add_df_out.groupby("rp_bucket").size().reset_index(name="n").sort_values("n", ascending=False).to_csv(out_by_rp, index=False)

    dbg_blocks.to_csv(out_blocks, index=False)

    print("\nWROTE\n-----")
    print(str(out_add_parq))
    print(str(out_add_csv))
    print(str(out_by_hour))
    print(str(out_by_side))
    print(str(out_by_dirscore))
    print(str(out_by_rp))
    print(str(out_blocks))


def main() -> None:
    ap = argparse.ArgumentParser(description="Analyze LOW+ policy on a joined parquet and report ADD diagnostics.")
    ap.add_argument("--joined", required=True, help="Path to joined/enriched parquet.")
    ap.add_argument("--outdir", required=True, help="Output folder for reports (csv/parquet).")
    ap.add_argument("--add-col", default="auto", help="ADD column to use (auto | is_add | any bool col). Default: auto.")
    ap.add_argument("--price-col", default="close", help="Price col used to synthesize returns if missing. Default: close.")
    ap.add_argument("--ret-cols", nargs="*", default=None, help="Explicit return columns (e.g. sret_4 sret_8). Optional.")
    args = ap.parse_args()

    # Default policy tuned to NOT kill your current is_add population:
    # - no bull_only (keep both sides)
    # - abs(dir_score) >= 0.30
    # - allow range_pctl down to 0.00 (your ADD includes 0.01..0.08)
    # - keep your time blocks + avoid your two bad rp micro-buckets if you want
    policy = LowPlusPolicy(
        allow_hours_utc=None,
        block_hours_utc=(1, 14, 17, 19, 23),
        bull_only=False,
        bear_only=False,
        forbid_neutral=True,
        dir_score_min=None,
        dir_score_max=None,
        dir_score_abs_min=0.30,
        dir_score_abs_max=None,
        allow_range_pctl=((0.00, 0.30),),
        block_range_pctl=((0.12, 0.14), (0.20, 0.22)),
    )

    horizons = [4, 8]

    print(f"[analyze_low_plus_policy] VERSION={VERSION}")
    print(f"[INFO] joined={args.joined}")
    print(f"[INFO] outdir={args.outdir}")
    print(f"[INFO] add_col={args.add_col}")
    print(f"[INFO] policy={policy}")
    print(f"[INFO] horizons={horizons} (used only if returns are missing)")
    print(f"[INFO] price_col={args.price_col}")

    run(args.joined, args.outdir, policy, args.ret_cols, horizons, args.price_col, args.add_col)


if __name__ == "__main__":
    main()
