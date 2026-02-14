# C:\Users\lolo_\PycharmProjects\binance_futures_data_lake\scripts\research\inspect_router_columns_and_counts_v1.py
# inspect_router_columns_and_counts_v1.py
from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import List, Dict, Optional

import pandas as pd


PARQUET_DEFAULT = r"C:\Users\lolo_\PycharmProjects\binance_futures_data_lake\data\research_debug\BTCUSDT\long_20260101_20260210\joined_20260101_20260210__enriched__router.parquet"


def _pick_col(cols: List[str], patterns: List[str]) -> Optional[str]:
    """Return first col matching any regex pattern (case-insensitive)."""
    for pat in patterns:
        rx = re.compile(pat, re.IGNORECASE)
        for c in cols:
            if rx.search(c):
                return c
    return None


def _find_all(cols: List[str], patterns: List[str]) -> List[str]:
    out = []
    for pat in patterns:
        rx = re.compile(pat, re.IGNORECASE)
        out.extend([c for c in cols if rx.search(c)])
    # unique preserve order
    seen = set()
    uniq = []
    for c in out:
        if c not in seen:
            uniq.append(c)
            seen.add(c)
    return uniq


def main(parquet_path: str) -> int:
    p = Path(parquet_path)
    if not p.exists():
        print(f"[ERROR] parquet not found: {p}")
        return 2

    # Read only schema first (fast) by reading zero rows
    try:
        df0 = pd.read_parquet(p, engine="pyarrow")
        cols = list(df0.columns)
    except Exception as e:
        print(f"[ERROR] failed reading parquet: {e}")
        return 3

    print("=== PARQUET ===")
    print(str(p))
    print(f"n_cols={len(cols)}")
    print()

    # 1) Print key columns presence
    required_base = ["dir_state", "dir_score", "dir_ready", "vol_state", "range_pctl", "tradable_final",
                     "delta_norm", "close_pos", "range_rel", "atr14"]
    print("=== REQUIRED BASE COLUMNS (presence) ===")
    for c in required_base:
        print(f"{c:16s} -> {'OK' if c in cols else 'MISSING'}")
    print()

    # 2) Detect if router_state already exists
    router_state_col = _pick_col(cols, [r"^router_state$", r"router.*state", r"regime_router", r"h1.*router.*state"])
    if router_state_col:
        print(f"=== FOUND router_state-like column: {router_state_col} ===")
    else:
        print("=== router_state-like column NOT FOUND ===")
    print()

    # 3) Find candidate H1 columns to reconstruct router_state
    # We'll look for ER, ATR percentile, EMA20/50/200, and slope columns.
    candidates: Dict[str, Optional[str]] = {}

    candidates["er_h1"] = _pick_col(cols, [r"\ber\b.*h1", r"h1.*\ber\b", r"efficiency.*h1", r"h1.*efficiency"])
    candidates["atrpctl_h1"] = _pick_col(cols, [r"atr.*pctl.*h1", r"h1.*atr.*pctl", r"atrpctl.*h1", r"h1.*atrpctl"])
    candidates["ema20_h1"] = _pick_col(cols, [r"ema[_\-]?20.*h1", r"h1.*ema[_\-]?20"])
    candidates["ema50_h1"] = _pick_col(cols, [r"ema[_\-]?50.*h1", r"h1.*ema[_\-]?50"])
    candidates["ema200_h1"] = _pick_col(cols, [r"ema[_\-]?200.*h1", r"h1.*ema[_\-]?200"])
    candidates["slope50_h1"] = _pick_col(cols, [r"slope.*ema.*50.*h1", r"h1.*slope.*ema.*50", r"slope.*50.*h1", r"h1.*slope.*50"])

    print("=== H1 ROUTER CANDIDATES (auto-detected) ===")
    for k, v in candidates.items():
        print(f"{k:12s} -> {v if v else 'NOT FOUND'}")
    print()

    # 4) Show all "router/h1" related columns for manual inspection
    print("=== ALL COLUMNS matching 'router|h1|ema|slope|er|atrpctl|atr_pctl' (for inspection) ===")
    h1_related = _find_all(cols, [r"router", r"\bh1\b", r"ema", r"slope", r"\ber\b", r"atr.*pctl", r"atrpctl"])
    for c in h1_related:
        print(" -", c)
    print()

    # 5) Now read minimal dataframe needed for stats
    needed = set([c for c in required_base if c in cols])
    if router_state_col:
        needed.add(router_state_col)
    else:
        # if we can compute router_state, we need H1 candidates
        for v in candidates.values():
            if v:
                needed.add(v)

    needed = sorted(list(needed))
    print(f"=== READING {len(needed)} columns for stats ===")
    df = pd.read_parquet(p, columns=needed, engine="pyarrow")
    print(f"rows={len(df)}")
    print()

    # 6) Basic distributions
    print("=== VALUE COUNTS ===")
    if "dir_state" in df.columns:
        print("\ndir_state:")
        print(df["dir_state"].value_counts(dropna=False).head(20))
    if "vol_state" in df.columns:
        print("\nvol_state:")
        print(df["vol_state"].value_counts(dropna=False).head(20))
    if "tradable_final" in df.columns:
        print("\ntradable_final:")
        print(df["tradable_final"].value_counts(dropna=False).head(20))
    print()

    # 7) Router_state computation if missing and inputs found
    if not router_state_col:
        can_compute = all([
            candidates["er_h1"], candidates["atrpctl_h1"],
            candidates["ema20_h1"], candidates["ema50_h1"], candidates["ema200_h1"],
            candidates["slope50_h1"],
        ])
        if can_compute:
            er = df[candidates["er_h1"]]
            atrp = df[candidates["atrpctl_h1"]]
            ema20 = df[candidates["ema20_h1"]]
            ema50 = df[candidates["ema50_h1"]]
            ema200 = df[candidates["ema200_h1"]]
            slope50 = df[candidates["slope50_h1"]]

            # Thresholds from your prompt (starter)
            ER_HI = 0.45
            ATRP_HI = 85.0
            SLOPE_HI = 0.15

            bull_ok = (er >= ER_HI) & (ema20 > ema50) & (ema50 > ema200) & (slope50 >= SLOPE_HI) & (atrp <= ATRP_HI)
            bear_ok = (er >= ER_HI) & (ema20 < ema50) & (ema50 < ema200) & (slope50 <= -SLOPE_HI) & (atrp <= ATRP_HI)

            df["router_state__computed"] = "BLOCK"
            df.loc[bull_ok, "router_state__computed"] = "BULL_OK"
            df.loc[bear_ok, "router_state__computed"] = "BEAR_OK"
            router_state_col = "router_state__computed"

            print("=== COMPUTED router_state__computed (since none found) ===")
            print(df[router_state_col].value_counts(dropna=False))
            print()
        else:
            print("=== router_state could NOT be computed (missing one or more H1 inputs). ===")
            print("Missing inputs:")
            for k in ["er_h1", "atrpctl_h1", "ema20_h1", "ema50_h1", "ema200_h1", "slope50_h1"]:
                if not candidates[k]:
                    print(" -", k)
            print()

    # 8) Compute the 3 regimes counts with a strict v1 definition (as discussed)
    print("=== REGIME COUNTS (strict v1) ===")
    has = lambda c: c in df.columns

    # Guards for missing columns
    if not (has("tradable_final") and has("dir_state") and has("dir_ready") and has("vol_state") and router_state_col and has(router_state_col)):
        print("[WARN] Not enough columns to compute regime counts. Need: tradable_final, dir_state, dir_ready, vol_state, router_state.")
        print("Present:", [c for c in ["tradable_final", "dir_state", "dir_ready", "vol_state", router_state_col] if has(c)])
        return 0

    tradable = df["tradable_final"] == True
    dir_ready = df["dir_ready"] == 1 if df["dir_ready"].dtype != bool else df["dir_ready"]
    vol_ok_trend = df["vol_state"].isin(["MID", "HIGH"])
    vol_ok_range = df["vol_state"].isin(["LOW"])  # strict
    rs = df[router_state_col]

    trend_bull = tradable & dir_ready & vol_ok_trend & (df["dir_state"] == "BULL") & (rs == "BULL_OK")
    trend_bear = tradable & dir_ready & vol_ok_trend & (df["dir_state"] == "BEAR") & (rs == "BEAR_OK")
    range_reg = tradable & dir_ready & vol_ok_range & (df["dir_state"] == "NEUTRAL") & (rs == "BLOCK")

    # mutually exclusive counts
    n_trend_bull = int(trend_bull.sum())
    n_trend_bear = int(trend_bear.sum())
    n_range = int(range_reg.sum())
    n_any = int((trend_bull | trend_bear | range_reg).sum())
    n_total = len(df)

    print(f"trend_bull_bars = {n_trend_bull}")
    print(f"trend_bear_bars = {n_trend_bear}")
    print(f"range_bars      = {n_range}")
    print(f"any_regime_bars = {n_any} / {n_total}  ({(n_any / max(n_total,1))*100:.2f}%)")
    print()

    # 9) Entry trigger counts (impulse + MR setup) under each regime
    print("=== ENTRY TRIGGER COUNTS (starter thresholds) ===")
    if not (has("delta_norm") and has("close_pos") and has("range_rel")):
        print("[WARN] Missing one of: delta_norm, close_pos, range_rel. Can't compute entry triggers.")
        return 0

    D = 0.20
    P = 0.65
    Rr = 0.80

    impulse_long = (df["delta_norm"] > D) & (df["close_pos"] > P) & (df["range_rel"] > Rr)
    impulse_short = (df["delta_norm"] < -D) & (df["close_pos"] < (1.0 - P)) & (df["range_rel"] > Rr)

    mr_long_setup = (df["close_pos"] < 0.25) & (df["delta_norm"] < -D) & (df["range_rel"] > Rr)
    mr_short_setup = (df["close_pos"] > 0.75) & (df["delta_norm"] > D) & (df["range_rel"] > Rr)

    print(f"Trend LONG triggers (bars)  = {int((trend_bull & impulse_long).sum())}")
    print(f"Trend SHORT triggers (bars) = {int((trend_bear & impulse_short).sum())}")
    print(f"Range MR LONG setups (bars) = {int((range_reg & mr_long_setup).sum())}")
    print(f"Range MR SHORT setups (bars)= {int((range_reg & mr_short_setup).sum())}")
    print()

    # 10) Recommend next action based on counts
    print("=== NEXT ACTION RECOMMENDATION ===")
    if n_any == 0:
        print("No regime bars found with strict v1 router. Relax conditions (ER threshold / slope / ATRpctl) or inspect router columns.")
    else:
        print("If trigger counts are > 0, proceed to event-driven backtest.")
        print("If trigger counts are too low, relax thresholds: D/P/Rr or allow vol_state='NA' after verifying why NA exists.")
    return 0


if __name__ == "__main__":
    parquet = sys.argv[1] if len(sys.argv) > 1 else PARQUET_DEFAULT
    raise SystemExit(main(parquet))
