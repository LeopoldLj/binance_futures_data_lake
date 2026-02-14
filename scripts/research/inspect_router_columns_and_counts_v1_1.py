# C:\Users\lolo_\PycharmProjects\binance_futures_data_lake\scripts\research\inspect_router_columns_and_counts_v1_1.py
# inspect_router_columns_and_counts_v1_1.py
from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import List, Dict, Optional

import pandas as pd


PARQUET_DEFAULT = r"C:\Users\lolo_\PycharmProjects\binance_futures_data_lake\data\research_debug\BTCUSDT\long_20260101_20260210\joined_20260101_20260210__enriched__router.parquet"


def _pick_col(cols: List[str], patterns: List[str]) -> Optional[str]:
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

    df0 = pd.read_parquet(p, engine="pyarrow")
    cols = list(df0.columns)

    print("=== PARQUET ===")
    print(str(p))
    print(f"n_cols={len(cols)}")
    print()

    required_base = ["dir_state", "dir_score", "dir_ready", "vol_state", "range_pctl", "tradable_final",
                     "delta_norm", "close_pos", "range_rel", "atr14", "router_mode_h1"]
    print("=== REQUIRED BASE COLUMNS (presence) ===")
    for c in required_base:
        print(f"{c:16s} -> {'OK' if c in cols else 'MISSING'}")
    print()

    # Router "state" detection: prefer router_mode_h1 if present
    router_state_col = None
    if "router_mode_h1" in cols:
        router_state_col = "router_mode_h1"
    else:
        router_state_col = _pick_col(cols, [r"^router_state$", r"router.*state", r"regime_router", r"h1.*router.*state"])

    if router_state_col:
        print(f"=== ROUTER COLUMN USED: {router_state_col} ===")
    else:
        print("=== router_state-like column NOT FOUND ===")
    print()

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

    print("=== ALL COLUMNS matching 'router|h1|ema|slope|er|atrpctl|atr_pctl' (for inspection) ===")
    h1_related = _find_all(cols, [r"router", r"\bh1\b", r"ema", r"slope", r"\ber\b", r"atr.*pctl", r"atrpctl"])
    for c in h1_related:
        print(" -", c)
    print()

    needed = set([c for c in required_base if c in cols])
    if router_state_col:
        needed.add(router_state_col)
    for v in candidates.values():
        if v:
            needed.add(v)
    needed = sorted(list(needed))

    print(f"=== READING {len(needed)} columns for stats ===")
    df = pd.read_parquet(p, columns=needed, engine="pyarrow")
    print(f"rows={len(df)}")
    print()

    print("=== VALUE COUNTS ===")
    print("\ndir_state:")
    print(df["dir_state"].value_counts(dropna=False).head(20))
    print("\nvol_state:")
    print(df["vol_state"].value_counts(dropna=False).head(20))
    print("\ntradable_final:")
    print(df["tradable_final"].value_counts(dropna=False).head(20))
    if router_state_col:
        print(f"\n{router_state_col}:")
        print(df[router_state_col].value_counts(dropna=False).head(50))
    print()

    if not router_state_col:
        print("[WARN] No router column available. Can't compute regimes.")
        return 0

    # --- Regime counts (we'll adapt mapping after we see router_mode_h1 values) ---
    print("=== REGIME COUNTS (proto) ===")

    tradable = df["tradable_final"] == True
    dir_ready = df["dir_ready"] == 1 if df["dir_ready"].dtype != bool else df["dir_ready"]
    rs = df[router_state_col].astype("string")

    # Heuristic mapping: detect bull/bear/block tokens
    rs_upper = rs.str.upper()
    is_bull = rs_upper.str.contains("BULL") | rs_upper.str.contains("UP")
    is_bear = rs_upper.str.contains("BEAR") | rs_upper.str.contains("DOWN")
    is_block = rs_upper.str.contains("BLOCK") | rs_upper.str.contains("RANGE") | rs_upper.str.contains("CHOP") | rs_upper.str.contains("NEUTRAL")

    vol_ok_trend = df["vol_state"].isin(["MID", "HIGH"])
    vol_ok_range = df["vol_state"].isin(["LOW"])

    trend_bull = tradable & dir_ready & vol_ok_trend & (df["dir_state"] == "BULL") & is_bull
    trend_bear = tradable & dir_ready & vol_ok_trend & (df["dir_state"] == "BEAR") & is_bear
    range_reg = tradable & dir_ready & vol_ok_range & (df["dir_state"] == "NEUTRAL") & is_block

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

    print("=== ENTRY TRIGGER COUNTS (starter thresholds) ===")
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

    print("=== NOTE ===")
    print("The mapping from router_mode_h1 -> bull/bear/block is heuristic (token-based).")
    print("After you paste router_mode_h1 unique values, we will lock an exact mapping.")
    return 0


if __name__ == "__main__":
    parquet = sys.argv[1] if len(sys.argv) > 1 else PARQUET_DEFAULT
    raise SystemExit(main(parquet))
