from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Any, List, Set

import numpy as np
import pandas as pd

# ======================================================================================
# BF Data Lake — Research — trade_diagnostics_mae_mfe_short.py
#
# But :
# - Sur une config SHORT (H, SL_K, TP_R) et une policy fixée, simuler les trades (entrée open+1)
# - Diagnostiquer :
#     * R final, reason (SL/TP/TIME)
#     * MAE_R : pire excursion adverse en R pendant la vie du trade (short => adverse = high - entry)
#     * MFE_R : meilleure excursion favorable en R (short => favorable = entry - low)
#     * Feasibility : combien de trades auraient touché TP si on avait laissé courir H, etc.
#
# Sorties :
# - Table globale + par reason + quantiles MAE/MFE
# - Optionnel : export CSV trades détaillés
# ======================================================================================


def _parse_int_list(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Diagnostics MAE/MFE for SHORT ATR-stop TP-R on policy-filtered ADD signals.")
    p.add_argument("--parquet", required=True, help="Path to joined_*__enriched.parquet")
    p.add_argument("--atr-len", type=int, default=14, help="ATR rolling length (default: 14)")

    # Trade config
    p.add_argument("--H", type=int, default=16, help="Horizon bars (default: 16)")
    p.add_argument("--sl-k", type=float, default=1.0, help="SL = entry + sl_k * ATR (default: 1.0)")
    p.add_argument("--tp-r", type=float, default=2.0, help="TP = entry - tp_r * risk (default: 2.0)")
    p.add_argument("--conservative-both-touch", action="store_true", default=True, help="If SL & TP touched same bar: assume SL first (default: True)")

    # Policy
    p.add_argument("--bear-only", action="store_true", help="If set: keep dir_state == BEAR only")
    p.add_argument("--forbid-neutral", action="store_true", default=True, help="If set: forbid dir_state == NEUTRAL (default: True)")
    p.add_argument("--dir-score-abs-min", type=float, default=0.3, help="Minimum abs(dir_score) (default: 0.3)")
    p.add_argument("--allow-rp-max", type=float, default=0.10, help="Allow range_pctl in [0, allow_rp_max) (default: 0.10)")
    p.add_argument("--block-hours", default="0,1,14,17,19,23", help="Comma list of UTC hours to block (default: 0,1,14,17,19,23)")

    # Output
    p.add_argument("--out-trades-csv", default=None, help="Optional path to write per-trade detailed CSV")
    return p.parse_args()


def _require_cols(df: pd.DataFrame, cols: List[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing required columns: {missing}. Available={list(df.columns)[:80]}...")


def _compute_atr(df: pd.DataFrame, atr_len: int) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr1 = (df["high"] - df["low"]).abs()
    tr2 = (df["high"] - prev_close).abs()
    tr3 = (df["low"] - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(atr_len, min_periods=atr_len).mean()


def _build_policy_mask(
    df: pd.DataFrame,
    block_hours_utc: Set[int],
    forbid_neutral: bool,
    bear_only: bool,
    dir_score_abs_min: float,
    allow_rp_max: float,
) -> pd.Series:
    ds = df["dir_state"].astype(str)
    pass_hours = ~df["hour_utc"].isin(block_hours_utc)

    pass_side = pd.Series(True, index=df.index)
    if forbid_neutral:
        pass_side = pass_side & (ds != "NEUTRAL")
    if bear_only:
        pass_side = pass_side & (ds == "BEAR")

    score = pd.to_numeric(df["dir_score"], errors="coerce")
    pass_score = score.abs() >= dir_score_abs_min

    rp = pd.to_numeric(df["range_pctl"], errors="coerce")
    pass_rp = (rp >= 0.0) & (rp < allow_rp_max)

    return pass_hours & pass_side & pass_score & pass_rp


def _perf(x: pd.Series) -> Dict[str, Any]:
    x = x.dropna()
    if len(x) == 0:
        return {"n": 0, "wr": np.nan, "mean": np.nan, "median": np.nan, "p05": np.nan, "p95": np.nan, "min": np.nan, "max": np.nan}
    return {
        "n": int(len(x)),
        "wr": float((x > 0).mean()),
        "mean": float(x.mean()),
        "median": float(x.median()),
        "p05": float(x.quantile(0.05)),
        "p95": float(x.quantile(0.95)),
        "min": float(x.min()),
        "max": float(x.max()),
    }


def main() -> int:
    args = _parse_args()

    parquet = Path(args.parquet)
    if not parquet.exists():
        raise FileNotFoundError(f"Parquet not found: {parquet}")

    BLOCK_HOURS_UTC = set(_parse_int_list(args.block_hours))

    df = pd.read_parquet(parquet)
    _require_cols(df, ["t", "dir_state", "dir_score", "range_pctl", "is_add", "open", "high", "low", "close"])

    tt = pd.to_datetime(df["t"], utc=True, errors="coerce")
    df["_t"] = tt
    df["hour_utc"] = tt.dt.hour
    df = df.sort_values("_t").reset_index(drop=True)

    df["atr"] = _compute_atr(df, int(args.atr_len))

    pass_lowp = _build_policy_mask(
        df=df,
        block_hours_utc=BLOCK_HOURS_UTC,
        forbid_neutral=bool(args.forbid_neutral),
        bear_only=bool(args.bear_only),
        dir_score_abs_min=float(args.dir_score_abs_min),
        allow_rp_max=float(args.allow_rp_max),
    )
    df["pass_lowp"] = pass_lowp

    sig = (df["is_add"] == True) & (df["pass_lowp"] == True)
    sig_idx_all = np.flatnonzero(sig.values)
    H = int(args.H)
    sig_idx = [i for i in sig_idx_all.tolist() if (i + 1) < len(df) and (i + H) < len(df)]

    trades: List[Dict[str, Any]] = []

    for i in sig_idx:
        entry_i = i + 1
        entry = float(df.loc[entry_i, "open"])
        atr_i = df.loc[entry_i, "atr"]
        if pd.isna(atr_i) or float(atr_i) <= 0:
            continue

        sl = entry + float(args.sl_k) * float(atr_i)
        risk = sl - entry
        tp = entry - float(args.tp_r) * risk

        # Track MAE/MFE during life
        mae_r = -np.inf
        mfe_r = -np.inf

        exit_reason = None
        exit_px = None
        exit_i = None

        for j in range(entry_i, entry_i + H):
            hi = float(df.loc[j, "high"])
            lo = float(df.loc[j, "low"])

            # For SHORT:
            # - adverse excursion (against us): high - entry  (>=0)
            # - favorable excursion: entry - low (>=0)
            adverse = hi - entry
            favorable = entry - lo

            mae_r = max(mae_r, adverse / risk) if risk > 0 else np.nan
            mfe_r = max(mfe_r, favorable / risk) if risk > 0 else np.nan

            hit_sl = hi >= sl
            hit_tp = lo <= tp

            if hit_sl and hit_tp:
                if bool(args.conservative_both_touch):
                    exit_reason = "SL_both"
                    exit_px = sl
                else:
                    exit_reason = "TP_both"
                    exit_px = tp
                exit_i = j
                break
            if hit_sl:
                exit_reason = "SL"
                exit_px = sl
                exit_i = j
                break
            if hit_tp:
                exit_reason = "TP"
                exit_px = tp
                exit_i = j
                break

        if exit_reason is None:
            exit_reason = "TIME"
            exit_i = entry_i + H - 1
            exit_px = float(df.loc[exit_i, "close"])

        pnl = entry - float(exit_px)
        R = pnl / risk if risk > 0 else np.nan

        trades.append(
            {
                "sig_i": i,
                "entry_i": entry_i,
                "exit_i": int(exit_i),
                "sig_t": df.loc[i, "_t"],
                "entry_t": df.loc[entry_i, "_t"],
                "exit_t": df.loc[int(exit_i), "_t"],
                "hour_utc": int(df.loc[i, "hour_utc"]) if not pd.isna(df.loc[i, "hour_utc"]) else None,
                "entry": entry,
                "sl": float(sl),
                "tp": float(tp),
                "exit": float(exit_px),
                "reason": exit_reason,
                "risk": float(risk),
                "R": float(R),
                "MAE_R": float(mae_r) if mae_r != -np.inf else np.nan,
                "MFE_R": float(mfe_r) if mfe_r != -np.inf else np.nan,
            }
        )

    tr = pd.DataFrame(trades)

    print("Config:")
    print(f"  H={H}  ATR_LEN={args.atr_len}  SL_K={args.sl_k}  TP_R={args.tp_r}  conservative_both_touch={args.conservative_both_touch}")
    print(f"Policy:")
    print(f"  bear_only={args.bear_only} forbid_neutral={args.forbid_neutral} dir_score_abs_min={args.dir_score_abs_min} allow_rp_max={args.allow_rp_max} block_hours={sorted(list(BLOCK_HOURS_UTC))}")
    print(f"Signals passing policy (raw): {int(sig.sum())}")
    print(f"Trades simulated: {len(tr)}")

    if len(tr) == 0:
        return 0

    # Global
    gR = _perf(tr["R"])
    gMAE = _perf(tr["MAE_R"])
    gMFE = _perf(tr["MFE_R"])

    print("\n=== GLOBAL (R) ===")
    print(pd.DataFrame([gR]).to_string(index=False))
    print("\n=== GLOBAL (MAE_R) ===")
    print(pd.DataFrame([gMAE]).to_string(index=False))
    print("\n=== GLOBAL (MFE_R) ===")
    print(pd.DataFrame([gMFE]).to_string(index=False))

    # By reason
    rows = []
    for k, g in tr.groupby("reason"):
        r = _perf(g["R"])
        r["reason"] = k
        rows.append(r)
    out_reason = pd.DataFrame(rows).sort_values("n", ascending=False)
    print("\n=== BY EXIT REASON (R) ===")
    print(out_reason.to_string(index=False))

    # Correlation-style insight (simple)
    print("\n=== QUICK CHECKS ===")
    print("Mean MAE_R on SL:", float(tr.loc[tr["reason"].str.startswith("SL"), "MAE_R"].mean()))
    print("Mean MFE_R on SL:", float(tr.loc[tr["reason"].str.startswith("SL"), "MFE_R"].mean()))
    print("Median MFE_R overall:", float(tr["MFE_R"].median()))
    print("Share trades with MFE_R >= TP_R:", float((tr["MFE_R"] >= float(args.tp_r)).mean()))

    if args.out_trades_csv:
        out_path = Path(args.out_trades_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        tr.to_csv(out_path, index=False)
        print(f"\n[OK] wrote: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
