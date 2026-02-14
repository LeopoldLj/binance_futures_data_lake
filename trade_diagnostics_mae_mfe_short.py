from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional, Set, Tuple

import numpy as np
import pandas as pd


# ======================================================================================
# BF Data Lake — Trade Diagnostics — MAE/MFE (SHORT) — trade_diagnostics_mae_mfe_short.py
#
# But :
# - Diagnostiquer les trades SHORT issus d’un signal "is_add" filtré par une policy Low+.
# - Simuler une exécution simple :
#     * entrée au open de la bougie suivante (i+1)
#     * SL = entry + SL_K * ATR
#     * TP = entry - TP_R * (SL-entry)
#     * sortie si SL/TP touchés dans les H barres, sinon TIME à close de la dernière barre
# - Calculer MAE/MFE en R jusqu’à l’exit :
#     * MAE_R = max((high - entry) / risk)   (adverse pour un short)
#     * MFE_R = max((entry - low) / risk)    (favorable pour un short)
#
# Hypothèse intrabar :
# - Si SL et TP touchés dans la même bougie, on peut forcer un scénario conservateur :
#     * SHORT : STOP d'abord (worst-case)
#
# Sorties :
# - CSV trades détaillés (1 ligne par trade)
# - Résumés console : R distribution + MAE/MFE stats + split par reason/hour/rp_bucket
# ======================================================================================


def _parse_set_int(csv: str) -> Set[int]:
    if csv is None or csv.strip() == "":
        return set()
    out: Set[int] = set()
    for part in csv.split(","):
        part = part.strip()
        if part == "":
            continue
        out.add(int(part))
    return out


def _rp_bucket(x: float) -> str:
    if pd.isna(x):
        return "NA"
    if x < 0.02:
        return "[0.00,0.02)"
    if x < 0.04:
        return "[0.02,0.04)"
    if x < 0.06:
        return "[0.04,0.06)"
    if x < 0.08:
        return "[0.06,0.08)"
    if x < 0.10:
        return "[0.08,0.10)"
    if x < 0.12:
        return "[0.10,0.12)"
    return ">=0.12"


def _perf_series(x: pd.Series) -> dict:
    x = x.dropna()
    if len(x) == 0:
        return {"n": 0, "mean": np.nan, "median": np.nan, "wr": np.nan, "std": np.nan, "p05": np.nan, "p95": np.nan, "min": np.nan, "max": np.nan}
    return {
        "n": int(len(x)),
        "mean": float(x.mean()),
        "median": float(x.median()),
        "wr": float((x > 0).mean()),
        "std": float(x.std(ddof=1)) if len(x) > 1 else np.nan,
        "p05": float(x.quantile(0.05)),
        "p95": float(x.quantile(0.95)),
        "min": float(x.min()),
        "max": float(x.max()),
    }


def _pf(x: pd.Series) -> float:
    x = x.dropna()
    if len(x) == 0:
        return np.nan
    pos = float(x[x > 0].sum())
    neg = float((-x[x < 0]).sum())
    return float(pos / neg) if neg > 0 else float("inf")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Trade diagnostics MAE/MFE for SHORT trades.")
    p.add_argument("--parquet", required=True, help="Path to enriched parquet (joined_...__enriched.parquet)")
    p.add_argument("--H", type=int, default=16, help="Horizon (bars) for time-exit window")
    p.add_argument("--atr-len", type=int, default=14, help="ATR lookback length (SMA TR)")
    p.add_argument("--sl-k", type=float, default=1.0, help="Stop = entry + sl_k * ATR (SHORT)")
    p.add_argument("--tp-r", type=float, default=2.0, help="Take profit in R multiples (SHORT)")
    p.add_argument("--dir-score-abs-min", type=float, default=0.3, help="Minimum abs(dir_score) to allow")
    p.add_argument("--allow-rp-max", type=float, default=0.10, help="Allow range_pctl in [0, allow_rp_max)")
    p.add_argument("--block-hours", default="0,1,14,17,19,23", help="CSV hours UTC to block, e.g. '0,1,14,17,19,23'")
    p.add_argument("--forbid-neutral", action="store_true", help="Forbid dir_state=NEUTRAL")
    p.add_argument("--bear-only", action="store_true", help="Restrict to dir_state=BEAR")
    p.add_argument("--conservative-both-touch", action="store_true", help="If SL and TP touched same bar, assume SL first")
    p.add_argument("--out-trades-csv", required=True, help="Output CSV path for trades details")
    return p.parse_args()


def main() -> int:
    args = _parse_args()

    parquet_path = Path(args.parquet)
    if not parquet_path.exists():
        raise FileNotFoundError(f"Parquet not found: {parquet_path}")

    out_csv = Path(args.out_trades_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    H = int(args.H)
    atr_len = int(args.atr_len)
    sl_k = float(args.sl_k)
    tp_r = float(args.tp_r)

    block_hours = _parse_set_int(args.block_hours)
    forbid_neutral = bool(args.forbid_neutral)
    bear_only = bool(args.bear_only)
    conservative_both = bool(args.conservative_both_touch)
    dir_score_abs_min = float(args.dir_score_abs_min)
    allow_rp_max = float(args.allow_rp_max)

    df = pd.read_parquet(parquet_path)

    need_cols = ["t", "dir_state", "dir_score", "range_pctl", "is_add", "open", "high", "low", "close"]
    missing = [c for c in need_cols if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing required columns: {missing}. Available={list(df.columns)[:60]}...")

    tt = pd.to_datetime(df["t"], utc=True, errors="coerce")
    df["_t"] = tt
    df["hour_utc"] = tt.dt.hour
    df["date_utc"] = tt.dt.date

    df = df.sort_values("_t").reset_index(drop=True)

    prev_close = df["close"].shift(1)
    tr1 = (df["high"] - df["low"]).abs()
    tr2 = (df["high"] - prev_close).abs()
    tr3 = (df["low"] - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df["atr"] = tr.rolling(atr_len, min_periods=atr_len).mean()

    ds = df["dir_state"].astype(str)
    pass_hours = ~df["hour_utc"].isin(block_hours)

    pass_side = pd.Series(True, index=df.index)
    if forbid_neutral:
        pass_side = pass_side & (ds != "NEUTRAL")
    if bear_only:
        pass_side = pass_side & (ds == "BEAR")

    score = pd.to_numeric(df["dir_score"], errors="coerce")
    pass_score = score.abs() >= dir_score_abs_min

    rp = pd.to_numeric(df["range_pctl"], errors="coerce")
    pass_rp = (rp >= 0.0) & (rp < allow_rp_max)

    df["pass_lowp"] = pass_hours & pass_side & pass_score & pass_rp
    df["rp_bucket"] = df["range_pctl"].map(_rp_bucket)

    sig = (df["is_add"] == True) & (df["pass_lowp"] == True)
    sig_idx_all = np.flatnonzero(sig.values)

    sig_idx = [i for i in sig_idx_all if (i + 1) < len(df) and (i + H) < len(df)]

    trades: List[dict] = []
    for i in sig_idx:
        entry_i = i + 1
        entry = float(df.loc[entry_i, "open"])
        atr_i = df.loc[entry_i, "atr"]
        if pd.isna(atr_i) or float(atr_i) <= 0:
            continue

        sl = entry + sl_k * float(atr_i)
        risk = sl - entry
        if risk <= 0:
            continue
        tp = entry - tp_r * risk

        mae_r = 0.0
        mfe_r = 0.0

        exit_reason: Optional[str] = None
        exit_px: Optional[float] = None
        exit_i: Optional[int] = None

        for j in range(entry_i, entry_i + H):
            hi = float(df.loc[j, "high"])
            lo = float(df.loc[j, "low"])

            mae_r = max(mae_r, (hi - entry) / risk)
            mfe_r = max(mfe_r, (entry - lo) / risk)

            hit_sl = hi >= sl
            hit_tp = lo <= tp

            if hit_sl and hit_tp:
                if conservative_both:
                    exit_reason = "SL_both"
                    exit_px = float(sl)
                else:
                    exit_reason = "TP_both"
                    exit_px = float(tp)
                exit_i = j
                break

            if hit_sl:
                exit_reason = "SL"
                exit_px = float(sl)
                exit_i = j
                break

            if hit_tp:
                exit_reason = "TP"
                exit_px = float(tp)
                exit_i = j
                break

        if exit_reason is None:
            exit_reason = "TIME"
            j = entry_i + H - 1
            exit_px = float(df.loc[j, "close"])
            exit_i = j

            hi = float(df.loc[exit_i, "high"])
            lo = float(df.loc[exit_i, "low"])
            mae_r = max(mae_r, (hi - entry) / risk)
            mfe_r = max(mfe_r, (entry - lo) / risk)

        pnl = entry - float(exit_px)
        r_mult = pnl / risk

        trades.append(
            {
                "sig_i": int(i),
                "entry_i": int(entry_i),
                "exit_i": int(exit_i),
                "sig_t": df.loc[i, "_t"],
                "entry_t": df.loc[entry_i, "_t"],
                "exit_t": df.loc[exit_i, "_t"],
                "hour_utc": int(df.loc[i, "hour_utc"]) if not pd.isna(df.loc[i, "hour_utc"]) else None,
                "date_utc": df.loc[i, "date_utc"],
                "dir_state": df.loc[i, "dir_state"],
                "dir_score": float(score.iloc[i]) if not pd.isna(score.iloc[i]) else np.nan,
                "range_pctl": float(rp.iloc[i]) if not pd.isna(rp.iloc[i]) else np.nan,
                "rp_bucket": df.loc[i, "rp_bucket"],
                "entry": float(entry),
                "sl": float(sl),
                "tp": float(tp),
                "exit": float(exit_px),
                "reason": exit_reason,
                "risk": float(risk),
                "pnl": float(pnl),
                "R": float(r_mult),
                "MAE_R": float(mae_r),
                "MFE_R": float(mfe_r),
            }
        )

    trdf = pd.DataFrame(trades)
    trdf.to_csv(out_csv, index=False)

    print("Config:")
    print(f"  parquet={parquet_path}")
    print(f"  H={H} atr_len={atr_len} sl_k={sl_k} tp_r={tp_r}")
    print(f"  policy: bear_only={bear_only} forbid_neutral={forbid_neutral} dir_score_abs_min={dir_score_abs_min} allow_rp_max={allow_rp_max} block_hours={sorted(list(block_hours))}")
    print(f"  conservative_both_touch={conservative_both}")
    print(f"\nSignals passing policy (raw): {int(sig.sum())}")
    print(f"Trades simulated: {len(trdf)}")
    print(f"Trades CSV: {out_csv}")

    if len(trdf) == 0:
        return 0

    R = trdf["R"]
    mae = trdf["MAE_R"]
    mfe = trdf["MFE_R"]

    gR = _perf_series(R)
    gR["pf"] = _pf(R)
    gMAE = _perf_series(mae)
    gMFE = _perf_series(mfe)

    print("\n=== GLOBAL R ===")
    print(pd.DataFrame([gR]).to_string(index=False))
    print("\n=== GLOBAL MAE_R ===")
    print(pd.DataFrame([gMAE]).to_string(index=False))
    print("\n=== GLOBAL MFE_R ===")
    print(pd.DataFrame([gMFE]).to_string(index=False))

    def _group_report(col: str, title: str) -> None:
        rows = []
        for k, g in trdf.groupby(col):
            rr = _perf_series(g["R"])
            rr["pf"] = _pf(g["R"])
            rr["MAE_mean"] = float(g["MAE_R"].mean()) if g["MAE_R"].notna().any() else np.nan
            rr["MFE_mean"] = float(g["MFE_R"].mean()) if g["MFE_R"].notna().any() else np.nan
            rr[col] = k
            rows.append(rr)
        out = pd.DataFrame(rows).sort_values("n", ascending=False)
        print(f"\n=== {title} (group={col}) ===")
        print(out.to_string(index=False))

    _group_report("reason", "BY EXIT REASON")
    _group_report("hour_utc", "BY HOUR")
    _group_report("rp_bucket", "BY RP_BUCKET")

    daily = trdf.groupby("date_utc")["R"].sum().reset_index()
    daily["win_day"] = daily["R"] > 0
    print("\n=== DAILY (sum R) ===")
    print("days:", len(daily), " mean_daily_R:", float(daily["R"].mean()), " wr_days:", float(daily["win_day"].mean()))
    print(daily.sort_values("date_utc").to_string(index=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
