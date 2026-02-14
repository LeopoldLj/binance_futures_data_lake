from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Set

import numpy as np
import pandas as pd


# ======================================================================================
# BF Data Lake — Trade Sweep (SHORT) — TP_R sweep (fixed policy + fixed H/SL_K)
# File: trade_sweep_tpR_short.py
#
# But :
# - Fixer la policy Low+ (hours, BEAR_ONLY, forbid_neutral, dir_score_abs_min, range_pctl<max)
# - Fixer l’exécution : SHORT, entrée au open (i+1), SL=entry+SL_K*ATR, TP=entry-TP_R*risk, TIME à H
# - Sweeper TP_R (liste) et sortir :
#     * table triée (mean_R, pf_R, wr, n)
#     * CSV complet (1 ligne par config)
#
# Remarques :
# - ATR = SMA(TR, atr_len) (simple, stable)
# - Intrabar : si SL et TP touchés même bougie => STOP first (conservatif) si activé
#
# Exemple :
# python trade_sweep_tpR_short.py `
#   --parquet "...\joined_20260103_20260110__enriched.parquet" `
#   --bear-only --forbid-neutral --conservative-both-touch `
#   --H 16 --atr-len 14 --sl-k 1.0 `
#   --tp-r-list "1.25,1.5,1.75,2.0,2.25,2.5" `
#   --dir-score-abs-min 0.3 --allow-rp-max 0.10 `
#   --block-hours "0,1,14,17,19,23" `
#   --out-grid-csv "...\reports\tpR_sweep_H16_sl1.csv"
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


def _parse_list_float(csv: str) -> List[float]:
    if csv is None or csv.strip() == "":
        return []
    out: List[float] = []
    for part in csv.split(","):
        part = part.strip()
        if part == "":
            continue
        out.append(float(part))
    return out


def _pf(x: pd.Series) -> float:
    x = x.dropna()
    if len(x) == 0:
        return float("nan")
    pos = float(x[x > 0].sum())
    neg = float((-x[x < 0]).sum())
    return float(pos / neg) if neg > 0 else float("inf")


def _perf_R(x: pd.Series) -> Dict[str, float]:
    x = x.dropna()
    if len(x) == 0:
        return {"n": 0, "wr": float("nan"), "mean_R": float("nan"), "median_R": float("nan"), "pf_R": float("nan"), "p05_R": float("nan"), "p95_R": float("nan"), "min_R": float("nan")}
    return {
        "n": float(len(x)),
        "wr": float((x > 0).mean()),
        "mean_R": float(x.mean()),
        "median_R": float(x.median()),
        "pf_R": float(_pf(x)),
        "p05_R": float(x.quantile(0.05)),
        "p95_R": float(x.quantile(0.95)),
        "min_R": float(x.min()),
    }


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sweep TP_R for SHORT execution (fixed policy + fixed H/SL_K).")
    p.add_argument("--parquet", required=True, help="Path to enriched parquet (joined_...__enriched.parquet)")
    p.add_argument("--H", type=int, default=16, help="Horizon (bars) for time-exit window")
    p.add_argument("--atr-len", type=int, default=14, help="ATR lookback length (SMA TR)")
    p.add_argument("--sl-k", type=float, default=1.0, help="Stop = entry + sl_k * ATR (SHORT)")
    p.add_argument("--tp-r-list", default="1.25,1.5,1.75,2.0,2.25,2.5", help="CSV list of TP_R values")
    p.add_argument("--dir-score-abs-min", type=float, default=0.3, help="Minimum abs(dir_score)")
    p.add_argument("--allow-rp-max", type=float, default=0.10, help="Allow range_pctl in [0, allow_rp_max)")
    p.add_argument("--block-hours", default="0,1,14,17,19,23", help="CSV hours UTC to block")
    p.add_argument("--forbid-neutral", action="store_true", help="Forbid dir_state=NEUTRAL")
    p.add_argument("--bear-only", action="store_true", help="Restrict to dir_state=BEAR")
    p.add_argument("--conservative-both-touch", action="store_true", help="If SL and TP touched same bar, assume SL first")
    p.add_argument("--out-grid-csv", required=True, help="Output CSV path for grid results")
    return p.parse_args()


def main() -> int:
    args = _parse_args()

    parquet_path = Path(args.parquet)
    if not parquet_path.exists():
        raise FileNotFoundError(f"Parquet not found: {parquet_path}")

    out_csv = Path(args.out_grid_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    H = int(args.H)
    atr_len = int(args.atr_len)
    sl_k = float(args.sl_k)
    tp_r_list = _parse_list_float(args.tp_r_list)

    if len(tp_r_list) == 0:
        raise ValueError("tp-r-list is empty.")

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

    sig = (df["is_add"] == True) & (df["pass_lowp"] == True)
    sig_idx_all = np.flatnonzero(sig.values)
    sig_idx = [i for i in sig_idx_all if (i + 1) < len(df) and (i + H) < len(df)]

    def run_one(tp_r: float) -> Dict[str, float]:
        Rs: List[float] = []
        n_sl = 0
        n_tp = 0
        n_time = 0
        n_skip_atr = 0

        for i in sig_idx:
            entry_i = i + 1
            entry = float(df.loc[entry_i, "open"])
            atr_i = df.loc[entry_i, "atr"]
            if pd.isna(atr_i) or float(atr_i) <= 0:
                n_skip_atr += 1
                continue

            sl = entry + sl_k * float(atr_i)
            risk = sl - entry
            if risk <= 0:
                n_skip_atr += 1
                continue

            tp = entry - tp_r * risk

            reason = None
            exit_px = None

            for j in range(entry_i, entry_i + H):
                hi = float(df.loc[j, "high"])
                lo = float(df.loc[j, "low"])
                hit_sl = hi >= sl
                hit_tp = lo <= tp

                if hit_sl and hit_tp:
                    if conservative_both:
                        reason = "SL_both"
                        exit_px = sl
                    else:
                        reason = "TP_both"
                        exit_px = tp
                    break
                if hit_sl:
                    reason = "SL"
                    exit_px = sl
                    break
                if hit_tp:
                    reason = "TP"
                    exit_px = tp
                    break

            if reason is None:
                reason = "TIME"
                j = entry_i + H - 1
                exit_px = float(df.loc[j, "close"])

            pnl = entry - float(exit_px)
            R = pnl / risk
            Rs.append(float(R))

            if reason.startswith("SL"):
                n_sl += 1
            elif reason.startswith("TP"):
                n_tp += 1
            else:
                n_time += 1

        s = pd.Series(Rs, dtype="float64")
        r = _perf_R(s)
        r.update({"H": float(H), "ATR_LEN": float(atr_len), "SL_K": float(sl_k), "TP_R": float(tp_r), "n_sl": float(n_sl), "n_tp": float(n_tp), "n_time": float(n_time), "n_skip_atr": float(n_skip_atr)})
        return r

    rows: List[Dict[str, float]] = []
    for tp_r in tp_r_list:
        rows.append(run_one(float(tp_r)))

    out = pd.DataFrame(rows)

    out.to_csv(out_csv, index=False)

    out_sorted = out.sort_values(["mean_R", "pf_R", "n"], ascending=[False, False, False])

    print("Config:")
    print(f"  parquet={parquet_path}")
    print(f"  policy: bear_only={bear_only} forbid_neutral={forbid_neutral} dir_score_abs_min={dir_score_abs_min} allow_rp_max={allow_rp_max} block_hours={sorted(list(block_hours))}")
    print(f"  exec: H={H} atr_len={atr_len} sl_k={sl_k} conservative_both_touch={conservative_both}")
    print(f"  signals passing policy (raw): {int(sig.sum())}  usable_signals: {len(sig_idx)}")
    print(f"  tp_r_list={tp_r_list}")
    print(f"  out_grid_csv={out_csv}")

    print("\n=== TOP (sorted) ===")
    print(out_sorted.to_string(index=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
