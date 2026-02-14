# File: trade_backtest_router_short_v2.py
# Directory: C:\Users\lolo_\PycharmProjects\binance_futures_data_lake\scripts\research\
# VERSION=2026-02-14d
#
# Ajouts:
# - dir_score_mode: abs (défaut) / raw_bear (dir_score <= -seuil) / raw_pos (dir_score >= seuil)
# - debug_funnel: imprime le nombre de lignes survivantes après chaque filtre
# - colonnes paramétrables: --atr-col (defaut atr_h1) / --rp-col (defaut low_plus)

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def _ensure_utc_datetime(s: pd.Series) -> pd.Series:
    x = pd.to_datetime(s, utc=True, errors="coerce")
    if x.isna().any():
        bad = int(x.isna().sum())
        raise ValueError(f"[KO] {bad} timestamps invalides après to_datetime(utc=True).")
    return x


def parse_int_list(csv: str) -> List[int]:
    if csv.strip() == "":
        return []
    return [int(x.strip()) for x in csv.split(",") if x.strip() != ""]


@dataclass(frozen=True)
class Policy:
    sl_k_trend: float
    tp_r_trend: float
    sl_k_range: float
    tp_r_range: float


@dataclass
class Trade:
    entry_i: int
    entry_t: pd.Timestamp
    entry_px: float
    regime: str
    sl_px: float
    tp_px: float
    exit_i: int
    exit_t: pd.Timestamp
    exit_px: float
    outcome: str
    r_mult: float
    mae: float
    mfe: float


def decide_policy(regime: str, pol: Policy) -> Tuple[float, float]:
    if regime == "TREND":
        return pol.sl_k_trend, pol.tp_r_trend
    if regime == "RANGE":
        return pol.sl_k_range, pol.tp_r_range
    return pol.sl_k_range, pol.tp_r_range


def _dir_score_pass(x: float, thr: float, mode: str) -> bool:
    if not np.isfinite(x):
        return False
    if mode == "abs":
        return abs(x) >= thr
    if mode == "raw_bear":
        return x <= -thr
    if mode == "raw_pos":
        return x >= thr
    raise ValueError(f"[KO] dir_score_mode invalide: {mode}")


def _funnel(debug: bool, name: str, mask: pd.Series) -> None:
    if debug:
        print(f"[FUNNEL] {name}: {int(mask.sum()):,} / {len(mask):,}")


def backtest_short(
    df: pd.DataFrame,
    ts_col: str,
    H: int,
    pol: Policy,
    conservative_both_touch: bool,
    forbid_neutral: bool,
    dir_score_min: float,
    dir_score_mode: str,
    rp_max: float,
    block_hours: List[int],
    atr_col: str,
    rp_col: str,
    use_tradable_final: bool,
    debug_funnel: bool,
) -> List[Trade]:
    req_base = ["open", "high", "low", "close", "dir_state", "dir_score", "hour_utc", "regime_h1"]
    miss_base = [c for c in req_base if c not in df.columns]
    if miss_base:
        raise ValueError(f"[KO] Colonnes manquantes (base): {miss_base}")

    if atr_col not in df.columns:
        raise ValueError(f"[KO] Colonne ATR introuvable: '{atr_col}'")
    if rp_col not in df.columns:
        raise ValueError(f"[KO] Colonne rp introuvable: '{rp_col}'")
    if use_tradable_final and "tradable_final" not in df.columns:
        raise ValueError("[KO] --use-tradable-final=1 mais colonne 'tradable_final' absente.")

    d = df.sort_values(ts_col).reset_index(drop=True).copy()

    # --- DEBUG funnel (sur lignes, pas sur trades) ---
    if debug_funnel:
        base = pd.Series(True, index=d.index)
        _funnel(True, "BASE", base)

        m_trad = base & ((d["tradable_final"] == 1) if use_tradable_final else True)
        _funnel(True, "tradable_final", m_trad)

        if forbid_neutral:
            m_neu = m_trad & (d["dir_state"].astype(str).str.upper() != "NEUTRAL")
        else:
            m_neu = m_trad
        _funnel(True, "forbid_neutral", m_neu)

        m_bear = m_neu & (d["dir_state"].astype(str).str.upper() == "BEAR")
        _funnel(True, "dir_state==BEAR", m_bear)

        # dir_score
        ds = pd.to_numeric(d["dir_score"], errors="coerce")
        m_ds = m_bear & ds.apply(lambda x: _dir_score_pass(float(x) if pd.notna(x) else np.nan, dir_score_min, dir_score_mode))
        _funnel(True, f"dir_score({dir_score_mode})>= {dir_score_min}", m_ds)

        # rp
        rp = pd.to_numeric(d[rp_col], errors="coerce")
        m_rp = m_ds & (rp <= rp_max)
        _funnel(True, f"{rp_col} <= {rp_max}", m_rp)

        # block hours
        m_hr = m_rp & (~d["hour_utc"].isin(block_hours))
        _funnel(True, "not block_hours", m_hr)

        # regime
        reg = d["regime_h1"].astype(str)
        m_reg = m_hr & (reg != "NA") & (reg != "nan") & (reg != "CHAOS")
        _funnel(True, "regime != CHAOS/NA", m_reg)

        # atr > 0
        atrs = pd.to_numeric(d[atr_col], errors="coerce")
        m_atr = m_reg & (atrs > 0)
        _funnel(True, f"{atr_col} > 0", m_atr)

        # aperçu de quelques rows survivantes
        sample = d.loc[m_atr, [ts_col, "dir_state", "dir_score", rp_col, "hour_utc", "regime_h1", atr_col]].head(10)
        print("[FUNNEL] sample survivors (max 10 rows):")
        print(sample.to_string(index=False))

    # --- backtest trades ---
    trades: List[Trade] = []
    n = len(d)
    i = 0

    while i < n - (H + 1):
        row = d.loc[i]

        if use_tradable_final:
            if pd.isna(row["tradable_final"]) or int(row["tradable_final"]) != 1:
                i += 1
                continue

        if forbid_neutral and str(row["dir_state"]).upper() == "NEUTRAL":
            i += 1
            continue
        if str(row["dir_state"]).upper() != "BEAR":
            i += 1
            continue

        ds = float(row["dir_score"]) if pd.notna(row["dir_score"]) else np.nan
        if not _dir_score_pass(ds, dir_score_min, dir_score_mode):
            i += 1
            continue

        rp = float(row[rp_col]) if pd.notna(row[rp_col]) else np.nan
        if not np.isfinite(rp) or rp > rp_max:
            i += 1
            continue

        if int(row["hour_utc"]) in block_hours:
            i += 1
            continue

        regime = str(row["regime_h1"]) if not pd.isna(row["regime_h1"]) else "NA"
        if regime == "CHAOS" or regime == "NA" or regime == "nan":
            i += 1
            continue

        atrv = float(row[atr_col]) if pd.notna(row[atr_col]) else np.nan
        if not np.isfinite(atrv) or atrv <= 0:
            i += 1
            continue

        sl_k, tp_r = decide_policy(regime, pol)

        entry_px = float(row["close"])
        risk = sl_k * atrv
        sl_px = entry_px + risk
        tp_px = entry_px - tp_r * risk

        exit_i = None
        exit_px = None
        outcome = None

        mae = 0.0
        mfe = 0.0

        for j in range(1, H + 1):
            hi = float(d.loc[i + j, "high"])
            lo = float(d.loc[i + j, "low"])

            mae = max(mae, max(0.0, hi - entry_px))
            mfe = max(mfe, max(0.0, entry_px - lo))

            hit_sl = hi >= sl_px
            hit_tp = lo <= tp_px

            if hit_sl and hit_tp:
                exit_i = i + j
                if conservative_both_touch:
                    exit_px = sl_px
                    outcome = "SL_BOTH"
                else:
                    exit_px = tp_px
                    outcome = "TP_BOTH"
                break
            if hit_sl:
                exit_i = i + j
                exit_px = sl_px
                outcome = "SL"
                break
            if hit_tp:
                exit_i = i + j
                exit_px = tp_px
                outcome = "TP"
                break

        if exit_i is None:
            exit_i = i + H
            exit_px = float(d.loc[exit_i, "close"])
            outcome = "TIME"

        pnl = entry_px - float(exit_px)
        r_mult = pnl / risk

        trades.append(
            Trade(
                entry_i=i,
                entry_t=pd.to_datetime(row[ts_col], utc=True),
                entry_px=entry_px,
                regime=regime,
                sl_px=sl_px,
                tp_px=tp_px,
                exit_i=int(exit_i),
                exit_t=pd.to_datetime(d.loc[exit_i, ts_col], utc=True),
                exit_px=float(exit_px),
                outcome=str(outcome),
                r_mult=float(r_mult),
                mae=float(mae),
                mfe=float(mfe),
            )
        )

        i = int(exit_i) + 1

    return trades


def summarize(trades: List[Trade]) -> Dict[str, object]:
    if len(trades) == 0:
        return {"n": 0}

    r = np.array([t.r_mult for t in trades], dtype=float)
    wins = r[r > 0]
    losses = r[r <= 0]

    gross_win = float(wins.sum()) if wins.size else 0.0
    gross_loss = float(-losses.sum()) if losses.size else 0.0
    pf = (gross_win / gross_loss) if gross_loss > 0 else float("inf")

    return {
        "n": int(len(trades)),
        "expectancy_r": float(r.mean()),
        "pf": float(pf),
        "win_rate": float((r > 0).mean()),
        "avg_win_r": float(wins.mean()) if wins.size else 0.0,
        "avg_loss_r": float(losses.mean()) if losses.size else 0.0,
        "median_r": float(np.median(r)),
        "p10_r": float(np.quantile(r, 0.10)),
        "p90_r": float(np.quantile(r, 0.90)),
    }


def trades_to_df(trades: List[Trade]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "entry_t": t.entry_t,
                "entry_px": t.entry_px,
                "regime": t.regime,
                "sl_px": t.sl_px,
                "tp_px": t.tp_px,
                "exit_t": t.exit_t,
                "exit_px": t.exit_px,
                "outcome": t.outcome,
                "r_mult": t.r_mult,
                "mae": t.mae,
                "mfe": t.mfe,
            }
            for t in trades
        ]
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--ts-col", default="t")
    ap.add_argument("--H", type=int, default=16)

    ap.add_argument("--dir-score-min", type=float, default=0.30)
    ap.add_argument("--dir-score-mode", type=str, default="abs", choices=["abs", "raw_bear", "raw_pos"])

    ap.add_argument("--rp-max", type=float, default=0.10)
    ap.add_argument("--rp-col", type=str, default="low_plus")
    ap.add_argument("--atr-col", type=str, default="atr_h1")

    ap.add_argument("--block-hours", type=str, default="")
    ap.add_argument("--forbid-neutral", type=int, default=1)
    ap.add_argument("--conservative-both-touch", type=int, default=1)
    ap.add_argument("--use-tradable-final", type=int, default=0)
    ap.add_argument("--debug-funnel", type=int, default=0)

    ap.add_argument("--sl-k-trend", type=float, default=1.6)
    ap.add_argument("--tp-r-trend", type=float, default=1.8)
    ap.add_argument("--sl-k-range", type=float, default=1.3)
    ap.add_argument("--tp-r-range", type=float, default=1.1)

    ap.add_argument("--out-trades-csv", type=str, default="")
    args = ap.parse_args()

    print("[trade_backtest_router_short_v2] VERSION=2026-02-14d")

    df = pd.read_parquet(args.input)
    if args.ts_col not in df.columns:
        raise ValueError(f"[KO] ts-col '{args.ts_col}' introuvable.")

    df = df.copy()
    df[args.ts_col] = _ensure_utc_datetime(df[args.ts_col])

    if "hour_utc" not in df.columns:
        df["hour_utc"] = pd.to_datetime(df[args.ts_col], utc=True).dt.hour.astype(int)

    print(f"[INFO] atr_col='{args.atr_col}', rp_col='{args.rp_col}', rp_max={args.rp_max}, dir_score_mode='{args.dir_score_mode}', dir_score_min={args.dir_score_min}, debug_funnel={int(args.debug_funnel)}")

    pol = Policy(
        sl_k_trend=args.sl_k_trend,
        tp_r_trend=args.tp_r_trend,
        sl_k_range=args.sl_k_range,
        tp_r_range=args.tp_r_range,
    )

    trades = backtest_short(
        df=df,
        ts_col=args.ts_col,
        H=args.H,
        pol=pol,
        conservative_both_touch=bool(args.conservative_both_touch),
        forbid_neutral=bool(args.forbid_neutral),
        dir_score_min=args.dir_score_min,
        dir_score_mode=args.dir_score_mode,
        rp_max=args.rp_max,
        block_hours=parse_int_list(args.block_hours),
        atr_col=args.atr_col,
        rp_col=args.rp_col,
        use_tradable_final=bool(args.use_tradable_final),
        debug_funnel=bool(args.debug_funnel),
    )

    s = summarize(trades)
    print("[STATS]", s)

    tdf = trades_to_df(trades)
    if len(tdf) > 0:
        print("[BY REGIME]")
        print(tdf.groupby("regime")["r_mult"].agg(["count", "mean", "median"]))
        print("[BY OUTCOME]")
        print(tdf["outcome"].value_counts())
        tdf["hour_utc"] = tdf["entry_t"].dt.hour
        print("[BY HOUR]")
        print(tdf.groupby("hour_utc")["r_mult"].agg(["count", "mean", "median"]).sort_index())

    if args.out_trades_csv.strip() != "":
        tdf.to_csv(args.out_trades_csv, index=False)
        print(f"[OK] wrote trades csv: {args.out_trades_csv}")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as e:
        print(f"[ERROR] {type(e).__name__}: {e}", file=sys.stderr)
        raise
