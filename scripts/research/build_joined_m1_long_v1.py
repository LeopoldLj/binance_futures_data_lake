# C:\Users\lolo_\PycharmProjects\binance_futures_data_lake\scripts\research\build_joined_m1_long_v1.py
# target_dir: C:\Users\lolo_\PycharmProjects\binance_futures_data_lake\scripts\research
# build_joined_m1_long_v1.py

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import List, Optional

import pandas as pd


VERSION = "2026-02-14a"


@dataclass
class Inputs:
    m1_path: str
    m15_dir_path: Optional[str]
    m5_vol_path: Optional[str]
    out_path: str
    ts_col_m1: str
    ts_col_m15: str
    ts_col_m5: str
    shift_m15_cols: List[str]
    shift_m5_cols: List[str]


def _read_parquet(path: str, cols: Optional[List[str]] = None) -> pd.DataFrame:
    if cols is None:
        return pd.read_parquet(path)
    return pd.read_parquet(path, columns=cols)


def _ensure_datetime_utc(s: pd.Series) -> pd.Series:
    t = pd.to_datetime(s, utc=True, errors="coerce")
    if t.isna().any():
        bad = int(t.isna().sum())
        raise ValueError(f"[KO] timestamp parse failed: {bad} NA after to_datetime(utc=True)")
    return t


def _prepare_ts(df: pd.DataFrame, ts_col: str, new_col: str) -> pd.DataFrame:
    if ts_col not in df.columns:
        raise ValueError(f"[KO] missing ts_col='{ts_col}' in columns={list(df.columns)[:50]}")
    out = df.copy()
    out[new_col] = _ensure_datetime_utc(out[ts_col])
    out = out.sort_values(new_col, kind="mergesort").reset_index(drop=True)
    return out


def _shift_cols(df: pd.DataFrame, cols: List[str], n: int = 1) -> pd.DataFrame:
    if not cols:
        return df
    miss = [c for c in cols if c not in df.columns]
    if miss:
        raise ValueError(f"[KO] shift cols missing: {miss}")
    out = df.copy()
    for c in cols:
        out[c] = out[c].shift(n)
    return out


def build_joined(inp: Inputs) -> pd.DataFrame:
    print(f"[build_joined_m1_long] VERSION={VERSION}")
    print(f"[INFO] m1={inp.m1_path}")
    print(f"[INFO] m15_dir={inp.m15_dir_path}")
    print(f"[INFO] m5_vol={inp.m5_vol_path}")
    print(f"[INFO] out={inp.out_path}")

    m1 = _read_parquet(inp.m1_path)
    m1 = _prepare_ts(m1, inp.ts_col_m1, "_t_m1")

    out = m1

    if inp.m15_dir_path:
        m15 = _read_parquet(inp.m15_dir_path)
        m15 = _prepare_ts(m15, inp.ts_col_m15, "_t_m15")
        m15 = _shift_cols(m15, inp.shift_m15_cols, n=1)

        # merge_asof backward: each M1 takes last M15 close <= t_m1
        out = pd.merge_asof(out, m15, left_on="_t_m1", right_on="_t_m15", direction="backward")

        bad = (out["_t_m15"].notna()) & (out["_t_m15"] > out["_t_m1"])
        if bad.any():
            raise RuntimeError(f"[KO] merge_asof M15 incohérent: {int(bad.sum())} lignes avec t_m15 > t_m1")

    if inp.m5_vol_path:
        m5 = _read_parquet(inp.m5_vol_path)
        m5 = _prepare_ts(m5, inp.ts_col_m5, "_t_m5")
        m5 = _shift_cols(m5, inp.shift_m5_cols, n=1)

        out = pd.merge_asof(out, m5, left_on="_t_m1", right_on="_t_m5", direction="backward")

        bad = (out["_t_m5"].notna()) & (out["_t_m5"] > out["_t_m1"])
        if bad.any():
            raise RuntimeError(f"[KO] merge_asof M5 incohérent: {int(bad.sum())} lignes avec t_m5 > t_m1")

    # Final checks (minimum set expected downstream)
    if "t" not in out.columns and inp.ts_col_m1 != "t":
        # keep canonical 't' for downstream scripts
        out["t"] = out[inp.ts_col_m1]

    out = out.drop(columns=[c for c in ["_t_m1", "_t_m15", "_t_m5"] if c in out.columns], errors="ignore")

    return out


def parse_args() -> Inputs:
    p = argparse.ArgumentParser(description="Build joined M1 dataset from M1 features + M15 dir regime + M5 vol regime (merge_asof backward, anti-lookahead)")
    p.add_argument("--m1", required=True, help="Path to M1 features parquet (must contain ts + OHLCV + micro feats)")
    p.add_argument("--m15-dir", default="", help="Path to M15 direction regime parquet (dir_state/dir_score/dir_ready...)")
    p.add_argument("--m5-vol", default="", help="Path to M5 vol regime parquet (vol_state/range_pctl/etc)")
    p.add_argument("--out", required=True, help="Output joined parquet path")

    p.add_argument("--ts-col-m1", default="t", help="Timestamp column name in M1 parquet")
    p.add_argument("--ts-col-m15", default="t", help="Timestamp column name in M15 parquet")
    p.add_argument("--ts-col-m5", default="t", help="Timestamp column name in M5 parquet")

    p.add_argument("--shift-m15-cols", default="", help="Comma-separated cols to shift(1) on M15 before merge (anti-lookahead). Example: dir_state,dir_score,dir_ready")
    p.add_argument("--shift-m5-cols", default="", help="Comma-separated cols to shift(1) on M5 before merge (anti-lookahead). Example: vol_state,range_pctl")

    a = p.parse_args()

    shift_m15_cols = [c.strip() for c in a.shift_m15_cols.split(",") if c.strip()]
    shift_m5_cols = [c.strip() for c in a.shift_m5_cols.split(",") if c.strip()]

    return Inputs(
        m1_path=a.m1,
        m15_dir_path=(a.m15_dir if a.m15_dir else None),
        m5_vol_path=(a.m5_vol if a.m5_vol else None),
        out_path=a.out,
        ts_col_m1=a.ts_col_m1,
        ts_col_m15=a.ts_col_m15,
        ts_col_m5=a.ts_col_m5,
        shift_m15_cols=shift_m15_cols,
        shift_m5_cols=shift_m5_cols,
    )


def main() -> int:
    inp = parse_args()
    joined = build_joined(inp)
    print(f"[INFO] joined rows={len(joined)} cols={len(joined.columns)}")
    joined.to_parquet(inp.out_path, index=False)
    print(f"[OK] wrote: {inp.out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
