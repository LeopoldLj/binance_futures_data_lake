# C:\Users\lolo_\PycharmProjects\binance_futures_data_lake\scripts\research\concat_parquets_v1.py
# VERSION=2026-02-14a
import argparse
from pathlib import Path
import pandas as pd


def _read_one(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"[KO] Missing file: {p}")
    return pd.read_parquet(p)


def concat_parquets(inputs: list[str], ts_col: str) -> pd.DataFrame:
    parts = []
    for p in inputs:
        df = _read_one(p)
        if ts_col not in df.columns:
            raise ValueError(f"[KO] '{p}' missing ts_col='{ts_col}'. Columns={list(df.columns)[:40]}")
        parts.append(df)

    out = pd.concat(parts, axis=0, ignore_index=True)

    t = pd.to_datetime(out[ts_col], utc=True, errors="coerce")
    if t.isna().any():
        bad = int(t.isna().sum())
        raise ValueError(f"[KO] {bad} rows have invalid '{ts_col}' timestamps after concat.")
    out["_t__"] = t

    out = out.sort_values("_t__", kind="mergesort").drop(columns=["_t__"])
    out = out.drop_duplicates(subset=[ts_col], keep="last").reset_index(drop=True)

    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Concat multiple parquet files, sort by timestamp, drop duplicates.")
    ap.add_argument("--inputs", required=True, help="Comma-separated list of parquet files")
    ap.add_argument("--out", required=True, help="Output parquet path")
    ap.add_argument("--ts-col", default="ts", help="Timestamp column name (default: ts)")
    args = ap.parse_args()

    inputs = [s.strip() for s in args.inputs.split(",") if s.strip()]
    if len(inputs) < 1:
        raise ValueError("[KO] --inputs is empty")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = concat_parquets(inputs, ts_col=args.ts_col)

    t = pd.to_datetime(df[args.ts_col], utc=True)
    print(f"[concat_parquets_v1] VERSION=2026-02-14a")
    print(f"[INFO] rows={len(df)} t_min={t.min()} t_max={t.max()} ts_col={args.ts_col}")
    df.to_parquet(out_path, index=False)
    print(f"[OK] wrote: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
