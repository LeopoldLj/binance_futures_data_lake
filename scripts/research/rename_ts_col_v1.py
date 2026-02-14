# C:\Users\lolo_\PycharmProjects\binance_futures_data_lake\scripts\research\rename_ts_col_v1.py
# VERSION=2026-02-14a
import argparse
from pathlib import Path
import pandas as pd


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--from-col", required=True)
    ap.add_argument("--to-col", required=True)
    args = ap.parse_args()

    inp = Path(args.inp)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    if not inp.exists():
        raise FileNotFoundError(f"[KO] missing: {inp}")

    df = pd.read_parquet(inp)
    if args.from_col not in df.columns:
        raise ValueError(f"[KO] '{args.from_col}' not in columns={list(df.columns)[:50]}")
    if args.to_col in df.columns:
        raise ValueError(f"[KO] '{args.to_col}' already exists in columns")

    df = df.rename(columns={args.from_col: args.to_col})
    df.to_parquet(out, index=False)
    print(f"[rename_ts_col_v1] VERSION=2026-02-14a")
    print(f"[OK] wrote: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
