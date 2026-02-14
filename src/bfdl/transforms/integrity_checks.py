from __future__ import annotations

from pathlib import Path
from typing import List

import pandas as pd


def _symbol_root(base_dir: str, symbol: str) -> Path:
    return Path(base_dir) / "data" / "raw" / "binance_um" / "klines_m1" / f"symbol={symbol.upper()}"


def _list_parquet_files(symbol_dir: Path) -> List[Path]:
    return sorted(symbol_dir.rglob("part-*.parquet"))


def _read_all_parquets(files: List[Path]) -> pd.DataFrame:
    dfs = []
    for p in files:
        dfs.append(pd.read_parquet(p))
    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)


def check_symbol(base_dir: str, symbol: str, show_missing_sample: int = 20) -> None:
    symbol = symbol.upper()
    root = _symbol_root(base_dir, symbol)
    files = _list_parquet_files(root)

    if not files:
        print(f"[KO] Aucun parquet trouvé pour {symbol} dans {root}")
        return

    df = _read_all_parquets(files)
    if df.empty:
        print(f"[KO] DataFrame vide pour {symbol}")
        return

    # Sécurité : ts dérivé de open_time_ms (anti-lookahead)
    df["ts"] = pd.to_datetime(df["open_time_ms"], unit="ms", utc=True)

    # Tri
    df_sorted = df.sort_values("open_time_ms").reset_index(drop=True)

    min_ts = df_sorted["ts"].iloc[0]
    max_ts = df_sorted["ts"].iloc[-1]
    n_total = int(len(df_sorted))

    # Duplicats exacts sur minute
    n_dups = int(df_sorted.duplicated(subset=["open_time_ms"]).sum())

    # Monotonicité stricte
    diffs = df_sorted["open_time_ms"].diff()
    non_increasing = int((diffs <= 0).sum())

    # Expected rows (minute grid)
    expected = int(((max_ts - min_ts).total_seconds() // 60) + 1)

    # Gaps stricts (FIX MAJEUR : NE PAS utiliser .values sur tz-aware)
    full_index = pd.date_range(start=min_ts, end=max_ts, freq="1min", tz="UTC")
    present_index = pd.DatetimeIndex(df_sorted["ts"])  # <- tz-aware OK
    missing = full_index.difference(present_index)
    n_missing = int(len(missing))

    print(f"=== Integrity check: {symbol} ===")
    print(f"Files: {len(files)}")
    print(f"Range: {min_ts.isoformat()} -> {max_ts.isoformat()}")
    print(f"Rows: {n_total}")
    print(f"Expected rows (minute grid): {expected}")
    print(f"Duplicates (open_time_ms): {n_dups}")
    print(f"Non-increasing steps: {non_increasing}")
    print(f"Missing minutes: {n_missing}")

    if n_missing > 0 and show_missing_sample > 0:
        print("First missing minutes (sample):")
        for t in missing[:show_missing_sample]:
            print(" -", t.isoformat())

    if n_dups > 0:
        dups = df_sorted[df_sorted.duplicated(subset=["open_time_ms"], keep=False)].copy()
        print(f"Dup range: {dups['ts'].min().isoformat()} -> {dups['ts'].max().isoformat()}")

    ok = (n_dups == 0) and (non_increasing == 0) and (n_missing == 0) and (n_total == expected)
    print("Verdict:", "[OK]" if ok else "[KO]")


if __name__ == "__main__":
    base_dir = str(Path.cwd())
    check_symbol(base_dir=base_dir, symbol="BTCUSDT")
