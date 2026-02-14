from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import pandas as pd


def _list_parquets(symbol_dir: Path) -> List[Path]:
    return sorted(symbol_dir.rglob("part-*.parquet"))


def _missing_ranges(min_ts: pd.Timestamp, max_ts: pd.Timestamp, present: pd.DatetimeIndex) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    full = pd.date_range(min_ts, max_ts, freq="1min", tz="UTC")
    missing = full.difference(present)
    if missing.empty:
        return []
    ranges = []
    start = missing[0]
    prev = missing[0]
    for t in missing[1:]:
        if (t - prev) == pd.Timedelta(minutes=1):
            prev = t
            continue
        ranges.append((start, prev))
        start = t
        prev = t
    ranges.append((start, prev))
    return ranges


def report(base_dir: str, symbol: str, top_n: int = 20) -> int:
    symbol = symbol.upper()
    symbol_dir = Path(base_dir) / "data" / "raw" / "binance_um" / "klines_m1" / f"symbol={symbol}"
    files = _list_parquets(symbol_dir)
    if not files:
        print("[KO] Aucun parquet.")
        return 2

    df = pd.concat([pd.read_parquet(p, columns=["open_time_ms"]) for p in files], ignore_index=True)
    df["ts"] = pd.to_datetime(df["open_time_ms"], unit="ms", utc=True)
    df.sort_values("open_time_ms", inplace=True)
    df.drop_duplicates(subset=["open_time_ms"], keep="last", inplace=True)
    df.reset_index(drop=True, inplace=True)

    min_ts = df["ts"].iloc[0]
    max_ts = df["ts"].iloc[-1]
    present = pd.DatetimeIndex(df["ts"])

    ranges = _missing_ranges(min_ts, max_ts, present)
    if not ranges:
        print("[OK] Aucun gap.")
        return 0

    lengths = [int(((b - a).total_seconds() // 60) + 1) for a, b in ranges]
    total = sum(lengths)

    print(f"Ranges gaps: {len(ranges)}")
    print(f"Total missing minutes: {total}")
    print(f"Largest gap minutes: {max(lengths)}")
    print(f"Median gap minutes: {int(pd.Series(lengths).median())}")

    top = sorted(zip(lengths, ranges), key=lambda x: x[0], reverse=True)[:top_n]
    print(f"Top {top_n} gaps:")
    for L, (a, b) in top:
        print(f"- {a.isoformat()} -> {b.isoformat()}  ({L} min)")

    return 0


if __name__ == "__main__":
    raise SystemExit(report(str(Path.cwd()), "BTCUSDT", top_n=20))
