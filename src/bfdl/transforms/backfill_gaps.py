from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests


BINANCE_UM_BASE_URL = "https://fapi.binance.com"
KLINES_ENDPOINT = "/fapi/v1/klines"
INTERVAL = "1m"
LIMIT = 1500


def _symbol_dir(base_dir: Path, symbol: str) -> Path:
    return base_dir / "data" / "raw" / "binance_um" / "klines_m1" / f"symbol={symbol.upper()}"


def _month_dir(symbol_dir: Path, year: int, month: int) -> Path:
    return symbol_dir / f"year={year}" / f"month={month:02d}"


def _list_all_parquets(symbol_dir: Path) -> List[Path]:
    return sorted(symbol_dir.rglob("part-*.parquet"))


def _fetch_klines(symbol: str, start_ms: int, end_ms: Optional[int]) -> List[list]:
    params = {"symbol": symbol.upper(), "interval": INTERVAL, "limit": LIMIT, "startTime": int(start_ms)}
    if end_ms is not None:
        params["endTime"] = int(end_ms)

    r = requests.get(BINANCE_UM_BASE_URL + KLINES_ENDPOINT, params=params, timeout=10)
    r.raise_for_status()
    return r.json()


def _klines_to_df(klines: List[list], symbol: str) -> pd.DataFrame:
    rows = []
    for k in klines:
        rows.append(
            {
                "ts": pd.to_datetime(k[0], unit="ms", utc=True),
                "open": float(k[1]),
                "high": float(k[2]),
                "low": float(k[3]),
                "close": float(k[4]),
                "volume_base": float(k[5]),
                "volume_quote": float(k[7]),
                "n_trades": int(k[8]),
                "taker_buy_base": float(k[9]),
                "taker_buy_quote": float(k[10]),
                "open_time_ms": int(k[0]),
                "close_time_ms": int(k[6]),
                "exchange": "binance",
                "market": "um_futures",
                "symbol": symbol.upper(),
            }
        )
    df = pd.DataFrame(rows)
    if not df.empty:
        df.sort_values("open_time_ms", inplace=True)
        df.reset_index(drop=True, inplace=True)
    return df


def _merge_write_month(symbol_dir: Path, year: int, month: int, df_new: pd.DataFrame, write_csv: bool) -> None:
    mdir = _month_dir(symbol_dir, year, month)
    mdir.mkdir(parents=True, exist_ok=True)
    existing_files = sorted(mdir.glob("part-*.parquet"))
    dfs = [df_new]

    for p in existing_files:
        dfs.append(pd.read_parquet(p))

    merged = pd.concat(dfs, ignore_index=True)
    merged["ts"] = pd.to_datetime(merged["open_time_ms"], unit="ms", utc=True)
    merged.sort_values("open_time_ms", inplace=True)
    merged.drop_duplicates(subset=["open_time_ms"], keep="last", inplace=True)
    merged.reset_index(drop=True, inplace=True)

    out_parquet = mdir / "part-000.parquet"
    merged.to_parquet(out_parquet, index=False)

    for p in existing_files:
        if p.name != "part-000.parquet":
            try:
                p.unlink()
            except OSError:
                pass

    if write_csv:
        out_csv = mdir / "part-000.csv"
        merged.to_csv(out_csv, index=False)
        for p in sorted(mdir.glob("part-*.csv")):
            if p.name != "part-000.csv":
                try:
                    p.unlink()
                except OSError:
                    pass


def _missing_ranges(min_ts: pd.Timestamp, max_ts: pd.Timestamp, present: pd.DatetimeIndex) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    full = pd.date_range(min_ts, max_ts, freq="1min", tz="UTC")
    missing = full.difference(present)
    if missing.empty:
        return []

    # Regroupe en ranges contigus
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


def backfill_gaps(base_dir: str, symbol: str, write_csv: bool = True, sleep_sec: float = 0.15, max_ranges: Optional[int] = None) -> int:
    base = Path(base_dir)
    symbol_dir = _symbol_dir(base, symbol)
    if not symbol_dir.exists():
        print(f"[KO] Symbol dir introuvable: {symbol_dir}")
        return 2

    files = _list_all_parquets(symbol_dir)
    if not files:
        print("[KO] Aucun parquet pour calculer les gaps.")
        return 2

    print(f"Lecture dataset ({len(files)} fichiers) pour détecter les gaps…")
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
        print("[OK] Aucun gap détecté.")
        return 0

    if max_ranges is not None:
        ranges = ranges[:max_ranges]

    total_missing = sum(int(((b - a).total_seconds() // 60) + 1) for a, b in ranges)
    print(f"Gaps: {len(ranges)} ranges, ~{total_missing} minutes manquantes sur {min_ts.isoformat()} -> {max_ts.isoformat()}")

    # Backfill range par range, mais chunk en blocs ≤ 1500 minutes
    for i, (a, b) in enumerate(ranges, start=1):
        start_ms = int(a.value // 1_000_000)
        end_ms = int(b.value // 1_000_000)

        # chunking: 1500 minutes -> 1500*60*1000 ms
        step_ms = LIMIT * 60 * 1000

        cur = start_ms
        while cur <= end_ms:
            chunk_end = min(cur + step_ms - 1, end_ms)
            klines = _fetch_klines(symbol, cur, chunk_end)
            if klines:
                df_new = _klines_to_df(klines, symbol)

                # On peut recevoir un peu plus large, on coupe strictement au range demandé (anti-lookahead)
                df_new = df_new[(df_new["open_time_ms"] >= cur) & (df_new["open_time_ms"] <= chunk_end)].copy()

                if not df_new.empty:
                    # peut traverser 2 mois si chunk sur frontière -> split par mois
                    df_new["year"] = df_new["ts"].dt.year
                    df_new["month"] = df_new["ts"].dt.month
                    for (yy, mm), g in df_new.groupby(["year", "month"]):
                        _merge_write_month(symbol_dir, int(yy), int(mm), g.drop(columns=["year", "month"]), write_csv)

            cur = chunk_end + 1
            time.sleep(sleep_sec)

        if i % 10 == 0:
            print(f"[INFO] ranges traités: {i}/{len(ranges)}")

    print("[OK] Backfill terminé.")
    return 0


if __name__ == "__main__":
    raise SystemExit(backfill_gaps(str(Path.cwd()), "BTCUSDT", write_csv=True, sleep_sec=0.15, max_ranges=None))
