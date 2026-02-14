from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import pandas as pd
import requests


BINANCE_UM_BASE_URL = "https://fapi.binance.com"
KLINES_ENDPOINT = "/fapi/v1/klines"
INTERVAL = "1m"
LIMIT = 1500


def _parse_utc_to_ms(s: str) -> int:
    ts = pd.Timestamp(s)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return int(ts.value // 1_000_000)


def _fetch_klines(symbol: str, start_ms: int, end_ms: Optional[int]) -> list:
    params = {"symbol": symbol.upper(), "interval": INTERVAL, "limit": LIMIT, "startTime": int(start_ms)}
    if end_ms is not None:
        params["endTime"] = int(end_ms)
    r = requests.get(BINANCE_UM_BASE_URL + KLINES_ENDPOINT, params=params, timeout=10)
    r.raise_for_status()
    return r.json()


def _klines_to_df(klines: list, symbol: str) -> pd.DataFrame:
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


def _month_dir(symbol_dir: Path, year: int, month: int) -> Path:
    return symbol_dir / f"year={year}" / f"month={month:02d}"


def _merge_write_month(symbol_dir: Path, year: int, month: int, df_new: pd.DataFrame, write_csv: bool) -> None:
    mdir = _month_dir(symbol_dir, year, month)
    mdir.mkdir(parents=True, exist_ok=True)
    existing = sorted(mdir.glob("part-*.parquet"))

    dfs = [df_new]
    for p in existing:
        dfs.append(pd.read_parquet(p))

    merged = pd.concat(dfs, ignore_index=True)
    merged["ts"] = pd.to_datetime(merged["open_time_ms"], unit="ms", utc=True)
    merged.sort_values("open_time_ms", inplace=True)
    merged.drop_duplicates(subset=["open_time_ms"], keep="last", inplace=True)
    merged.reset_index(drop=True, inplace=True)

    out_parquet = mdir / "part-000.parquet"
    merged.to_parquet(out_parquet, index=False)

    for p in existing:
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


def backfill_range(
    base_dir: str,
    symbol: str,
    start_utc: str,
    end_utc: str,
    write_csv: bool = True,
    sleep_sec: float = 0.15,
) -> int:
    base = Path(base_dir)
    symbol_dir = base / "data" / "raw" / "binance_um" / "klines_m1" / f"symbol={symbol.upper()}"
    if not symbol_dir.exists():
        print(f"[KO] symbol dir introuvable: {symbol_dir}")
        return 2

    start_ms = _parse_utc_to_ms(start_utc)
    end_ms = _parse_utc_to_ms(end_utc)

    step_ms = LIMIT * 60 * 1000  # 1500 minutes

    cur = start_ms
    n_calls = 0
    while cur <= end_ms:
        chunk_end = min(cur + step_ms - 1, end_ms)

        klines = _fetch_klines(symbol, cur, chunk_end)
        n_calls += 1

        if not klines:
            # Binance peut renvoyer vide sur certains segments (source gap)
            if n_calls % 200 == 0:
                print(f"[WARN] vide sur {pd.to_datetime(cur, unit='ms', utc=True).isoformat()} -> {pd.to_datetime(chunk_end, unit='ms', utc=True).isoformat()}")
            cur = chunk_end + 1
            time.sleep(sleep_sec)
            continue

        df_new = _klines_to_df(klines, symbol)
        df_new = df_new[(df_new["open_time_ms"] >= cur) & (df_new["open_time_ms"] <= chunk_end)].copy()

        if not df_new.empty:
            df_new["year"] = df_new["ts"].dt.year
            df_new["month"] = df_new["ts"].dt.month
            for (yy, mm), g in df_new.groupby(["year", "month"]):
                _merge_write_month(symbol_dir, int(yy), int(mm), g.drop(columns=["year", "month"]), write_csv)

        if n_calls % 200 == 0:
            print(f"[INFO] calls={n_calls} cur={pd.to_datetime(cur, unit='ms', utc=True).isoformat()}")

        cur = chunk_end + 1
        time.sleep(sleep_sec)

    print("[OK] backfill_range terminé.")
    return 0


if __name__ == "__main__":
    # Range manquant détecté chez toi
    raise SystemExit(
        backfill_range(
            base_dir=str(Path.cwd()),
            symbol="BTCUSDT",
            start_utc="2019-10-02T01:00:00Z",
            end_utc="2022-12-31T23:59:00Z",
            write_csv=True,
            sleep_sec=0.15,
        )
    )
