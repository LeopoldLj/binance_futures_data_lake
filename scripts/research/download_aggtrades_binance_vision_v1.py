#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import hashlib
import re
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, Optional

import pandas as pd
import requests


VERSION = "2026-02-15-aggtrades-binance-vision-v1"
DEFAULT_BASE_URL = "https://data.binance.vision"


@dataclass
class Job:
    label: str
    key: str


def parse_month(s: str) -> pd.Timestamp:
    try:
        t = pd.Timestamp(f"{s}-01", tz="UTC")
    except Exception as e:
        raise ValueError(f"Invalid month '{s}', expected YYYY-MM") from e
    return t


def month_range(start_month: str, end_month: str) -> Iterable[pd.Timestamp]:
    a = parse_month(start_month)
    b = parse_month(end_month)
    if b < a:
        raise ValueError("end-month must be >= start-month")
    x = a
    while x <= b:
        yield x
        x = x + pd.offsets.MonthBegin(1)


def build_jobs(symbol: str, start_month: str, end_month: str, periodicity: str) -> list[Job]:
    out: list[Job] = []
    symbol = symbol.upper()
    if periodicity == "monthly":
        for m in month_range(start_month, end_month):
            ym = m.strftime("%Y-%m")
            fname = f"{symbol}-aggTrades-{ym}.zip"
            key = f"data/futures/um/monthly/aggTrades/{symbol}/{fname}"
            out.append(Job(label=ym, key=key))
        return out

    # daily mode from month range boundaries
    a = parse_month(start_month)
    b = parse_month(end_month) + pd.offsets.MonthEnd(0)
    d = a
    while d <= b:
        ymd = d.strftime("%Y-%m-%d")
        fname = f"{symbol}-aggTrades-{ymd}.zip"
        key = f"data/futures/um/daily/aggTrades/{symbol}/{fname}"
        out.append(Job(label=ymd, key=key))
        d = d + pd.Timedelta(days=1)
    return out


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def parse_checksum(text: str) -> str:
    # Typical format: "<sha256>  <filename>"
    first = text.strip().split()[0]
    if not re.fullmatch(r"[A-Fa-f0-9]{64}", first):
        raise ValueError("Invalid CHECKSUM content")
    return first.lower()


def http_download(session: requests.Session, url: str, out: Path, timeout_sec: int) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with session.get(url, stream=True, timeout=timeout_sec) as r:
        if r.status_code == 404:
            raise FileNotFoundError(url)
        r.raise_for_status()
        with open(out, "wb") as f:
            for chunk in r.iter_content(chunk_size=8 * 1024 * 1024):
                if chunk:
                    f.write(chunk)


def parse_bool_series(s: pd.Series) -> pd.Series:
    st = s.astype("string").str.lower()
    return st.isin(["true", "1", "t", "yes", "y"])


def normalize_agg_chunk(raw: pd.DataFrame, symbol: str) -> pd.DataFrame:
    n = raw.shape[1]
    if n < 7:
        raise ValueError(f"Unexpected aggTrades csv columns={n}, expected >=7")

    cols7 = [
        "agg_trade_id",
        "price",
        "qty",
        "first_trade_id",
        "last_trade_id",
        "transact_time_ms",
        "is_buyer_maker",
    ]
    cols8 = cols7 + ["is_best_match"]
    raw = raw.copy()
    raw.columns = cols8[:n]

    out = pd.DataFrame()
    out["agg_trade_id"] = pd.to_numeric(raw["agg_trade_id"], errors="coerce").astype("Int64")
    out["price"] = pd.to_numeric(raw["price"], errors="coerce").astype("float64")
    out["qty"] = pd.to_numeric(raw["qty"], errors="coerce").astype("float64")
    out["first_trade_id"] = pd.to_numeric(raw["first_trade_id"], errors="coerce").astype("Int64")
    out["last_trade_id"] = pd.to_numeric(raw["last_trade_id"], errors="coerce").astype("Int64")
    out["transact_time_ms"] = pd.to_numeric(raw["transact_time_ms"], errors="coerce").astype("Int64")
    out["is_buyer_maker"] = parse_bool_series(raw["is_buyer_maker"]).astype("boolean")

    if "is_best_match" in raw.columns:
        out["is_best_match"] = parse_bool_series(raw["is_best_match"]).astype("boolean")
    else:
        out["is_best_match"] = pd.Series([pd.NA] * len(raw), dtype="boolean")

    out = out.dropna(subset=["price", "qty", "transact_time_ms"]).copy()
    out["transact_time_ms"] = out["transact_time_ms"].astype("int64")
    out["ts"] = pd.to_datetime(out["transact_time_ms"], unit="ms", utc=True)
    out["quote_qty"] = out["price"] * out["qty"]
    out["symbol"] = symbol
    out["exchange"] = "binance"
    out["market"] = "um_futures"
    # buyer_is_maker=True => taker was sell side
    out["taker_side"] = out["is_buyer_maker"].map({True: "SELL", False: "BUY"}).astype("string")

    return out[
        [
            "ts",
            "transact_time_ms",
            "agg_trade_id",
            "first_trade_id",
            "last_trade_id",
            "price",
            "qty",
            "quote_qty",
            "is_buyer_maker",
            "is_best_match",
            "taker_side",
            "symbol",
            "exchange",
            "market",
        ]
    ]


def iter_zip_csv_chunks(zip_path: Path, chunk_rows: int) -> Iterator[pd.DataFrame]:
    with zipfile.ZipFile(zip_path, "r") as zf:
        names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        if not names:
            raise FileNotFoundError(f"No CSV found in ZIP: {zip_path}")
        csv_name = names[0]
        with zf.open(csv_name, "r") as f:
            for chunk in pd.read_csv(f, header=None, chunksize=chunk_rows, low_memory=False):
                yield chunk


def next_part_idx(dir_path: Path, source_key: str, cache: Dict[str, int]) -> int:
    if source_key in cache:
        cache[source_key] += 1
        return cache[source_key]

    max_idx = -1
    if dir_path.exists():
        for p in dir_path.glob("part-*.parquet"):
            m = re.search(r"part-(\d+)\.parquet$", p.name)
            if m:
                max_idx = max(max_idx, int(m.group(1)))
    cache[source_key] = max_idx + 1
    return cache[source_key]


def write_partitioned(df: pd.DataFrame, out_root: Path, symbol: str, part_cache: Dict[str, int]) -> int:
    rows = 0
    symbol = symbol.upper()
    years = df["ts"].dt.year
    months = df["ts"].dt.month

    for (y, m), g in df.groupby([years, months], sort=False):
        out_dir = out_root / f"symbol={symbol}" / f"year={int(y):04d}" / f"month={int(m):02d}"
        out_dir.mkdir(parents=True, exist_ok=True)
        idx = next_part_idx(out_dir, str(out_dir), part_cache)
        out_file = out_dir / f"part-{idx:06d}.parquet"
        g.to_parquet(out_file, index=False)
        rows += len(g)
    return rows


def make_session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": "bfdl-aggtrades-downloader/1.0"})
    return s


def main() -> int:
    p = argparse.ArgumentParser(description="Download Binance Vision UM aggTrades and convert ZIP -> Parquet directly.")
    p.add_argument("--symbol", default="BTCUSDT", help="Symbol, e.g. BTCUSDT")
    p.add_argument("--start-month", required=True, help="Start month YYYY-MM")
    p.add_argument("--end-month", required=True, help="End month YYYY-MM (inclusive)")
    p.add_argument("--periodicity", choices=["monthly", "daily"], default="monthly", help="Archive periodicity")
    p.add_argument("--out-root", default="data/raw/binance_um/aggtrades", help="Output root for partitioned parquet")
    p.add_argument("--tmp-dir", default="data/tmp/binance_vision_aggtrades", help="Temporary ZIP download folder")
    p.add_argument("--base-url", default=DEFAULT_BASE_URL, help="Binance Vision base URL")
    p.add_argument("--chunk-rows", type=int, default=1_000_000, help="CSV rows per processing chunk")
    p.add_argument("--timeout-sec", type=int, default=120, help="HTTP timeout seconds")
    p.add_argument("--sleep-sec", type=float, default=0.2, help="Sleep between files")
    p.add_argument("--no-checksum", action="store_true", help="Disable .CHECKSUM verification")
    p.add_argument("--keep-zip", action="store_true", help="Keep downloaded ZIP files")
    args = p.parse_args()

    symbol = args.symbol.upper()
    out_root = Path(args.out_root)
    tmp_dir = Path(args.tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    jobs = build_jobs(symbol, args.start_month, args.end_month, args.periodicity)
    print(f"[download_aggtrades_binance_vision_v1] VERSION={VERSION}")
    print(f"[INFO] symbol={symbol} jobs={len(jobs)} periodicity={args.periodicity}")
    print(f"[INFO] out_root={out_root}")
    print(f"[INFO] tmp_dir={tmp_dir}")

    total_rows = 0
    done_files = 0
    part_cache: Dict[str, int] = {}

    session = make_session()
    try:
        for j in jobs:
            zip_url = f"{args.base_url.rstrip('/')}/{j.key}"
            zip_name = Path(j.key).name
            zip_path = tmp_dir / zip_name

            print(f"\n[FILE] {j.label} -> {zip_name}")
            http_download(session, zip_url, zip_path, timeout_sec=args.timeout_sec)
            print(f"[OK] downloaded={zip_path} size_mb={zip_path.stat().st_size / (1024**2):.2f}")

            if not args.no_checksum:
                c_url = zip_url + ".CHECKSUM"
                r = session.get(c_url, timeout=args.timeout_sec)
                if r.status_code == 404:
                    raise FileNotFoundError(c_url)
                r.raise_for_status()
                expected = parse_checksum(r.text)
                actual = sha256_file(zip_path)
                if expected != actual:
                    raise ValueError(f"CHECKSUM mismatch for {zip_name}: expected={expected} actual={actual}")
                print("[OK] checksum")

            file_rows = 0
            for i, raw_chunk in enumerate(iter_zip_csv_chunks(zip_path, chunk_rows=args.chunk_rows), start=1):
                norm = normalize_agg_chunk(raw_chunk, symbol=symbol)
                wrote = write_partitioned(norm, out_root=out_root, symbol=symbol, part_cache=part_cache)
                file_rows += wrote
                total_rows += wrote
                print(f"[CHUNK] idx={i} rows_in={len(raw_chunk)} rows_out={wrote} file_rows={file_rows}")

            done_files += 1
            print(f"[DONE] label={j.label} rows={file_rows}")

            if not args.keep_zip and zip_path.exists():
                zip_path.unlink()

            time.sleep(args.sleep_sec)
    finally:
        session.close()

    print(f"\n[OK] completed_files={done_files} total_rows={total_rows}")
    print(f"[OK] parquet_root={out_root / f'symbol={symbol}'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
