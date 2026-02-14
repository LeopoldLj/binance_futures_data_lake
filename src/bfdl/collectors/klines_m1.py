from __future__ import annotations

import json
import secrets
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import pandas as pd
import requests


BINANCE_UM_BASE_URL = "https://fapi.binance.com"
KLINES_ENDPOINT = "/fapi/v1/klines"
INTERVAL = "1m"
LIMIT = 1500


@dataclass
class KlinesM1Config:
    symbol: str
    base_dir: str
    write_csv: bool
    start_date_utc: Optional[str] = None
    end_date_utc: Optional[str] = None
    sleep_sec: float = 0.15
    safe_lag_minutes: int = 2  # ne pas collecter les 2 dernières minutes (bougies potentiellement instables)


class BinanceKlinesM1Collector:
    def run(self, cfg: KlinesM1Config) -> None:
        symbol = cfg.symbol.upper()

        root = Path(cfg.base_dir) / "data" / "raw" / "binance_um" / "klines_m1" / f"symbol={symbol}"
        root.mkdir(parents=True, exist_ok=True)

        checkpoint_path = root / "_checkpoint.json"
        meta_path = root / "_meta.json"

        checkpoint_ms = self._load_checkpoint(checkpoint_path)
        start_from_date_ms = self._parse_utc_to_ms(cfg.start_date_utc)
        end_ms = self._parse_utc_to_ms(cfg.end_date_utc)

        start_time_ms = checkpoint_ms if checkpoint_ms is not None else start_from_date_ms

        if not meta_path.exists():
            self._write_meta(meta_path, symbol)

        print(f"[START] symbol={symbol} interval={INTERVAL} limit={LIMIT} write_csv={cfg.write_csv}")
        print(f"[START] checkpoint_ms={checkpoint_ms} start_date_ms={start_from_date_ms} end_ms={end_ms}")
        print(f"[START] start_time_ms={start_time_ms}")

        n_loops = 0
        while True:
            n_loops += 1

            now_ms = int(pd.Timestamp.now(tz="UTC").value // 1_000_000)
            safe_end_ms = now_ms - int(cfg.safe_lag_minutes) * 60 * 1000

            if start_time_ms is not None and start_time_ms > safe_end_ms:
                print(f"[STOP] up-to-date (within {cfg.safe_lag_minutes}m). start={start_time_ms} safe_end={safe_end_ms}")
                break

            if end_ms is not None and start_time_ms is not None and start_time_ms > end_ms:
                print("[STOP] end_date_utc reached.")
                break

            print(f"[RUN] loop={n_loops} start_time_ms={start_time_ms}")

            klines = self._fetch_klines(symbol, start_time_ms)
            print(f"[FETCH] n={len(klines)}")

            if not klines:
                print("[STOP] Binance returned 0 klines (no new data for this startTime).")
                break

            df = self._klines_to_df(klines, symbol)

            if end_ms is not None:
                df = df[df["open_time_ms"] <= end_ms].copy()
                if df.empty:
                    print("[STOP] all fetched klines are after end_date_utc.")
                    break

            new_rows = int(len(df))
            self._write_month_staging(df, root)
            print(f"[NEW] new_rows_fetched={new_rows}")

            # incrément strict : prochaine milliseconde après la dernière open_time_ms
            start_time_ms = int(df["open_time_ms"].max()) + 1
            self._save_checkpoint(checkpoint_path, start_time_ms)
            print(f"[CKPT] next_start_time_ms={start_time_ms}")

            time.sleep(cfg.sleep_sec)

        print("[DONE] collector finished.")

    def _fetch_klines(self, symbol: str, start_time_ms: Optional[int]) -> List[list]:
        params = {"symbol": symbol, "interval": INTERVAL, "limit": LIMIT}
        if start_time_ms is not None:
            params["startTime"] = int(start_time_ms)

        r = requests.get(BINANCE_UM_BASE_URL + KLINES_ENDPOINT, params=params, timeout=10)
        r.raise_for_status()
        return r.json()

    def _klines_to_df(self, klines: List[list], symbol: str) -> pd.DataFrame:
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
                    "symbol": symbol,
                }
            )

        df = pd.DataFrame(rows)
        df.sort_values("open_time_ms", inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df

    # ----------------------------------------------------------------------------------
    # STAGING APPEND (rapide) — FIX ANTI-OVERWRITE
    #
    # Problème : lors de backfill massif, plusieurs écritures peuvent tomber la même seconde,
    # ce qui peut écraser un fichier si le nom est uniquement à la seconde.
    #
    # Fix : nom de fichier staging unique avec :
    # - timestamp UTC à la milliseconde
    # - + nonce aléatoire court (secrets.token_hex)
    # ----------------------------------------------------------------------------------
    def _write_month_staging(self, df: pd.DataFrame, root: Path) -> None:
        df2 = df.copy()
        df2["year"] = df2["ts"].dt.year
        df2["month"] = df2["ts"].dt.month

        now = pd.Timestamp.now(tz="UTC")
        # ex: 20260209T125532_847Z
        stamp = now.strftime("%Y%m%dT%H%M%S")
        ms = f"{int(now.microsecond / 1000):03d}Z"
        # nonce court pour éviter collision même si même ms
        nonce = secrets.token_hex(2)  # 4 hex chars
        fname = f"stage-{stamp}_{ms}_{nonce}.parquet"

        for (year, month), g in df2.groupby(["year", "month"]):
            month_dir = root / f"year={int(year)}" / f"month={int(month):02d}"
            staging_dir = month_dir / "staging"
            staging_dir.mkdir(parents=True, exist_ok=True)

            out = staging_dir / fname
            g_out = g.drop(columns=["year", "month"])
            g_out.to_parquet(out, index=False)

            print(f"[STAGE] year={int(year)} month={int(month):02d} rows={len(g_out)} file={out.name}")

    def _load_checkpoint(self, path: Path) -> Optional[int]:
        if not path.exists():
            return None
        with open(path, "r", encoding="utf-8") as f:
            return int(json.load(f)["next_start_time_ms"])

    def _save_checkpoint(self, path: Path, next_start_time_ms: int) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "next_start_time_ms": int(next_start_time_ms),
                    "updated_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
                },
                f,
                indent=2,
            )

    def _write_meta(self, path: Path, symbol: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "symbol": symbol,
                    "exchange": "binance",
                    "market": "um_futures",
                    "interval": "1m",
                    "created_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
                },
                f,
                indent=2,
            )

    def _parse_utc_to_ms(self, s: Optional[str]) -> Optional[int]:
        if not s:
            return None
        ts = pd.Timestamp(s)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        else:
            ts = ts.tz_convert("UTC")
        return int(ts.value // 1_000_000)  # ns -> ms


def collect_klines_m1(
    symbol: str,
    base_dir: str,
    start_date_utc: Optional[str] = None,
    end_date_utc: Optional[str] = None,
    write_csv: bool = True,
) -> None:
    cfg = KlinesM1Config(
        symbol=symbol,
        base_dir=base_dir,
        write_csv=write_csv,
        start_date_utc=start_date_utc,
        end_date_utc=end_date_utc,
    )
    BinanceKlinesM1Collector().run(cfg)
