from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import pandas as pd


def _symbol_dir(base_dir: Path, symbol: str) -> Path:
    return base_dir / "data" / "raw" / "binance_um" / "klines_m1" / f"symbol={symbol.upper()}"


def _parquet_files(symbol_dir: Path) -> List[Path]:
    return sorted(symbol_dir.rglob("part-*.parquet"))


def last_minute(base_dir: str, symbol: str = "BTCUSDT") -> int:
    """
    Lecture "light" :
    - lit uniquement la colonne open_time_ms sur tous les part-*.parquet
    - calcule le max open_time_ms (dernière minute disponible)
    - imprime UTC + Europe/Paris
    """
    base = Path(base_dir)
    sdir = _symbol_dir(base, symbol)
    if not sdir.exists():
        print(f"[KO] Symbol dir inexistant: {sdir}")
        return 2

    files = _parquet_files(sdir)
    if not files:
        print("[KO] Aucun parquet trouvé.")
        return 2

    max_ms: Optional[int] = None

    # On scanne les fichiers : comme tu n'en as que 77, c'est rapide
    for p in files:
        try:
            col = pd.read_parquet(p, columns=["open_time_ms"])["open_time_ms"]
            if col.empty:
                continue
            m = int(col.max())
            if max_ms is None or m > max_ms:
                max_ms = m
        except Exception as e:
            print(f"[KO] Lecture parquet impossible: {p} ({e})")
            return 2

    if max_ms is None:
        print("[KO] Aucun open_time_ms trouvé.")
        return 2

    ts_utc = pd.to_datetime(max_ms, unit="ms", utc=True)
    ts_paris = ts_utc.tz_convert("Europe/Paris")

    print(f"Last minute (UTC):   {ts_utc.isoformat()}")
    print(f"Last minute (Paris): {ts_paris.isoformat()}")

    return 0


if __name__ == "__main__":
    raise SystemExit(last_minute(str(Path.cwd()), "BTCUSDT"))
