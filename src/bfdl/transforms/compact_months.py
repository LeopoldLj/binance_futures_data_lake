from __future__ import annotations

from pathlib import Path
from typing import List

import pandas as pd


def _month_dirs(symbol_dir: Path) -> List[Path]:
    return sorted([p for p in symbol_dir.rglob("month=*") if p.is_dir()])


def compact_symbol(base_dir: str, symbol: str, write_csv: bool = True) -> int:
    symbol = symbol.upper()
    symbol_dir = Path(base_dir) / "data" / "raw" / "binance_um" / "klines_m1" / f"symbol={symbol}"
    if not symbol_dir.exists():
        print(f"[KO] symbol dir introuvable: {symbol_dir}")
        return 2

    months = _month_dirs(symbol_dir)
    if not months:
        print(f"[KO] aucun month=MM trouvé sous: {symbol_dir}")
        return 2

    print(f"Compaction {symbol}: {len(months)} mois détectés")

    for mdir in months:
        parquet_files = sorted(mdir.glob("part-*.parquet"))
        if not parquet_files:
            continue

        dfs = []
        for p in parquet_files:
            dfs.append(pd.read_parquet(p))
        df = pd.concat(dfs, ignore_index=True)

        if df.empty:
            continue

        # ts = open_time_ms (anti-lookahead)
        df["ts"] = pd.to_datetime(df["open_time_ms"], unit="ms", utc=True)

        before = int(len(df))
        df.sort_values("open_time_ms", inplace=True)
        df.drop_duplicates(subset=["open_time_ms"], keep="last", inplace=True)
        df.reset_index(drop=True, inplace=True)
        after = int(len(df))

        out_parquet = mdir / "part-000.parquet"
        df.to_parquet(out_parquet, index=False)

        # Nettoyage des autres parts parquet
        for p in parquet_files:
            if p.name != "part-000.parquet":
                try:
                    p.unlink()
                except OSError:
                    pass

        if write_csv:
            out_csv = mdir / "part-000.csv"
            df.to_csv(out_csv, index=False)

            for p in sorted(mdir.glob("part-*.csv")):
                if p.name != "part-000.csv":
                    try:
                        p.unlink()
                    except OSError:
                        pass

        if before != after or len(parquet_files) > 1:
            print(f"[OK] {mdir.parent.name}/{mdir.name}: {before} -> {after} rows, {len(parquet_files)} files -> 1")

    print("[OK] Compaction terminée.")
    return 0


if __name__ == "__main__":
    raise SystemExit(compact_symbol(str(Path.cwd()), "BTCUSDT", write_csv=True))
