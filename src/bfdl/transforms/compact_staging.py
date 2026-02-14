from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import pandas as pd

# ======================================================================================
# BF Data Lake — compact_staging.py
#
# But :
# - Fusionner les fichiers staging (stage-*.parquet) vers le parquet canonique part-000.parquet
# - Déduplication stricte sur open_time_ms + tri strict (anti-lookahead conservé)
# - Staging append -> compaction (source of truth = parquet)
#
# Chemin :
# data/raw/binance_um/klines_m1/
#   symbol=XXX/
#     year=YYYY/
#       month=MM/
#         part-000.parquet
#         staging/
#           stage-*.parquet
#
# CLI :
# - --symbol BTCUSDT (obligatoire)
# - --write-csv (optionnel)
#
# Exit codes :
# - 0 : OK (même si rien à merger)
# - 2 : erreur technique (répertoire introuvable / structure invalide)
# ======================================================================================


def _symbol_dir(base_dir: Path, symbol: str) -> Path:
    return base_dir / "data" / "raw" / "binance_um" / "klines_m1" / f"symbol={symbol.upper()}"


def _month_dirs(symbol_dir: Path) -> List[Path]:
    return sorted([p for p in symbol_dir.rglob("month=*") if p.is_dir()])


def compact_staging(base_dir: str, symbol: str, write_csv: bool = False) -> int:
    base = Path(base_dir)
    symbol_dir = _symbol_dir(base, symbol)

    if not symbol_dir.exists():
        print(f"[KO] symbol dir introuvable: {symbol_dir}")
        return 2

    months = _month_dirs(symbol_dir)
    if not months:
        print("[KO] aucun month=MM trouvé.")
        return 2

    total_staged = 0
    total_merged = 0

    for mdir in months:
        staging_dir = mdir / "staging"
        staged_files = sorted(staging_dir.glob("stage-*.parquet")) if staging_dir.exists() else []
        if not staged_files:
            continue

        total_staged += len(staged_files)

        dfs = []

        main_parquet = mdir / "part-000.parquet"
        if main_parquet.exists():
            dfs.append(pd.read_parquet(main_parquet))

        for p in staged_files:
            dfs.append(pd.read_parquet(p))

        merged = pd.concat(dfs, ignore_index=True)
        if merged.empty:
            for p in staged_files:
                try:
                    p.unlink()
                except OSError:
                    pass
            continue

        # Sécurité : ts dérivé de open_time_ms (anti-lookahead)
        merged["ts"] = pd.to_datetime(merged["open_time_ms"], unit="ms", utc=True)

        # Tri strict + déduplication
        merged.sort_values("open_time_ms", inplace=True)
        before = int(len(merged))
        merged.drop_duplicates(subset=["open_time_ms"], keep="last", inplace=True)
        merged.reset_index(drop=True, inplace=True)
        after = int(len(merged))

        merged.to_parquet(main_parquet, index=False)
        if write_csv:
            merged.to_csv(mdir / "part-000.csv", index=False)

        # Purge staging (déjà compacté)
        for p in staged_files:
            try:
                p.unlink()
            except OSError:
                pass

        total_merged += 1
        print(f"[OK] {mdir.parent.name}/{mdir.name}: staged_files={len(staged_files)} rows(before={before}, after={after})")

    print(f"[DONE] months_merged={total_merged} total_staged_files={total_staged}")
    return 0


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compact staging -> part-000.parquet (RAW M1).")
    p.add_argument("--symbol", required=True, help="Symbol to compact, e.g. BTCUSDT / ETHUSDT")
    p.add_argument("--write-csv", action="store_true", help="Also write part-000.csv (optional)")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    raise SystemExit(compact_staging(str(Path.cwd()), args.symbol, write_csv=bool(args.write_csv)))
