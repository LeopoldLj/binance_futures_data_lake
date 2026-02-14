from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional

# ======================================================================================
# BF Data Lake — CLI Agrégation TF dérivés — aggregate.py
#
# But :
# - Point d’entrée CLI pour générer les TF dérivés (M5 / H1 / H4) à partir du RAW M1.
# - Respect strict de l’anti-lookahead :
#     * ts = open_time_ms (UTC) au niveau bucket
#     * on ne produit que des buckets complets (géré dans aggregate_tf.py)
# - Sorties écrites dans :
#     data/derived/binance_um/klines_{m5|h1|h4}/symbol=.../year=YYYY/month=MM/part-000.parquet
#
# Multi-symboles :
# - Par défaut, lit config/symbols.yml (format simple) :
#     symbols:
#       - BTCUSDT
#       - ETHUSDT
#
# Options :
# - --tf all|m5|h1|h4 : sélectionner les TF à produire
# - --symbol BTCUSDT  : ne traiter qu’un symbole (override)
# - --symbols-file    : chemin vers le fichier YAML (default: config/symbols.yml)
# - --audit           : lance un audit "non-bloquant" après chaque TF (prints [OK]/[KO])
#
# Important PROD :
# - Pour un audit bloquant (exit codes 0/1/2), utiliser :
#     python -m bfdl.transforms.audit_derived
# ======================================================================================

from bfdl.transforms.aggregate_tf import aggregate_symbol_tf
from bfdl.transforms.audit_derived import audit_symbol_tf


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Aggregate derived TFs from raw M1 (anti-lookahead).")
    p.add_argument("--tf", default="all", choices=["all", "m5", "h1", "h4"], help="Target timeframe")
    p.add_argument("--symbol", default=None, help="Symbol override (if set, ignores symbols.yml), e.g. BTCUSDT")
    p.add_argument("--symbols-file", default="config/symbols.yml", help="Path to symbols.yml (default: config/symbols.yml)")
    p.add_argument("--audit", action="store_true", help="Run a non-blocking audit after aggregation (prints only)")
    return p.parse_args()


def _read_symbols_file(base_dir: str, rel_path: str) -> List[str]:
    # Parser minimaliste (pas de dépendance PyYAML).
    # Supporte exactement le format :
    # symbols:
    #   - BTCUSDT
    #   - ETHUSDT
    path = Path(base_dir) / rel_path
    if not path.exists():
        print(f"[KO] symbols file not found: {path}")
        return []

    syms: List[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("#"):
            continue
        if line.startswith("- "):
            sym = line[2:].strip().strip('"').strip("'")
            if sym:
                syms.append(sym.upper())
        if line.startswith("  - "):
            sym = line[4:].strip().strip('"').strip("'")
            if sym:
                syms.append(sym.upper())

    # Dédup conservateur (ordre conservé)
    out: List[str] = []
    seen = set()
    for s in syms:
        if s not in seen:
            out.append(s)
            seen.add(s)

    return out


def main() -> int:
    args = _parse_args()
    base_dir = str(Path.cwd())

    tfs: List[str] = ["m5", "h1", "h4"] if args.tf == "all" else [args.tf]

    if args.symbol:
        symbols = [args.symbol.upper()]
    else:
        symbols = _read_symbols_file(base_dir=base_dir, rel_path=args.symbols_file)
        if not symbols:
            print("[KO] No symbols to process.")
            return 2

    for symbol in symbols:
        for tf_name in tfs:
            aggregate_symbol_tf(base_dir=base_dir, symbol=symbol, tf_name=tf_name, show_summary=True)
            if args.audit:
                audit_symbol_tf(base_dir=base_dir, symbol=symbol, tf_name=tf_name, show_missing_sample=20)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
