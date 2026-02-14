from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

# ======================================================================================
# BF Data Lake — CLI Collect (RAW M1) — collect.py
#
# But :
# - Point d’entrée CLI PROD pour collecter les klines M1 (RAW) sur Binance UM Futures.
# - Multi-symboles par défaut via config/symbols.yml :
#     symbols:
#       - BTCUSDT
#       - ETHUSDT
#
# Options :
# - --symbol BTCUSDT  : ne traiter qu’un symbole (override)
# - --symbols-file    : chemin vers le fichier YAML (default: config/symbols.yml)
#
# Notes :
# - La logique safe lag / checkpoint / staging est gérée par collectors/klines_m1.py
# - On appelle l’entrée réelle du collector : collect_klines_m1(base_dir=..., symbol=...)
# ======================================================================================

from bfdl.collectors.klines_m1 import collect_klines_m1


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Collect RAW M1 klines from Binance UM Futures.")
    p.add_argument("--symbol", default=None, help="Symbol override (if set, ignores symbols.yml), e.g. BTCUSDT")
    p.add_argument("--symbols-file", default="config/symbols.yml", help="Path to symbols.yml (default: config/symbols.yml)")
    return p.parse_args()


def _read_symbols_file(base_dir: str, rel_path: str) -> List[str]:
    # Parser minimaliste (sans PyYAML), supporte exactement :
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

    if args.symbol:
        symbols = [args.symbol.upper()]
    else:
        symbols = _read_symbols_file(base_dir=base_dir, rel_path=args.symbols_file)
        if not symbols:
            print("[KO] No symbols to process.")
            return 2

    for sym in symbols:
        collect_klines_m1(base_dir=base_dir, symbol=sym)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
