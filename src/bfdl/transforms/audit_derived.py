from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import pandas as pd


# ======================================================================================
# BF Data Lake — Derived TF Audit (anti-lookahead) — audit_derived.py
#
# But :
# - Auditer l’intégrité des TimeFrames dérivés (M5 / H1 / H4) générés depuis le RAW M1.
# - Les datasets dérivés doivent être PROD-grade, au même niveau que le M1 :
#     * pas de doublons sur open_time_ms (clé unique du bucket)
#     * monotonie stricte (open_time_ms strictement croissant)
#     * aucune bougie manquante sur la grille du TF (5min / 60min / 240min)
#     * Rows == Expected rows sur la grille TF (min_ts -> max_ts)
#
# Multi-symboles :
# - Par défaut, lit config/symbols.yml :
#     symbols:
#       - BTCUSDT
#       - ETHUSDT
#
# Intégration PROD :
# - Exit codes :
#     * exit 0 : OK
#     * exit 1 : problème DATA (gaps/duplicates/monotonicité/rows mismatch) sur au moins 1 symbole/TF
#     * exit 2 : erreur TECHNIQUE (exception non prévue)
# ======================================================================================


_MS_PER_MIN = 60_000


def _tf_minutes(tf_name: str) -> int:
    tf = tf_name.lower()
    if tf == "m5":
        return 5
    if tf == "h1":
        return 60
    if tf == "h4":
        return 240
    raise ValueError("tf_name must be one of: m5, h1, h4")


def _symbol_root_derived(base_dir: str, tf_name: str, symbol: str) -> Path:
    return Path(base_dir) / "data" / "derived" / "binance_um" / f"klines_{tf_name.lower()}" / f"symbol={symbol.upper()}"


def _list_parquet_files(symbol_dir: Path) -> List[Path]:
    return sorted(symbol_dir.rglob("part-*.parquet"))


def _read_all_parquets(files: List[Path], columns: Optional[List[str]] = None) -> pd.DataFrame:
    dfs = []
    for p in files:
        dfs.append(pd.read_parquet(p, columns=columns))
    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)


def _read_symbols_file(base_dir: str, rel_path: str) -> List[str]:
    # Parser minimaliste, sans PyYAML.
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


def audit_symbol_tf(base_dir: str, symbol: str, tf_name: str, show_missing_sample: int = 20) -> bool:
    symbol = symbol.upper()
    tf_name = tf_name.lower()

    n_min = _tf_minutes(tf_name)
    step_ms = n_min * _MS_PER_MIN

    root = _symbol_root_derived(base_dir, tf_name, symbol)
    files = _list_parquet_files(root)

    if not files:
        print(f"[KO] Aucun parquet trouvé pour {symbol} {tf_name} dans {root}")
        return False

    df = _read_all_parquets(files)
    if df.empty:
        print(f"[KO] DataFrame vide pour {symbol} {tf_name}")
        return False

    # Sécurité : ts dérivé de open_time_ms (anti-lookahead)
    df["ts"] = pd.to_datetime(df["open_time_ms"], unit="ms", utc=True)

    # Tri canonique
    df_sorted = df.sort_values("open_time_ms").reset_index(drop=True)

    min_ts = df_sorted["ts"].iloc[0]
    max_ts = df_sorted["ts"].iloc[-1]
    n_total = int(len(df_sorted))

    # Doublons sur open_time_ms
    n_dups = int(df_sorted.duplicated(subset=["open_time_ms"]).sum())

    # Monotonicité stricte
    diffs = df_sorted["open_time_ms"].diff()
    non_increasing = int((diffs <= 0).sum())

    # Expected rows sur grille TF
    min_ot = int(df_sorted["open_time_ms"].iloc[0])
    max_ot = int(df_sorted["open_time_ms"].iloc[-1])
    expected = int(((max_ot - min_ot) // step_ms) + 1)

    # Gaps stricts sur grille TF (tz-aware)
    full_index = pd.date_range(start=min_ts, end=max_ts, freq=f"{n_min}min", tz="UTC")
    present_index = pd.DatetimeIndex(df_sorted["ts"])
    missing = full_index.difference(present_index)
    n_missing = int(len(missing))

    print(f"=== Derived audit: {symbol} ({tf_name}) ===")
    print(f"Files: {len(files)}")
    print(f"Range: {min_ts.isoformat()} -> {max_ts.isoformat()}")
    print(f"Rows: {n_total}")
    print(f"Expected rows (grid {n_min}min): {expected}")
    print(f"Duplicates (open_time_ms): {n_dups}")
    print(f"Non-increasing steps: {non_increasing}")
    print(f"Missing buckets: {n_missing}")

    if n_missing > 0 and show_missing_sample > 0:
        print("First missing buckets (sample):")
        for t in missing[:show_missing_sample]:
            print(" -", t.isoformat())

    ok = (n_dups == 0) and (non_increasing == 0) and (n_missing == 0) and (n_total == expected)
    print("Verdict:", "[OK]" if ok else "[KO]")

    return ok


def audit_symbol_all(base_dir: str, symbol: str, tfs: Optional[List[str]] = None) -> bool:
    tf_list = tfs or ["m5", "h1", "h4"]
    ok_all = True
    for tf_name in tf_list:
        ok_all = audit_symbol_tf(base_dir=base_dir, symbol=symbol, tf_name=tf_name, show_missing_sample=20) and ok_all
    return ok_all


def audit_all_symbols(base_dir: str, symbols: List[str], tfs: Optional[List[str]] = None) -> bool:
    ok_all = True
    for sym in symbols:
        ok_all = audit_symbol_all(base_dir=base_dir, symbol=sym, tfs=tfs) and ok_all
    return ok_all


if __name__ == "__main__":
    # PROD behavior:
    # - exit 0 si tout est OK
    # - exit 1 si problème DATA sur au moins un symbole/TF
    # - exit 2 si erreur TECHNIQUE
    try:
        base_dir = str(Path.cwd())
        symbols = _read_symbols_file(base_dir=base_dir, rel_path="config/symbols.yml")

        if not symbols:
            print("[KO] No symbols found in config/symbols.yml")
            raise SystemExit(2)

        ok = audit_all_symbols(base_dir=base_dir, symbols=symbols, tfs=["m5", "h1", "h4"])

        if not ok:
            raise SystemExit(1)

        raise SystemExit(0)

    except SystemExit:
        raise
    except Exception as e:
        print("[KO] Erreur technique audit_derived:", repr(e))
        raise SystemExit(2)
