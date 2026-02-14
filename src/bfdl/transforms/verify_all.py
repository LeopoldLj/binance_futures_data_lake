from __future__ import annotations

import inspect
import json
import os
import sys
import time
from pathlib import Path
from typing import List, Tuple

import pandas as pd
import requests


# ======================================================================================
# BF Data Lake Doctor — verify_all.py
#
# But :
# - Vérifier que ton environnement exécute bien le package bfdl depuis ./src (et pas site-packages)
# - Vérifier l'accès à l'API Binance (sanity check)
# - Vérifier la présence et la conformité de _meta.json et _checkpoint.json
# - Vérifier le schéma des Parquets (colonnes attendues)
# - Vérifier l’intégrité temporelle M1 :
#     * pas de doublons open_time_ms
#     * monotonie stricte (pas de pas non-croissant)
#     * aucune minute manquante sur la plage observée
#     * Rows == Expected rows sur la grille minute (min_ts -> max_ts)
#
# + Ajout demandé :
# - Afficher l’heure de la dernière minute disponible en UTC et en Europe/Paris.
#
# + Patch CLI :
# - Supporte: python -m bfdl.transforms.verify_all --symbol ETHUSDT
# ======================================================================================


# Colonnes attendues dans la "source of truth" parquet
REQUIRED_COLS = [
    "ts",               # timestamp UTC basé sur open_time_ms (anti-lookahead)
    "open",
    "high",
    "low",
    "close",
    "volume_base",
    "volume_quote",
    "n_trades",
    "taker_buy_base",
    "taker_buy_quote",
    "open_time_ms",
    "close_time_ms",
    "exchange",
    "market",
    "symbol",
]

# Endpoint test (sanity)
BINANCE_TEST_URL = "https://fapi.binance.com/fapi/v1/klines"


def _symbol_root(base_dir: Path, symbol: str) -> Path:
    # Racine de stockage d’un symbole
    return base_dir / "data" / "raw" / "binance_um" / "klines_m1" / f"symbol={symbol.upper()}"


def _list_parquet_files(symbol_dir: Path) -> List[Path]:
    # Liste tous les part-*.parquet (hors staging qui est sous month=MM/staging/)
    return sorted(symbol_dir.rglob("part-*.parquet"))


def _now_ms() -> int:
    # now en ms (UTC), utilisé pour vérifier que le checkpoint n’est pas dans le futur
    return int(time.time() * 1000)


def _print_kv(k: str, v: str) -> None:
    # Affichage aligné type "doctor"
    print(f"{k:<28} {v}")


def _ok(msg: str) -> None:
    print(f"[OK] {msg}")


def _ko(msg: str) -> None:
    print(f"[KO] {msg}")


def check_import_paths() -> bool:
    """
    Vérifie que bfdl est importé depuis ./src (et pas depuis un package installé).
    Vérifie aussi la signature de collect_klines_m1.
    """
    ok = True
    _print_kv("sys.executable:", sys.executable)
    _print_kv("cwd:", str(Path.cwd()))
    _print_kv("PYTHONPATH:", os.environ.get("PYTHONPATH", ""))

    try:
        import bfdl
        import bfdl.collectors.klines_m1 as m
        _print_kv("bfdl.__file__:", str(Path(bfdl.__file__)))
        _print_kv("klines_m1.__file__:", str(Path(m.__file__)))

        if "site-packages" in str(bfdl.__file__).lower():
            _ko("bfdl importé depuis site-packages (mauvais). Tu dois importer depuis ./src.")
            ok = False
        else:
            _ok("bfdl importé depuis src (bon).")
    except Exception as e:
        _ko(f"Import bfdl impossible: {e}")
        return False

    try:
        from bfdl.collectors.klines_m1 import collect_klines_m1
        sig = str(inspect.signature(collect_klines_m1))
        _print_kv("collect_klines_m1:", sig)

        # On veut bien retrouver les arguments de range (même si en prod tu les laisses à None)
        if "start_date_utc" not in sig:
            _ko("La fonction collect_klines_m1 n'a pas start_date_utc (mauvaise version).")
            ok = False
        if "end_date_utc" not in sig:
            _ko("La fonction collect_klines_m1 n'a pas end_date_utc (mauvaise version).")
            ok = False

        if ok:
            _ok("Signature collect_klines_m1 OK.")
    except Exception as e:
        _ko(f"Signature collect_klines_m1 non vérifiable: {e}")
        ok = False

    return ok


def check_binance_api(symbol: str) -> bool:
    """
    Ping simple : on demande 5 klines M1 au marché UM Futures.
    Objectif : vérifier réseau + endpoint + symbol.
    """
    try:
        r = requests.get(
            BINANCE_TEST_URL,
            params={"symbol": symbol.upper(), "interval": "1m", "limit": 5},
            timeout=10,
        )
        _print_kv("Binance status:", str(r.status_code))
        if r.status_code != 200:
            _ko(f"API Binance KO: {r.text[:200]}")
            return False

        data = r.json()
        if not isinstance(data, list) or len(data) == 0:
            _ko("API Binance répond mais renvoie une liste vide (anormal).")
            return False

        _ok("API Binance OK (klines reçues).")
        return True
    except Exception as e:
        _ko(f"API Binance inaccessible: {e}")
        return False


def check_meta_checkpoint(symbol_dir: Path) -> bool:
    """
    Vérifie :
    - _meta.json présent et contient les champs attendus (dont created_at_utc)
    - _checkpoint.json présent et cohérent (pas dans le futur)
    """
    ok = True
    meta = symbol_dir / "_meta.json"
    ckpt = symbol_dir / "_checkpoint.json"

    if meta.exists():
        try:
            j = json.loads(meta.read_text(encoding="utf-8"))
            _ok("_meta.json présent.")
            for k in ["symbol", "exchange", "market", "interval", "created_at_utc"]:
                if k not in j:
                    _ko(f"_meta.json: champ manquant {k}")
                    ok = False
        except Exception as e:
            _ko(f"_meta.json illisible: {e}")
            ok = False
    else:
        _ko("_meta.json absent.")
        ok = False

    if ckpt.exists():
        try:
            j = json.loads(ckpt.read_text(encoding="utf-8"))
            _ok("_checkpoint.json présent.")
            nst = int(j.get("next_start_time_ms"))
            _print_kv("next_start_time_ms:", str(nst))

            now_ms = _now_ms()
            _print_kv("now_ms:", str(now_ms))

            # Tolérance : 5 minutes dans le futur max (sinon on risque un "silent no-op")
            if nst > now_ms + 5 * 60 * 1000:
                _ko("Checkpoint dans le futur (Binance renverra vide).")
                ok = False
            else:
                _ok("Checkpoint cohérent (pas dans le futur).")
        except Exception as e:
            _ko(f"_checkpoint.json illisible: {e}")
            ok = False
    else:
        _ko("_checkpoint.json absent.")
        ok = False

    return ok


def check_parquet_schema(symbol_dir: Path, max_files_scan: int = 50) -> Tuple[bool, List[Path]]:
    """
    Vérifie qu'il existe des Parquets et qu'un sous-ensemble (max_files_scan)
    respecte le schéma attendu (REQUIRED_COLS).
    """
    files = _list_parquet_files(symbol_dir)
    if not files:
        _ko("Aucun fichier parquet trouvé.")
        return False, []

    _ok(f"Parquets trouvés: {len(files)}")
    ok = True

    scan_files = files[:max_files_scan] if len(files) > max_files_scan else files
    for p in scan_files:
        try:
            df = pd.read_parquet(p)
            missing_cols = [c for c in REQUIRED_COLS if c not in df.columns]
            if missing_cols:
                _ko(f"{p}: colonnes manquantes: {missing_cols}")
                ok = False
                break
        except Exception as e:
            _ko(f"Lecture parquet KO {p}: {e}")
            ok = False
            break

    if ok:
        _ok(f"Schéma OK (sur {len(scan_files)} fichiers scannés).")

    return ok, files


def check_integrity_full(files: List[Path], gaps_sample: int = 0) -> bool:
    """
    Check complet (lourd) :
    - concat tous les fichiers Parquet
    - trie par open_time_ms
    - calcule :
        * duplicats open_time_ms
        * pas non-croissants
        * expected rows sur la grille minute
        * minutes manquantes
    - imprime la dernière minute en UTC et en Europe/Paris (demande utilisateur)
    """
    try:
        df = pd.concat([pd.read_parquet(p) for p in files], ignore_index=True)
    except Exception as e:
        _ko(f"Concat parquet KO: {e}")
        return False

    if df.empty:
        _ko("Dataset vide.")
        return False

    # Anti-lookahead : ts = open_time_ms
    df["ts"] = pd.to_datetime(df["open_time_ms"], unit="ms", utc=True)
    df_sorted = df.sort_values("open_time_ms").reset_index(drop=True)

    min_ts = df_sorted["ts"].iloc[0]
    max_ts = df_sorted["ts"].iloc[-1]
    n_total = int(len(df_sorted))

    # Dernière minute (UTC / Paris)
    _print_kv("Last minute (UTC):", max_ts.isoformat())
    try:
        _print_kv("Last minute (Paris):", max_ts.tz_convert("Europe/Paris").isoformat())
    except Exception as e:
        _print_kv("Last minute (Paris):", f"[WARN] tz_convert failed: {e}")

    # Duplicats open_time_ms
    n_dups = int(df_sorted.duplicated(subset=["open_time_ms"]).sum())

    # Monotonie : diff en ms (si <= 0 => pas non-croissant)
    diffs = df_sorted["open_time_ms"].diff()
    non_increasing = int((diffs <= 0).sum())

    # Rows attendues sur la grille minute complète
    expected = int(((max_ts - min_ts).total_seconds() // 60) + 1)

    # Minutes manquantes
    full_index = pd.date_range(start=min_ts, end=max_ts, freq="1min", tz="UTC")
    present_index = pd.DatetimeIndex(df_sorted["ts"])
    missing = full_index.difference(present_index)
    n_missing = int(len(missing))

    print("=== DATA INTEGRITY SUMMARY ===")
    _print_kv("Range:", f"{min_ts.isoformat()} -> {max_ts.isoformat()}")
    _print_kv("Rows:", str(n_total))
    _print_kv("Expected rows:", str(expected))
    _print_kv("Duplicates:", str(n_dups))
    _print_kv("Non-increasing:", str(non_increasing))
    _print_kv("Missing minutes:", str(n_missing))

    ok = True

    if n_dups > 0:
        _ko("Doublons détectés.")
        ok = False
    else:
        _ok("Pas de doublons.")

    if non_increasing > 0:
        _ko("Non-increasing steps détectés.")
        ok = False
    else:
        _ok("Monotonicité OK.")

    if n_missing > 0:
        _ko("Trous temporels détectés (gaps).")
        if gaps_sample > 0:
            print("First missing minutes (sample):")
            for t in missing[:gaps_sample]:
                print(" -", t.isoformat())
        ok = False
    else:
        _ok("Pas de trous (minute grid complète).")

    if n_total != expected:
        _ko("Rows != Expected rows (completude KO sur la plage observée).")
        ok = False
    else:
        _ok("Rows == Expected rows (completude OK sur la plage observée).")

    return ok


def verify_all(symbol: str = "BTCUSDT") -> int:
    """
    Point d'entrée.
    Retour :
    - 0 si tout OK
    - 1 si KO
    - 2 si KO critique (symbol dir absent)
    """
    print("=== BF Data Lake Doctor ===")
    base_dir = Path.cwd()

    ok1 = check_import_paths()
    ok2 = check_binance_api(symbol)

    sym_dir = _symbol_root(base_dir, symbol)
    _print_kv("Symbol dir:", str(sym_dir))
    if not sym_dir.exists():
        _ko("Symbol dir inexistant.")
        return 2

    ok3 = check_meta_checkpoint(sym_dir)
    ok4, files = check_parquet_schema(sym_dir, max_files_scan=50)

    ok5 = False
    if files:
        # gaps_sample=0 pour ne pas spammer la console
        ok5 = check_integrity_full(files, gaps_sample=0)

    print("=== VERDICT GLOBAL ===")
    all_ok = ok1 and ok2 and ok3 and ok4 and ok5
    if all_ok:
        _ok("Tout est OK.")
        return 0

    _ko("Il y a au moins un problème à corriger (voir [KO]).")
    return 1


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="BF Data Lake Doctor — verify_all")
    parser.add_argument("--symbol", type=str, default="BTCUSDT", help="Symbol (ex: BTCUSDT, ETHUSDT)")
    args = parser.parse_args()

    raise SystemExit(verify_all(args.symbol))
