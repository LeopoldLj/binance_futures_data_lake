from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def fix_meta(base_dir: str, symbol: str) -> int:
    symbol = symbol.upper()
    meta_path = Path(base_dir) / "data" / "raw" / "binance_um" / "klines_m1" / f"symbol={symbol}" / "_meta.json"
    if not meta_path.exists():
        print(f"[KO] _meta.json introuvable: {meta_path}")
        return 2

    j = json.loads(meta_path.read_text(encoding="utf-8"))
    changed = False

    if "created_at_utc" not in j:
        j["created_at_utc"] = pd.Timestamp.now(tz="UTC").isoformat()
        changed = True

    if changed:
        meta_path.write_text(json.dumps(j, indent=2), encoding="utf-8")
        print(f"[OK] _meta.json patché: created_at_utc ajouté ({symbol})")
    else:
        print(f"[OK] _meta.json déjà conforme ({symbol})")

    return 0


if __name__ == "__main__":
    raise SystemExit(fix_meta(str(Path.cwd()), "BTCUSDT"))
