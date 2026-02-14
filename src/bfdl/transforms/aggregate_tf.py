from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import json
import pandas as pd


_MS_PER_MIN = 60_000


@dataclass(frozen=True)
class TfSpec:
    name: str
    n_min: int


_TF_SPECS: Dict[str, TfSpec] = {
    "m5": TfSpec("m5", 5),
    "h1": TfSpec("h1", 60),
    "h4": TfSpec("h4", 240),
}


def _raw_m1_symbol_root(base_dir: str, symbol: str) -> Path:
    return Path(base_dir) / "data" / "raw" / "binance_um" / "klines_m1" / f"symbol={symbol.upper()}"


def _derived_tf_symbol_root(base_dir: str, tf_name: str, symbol: str) -> Path:
    return Path(base_dir) / "data" / "derived" / "binance_um" / f"klines_{tf_name}" / f"symbol={symbol.upper()}"


def _list_parquet_files(symbol_dir: Path) -> List[Path]:
    return sorted(symbol_dir.rglob("part-*.parquet"))


def _read_all_parquets(files: List[Path], columns: Optional[List[str]] = None) -> pd.DataFrame:
    dfs = []
    for p in files:
        dfs.append(pd.read_parquet(p, columns=columns))
    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)


def _floor_to_bucket_open_ms(open_time_ms: pd.Series, n_min: int) -> pd.Series:
    bucket_ms = n_min * _MS_PER_MIN
    return (open_time_ms // bucket_ms) * bucket_ms


def _compute_bucket_completeness(df_m1_sorted: pd.DataFrame, n_min: int) -> pd.DataFrame:
    bucket_ms = n_min * _MS_PER_MIN

    g = df_m1_sorted.groupby("bucket_open_ms", sort=True)

    cnt = g["open_time_ms"].size().rename("cnt")
    mn = g["open_time_ms"].min().rename("mn")
    mx = g["open_time_ms"].max().rename("mx")
    nun = g["open_time_ms"].nunique().rename("nun")

    comp = pd.concat([cnt, mn, mx, nun], axis=1).reset_index()

    comp["is_complete"] = (comp["cnt"] == n_min) & (comp["nun"] == n_min) & ((comp["mx"] - comp["mn"]) == (n_min - 1) * _MS_PER_MIN)
    comp["bucket_close_time_ms"] = comp["bucket_open_ms"] + bucket_ms - 1
    return comp[["bucket_open_ms", "bucket_close_time_ms", "is_complete"]]


def _aggregate_complete_buckets(df_m1_sorted: pd.DataFrame, tf: TfSpec) -> pd.DataFrame:
    df_m1_sorted["ts"] = pd.to_datetime(df_m1_sorted["open_time_ms"], unit="ms", utc=True)

    df_m1_sorted["bucket_open_ms"] = _floor_to_bucket_open_ms(df_m1_sorted["open_time_ms"], tf.n_min)

    comp = _compute_bucket_completeness(df_m1_sorted, tf.n_min)
    complete_opens = comp.loc[comp["is_complete"], "bucket_open_ms"]

    if complete_opens.empty:
        return pd.DataFrame()

    df_ok = df_m1_sorted[df_m1_sorted["bucket_open_ms"].isin(complete_opens)].copy()

    df_ok = df_ok.sort_values(["bucket_open_ms", "open_time_ms"]).reset_index(drop=True)

    g = df_ok.groupby("bucket_open_ms", sort=True)

    out = pd.DataFrame({
        "open_time_ms": g["bucket_open_ms"].first(),
        "close_time_ms": g["open_time_ms"].last() + _MS_PER_MIN - 1,
        "open": g["open"].first(),
        "high": g["high"].max(),
        "low": g["low"].min(),
        "close": g["close"].last(),
        "volume_base": g["volume_base"].sum(),
        "volume_quote": g["volume_quote"].sum(),
        "n_trades": g["n_trades"].sum(),
        "taker_buy_base": g["taker_buy_base"].sum(),
        "taker_buy_quote": g["taker_buy_quote"].sum(),
        "exchange": g["exchange"].first(),
        "market": g["market"].first(),
        "symbol": g["symbol"].first(),
    }).reset_index(drop=True)

    out["ts"] = pd.to_datetime(out["open_time_ms"], unit="ms", utc=True)

    out = out.sort_values("open_time_ms").reset_index(drop=True)

    out = out.drop_duplicates(subset=["open_time_ms"], keep="last").reset_index(drop=True)

    return out


def _year_month_from_open_time_ms(open_time_ms: int) -> Tuple[int, int]:
    ts = pd.to_datetime(open_time_ms, unit="ms", utc=True)
    return int(ts.year), int(ts.month)


def _month_dir(root: Path, year: int, month: int) -> Path:
    return root / f"year={year:04d}" / f"month={month:02d}"


def _load_checkpoint(path: Path) -> Optional[int]:
    if not path.exists():
        return None
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    v = obj.get("next_bucket_open_time_ms")
    if v is None:
        return None
    try:
        return int(v)
    except Exception:
        return None


def _save_checkpoint(path: Path, next_bucket_open_time_ms: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "next_bucket_open_time_ms": int(next_bucket_open_time_ms),
        "updated_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _ensure_meta(path: Path, tf: TfSpec, symbol: str) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "exchange": "binance",
        "market": "um_futures",
        "symbol": symbol.upper(),
        "tf": tf.name,
        "n_minutes": tf.n_min,
        "anti_lookahead": True,
        "source": "derived_from_raw_m1_partitions",
        "created_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _merge_write_month_part(month_dir: Path, df_month: pd.DataFrame) -> None:
    month_dir.mkdir(parents=True, exist_ok=True)
    part_path = month_dir / "part-000.parquet"

    if part_path.exists():
        df_prev = pd.read_parquet(part_path)
        df_all = pd.concat([df_prev, df_month], ignore_index=True)
    else:
        df_all = df_month.copy()

    df_all["ts"] = pd.to_datetime(df_all["open_time_ms"], unit="ms", utc=True)
    df_all = df_all.sort_values("open_time_ms").drop_duplicates(subset=["open_time_ms"], keep="last").reset_index(drop=True)

    tmp_path = month_dir / "part-000.tmp.parquet"
    df_all.to_parquet(tmp_path, index=False)
    tmp_path.replace(part_path)


def aggregate_symbol_tf(base_dir: str, symbol: str, tf_name: str, show_summary: bool = True) -> None:
    symbol = symbol.upper()
    if tf_name not in _TF_SPECS:
        raise ValueError(f"Unsupported tf: {tf_name}. Use one of: {list(_TF_SPECS.keys())}")

    tf = _TF_SPECS[tf_name]

    raw_root = _raw_m1_symbol_root(base_dir, symbol)
    raw_files = _list_parquet_files(raw_root)
    if not raw_files:
        print(f"[KO] Aucun parquet M1 trouvé pour {symbol} dans {raw_root}")
        return

    cols = [
        "open_time_ms", "open", "high", "low", "close",
        "volume_base", "volume_quote", "n_trades", "taker_buy_base", "taker_buy_quote",
        "close_time_ms", "exchange", "market", "symbol",
    ]
    df_m1 = _read_all_parquets(raw_files, columns=cols)
    if df_m1.empty:
        print(f"[KO] DataFrame M1 vide pour {symbol}")
        return

    df_m1 = df_m1.sort_values("open_time_ms").drop_duplicates(subset=["open_time_ms"], keep="last").reset_index(drop=True)

    der_root = _derived_tf_symbol_root(base_dir, tf.name, symbol)
    meta_path = der_root / "_meta.json"
    ckpt_path = der_root / "_checkpoint.json"
    _ensure_meta(meta_path, tf=tf, symbol=symbol)

    ckpt_next = _load_checkpoint(ckpt_path)
    if ckpt_next is not None:
        df_m1 = df_m1[df_m1["open_time_ms"] >= (ckpt_next)].copy()
        df_m1 = df_m1.sort_values("open_time_ms").reset_index(drop=True)

    if df_m1.empty:
        if show_summary:
            print(f"[OK] Rien à agréger ({symbol}, {tf.name})")
        return

    df_tf = _aggregate_complete_buckets(df_m1, tf=tf)
    if df_tf.empty:
        if show_summary:
            print(f"[OK] Aucun bucket complet à produire ({symbol}, {tf.name})")
        return

    groups = df_tf.groupby([df_tf["ts"].dt.year, df_tf["ts"].dt.month], sort=True)
    n_written = 0
    for (y, m), df_month in groups:
        md = _month_dir(der_root, int(y), int(m))
        _merge_write_month_part(md, df_month.drop(columns=["ts"], errors="ignore"))
        n_written += int(len(df_month))

    last_bucket_open = int(df_tf["open_time_ms"].iloc[-1])
    next_bucket_open = last_bucket_open + tf.n_min * _MS_PER_MIN
    _save_checkpoint(ckpt_path, next_bucket_open)

    if show_summary:
        min_ts = pd.to_datetime(int(df_tf["open_time_ms"].iloc[0]), unit="ms", utc=True)
        max_ts = pd.to_datetime(int(df_tf["open_time_ms"].iloc[-1]), unit="ms", utc=True)
        print(f"[OK] Aggregated {symbol} -> {tf.name} | rows={n_written} | range={min_ts.isoformat()} -> {max_ts.isoformat()}")


def aggregate_symbol_all(base_dir: str, symbol: str, tfs: Optional[List[str]] = None) -> None:
    tf_list = tfs or ["m5", "h1", "h4"]
    for tf_name in tf_list:
        aggregate_symbol_tf(base_dir=base_dir, symbol=symbol, tf_name=tf_name, show_summary=True)


if __name__ == "__main__":
    base_dir = str(Path.cwd())
    aggregate_symbol_all(base_dir=base_dir, symbol="BTCUSDT", tfs=["m5", "h1", "h4"])
