import argparse
from dataclasses import dataclass
from pathlib import Path
import pandas as pd


VERSION = "2026-02-13c"


@dataclass(frozen=True)
class EnrichConfig:
    th: float
    dir_abs_min: float


def _require_cols(df: pd.DataFrame, cols: list[str], ctx: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"[{ctx}] missing required cols: {missing}. cols={list(df.columns)}")


def enrich(df: pd.DataFrame, cfg: EnrichConfig) -> pd.DataFrame:
    _require_cols(df, ["t", "dir_state", "dir_score", "vol_state", "range_pctl"], "enrich")

    out = df.copy()

    # LOW+ is purely a range_pctl threshold
    out["low_plus"] = (out["range_pctl"] <= cfg.th)

    # Base market readiness: classic gating (LOW blocks baseline)
    out["market_ready_base"] = out["vol_state"].isin(["MID", "HIGH"])

    # Direction readiness (simple default)
    dir_abs = out["dir_score"].abs()
    dir_non_neutral = out["dir_state"].astype(str).str.upper().ne("NEUTRAL")
    out["dir_ready"] = dir_non_neutral & (dir_abs >= cfg.dir_abs_min)

    # Baseline tradable (what you'd trade WITHOUT LOW+)
    out["tradable_base"] = out["market_ready_base"] & out["dir_ready"]

    # LOW+ override tradable (what you'd trade ONLY because LOW+ is on)
    out["tradable_override"] = out["low_plus"] & out["dir_ready"]

    # Final tradable (router output if you allow the override)
    out["tradable_final"] = out["tradable_base"] | out["tradable_override"]

    # True ADD = override-only (delta vs baseline)
    out["is_add"] = out["tradable_override"] & ~out["tradable_base"]

    # Reasons (debug)
    out["market_ready_reason"] = out["market_ready_base"].map({True: "VOL_OK_BASE", False: "VOL_BLOCKED_BASE"})
    out["override_reason"] = ""
    out.loc[out["low_plus"] & out["dir_ready"], "override_reason"] = "LOW_PLUS_AND_DIR_READY"
    out.loc[out["low_plus"] & ~out["dir_ready"], "override_reason"] = "LOW_PLUS_BUT_DIR_NOT_READY"

    out["block_reason_base"] = ""
    out.loc[~out["dir_ready"], "block_reason_base"] = "DIR_NOT_READY"
    out.loc[out["dir_ready"] & ~out["market_ready_base"], "block_reason_base"] = "VOL_BLOCKED_BASE"

    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Enrich a joined parquet with recomputed low_plus and baseline/override tradability (ADD diagnostics).")
    ap.add_argument("--joined", required=True, help="Path to joined parquet.")
    ap.add_argument("--out", required=True, help="Output parquet path.")
    ap.add_argument("--th", type=float, default=0.14, help="LOW+ threshold on range_pctl (default: 0.14).")
    ap.add_argument("--dir-abs-min", type=float, default=0.30, help="Min abs(dir_score) for dir_ready (default: 0.30).")
    args = ap.parse_args()

    joined_path = Path(args.joined)
    out_path = Path(args.out)

    print(f"[enrich_joined_low_plus] VERSION={VERSION}")
    print(f"[INFO] joined={joined_path}")
    print(f"[INFO] out={out_path}")
    print(f"[INFO] th={args.th} dir_abs_min={args.dir_abs_min}")

    df = pd.read_parquet(joined_path)

    cfg = EnrichConfig(
        th=args.th,
        dir_abs_min=args.dir_abs_min,
    )
    out = enrich(df, cfg)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False)

    print("[OK] wrote:", out_path)
    print("[STATS] market_ready_base_rate=", float(out["market_ready_base"].mean()))
    print("[STATS] low_plus_rate=", float(out["low_plus"].mean()))
    print("[STATS] tradable_base_rate=", float(out["tradable_base"].mean()))
    print("[STATS] tradable_override_rate=", float(out["tradable_override"].mean()))
    print("[STATS] tradable_final_rate=", float(out["tradable_final"].mean()))
    print("[STATS] is_add_rate=", float(out["is_add"].mean()))


if __name__ == "__main__":
    main()
