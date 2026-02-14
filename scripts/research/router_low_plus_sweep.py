# scripts/research/router_low_plus_filters.py
# VERSION=2026-02-12a
# Full LOW+ gating logic driven by your reports:
# - hours: from ADD_by_hour.csv
# - side:  from ADD_by_side.csv (BULL only)
# - dir_score: from ADD_by_dirscore.csv (keep only positive bin)
# - range_pctl: from ADD_by_rangepctl_bucket.csv (exclude worst bucket, prefer 0.14-0.20)

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Set, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class LowPlusConfig:
    allow_hours_utc: Set[int] = None
    dir_score_min: float = 0.299
    bull_only: bool = True
    forbid_neutral: bool = True

    # Range percentile gating:
    # - allow_range_pctl: union of allowed intervals (lo, hi), hi exclusive by default
    # - block_range_pctl: union of blocked intervals (lo, hi)
    allow_range_pctl: Tuple[Tuple[float, float], ...] = ((0.14, 0.20),)
    block_range_pctl: Tuple[Tuple[float, float], ...] = ((0.20, 0.22),)

    def __post_init__(self):
        if self.allow_hours_utc is None:
            object.__setattr__(self, "allow_hours_utc", {2, 4, 11, 13, 17})


def _to_utc_hour(series_t: pd.Series) -> pd.Series:
    t = pd.to_datetime(series_t, utc=True, errors="coerce")
    return t.dt.hour


def _as_bool(s: pd.Series) -> pd.Series:
    # accepts bool, 0/1, "true"/"false"
    if s.dtype == bool:
        return s
    if pd.api.types.is_numeric_dtype(s):
        return s.fillna(0).astype(int).astype(bool)
    return s.astype("string").str.lower().isin(["1", "true", "t", "yes", "y"])


def _in_any_interval(x: pd.Series, intervals: Iterable[Tuple[float, float]]) -> pd.Series:
    # intervals are [lo, hi) by default
    mask = pd.Series(False, index=x.index)
    for lo, hi in intervals:
        mask = mask | (x.ge(lo) & x.lt(hi))
    return mask


def apply_low_plus_override_filter(j: pd.DataFrame, cfg: LowPlusConfig | None = None) -> pd.DataFrame:
    """
    Expects columns:
      - t (datetime-like or str)        : timestamp (UTC preferred)
      - dir_state (str)                : {"BULL","BEAR","NEUTRAL"}
      - dir_score (float)
      - range_pctl (float)             : 0..1
      - market_ready (bool or 0/1)     : baseline readiness
      - low_plus (bool or 0/1)         : candidate LOW+ override set

    Produces:
      - hour_utc (int)
      - low_plus_allow (bool)
      - market_ready_override (bool)
      - low_plus_block_reason (string) : first failing reason (debug)
    """
    if cfg is None:
        cfg = LowPlusConfig()

    jj = j.copy()

    # Parse / coerce
    jj["hour_utc"] = _to_utc_hour(jj["t"])
    market_ready = _as_bool(jj["market_ready"])
    low_plus = _as_bool(jj["low_plus"])

    dir_state = jj["dir_state"].astype("string")
    dir_score = pd.to_numeric(jj["dir_score"], errors="coerce")
    range_pctl = pd.to_numeric(jj["range_pctl"], errors="coerce")

    # Gates
    g_hour = jj["hour_utc"].isin(cfg.allow_hours_utc)

    g_not_neutral = dir_state.ne("NEUTRAL") if cfg.forbid_neutral else pd.Series(True, index=jj.index)
    g_side = dir_state.eq("BULL") if cfg.bull_only else dir_state.isin(["BULL", "BEAR"])
    g_dir = dir_score.ge(cfg.dir_score_min)

    # Range gating
    g_range_allow = _in_any_interval(range_pctl, cfg.allow_range_pctl)
    g_range_block = _in_any_interval(range_pctl, cfg.block_range_pctl)
    g_range = g_range_allow & (~g_range_block)

    low_plus_allow = low_plus & g_hour & g_not_neutral & g_side & g_dir & g_range

    # Debug reason (first failing rule)
    reason = pd.Series("", index=jj.index, dtype="string")
    reason = reason.mask(low_plus & ~g_hour, "hour")
    reason = reason.mask(low_plus & g_hour & ~g_not_neutral, "neutral")
    reason = reason.mask(low_plus & g_hour & g_not_neutral & ~g_side, "side")
    reason = reason.mask(low_plus & g_hour & g_not_neutral & g_side & ~g_dir, "dir_score")
    reason = reason.mask(low_plus & g_hour & g_not_neutral & g_side & g_dir & ~g_range_allow, "range_not_allowed")
    reason = reason.mask(low_plus & g_hour & g_not_neutral & g_side & g_dir & g_range_allow & g_range_block, "range_blocked")

    jj["low_plus_allow"] = low_plus_allow
    jj["low_plus_block_reason"] = reason

    # Override
    jj["market_ready_override"] = market_ready | low_plus_allow

    return jj


# -----------------------------
# Minimal self-check helper
# -----------------------------
def summarize_low_plus_gating(jj: pd.DataFrame) -> pd.DataFrame:
    # quick counts by reason to see what you are filtering out
    tmp = jj.loc[jj["low_plus"].astype(bool)].copy()
    out = (
        tmp.groupby("low_plus_block_reason", dropna=False)
        .size()
        .rename("n")
        .reset_index()
        .sort_values("n", ascending=False)
    )
    return out
