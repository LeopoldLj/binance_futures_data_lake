import pandas as pd
import numpy as np

P = r"C:\Users\lolo_\PycharmProjects\binance_futures_data_lake\data\research_debug\BTCUSDT\joined_20260103_20260110__enriched.parquet"

H = 8  # horizon bars
DIR_STATE_BEAR = "BEAR"
DIR_STATE_BULL = "BULL"
DIR_STATE_NEUTRAL = "NEUTRAL"

# --- Load
df = pd.read_parquet(P)

need_cols = ["t","dir_state","dir_score","range_pctl","is_add","close"]
missing = [c for c in need_cols if c not in df.columns]
if missing:
    raise RuntimeError(f"Missing required columns: {missing}")

# --- Ensure datetime + hour/day
tt = pd.to_datetime(df["t"], utc=True, errors="coerce")
df["_t"] = tt
df["hour_utc"] = tt.dt.hour
df["date_utc"] = tt.dt.date

# --- Build returns if absent (log returns)
sret_col = None
if "sret_8" in df.columns:
    sret_col = "sret_8"
else:
    df["sret_8"] = np.log(df["close"].shift(-H) / df["close"])
    sret_col = "sret_8"

# Optional: also compute sret_4 for diagnostics
if "sret_4" not in df.columns:
    df["sret_4"] = np.log(df["close"].shift(-4) / df["close"])

# --- LowPlusPolicy v2026-02-13c (base), with your refined decisions
# Base policy parameters
block_hours_utc_base = set([1,14,17,19,23])
forbid_neutral = True
dir_score_abs_min = 0.3

# Refined proposal knobs
ADD_BLOCK_HOUR_0 = True  # kill-switch hour 0
BEAR_ONLY = True         # short-only decision from analysis

# --- Apply base pass mask (policy)
pass_hours = ~df["hour_utc"].isin(block_hours_utc_base)
if ADD_BLOCK_HOUR_0:
    pass_hours = pass_hours & (df["hour_utc"] != 0)

ds = df["dir_state"].astype(str)
pass_side = pd.Series(True, index=df.index)
if forbid_neutral:
    pass_side = pass_side & (ds != DIR_STATE_NEUTRAL)

if BEAR_ONLY:
    pass_side = pass_side & (ds == DIR_STATE_BEAR)

score = pd.to_numeric(df["dir_score"], errors="coerce")
pass_score = score.abs() >= dir_score_abs_min

# NOTE: old allow_range_pctl was (0.0,0.3) with block micro-windows.
# Here we evaluate 2 clean variants: <0.10 and <0.08 (no micro blocks).
rp = pd.to_numeric(df["range_pctl"], errors="coerce")

# --- Buckets for reporting
def rp_bucket(x):
    if pd.isna(x): return "NA"
    if x < 0.02: return "[0.00,0.02)"
    if x < 0.04: return "[0.02,0.04)"
    if x < 0.06: return "[0.04,0.06)"
    if x < 0.08: return "[0.06,0.08)"
    if x < 0.10: return "[0.08,0.10)"
    if x < 0.12: return "[0.10,0.12)"
    return ">=0.12"

df["rp_bucket"] = rp.map(rp_bucket)

# --- Helper metrics
def perf_table(x: pd.Series) -> dict:
    x = x.dropna()
    if len(x) == 0:
        return {"n":0,"mean":np.nan,"median":np.nan,"wr":np.nan,"pf":np.nan,"std":np.nan,"p05":np.nan,"p95":np.nan}
    mean = float(x.mean())
    med = float(x.median())
    wr = float((x > 0).mean())
    std = float(x.std(ddof=1)) if len(x) > 1 else np.nan
    p05 = float(x.quantile(0.05))
    p95 = float(x.quantile(0.95))
    pos_sum = float(x[x > 0].sum())
    neg_sum = float((-x[x < 0]).sum())
    pf = float(pos_sum / neg_sum) if neg_sum > 0 else np.inf
    return {"n":int(len(x)),"mean":mean,"median":med,"wr":wr,"pf":pf,"std":std,"p05":p05,"p95":p95}

def report(name: str, dsub: pd.DataFrame):
    r = perf_table(dsub[sret_col])
    print(f"\n=== {name} ===")
    print(pd.DataFrame([r]).to_string(index=False))

def report_group(name: str, dsub: pd.DataFrame, group_col: str, sort_by: str = "n"):
    print(f"\n=== {name} (group={group_col}) ===")
    rows = []
    for k, g in dsub.groupby(group_col):
        rr = perf_table(g[sret_col])
        rr[group_col] = k
        rows.append(rr)
    out = pd.DataFrame(rows)
    if sort_by in out.columns:
        out = out.sort_values(sort_by, ascending=False)
    print(out.to_string(index=False))

def build_signals(rp_max: float, tag: str):
    pass_rp = (rp >= 0.0) & (rp < rp_max)
    mask = (df["is_add"] == True) & pass_hours & pass_side & pass_score & pass_rp
    dsig = df[mask].copy()
    dsig["variant"] = tag
    return dsig

# --- Build 2 variants
v1 = build_signals(0.10, "P1_rp_lt_0.10")
v2 = build_signals(0.08, "P2_rp_lt_0.08")

print("Config:")
print(f"  Horizon H={H}")
print(f"  BEAR_ONLY={BEAR_ONLY}")
print(f"  dir_score_abs_min={dir_score_abs_min}")
print(f"  block_hours_base={sorted(block_hours_utc_base)}")
print(f"  ADD_BLOCK_HOUR_0={ADD_BLOCK_HOUR_0}")

print(f"\nSignals count: P1={len(v1)}  P2={len(v2)}")

# --- Global reports
report("P1 GLOBAL", v1)
report("P2 GLOBAL", v2)

# --- Side split (should be BEAR only, but keep report anyway)
report_group("P1 BY SIDE", v1, "dir_state")
report_group("P2 BY SIDE", v2, "dir_state")

# --- Range buckets
report_group("P1 BY RP_BUCKET", v1, "rp_bucket")
report_group("P2 BY RP_BUCKET", v2, "rp_bucket")

# --- Hours
report_group("P1 BY HOUR", v1, "hour_utc")
report_group("P2 BY HOUR", v2, "hour_utc")

# --- Daily aggregation (sum of returns per day + daily winrate)
def daily_report(name: str, dsub: pd.DataFrame):
    if len(dsub) == 0:
        print(f"\n=== {name} DAILY ===\n(no signals)")
        return
    daily = dsub.groupby("date_utc")[sret_col].sum().reset_index()
    daily["win_day"] = daily[sret_col] > 0
    print(f"\n=== {name} DAILY ===")
    print("days:", len(daily), " mean_daily_sum:", float(daily[sret_col].mean()), " wr_days:", float(daily['win_day'].mean()))
    print(daily.sort_values("date_utc").to_string(index=False))

daily_report("P1", v1)
daily_report("P2", v2)

# --- Bonus: Compare with "all sides" (BEAR_ONLY False), to quantify damage of allowing BULL
BEAR_ONLY_2 = False
pass_side2 = pd.Series(True, index=df.index)
if forbid_neutral:
    pass_side2 = pass_side2 & (ds != DIR_STATE_NEUTRAL)

def build_signals_all_sides(rp_max: float, tag: str):
    pass_rp = (rp >= 0.0) & (rp < rp_max)
    mask = (df["is_add"] == True) & pass_hours & pass_side2 & pass_score & pass_rp
    dsig = df[mask].copy()
    dsig["variant"] = tag
    return dsig

v1_all = build_signals_all_sides(0.10, "P1_ALLSIDES_rp_lt_0.10")
v2_all = build_signals_all_sides(0.08, "P2_ALLSIDES_rp_lt_0.08")

report("P1 ALL SIDES", v1_all)
report("P2 ALL SIDES", v2_all)
report_group("P1 ALL SIDES BY SIDE", v1_all, "dir_state")
report_group("P2 ALL SIDES BY SIDE", v2_all, "dir_state")
