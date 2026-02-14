import pandas as pd
import numpy as np

P = r"C:\Users\lolo_\PycharmProjects\binance_futures_data_lake\data\research_debug\BTCUSDT\joined_20260103_20260110__enriched.parquet"
df = pd.read_parquet(P)

need_cols = ["t","dir_state","dir_score","range_pctl","is_add","close"]
missing = [c for c in need_cols if c not in df.columns]
if missing:
    raise RuntimeError(f"Missing required columns: {missing}")

# --- returns (log)
sret4_col = next((c for c in df.columns if c.lower() in ["sret_4","sret4"]), None)
sret8_col = next((c for c in df.columns if c.lower() in ["sret_8","sret8"]), None)

if sret4_col is None:
    df["sret_4"] = np.log(df["close"].shift(-4) / df["close"])
    sret4_col = "sret_4"
    print("Built sret_4 from close (log-return, horizon 4).")
else:
    print(f"Using existing {sret4_col} for sret_4.")

if sret8_col is None:
    df["sret_8"] = np.log(df["close"].shift(-8) / df["close"])
    sret8_col = "sret_8"
    print("Built sret_8 from close (log-return, horizon 8).")
else:
    print(f"Using existing {sret8_col} for sret_8.")

# --- LowPlusPolicy v2026-02-13c
# allow_hours_utc=None
block_hours_utc = set([1,14,17,19,23])
bull_only = False
bear_only = False
forbid_neutral = True
dir_score_min = None
dir_score_max = None
dir_score_abs_min = 0.3
dir_score_abs_max = None
allow_range_pctl = [(0.0, 0.3)]
block_range_pctl = [(0.12, 0.14), (0.2, 0.22)]

# hour
tt = pd.to_datetime(df["t"], utc=True, errors="coerce")
hour_utc = tt.dt.hour

pass_hours = ~hour_utc.isin(block_hours_utc)

# side
ds = df["dir_state"].astype(str)
pass_side = pd.Series(True, index=df.index)
if forbid_neutral:
    pass_side = pass_side & (ds != "NEUTRAL")
if bull_only:
    pass_side = pass_side & (ds == "BULL")
if bear_only:
    pass_side = pass_side & (ds == "BEAR")

# dir_score
score = pd.to_numeric(df["dir_score"], errors="coerce")
pass_score = pd.Series(True, index=df.index)
if dir_score_min is not None:
    pass_score = pass_score & (score >= dir_score_min)
if dir_score_max is not None:
    pass_score = pass_score & (score <= dir_score_max)
abs_score = score.abs()
if dir_score_abs_min is not None:
    pass_score = pass_score & (abs_score >= dir_score_abs_min)
if dir_score_abs_max is not None:
    pass_score = pass_score & (abs_score <= dir_score_abs_max)

# range_pctl allow/block
rp = pd.to_numeric(df["range_pctl"], errors="coerce")
pass_rp_allow = pd.Series(False, index=df.index)
for lo, hi in allow_range_pctl:
    pass_rp_allow = pass_rp_allow | ((rp >= lo) & (rp < hi))

pass_rp_block = pd.Series(True, index=df.index)
for lo, hi in block_range_pctl:
    pass_rp_block = pass_rp_block & ~((rp >= lo) & (rp < hi))

pass_lowp = pass_hours & pass_side & pass_score & pass_rp_allow & pass_rp_block
df["pass_lowp_20260213c"] = pass_lowp

# filter: ADD + policy pass
d = df[(df["is_add"] == True) & (df["pass_lowp_20260213c"] == True)].copy()

print(f"\nn_add_policy = {len(d)} (is_add & pass_lowp_20260213c)")

# hour on filtered
t2 = pd.to_datetime(d["t"], utc=True, errors="coerce")
d["hour_utc"] = t2.dt.hour

# buckets
def bucket(x):
    if pd.isna(x): return "NA"
    if x < 0.02: return "[0.00,0.02)"
    if x < 0.04: return "[0.02,0.04)"
    if x < 0.06: return "[0.04,0.06)"
    if x < 0.08: return "[0.06,0.08)"
    if x < 0.10: return "[0.08,0.10)"
    if x < 0.12: return "[0.10,0.12)"
    return ">=0.12"

d["rp_bucket"] = d["range_pctl"].map(bucket)

def agg(g):
    s8 = g[sret8_col].dropna()
    s4 = g[sret4_col].dropna()
    return pd.Series({
        "n": int(len(g)),
        "mean_sret8": float(s8.mean()) if len(s8) else float("nan"),
        "wr_sret8": float((s8 > 0).mean()) if len(s8) else float("nan"),
        "mean_sret4": float(s4.mean()) if len(s4) else float("nan"),
        "std_sret8": float(s8.std(ddof=1)) if len(s8) > 1 else float("nan"),
    })

print("\n=== BY SIDE (policy-filtered) ===")
print(d.groupby("dir_state").apply(agg).reset_index().to_string(index=False))

print("\n=== BY RP_BUCKET (policy-filtered) ===")
out_bucket = d.groupby("rp_bucket").apply(agg).reset_index()
print(out_bucket.sort_values("rp_bucket").to_string(index=False))

print("\n=== BY HOUR (policy-filtered) ===")
out_hour = d.groupby("hour_utc").apply(agg).reset_index()
print(out_hour.sort_values("n", ascending=False).to_string(index=False))

print("\n=== BY SIDE x RP_BUCKET (policy-filtered) ===")
out_sb = d.groupby(["dir_state","rp_bucket"]).apply(agg).reset_index()
print(out_sb.sort_values(["dir_state","rp_bucket"]).to_string(index=False))
