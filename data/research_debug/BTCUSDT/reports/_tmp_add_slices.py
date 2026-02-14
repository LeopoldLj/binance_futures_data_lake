import pandas as pd
import numpy as np

P = r"C:\Users\lolo_\PycharmProjects\binance_futures_data_lake\data\research_debug\BTCUSDT\joined_20260103_20260110__enriched.parquet"

df = pd.read_parquet(P)

# --- sanity checks
need_cols = ["t", "dir_state", "dir_score", "range_pctl", "is_add", "close"]
missing = [c for c in need_cols if c not in df.columns]
if missing:
    raise RuntimeError(f"Missing required columns: {missing}. Available cols={list(df.columns)[:50]}...")

# --- detect return cols
cols_lower = {c: c.lower() for c in df.columns}
ret_like = [c for c, cl in cols_lower.items() if ("ret" in cl) or ("return" in cl) or (cl.startswith("sret"))]
print("Return-like columns detected:", ret_like)

# prefer exact names if exist
sret4_col = None
sret8_col = None
for c in df.columns:
    cl = c.lower()
    if cl == "sret_4" or cl == "sret4":
        sret4_col = c
    if cl == "sret_8" or cl == "sret8":
        sret8_col = c

# If missing, rebuild from close
# We'll compute LOG returns: log(close[t+h]/close[t])
# This is consistent, scale-invariant, and safe for BTC.
if sret4_col is None:
    df["sret_4"] = np.log(df["close"].shift(-4) / df["close"])
    sret4_col = "sret_4"
    print("Built sret_4 from close as log-return horizon 4 bars.")
else:
    print(f"Using existing column for sret_4: {sret4_col}")

if sret8_col is None:
    df["sret_8"] = np.log(df["close"].shift(-8) / df["close"])
    sret8_col = "sret_8"
    print("Built sret_8 from close as log-return horizon 8 bars.")
else:
    print(f"Using existing column for sret_8: {sret8_col}")

# Work only on ADD
d = df[df["is_add"] == True].copy()

# hour_utc
t = pd.to_datetime(d["t"], utc=True, errors="coerce")
d["hour_utc"] = t.dt.hour

# range buckets
def bucket(x):
    if pd.isna(x):
        return "NA"
    if x < 0.02:
        return "[0.00,0.02)"
    if x < 0.04:
        return "[0.02,0.04)"
    if x < 0.06:
        return "[0.04,0.06)"
    if x < 0.08:
        return "[0.06,0.08)"
    if x < 0.10:
        return "[0.08,0.10)"
    if x < 0.12:
        return "[0.10,0.12)"
    return ">=0.12"

d["rp_bucket"] = d["range_pctl"].map(bucket)

# aggregation
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

print("\n=== BY SIDE ===")
print(d.groupby("dir_state").apply(agg).reset_index().to_string(index=False))

print("\n=== BY RP_BUCKET ===")
out_bucket = d.groupby("rp_bucket").apply(agg).reset_index()
print(out_bucket.sort_values("rp_bucket").to_string(index=False))

print("\n=== BY HOUR ===")
out_hour = d.groupby("hour_utc").apply(agg).reset_index()
print(out_hour.sort_values("n", ascending=False).to_string(index=False))

# Extra: side x bucket (this is what we actually need)
print("\n=== BY SIDE x RP_BUCKET ===")
out_sb = d.groupby(["dir_state", "rp_bucket"]).apply(agg).reset_index()
print(out_sb.sort_values(["dir_state","rp_bucket"]).to_string(index=False))
