import pandas as pd
import numpy as np

P = r"C:\Users\lolo_\PycharmProjects\binance_futures_data_lake\data\research_debug\BTCUSDT\joined_20260103_20260110__enriched.parquet"

# -----------------------------
# CONFIG (policy + execution)
# -----------------------------
H = 8                 # time-exit horizon bars after entry
ATR_LEN = 14          # ATR lookback
SL_ATR_K = 1.0        # stop = entry + k*ATR (short)
TP_R = 2.0            # take profit in R (e.g. 2R)
DIR_SCORE_ABS_MIN = 0.3

# Policy decisions (v2026-02-13d)
BLOCK_HOURS_UTC = set([0, 1, 14, 17, 19, 23])
FORBID_NEUTRAL = True
BEAR_ONLY = True
ALLOW_RP_MAX = 0.10   # allow_range_pctl = (0.0, 0.10)

# Conservative intrabar rule if both SL and TP touched same bar
# For a SHORT: assume STOP first (worst-case)
CONSERVATIVE_BOTH_TOUCH = True

# -----------------------------
# LOAD + SANITY
# -----------------------------
df = pd.read_parquet(P)

need_cols = ["t","dir_state","dir_score","range_pctl","is_add","open","high","low","close"]
missing = [c for c in need_cols if c not in df.columns]
if missing:
    raise RuntimeError(f"Missing required columns: {missing}")

tt = pd.to_datetime(df["t"], utc=True, errors="coerce")
df["_t"] = tt
df["hour_utc"] = tt.dt.hour
df["date_utc"] = tt.dt.date

# ensure sorted by time (important for shifts)
df = df.sort_values("_t").reset_index(drop=True)

# -----------------------------
# ATR (classic)
# -----------------------------
prev_close = df["close"].shift(1)
tr1 = (df["high"] - df["low"]).abs()
tr2 = (df["high"] - prev_close).abs()
tr3 = (df["low"] - prev_close).abs()
tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
atr = tr.rolling(ATR_LEN, min_periods=ATR_LEN).mean()
df["atr"] = atr

# -----------------------------
# POLICY MASK (v2026-02-13d)
# -----------------------------
ds = df["dir_state"].astype(str)

pass_hours = ~df["hour_utc"].isin(BLOCK_HOURS_UTC)

pass_side = pd.Series(True, index=df.index)
if FORBID_NEUTRAL:
    pass_side = pass_side & (ds != "NEUTRAL")
if BEAR_ONLY:
    pass_side = pass_side & (ds == "BEAR")

score = pd.to_numeric(df["dir_score"], errors="coerce")
pass_score = score.abs() >= DIR_SCORE_ABS_MIN

rp = pd.to_numeric(df["range_pctl"], errors="coerce")
pass_rp = (rp >= 0.0) & (rp < ALLOW_RP_MAX)

df["pass_lowp_20260213d"] = pass_hours & pass_side & pass_score & pass_rp

# -----------------------------
# SIGNALS -> TRADES
# entry on next bar open
# -----------------------------
sig = (df["is_add"] == True) & (df["pass_lowp_20260213d"] == True)
sig_idx = np.flatnonzero(sig.values)

# drop signals too close to end (need next open and H bars forward)
sig_idx = [i for i in sig_idx if (i + 1) < len(df) and (i + H) < len(df)]

# rp buckets for reporting
def rp_bucket(x):
    if pd.isna(x): return "NA"
    if x < 0.02: return "[0.00,0.02)"
    if x < 0.04: return "[0.02,0.04)"
    if x < 0.06: return "[0.04,0.06)"
    if x < 0.08: return "[0.06,0.08)"
    if x < 0.10: return "[0.08,0.10)"
    if x < 0.12: return "[0.10,0.12)"
    return ">=0.12"

df["rp_bucket"] = df["range_pctl"].map(rp_bucket)

trades = []
for i in sig_idx:
    # signal at i
    entry_i = i + 1
    entry_t = df.loc[entry_i, "_t"]
    entry = float(df.loc[entry_i, "open"])
    atr_i = df.loc[entry_i, "atr"]
    if pd.isna(atr_i) or atr_i <= 0:
        continue

    sl = entry + SL_ATR_K * float(atr_i)              # SHORT SL above entry
    risk = sl - entry
    tp = entry - TP_R * risk                          # SHORT TP below entry

    # walk forward bars entry_i .. entry_i+H-1
    exit_t = None
    exit_px = None
    exit_reason = None
    exit_bar = None

    for j in range(entry_i, entry_i + H):
        hi = float(df.loc[j, "high"])
        lo = float(df.loc[j, "low"])

        hit_sl = hi >= sl
        hit_tp = lo <= tp

        if hit_sl and hit_tp:
            if CONSERVATIVE_BOTH_TOUCH:
                exit_px = sl
                exit_reason = "SL_both"
            else:
                exit_px = tp
                exit_reason = "TP_both"
            exit_t = df.loc[j, "_t"]
            exit_bar = j
            break

        if hit_sl:
            exit_px = sl
            exit_reason = "SL"
            exit_t = df.loc[j, "_t"]
            exit_bar = j
            break

        if hit_tp:
            exit_px = tp
            exit_reason = "TP"
            exit_t = df.loc[j, "_t"]
            exit_bar = j
            break

    if exit_reason is None:
        # time exit at close of bar entry_i+H-1
        j = entry_i + H - 1
        exit_px = float(df.loc[j, "close"])
        exit_reason = "TIME"
        exit_t = df.loc[j, "_t"]
        exit_bar = j

    # SHORT PnL in log-return and in R
    # In price terms: pnl = entry - exit
    pnl = entry - float(exit_px)
    r_mult = pnl / risk if risk > 0 else np.nan

    trades.append({
        "sig_i": i,
        "entry_i": entry_i,
        "exit_i": exit_bar,
        "sig_t": df.loc[i, "_t"],
        "entry_t": entry_t,
        "exit_t": exit_t,
        "hour_utc": int(df.loc[i, "hour_utc"]) if not pd.isna(df.loc[i, "hour_utc"]) else None,
        "rp_bucket": df.loc[i, "rp_bucket"],
        "entry": entry,
        "sl": sl,
        "tp": tp,
        "exit": float(exit_px),
        "reason": exit_reason,
        "risk": risk,
        "pnl": pnl,
        "R": float(r_mult),
    })

tr = pd.DataFrame(trades)

print("Config:")
print(f"  H={H}  ATR_LEN={ATR_LEN}  SL_ATR_K={SL_ATR_K}  TP_R={TP_R}")
print(f"  Policy: BEAR_ONLY={BEAR_ONLY}  DIR_SCORE_ABS_MIN={DIR_SCORE_ABS_MIN}  ALLOW_RP_MAX={ALLOW_RP_MAX}  BLOCK_HOURS={sorted(list(BLOCK_HOURS_UTC))}")
print(f"  CONSERVATIVE_BOTH_TOUCH={CONSERVATIVE_BOTH_TOUCH}")

print(f"\nSignals passing policy: {int(sig.sum())}")
print(f"Trades simulated: {len(tr)}")

# -----------------------------
# REPORTING
# -----------------------------
def perf_R(x: pd.Series) -> dict:
    x = x.dropna()
    if len(x) == 0:
        return {"n":0,"wr":np.nan,"mean_R":np.nan,"median_R":np.nan,"pf_R":np.nan,"p05_R":np.nan,"p95_R":np.nan,"min_R":np.nan}
    wr = float((x > 0).mean())
    mean = float(x.mean())
    med = float(x.median())
    p05 = float(x.quantile(0.05))
    p95 = float(x.quantile(0.95))
    mn = float(x.min())
    pos = float(x[x > 0].sum())
    neg = float((-x[x < 0]).sum())
    pf = float(pos / neg) if neg > 0 else np.inf
    return {"n":int(len(x)),"wr":wr,"mean_R":mean,"median_R":med,"pf_R":pf,"p05_R":p05,"p95_R":p95,"min_R":mn}

def print_table(title: str, df_out: pd.DataFrame):
    print(f"\n=== {title} ===")
    print(df_out.to_string(index=False))

# Global
glob = perf_R(tr["R"])
print_table("GLOBAL (R-multiples)", pd.DataFrame([glob]))

# By reason
rows = []
for k, g in tr.groupby("reason"):
    rr = perf_R(g["R"])
    rr["reason"] = k
    rows.append(rr)
print_table("BY EXIT REASON", pd.DataFrame(rows).sort_values("n", ascending=False))

# By hour (only if enough trades per hour)
rows = []
for k, g in tr.groupby("hour_utc"):
    rr = perf_R(g["R"])
    rr["hour_utc"] = k
    rows.append(rr)
print_table("BY HOUR", pd.DataFrame(rows).sort_values("n", ascending=False))

# By rp_bucket
rows = []
for k, g in tr.groupby("rp_bucket"):
    rr = perf_R(g["R"])
    rr["rp_bucket"] = k
    rows.append(rr)
print_table("BY RP_BUCKET", pd.DataFrame(rows).sort_values("n", ascending=False))

# Daily sum of R (rough risk-normalized day PnL)
if len(tr) > 0:
    tr["date_utc"] = pd.to_datetime(tr["sig_t"], utc=True).dt.date
    daily = tr.groupby("date_utc")["R"].sum().reset_index()
    daily["win_day"] = daily["R"] > 0
    print(f"\n=== DAILY (sum R) ===")
    print("days:", len(daily), " mean_daily_R:", float(daily["R"].mean()), " wr_days:", float(daily["win_day"].mean()))
    print(daily.sort_values("date_utc").to_string(index=False))
