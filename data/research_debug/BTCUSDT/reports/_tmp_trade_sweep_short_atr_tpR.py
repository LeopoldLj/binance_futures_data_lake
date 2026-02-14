import pandas as pd
import numpy as np

P = r"C:\Users\lolo_\PycharmProjects\binance_futures_data_lake\data\research_debug\BTCUSDT\joined_20260103_20260110__enriched.parquet"

# -----------------------------
# POLICY (fixed) v2026-02-13d
# -----------------------------
DIR_SCORE_ABS_MIN = 0.3
BLOCK_HOURS_UTC = set([0, 1, 14, 17, 19, 23])
FORBID_NEUTRAL = True
BEAR_ONLY = True
ALLOW_RP_MAX = 0.10
CONSERVATIVE_BOTH_TOUCH = True

# -----------------------------
# SWEEP GRID
# -----------------------------
ATR_LEN = 14
H_LIST = [8, 12, 16]
SL_K_LIST = [1.0, 1.5, 2.0]
TP_R_LIST = [1.5, 2.0, 2.5]

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
df = df.sort_values("_t").reset_index(drop=True)

# -----------------------------
# ATR
# -----------------------------
prev_close = df["close"].shift(1)
tr1 = (df["high"] - df["low"]).abs()
tr2 = (df["high"] - prev_close).abs()
tr3 = (df["low"] - prev_close).abs()
tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
df["atr"] = tr.rolling(ATR_LEN, min_periods=ATR_LEN).mean()

# -----------------------------
# POLICY MASK
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

df["pass_lowp"] = pass_hours & pass_side & pass_score & pass_rp

sig = (df["is_add"] == True) & (df["pass_lowp"] == True)
sig_idx_all = np.flatnonzero(sig.values)

def perf_R(x: pd.Series) -> dict:
    x = x.dropna()
    if len(x) == 0:
        return {"n":0,"wr":np.nan,"mean_R":np.nan,"pf_R":np.nan,"p05_R":np.nan,"p95_R":np.nan,"min_R":np.nan}
    wr = float((x > 0).mean())
    mean = float(x.mean())
    p05 = float(x.quantile(0.05))
    p95 = float(x.quantile(0.95))
    mn = float(x.min())
    pos = float(x[x > 0].sum())
    neg = float((-x[x < 0]).sum())
    pf = float(pos / neg) if neg > 0 else np.inf
    return {"n":int(len(x)),"wr":wr,"mean_R":mean,"pf_R":pf,"p05_R":p05,"p95_R":p95,"min_R":mn}

def run_one(H: int, sl_k: float, tp_r: float):
    sig_idx = [i for i in sig_idx_all if (i + 1) < len(df) and (i + H) < len(df)]
    Rs = []
    n_sl = 0
    n_tp = 0
    n_time = 0

    for i in sig_idx:
        entry_i = i + 1
        entry = float(df.loc[entry_i, "open"])
        atr_i = df.loc[entry_i, "atr"]
        if pd.isna(atr_i) or atr_i <= 0:
            continue

        sl = entry + sl_k * float(atr_i)
        risk = sl - entry
        tp = entry - tp_r * risk

        reason = None
        exit_px = None

        for j in range(entry_i, entry_i + H):
            hi = float(df.loc[j, "high"])
            lo = float(df.loc[j, "low"])
            hit_sl = hi >= sl
            hit_tp = lo <= tp

            if hit_sl and hit_tp:
                if CONSERVATIVE_BOTH_TOUCH:
                    reason = "SL_both"
                    exit_px = sl
                else:
                    reason = "TP_both"
                    exit_px = tp
                break
            if hit_sl:
                reason = "SL"
                exit_px = sl
                break
            if hit_tp:
                reason = "TP"
                exit_px = tp
                break

        if reason is None:
            reason = "TIME"
            j = entry_i + H - 1
            exit_px = float(df.loc[j, "close"])

        pnl = entry - float(exit_px)
        R = pnl / risk if risk > 0 else np.nan
        Rs.append(R)

        if reason.startswith("SL"):
            n_sl += 1
        elif reason.startswith("TP"):
            n_tp += 1
        else:
            n_time += 1

    s = pd.Series(Rs, dtype="float64")
    r = perf_R(s)
    r.update({"H":H,"SL_K":sl_k,"TP_R":tp_r,"n_sl":n_sl,"n_tp":n_tp,"n_time":n_time})
    return r

rows = []
for H in H_LIST:
    for sl_k in SL_K_LIST:
        for tp_r in TP_R_LIST:
            rows.append(run_one(H, sl_k, tp_r))

out = pd.DataFrame(rows)

# Rank: primary mean_R, secondary pf_R, then n desc
out = out.sort_values(["mean_R","pf_R","n"], ascending=[False,False,False])

print("Policy fixed:")
print(f"  BEAR_ONLY={BEAR_ONLY}  DIR_SCORE_ABS_MIN={DIR_SCORE_ABS_MIN}  ALLOW_RP_MAX={ALLOW_RP_MAX}  BLOCK_HOURS={sorted(list(BLOCK_HOURS_UTC))}")
print(f"Signals passing policy (raw): {int(sig.sum())}")

print("\n=== TOP 15 CONFIGS ===")
print(out.head(15).to_string(index=False))

print("\n=== FULL TABLE (sorted) ===")
print(out.to_string(index=False))
