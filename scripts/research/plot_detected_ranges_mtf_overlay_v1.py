from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import pandas as pd


VERSION = "2026-02-15-plot-detected-ranges-mtf-overlay-v1-svg"

TF_RULES = {
    "MN1": "ME",
    "W1": "W-MON",
    "D1": "1D",
    "H4": "4h",
    "H1": "1h",
    "M30": "30min",
}

TF_COLORS = {
    "W1": "#8E44AD",
    "D1": "#2E86C1",
    "H4": "#17A589",
    "H1": "#F39C12",
    "M30": "#C0392B",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot one chart with overlayed range boxes from multiple TFs.")
    p.add_argument("--input", required=True, help="Input parquet with ts/open/high/low/close.")
    p.add_argument("--boxes-csv", required=True, help="Range boxes CSV.")
    p.add_argument("--output-svg", required=True, help="Output SVG file.")
    p.add_argument("--base-tf", default="M30", choices=list(TF_RULES.keys()), help="Base TF for candles.")
    p.add_argument("--overlay-tfs", default="W1,D1,H4,H1,M30", help="Comma-separated TF list to overlay.")
    p.add_argument("--last-bars", type=int, default=240, help="Last N bars of base TF.")
    p.add_argument("--max-boxes-per-tf", type=int, default=8, help="Max boxes per TF.")
    p.add_argument(
        "--style-profile",
        default="balanced",
        choices=["balanced", "global_clean", "micro_precise"],
        help="Visual profile for TF overlays.",
    )
    p.add_argument("--width", type=int, default=1600)
    p.add_argument("--height", type=int, default=760)
    return p.parse_args()


def auto_detect_ts_col(df: pd.DataFrame) -> str:
    for c in ["t_x", "ts", "timestamp", "time", "datetime", "open_time", "open_ts", "time_utc", "ts_utc", "t"]:
        if c in df.columns:
            return c
    dt_cols = [c for c in df.columns if str(df[c].dtype).startswith("datetime64")]
    if dt_cols:
        return dt_cols[0]
    raise RuntimeError("Cannot auto-detect timestamp column.")


def normalize_ts_series(s: pd.Series) -> pd.Series:
    if str(s.dtype).startswith("datetime64"):
        return pd.to_datetime(s, utc=True, errors="coerce").dt.tz_convert(None)
    if pd.api.types.is_integer_dtype(s.dtype):
        return pd.to_datetime(s, unit="ms", utc=True).dt.tz_convert(None)
    return pd.to_datetime(s, utc=True, errors="coerce").dt.tz_convert(None)


def ensure_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    ts_col = auto_detect_ts_col(x)
    x[ts_col] = normalize_ts_series(x[ts_col])
    x = x.dropna(subset=[ts_col]).set_index(ts_col).sort_index()
    x = x[~x.index.duplicated(keep="last")]
    req = ["open", "high", "low", "close"]
    miss = [c for c in req if c not in x.columns]
    if miss:
        raise RuntimeError(f"Missing OHLC columns: {miss}")
    return x[req].copy()


def resample_ohlc(df_ohlc: pd.DataFrame, rule: str) -> pd.DataFrame:
    return (
        df_ohlc.resample(rule, label="left", closed="left")
        .agg({"open": "first", "high": "max", "low": "min", "close": "last"})
        .dropna()
    )


def _esc(s: str) -> str:
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _tf_style(tf: str, profile: str) -> Dict[str, float]:
    # default balanced
    style = {"fill_alpha": 0.10, "stroke_width": 1.4, "dash": ""}
    if profile == "global_clean":
        if tf in {"MN1", "W1"}:
            return {"fill_alpha": 0.03, "stroke_width": 2.0, "dash": "8,4"}
        if tf in {"D1", "H4"}:
            return {"fill_alpha": 0.06, "stroke_width": 1.6, "dash": ""}
        return {"fill_alpha": 0.12, "stroke_width": 1.2, "dash": ""}
    if profile == "micro_precise":
        if tf in {"MN1", "W1"}:
            return {"fill_alpha": 0.02, "stroke_width": 1.8, "dash": "6,4"}
        if tf in {"D1", "H4"}:
            return {"fill_alpha": 0.04, "stroke_width": 1.4, "dash": ""}
        if tf == "H1":
            return {"fill_alpha": 0.12, "stroke_width": 1.5, "dash": ""}
        if tf == "M30":
            return {"fill_alpha": 0.18, "stroke_width": 1.8, "dash": ""}
    return style


def main() -> int:
    args = parse_args()
    in_path = Path(args.input)
    boxes_path = Path(args.boxes_csv)
    out_svg = Path(args.output_svg)
    base_tf = str(args.base_tf).upper()
    overlay_tfs = [x.strip().upper() for x in str(args.overlay_tfs).split(",") if x.strip()]

    if not in_path.exists():
        raise FileNotFoundError(f"Input parquet not found: {in_path}")
    if not boxes_path.exists():
        raise FileNotFoundError(f"Boxes CSV not found: {boxes_path}")
    bad = [tf for tf in [base_tf] + overlay_tfs if tf not in TF_RULES]
    if bad:
        raise RuntimeError(f"Unsupported TF(s): {bad}. Allowed={list(TF_RULES.keys())}")

    print(f"[plot_detected_ranges_mtf_overlay_v1] VERSION={VERSION}")
    print(f"[INFO] input={in_path}")
    print(f"[INFO] boxes_csv={boxes_path}")
    print(f"[INFO] base_tf={base_tf} overlay_tfs={overlay_tfs}")
    print(f"[INFO] style_profile={args.style_profile}")

    raw = pd.read_parquet(in_path, engine="pyarrow")
    base = ensure_ohlc(raw)
    candles = resample_ohlc(base, TF_RULES[base_tf]).tail(int(args.last_bars)).copy()
    if candles.empty:
        raise RuntimeError("No candle data in selected window.")

    boxes = pd.read_csv(boxes_path)
    for c in ["start_ts", "end_ts"]:
        if c in boxes.columns:
            boxes[c] = pd.to_datetime(boxes[c], utc=True, errors="coerce").dt.tz_convert(None)
    boxes = boxes.dropna(subset=["start_ts", "end_ts"])

    v_start = candles.index.min()
    v_end = candles.index.max()

    width = int(args.width)
    height = int(args.height)
    m_left, m_right, m_top, m_bot = 78, 18, 45, 56
    pw = width - m_left - m_right
    ph = height - m_top - m_bot

    x0 = candles.index.min().value
    x1 = candles.index.max().value
    y_min = float(candles["low"].min())
    y_max = float(candles["high"].max())
    if not boxes.empty:
        vb = boxes[(boxes["end_ts"] >= v_start) & (boxes["start_ts"] <= v_end)]
        if not vb.empty:
            y_min = min(y_min, float(vb["range_low"].min()))
            y_max = max(y_max, float(vb["range_high"].max()))
    if y_max <= y_min:
        y_max = y_min + 1.0

    def xmap(ts: pd.Timestamp) -> float:
        if x1 == x0:
            return float(m_left)
        return m_left + ((ts.value - x0) / (x1 - x0)) * pw

    def ymap(v: float) -> float:
        return m_top + (1.0 - ((v - y_min) / (y_max - y_min))) * ph

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect x="0" y="0" width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{m_left}" y="24" font-family="Arial" font-size="17" fill="#111">MTF Ranges Overlay ({_esc(base_tf)} candles)</text>',
        f'<rect x="{m_left}" y="{m_top}" width="{pw}" height="{ph}" fill="#fafafa" stroke="#d0d0d0"/>',
    ]

    for i in range(6):
        y = m_top + (i / 5.0) * ph
        pv = y_max - (i / 5.0) * (y_max - y_min)
        parts.append(f'<line x1="{m_left}" y1="{y:.2f}" x2="{m_left + pw}" y2="{y:.2f}" stroke="#e6e6e6" stroke-width="1"/>')
        parts.append(f'<text x="8" y="{y + 4:.2f}" font-family="Arial" font-size="11" fill="#666">{pv:,.0f}</text>')

    # Overlay boxes by TF
    for tf in overlay_tfs:
        color = TF_COLORS.get(tf, "#555")
        style = _tf_style(tf, profile=str(args.style_profile))
        sub = boxes[boxes["tf"] == tf].copy()
        sub = sub[(sub["end_ts"] >= v_start) & (sub["start_ts"] <= v_end)]
        if sub.empty:
            continue
        if "strength" in sub.columns:
            sub = sub.sort_values("strength", ascending=False)
        sub = sub.head(int(args.max_boxes_per_tf))

        for _, r in sub.iterrows():
            st = max(pd.Timestamp(r["start_ts"]), v_start)
            en = min(pd.Timestamp(r["end_ts"]), v_end)
            xl = xmap(st)
            xr = xmap(en)
            yt = ymap(float(r["range_high"]))
            yb = ymap(float(r["range_low"]))
            h = max(1.0, yb - yt)
            parts.append(
                f'<rect x="{xl:.2f}" y="{yt:.2f}" width="{max(1.0, xr-xl):.2f}" height="{h:.2f}" fill="{color}" fill-opacity="{style["fill_alpha"]:.3f}" stroke="{color}" stroke-width="{style["stroke_width"]:.2f}"'
                f' stroke-dasharray="{style["dash"]}"/>'
            )

    # Candles
    n = len(candles)
    dx = pw / (n - 1) if n > 1 else pw
    body_w = max(1.0, min(8.0, dx * 0.55))
    for ts, row in candles.iterrows():
        x = xmap(ts)
        o = float(row["open"])
        h = float(row["high"])
        l = float(row["low"])
        c = float(row["close"])
        yh = ymap(h)
        yl = ymap(l)
        yo = ymap(o)
        yc = ymap(c)
        bullish = c >= o
        col = "#1e8449" if bullish else "#c0392b"
        parts.append(f'<line x1="{x:.2f}" y1="{yh:.2f}" x2="{x:.2f}" y2="{yl:.2f}" stroke="{col}" stroke-width="1"/>')
        y_top = min(yo, yc)
        y_bot = max(yo, yc)
        h_body = max(1.0, y_bot - y_top)
        parts.append(
            f'<rect x="{x - body_w/2:.2f}" y="{y_top:.2f}" width="{body_w:.2f}" height="{h_body:.2f}" fill="{col}" fill-opacity="0.85" stroke="{col}" stroke-width="1"/>'
        )

    # X-axis labels
    n_ticks = 8
    for i in range(n_ticks + 1):
        frac = i / n_ticks
        idx = int(frac * (len(candles) - 1))
        ts = candles.index[idx]
        x = xmap(ts)
        parts.append(f'<line x1="{x:.2f}" y1="{m_top + ph}" x2="{x:.2f}" y2="{m_top + ph + 5}" stroke="#666" stroke-width="1"/>')
        parts.append(f'<text x="{x - 36:.2f}" y="{m_top + ph + 20}" font-family="Arial" font-size="11" fill="#555">{ts.strftime("%Y-%m-%d")}</text>')

    # Legend by TF
    lx = m_left + 8
    ly = m_top + 16
    for tf in overlay_tfs:
        color = TF_COLORS.get(tf, "#555")
        parts.append(f'<rect x="{lx}" y="{ly-8}" width="18" height="10" fill="{color}" fill-opacity="0.20" stroke="{color}" stroke-width="1.4"/>')
        parts.append(f'<text x="{lx+24}" y="{ly}" font-family="Arial" font-size="11" fill="#444">Range {tf}</text>')
        ly += 16

    parts.append("</svg>")
    out_svg.parent.mkdir(parents=True, exist_ok=True)
    out_svg.write_text("\n".join(parts), encoding="utf-8")
    print(f"[OK] chart={out_svg}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
