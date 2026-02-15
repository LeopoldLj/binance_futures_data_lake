from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


VERSION = "2026-02-15-plot-detected-ranges-v1-svg"

TF_RULES = {
    "MN1": "ME",
    "W1": "W-MON",
    "D1": "1D",
    "H4": "4h",
    "H1": "1h",
    "M30": "30min",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot detected range boxes as SVG (no matplotlib dependency).")
    p.add_argument("--input", required=True, help="Input parquet with ts/open/high/low/close.")
    p.add_argument("--boxes-csv", required=True, help="Range boxes CSV from build_range_boxes_from_flats_v2.py")
    p.add_argument("--output-dir", required=True, help="Output folder for SVG charts.")
    p.add_argument("--tfs", default="D1,H4,H1,M30", help="Comma-separated TF list.")
    p.add_argument("--last-bars", type=int, default=220, help="Plot only last N bars per TF.")
    p.add_argument("--max-boxes", type=int, default=12, help="Max boxes to overlay per TF.")
    p.add_argument("--width", type=int, default=1400)
    p.add_argument("--height", type=int, default=620)
    p.add_argument("--kijun-len", type=int, default=26)
    p.add_argument("--senkou-b-len", type=int, default=52)
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


def compute_ichimoku_lines(df_tf: pd.DataFrame, kijun_len: int, senkou_b_len: int) -> pd.DataFrame:
    h = df_tf["high"]
    l = df_tf["low"]
    kijun = (h.rolling(kijun_len, min_periods=kijun_len).max() + l.rolling(kijun_len, min_periods=kijun_len).min()) / 2.0
    ssb = (h.rolling(senkou_b_len, min_periods=senkou_b_len).max() + l.rolling(senkou_b_len, min_periods=senkou_b_len).min()) / 2.0
    return pd.DataFrame({"kijun": kijun, "ssb": ssb}, index=df_tf.index)


def _esc(s: str) -> str:
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def render_svg(
    tf: str,
    view: pd.DataFrame,
    lines: pd.DataFrame,
    boxes: pd.DataFrame,
    out_path: Path,
    width: int,
    height: int,
) -> None:
    m_left, m_right, m_top, m_bot = 70, 20, 40, 50
    pw = width - m_left - m_right
    ph = height - m_top - m_bot

    x0 = view.index.min().value
    x1 = view.index.max().value
    y_min = float(min(view["low"].min(), boxes["range_low"].min() if not boxes.empty else view["low"].min()))
    y_max = float(max(view["high"].max(), boxes["range_high"].max() if not boxes.empty else view["high"].max()))
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
        f'<text x="{m_left}" y="24" font-family="Arial" font-size="16" fill="#111">Detected Ranges ({_esc(tf)})</text>',
        f'<rect x="{m_left}" y="{m_top}" width="{pw}" height="{ph}" fill="#fafafa" stroke="#d0d0d0"/>',
    ]

    # Horizontal grid
    for i in range(6):
        y = m_top + (i / 5.0) * ph
        pv = y_max - (i / 5.0) * (y_max - y_min)
        parts.append(f'<line x1="{m_left}" y1="{y:.2f}" x2="{m_left + pw}" y2="{y:.2f}" stroke="#e6e6e6" stroke-width="1"/>')
        parts.append(f'<text x="8" y="{y + 4:.2f}" font-family="Arial" font-size="11" fill="#666">{pv:,.0f}</text>')

    # Boxes
    for _, r in boxes.iterrows():
        st = pd.Timestamp(r["start_ts"])
        en = pd.Timestamp(r["end_ts"])
        if en < view.index.min() or st > view.index.max():
            continue
        st = max(st, view.index.min())
        en = min(en, view.index.max())
        xl = xmap(st)
        xr = xmap(en)
        yt = ymap(float(r["range_high"]))
        yb = ymap(float(r["range_low"]))
        h = max(1.0, yb - yt)
        status = str(r.get("status", "ACTIVE"))
        if status == "BREAKOUT_UP":
            fill, stroke = "#7DCEA0", "#27AE60"
        elif status == "BREAKOUT_DOWN":
            fill, stroke = "#F5B7B1", "#C0392B"
        else:
            fill, stroke = "#AED6F1", "#2E86C1"
        parts.append(
            f'<rect x="{xl:.2f}" y="{yt:.2f}" width="{max(1.0, xr - xl):.2f}" height="{h:.2f}" fill="{fill}" fill-opacity="0.25" stroke="{stroke}" stroke-width="1"/>'
        )

    # Candlesticks
    n = len(view)
    if n > 1:
        dx = pw / (n - 1)
    else:
        dx = pw
    body_w = max(1.0, min(8.0, dx * 0.55))

    for ts, row in view.iterrows():
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
        # Wick
        parts.append(f'<line x1="{x:.2f}" y1="{yh:.2f}" x2="{x:.2f}" y2="{yl:.2f}" stroke="{col}" stroke-width="1"/>')
        # Body
        y_top = min(yo, yc)
        y_bot = max(yo, yc)
        h_body = max(1.0, y_bot - y_top)
        parts.append(
            f'<rect x="{x - body_w/2:.2f}" y="{y_top:.2f}" width="{body_w:.2f}" height="{h_body:.2f}" '
            f'fill="{col}" fill-opacity="0.85" stroke="{col}" stroke-width="1"/>'
        )

    # Kijun / SSB lines
    if "kijun" in lines.columns:
        kij = lines["kijun"].dropna()
        if not kij.empty:
            pts = " ".join(f"{xmap(ts):.2f},{ymap(float(v)):.2f}" for ts, v in kij.items())
            parts.append(f'<polyline fill="none" stroke="#5DADE2" stroke-width="1" points="{pts}"/>')
    if "ssb" in lines.columns:
        ssb = lines["ssb"].dropna()
        if not ssb.empty:
            pts = " ".join(f"{xmap(ts):.2f},{ymap(float(v)):.2f}" for ts, v in ssb.items())
            parts.append(f'<polyline fill="none" stroke="#1F5FBF" stroke-width="2" points="{pts}"/>')

    # X-axis labels
    n_ticks = 8
    for i in range(n_ticks + 1):
        frac = i / n_ticks
        idx = int(frac * (len(view) - 1))
        ts = view.index[idx]
        x = xmap(ts)
        parts.append(f'<line x1="{x:.2f}" y1="{m_top + ph}" x2="{x:.2f}" y2="{m_top + ph + 5}" stroke="#666" stroke-width="1"/>')
        parts.append(f'<text x="{x - 36:.2f}" y="{m_top + ph + 20}" font-family="Arial" font-size="11" fill="#555">{ts.strftime("%Y-%m-%d")}</text>')

    # Legend
    lx = m_left + 8
    ly = m_top + 16
    parts.append(f'<line x1="{lx}" y1="{ly}" x2="{lx+18}" y2="{ly}" stroke="#5DADE2" stroke-width="1"/><text x="{lx+24}" y="{ly+4}" font-family="Arial" font-size="11" fill="#444">Kijun</text>')
    ly += 16
    parts.append(f'<line x1="{lx}" y1="{ly}" x2="{lx+18}" y2="{ly}" stroke="#1F5FBF" stroke-width="2"/><text x="{lx+24}" y="{ly+4}" font-family="Arial" font-size="11" fill="#444">SSB</text>')
    ly += 16
    parts.append(f'<rect x="{lx}" y="{ly-8}" width="18" height="10" fill="#AED6F1" fill-opacity="0.35" stroke="#2E86C1"/><text x="{lx+24}" y="{ly}" font-family="Arial" font-size="11" fill="#444">Range ACTIVE</text>')
    ly += 16
    parts.append(f'<rect x="{lx}" y="{ly-8}" width="18" height="10" fill="#7DCEA0" fill-opacity="0.35" stroke="#27AE60"/><text x="{lx+24}" y="{ly}" font-family="Arial" font-size="11" fill="#444">Breakout UP</text>')
    ly += 16
    parts.append(f'<rect x="{lx}" y="{ly-8}" width="18" height="10" fill="#F5B7B1" fill-opacity="0.35" stroke="#C0392B"/><text x="{lx+24}" y="{ly}" font-family="Arial" font-size="11" fill="#444">Breakout DOWN</text>')

    parts.append("</svg>")
    out_path.write_text("\n".join(parts), encoding="utf-8")


def main() -> int:
    args = parse_args()
    in_path = Path(args.input)
    boxes_path = Path(args.boxes_csv)
    out_dir = Path(args.output_dir)
    if not in_path.exists():
        raise FileNotFoundError(f"Input parquet not found: {in_path}")
    if not boxes_path.exists():
        raise FileNotFoundError(f"Boxes CSV not found: {boxes_path}")

    tfs = [x.strip().upper() for x in str(args.tfs).split(",") if x.strip()]
    bad = [tf for tf in tfs if tf not in TF_RULES]
    if bad:
        raise RuntimeError(f"Unsupported TF(s): {bad}. Allowed={list(TF_RULES.keys())}")

    print(f"[plot_detected_ranges_v1] VERSION={VERSION}")
    print(f"[INFO] input={in_path}")
    print(f"[INFO] boxes_csv={boxes_path}")

    raw = pd.read_parquet(in_path, engine="pyarrow")
    base = ensure_ohlc(raw)
    boxes = pd.read_csv(boxes_path)
    for c in ["start_ts", "end_ts", "breakout_ts"]:
        if c in boxes.columns:
            boxes[c] = pd.to_datetime(boxes[c], utc=True, errors="coerce").dt.tz_convert(None)
    out_dir.mkdir(parents=True, exist_ok=True)

    for tf in tfs:
        tf_ohlc = resample_ohlc(base, TF_RULES[tf])
        if tf_ohlc.empty:
            continue
        view = tf_ohlc.tail(int(args.last_bars)).copy()
        tf_lines = compute_ichimoku_lines(tf_ohlc, kijun_len=int(args.kijun_len), senkou_b_len=int(args.senkou_b_len))
        line_view = tf_lines.reindex(view.index)
        b = boxes[boxes["tf"] == tf].copy()
        if not b.empty:
            b = b[(b["end_ts"] >= view.index.min()) & (b["start_ts"] <= view.index.max())]
            if "strength" in b.columns:
                b = b.sort_values("strength", ascending=False)
            b = b.head(int(args.max_boxes))
        out_svg = out_dir / f"ranges_{tf.lower()}.svg"
        render_svg(
            tf=tf,
            view=view,
            lines=line_view,
            boxes=b,
            out_path=out_svg,
            width=int(args.width),
            height=int(args.height),
        )
        print(f"[OK] chart={out_svg}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
