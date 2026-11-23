#!/usr/bin/env python3
"""
oui_router_dashboard.py  —  multi-experiment anomaly dashboard

One Streamlit app, several experiment pages (picked in the sidebar), each reading
its own results folder under <project>/results/.  dashboard.py is NOT touched.

Pages
  • OUI × Router — windows            : oui_router_anomaly.py
        results/oui_router/<tag>/window_scores.csv, series_summary.csv
        + oui_windows parquet for the channel signal (per-window KDE & KL)
  • KL anomaly — flows               : anomaly_kl.py
        results/anomaly_kl/<run>/scores.csv, anomalous_flows.csv, regions.csv,
        region_summary.csv, run_config.json   (embedding map from pc1/pc2)
  • KDE anomaly — flows              : kde_anomaly.py
        results/kde/<run>/scores.csv, anomalous_flows.csv, regions.csv, run_config.json

Device detection (results/device_detection, check_random_oui.py) is NOT included.

The experiment selector is a sidebar radio rather than st.tabs so that only the
active page loads its data (the flow parquet is large).

RUN
    streamlit run oui_router_dashboard.py
"""
import glob
import json
import math
import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
from plotly.subplots import make_subplots

st.set_page_config(layout="wide")

# ----------------------------------------------------------------------------
# Project paths (all overridable in the sidebar)
# ----------------------------------------------------------------------------
PROJECT_ROOT = "/mnt/anomaly_detection"
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "parquet")   # holds the parquet files below
DEF_OUI_RUNS = os.path.join(PROJECT_ROOT, "results", "oui_router")
DEF_WINDOWS_PARQUET = os.path.join(DATA_DIR, "oui_windows.parquet")     # OUI window signal
DEF_SERIES_PARQUET = os.path.join(DATA_DIR, "oui_series.parquet")       # (pre-window OUI series)
DEF_KL_RUNS = os.path.join(PROJECT_ROOT, "results", "anomaly_kl")
DEF_SPLIT_RUNS = os.path.join(PROJECT_ROOT, "results_dashboard")   # original split KL layout
DEF_KDE_RUNS = os.path.join(PROJECT_ROOT, "results", "kde")
DEF_OA_RUNS = os.path.join(PROJECT_ROOT, "results", "oui_anomaly")
DEF_SLIDING_RUNS = os.path.join(PROJECT_ROOT, "results", "sliding")   # forecast_sliding_anomaly.py
DEF_FLOW_PARQUET = os.path.join(DATA_DIR, "new_data_aggregated_60s.parquet")
# the per-flow signal lives in a parquet whose BIN must match the run that produced the scores
FLOW_PARQUET_BY_BIN = {
    "1min": "new_data_aggregated_60s.parquet", "60s": "new_data_aggregated_60s.parquet",
    "1m": "new_data_aggregated_60s.parquet", "1t": "new_data_aggregated_60s.parquet",
    "5min": "new_data_aggregated_5min.parquet", "300s": "new_data_aggregated_5min.parquet",
    "5m": "new_data_aggregated_5min.parquet", "5t": "new_data_aggregated_5min.parquet",
}

DEFAULT_WIN_CHANNELS = ["packet_count", "bytes", "n_conns"]

# ----------------------------------------------------------------------------
# Theme handling (light / dark)
# ----------------------------------------------------------------------------
THEMES = {
    "light": dict(
        bg="#ffffff", fg="#111111", grid="#eeeeee", vline="#555555",
        bar_bg="#f2f2f2", bar_fg="#111111", hint="#444444",
        kde="#d62728", kl="#ff7f0e", both="#d6117f", normal="#1f77b4", muted="#cccccc",
        shade_kde="rgba(214,39,40,0.14)", shade_kl="rgba(255,127,14,0.16)",
        shade_both="rgba(214,17,127,0.18)", shade_anom="rgba(214,39,40,0.14)",
    ),
    "dark": dict(
        bg="#171614", fg="#e6e6e6", grid="#2f313b", vline="#999999",
        bar_bg="#1e1e1e", bar_fg="#dddddd", hint="#aaaaaa",
        kde="#ff5a5f", kl="#ffa23a", both="#ff3fd0", normal="#4aa3ff", muted="#9aa0a6",
        shade_kde="rgba(255,90,95,0.16)", shade_kl="rgba(255,162,58,0.18)",
        shade_both="rgba(255,63,208,0.24)", shade_anom="rgba(255,90,95,0.16)",
    ),
}
REGION_PALETTE = ["#4aa3ff", "#ffa23a", "#2ca02c", "#d6117f", "#9467bd",
                  "#17becf", "#e377c2", "#bcbd22", "#8c564b", "#ff5a5f",
                  "#1f77b4", "#7f7f7f", "#aec7e8", "#98df8a", "#c5b0d5"]


def detect_theme():
    try:
        t = st.context.theme.type
        if t in ("light", "dark"):
            return t
    except Exception:
        pass
    return None


detected = detect_theme()
theme_options = ["Dark", "Light"] + (["Auto"] if detected else [])
plot_theme = st.sidebar.radio("Plot theme", theme_options, horizontal=True, index=0)
theme_name = detected if plot_theme == "Auto" else plot_theme.lower()
TH = THEMES[theme_name]
FONT = dict(color=TH["fg"], size=10)


# ----------------------------------------------------------------------------
# Shared: zoomable scroll frame + plotly helper
# ----------------------------------------------------------------------------
def show_scroll(fig):
    W, H = int(fig.layout.width), int(fig.layout.height)
    plot_html = fig.to_html(include_plotlyjs="cdn", full_html=False, config={"displaylogo": False})
    page = """
    <style>
    html, body { margin:0; height:100%; font-family:sans-serif; }
    body { display:flex; flex-direction:column; height:100vh; background:__BG__; }
    #bar { display:flex; align-items:center; gap:8px; padding:4px 8px;
            font-size:12px; color:__BAR_FG__; background:__BAR_BG__; flex:0 0 auto; }
    #bar input[type=range] { width:180px; }
    #bar button { font-size:12px; padding:1px 8px; cursor:pointer; }
    #lab { width:42px; text-align:right; }
    #vp { flex:1 1 auto; overflow:auto; background:__BG__; }
    #sizer { position:relative; }
    #inner { position:absolute; top:0; left:0; transform-origin:0 0; }
    </style>
    <div id="bar">
    <span>Zoom</span><input id="zoom" type="range" min="0.1" max="4" step="0.01" value="1">
    <span id="lab">100%</span><button id="fit">Fit</button><button id="reset">100%</button>
    <span style="color:__HINT__">pinch or Ctrl/Cmd + scroll over the plot</span>
    </div>
    <div id="vp"><div id="sizer"><div id="inner">__PLOT__</div></div></div>
    <script>
    (function () {
    const W = __W__, H = __H__, MAX = 4;
    const vp = document.getElementById('vp'), sizer = document.getElementById('sizer');
    const inner = document.getElementById('inner'), slider = document.getElementById('zoom');
    const lab = document.getElementById('lab'); let s = 1;
    function minScale() { return Math.min(1, vp.clientWidth / W); }
    function setScale(ns, mx, my) {
        if (mx === undefined) { mx = vp.clientWidth / 2; my = vp.clientHeight / 2; }
        const lo = minScale(); ns = Math.max(lo, Math.min(MAX, ns));
        const cx = (vp.scrollLeft + mx) / s, cy = (vp.scrollTop + my) / s;
        s = ns; inner.style.transform = 'scale(' + s + ')';
        sizer.style.width = (W * s) + 'px'; sizer.style.height = (H * s) + 'px';
        vp.scrollLeft = cx * s - mx; vp.scrollTop = cy * s - my;
        slider.min = lo; slider.value = s; lab.textContent = Math.round(s * 100) + '%';
    }
    vp.addEventListener('wheel', function (e) {
        if (!(e.ctrlKey || e.metaKey)) return; e.preventDefault();
        const r = vp.getBoundingClientRect(); const d = Math.max(-25, Math.min(25, e.deltaY));
        setScale(s * Math.exp(-d * 0.01), e.clientX - r.left, e.clientY - r.top);
    }, { passive: false });
    let startS = 1;
    vp.addEventListener('gesturestart', function (e) { e.preventDefault(); startS = s; });
    vp.addEventListener('gesturechange', function (e) {
        e.preventDefault(); const r = vp.getBoundingClientRect();
        setScale(startS * e.scale, e.clientX - r.left, e.clientY - r.top);
    });
    vp.addEventListener('gestureend', function (e) { e.preventDefault(); });
    slider.addEventListener('input', function () { setScale(parseFloat(slider.value)); });
    document.getElementById('fit').addEventListener('click', function () { setScale(minScale()); });
    document.getElementById('reset').addEventListener('click', function () { setScale(1); });
    window.addEventListener('resize', function () { setScale(s); });
    setScale(1);
    })();
    </script>
    """
    page = (page.replace("__PLOT__", plot_html).replace("__W__", str(W)).replace("__H__", str(H))
                .replace("__BG__", TH["bg"]).replace("__BAR_BG__", TH["bar_bg"])
                .replace("__BAR_FG__", TH["bar_fg"]).replace("__HINT__", TH["hint"]))
    components.html(page, height=H + 60, scrolling=False)


def show(fig):
    try:
        st.plotly_chart(fig, width="stretch", theme=None)
    except TypeError:
        st.plotly_chart(fig, use_container_width=True, theme=None)


def pretty(ch):
    return (ch.replace("packet_count", "pkts").replace("packet_size", "pkt_size")
              .replace("mean_", "avg_").replace("fwd_", "fwd ").replace("bwd_", "bwd "))


def list_runs(runs_dir, skip=("_filter_cache",)):
    if not os.path.isdir(runs_dir):
        return []
    return sorted([d for d in os.listdir(runs_dir)
                   if os.path.isdir(os.path.join(runs_dir, d)) and d not in skip], reverse=True)


def pick_parquet(label, data_dir, preferred_path):
    """Sidebar picker for a data parquet. Lists the real .parquet files in `data_dir` as a
    dropdown (so the right file is chosen, not typed), preselecting `preferred_path`'s basename.
    If the folder has none, falls back to a free-text path. Returns an absolute path."""
    pref = os.path.basename(preferred_path)
    files = sorted(os.path.basename(f) for f in glob.glob(os.path.join(data_dir, "*.parquet"))) \
        if os.path.isdir(data_dir) else []
    if files:
        idx = files.index(pref) if pref in files else 0
        return os.path.join(data_dir, st.sidebar.selectbox(label, files, index=idx))
    return st.sidebar.text_input(label + " (full path)", preferred_path)


# ----------------------------------------------------------------------------
# Shared embedding-experiment helpers (anomaly_kl, kde, oui_anomaly)
# ----------------------------------------------------------------------------
def flag_mask(sc):
    """A boolean 'flagged as anomaly' mask, whatever the method's column is."""
    for col in ("anomaly", "survivor", "sig_anomaly"):
        if col in sc.columns:
            s = sc[col]
            if s.dtype == object:
                return s.map(lambda v: str(v).strip().lower() in ("true", "1", "1.0")).astype(bool)
            return s.astype(bool)
    return pd.Series(False, index=sc.index)


def emb_map(sc, score_col, id_col, hover_cols, title, max_regions=12):
    """PCA embedding scatter (pc1 vs pc2): grey normal, red flagged-no-region, colour per region.
    With many regions, only the largest `max_regions` get a distinct colour + legend entry; the
    rest are bucketed into one 'other regions' trace so the map stays readable and fast."""
    if not {"pc1", "pc2"} <= set(sc.columns):
        return None
    flag = flag_mask(sc)
    reg = sc["region_id"] if "region_id" in sc.columns else pd.Series(-1, index=sc.index)
    hover = [c for c in [id_col, score_col] + list(hover_cols) if c in sc.columns]

    def cdata(df):
        return df[hover].to_numpy() if hover else None

    htmpl = "<br>".join([f"%{{customdata[{i}]}}" for i in range(len(hover))]) + "<extra></extra>" if hover else None
    fig = go.Figure()
    normal = sc[(~flag) & (reg < 0)]
    if len(normal):
        fig.add_trace(go.Scatter(x=normal["pc1"], y=normal["pc2"], mode="markers",
                                 marker=dict(size=4, color=TH["muted"]), name="normal / reference",
                                 customdata=cdata(normal), hovertemplate=htmpl))
    nr = sc[flag & (reg < 0)]
    if len(nr):
        fig.add_trace(go.Scatter(x=nr["pc1"], y=nr["pc2"], mode="markers",
                                 marker=dict(size=7, color=TH["kde"], symbol="circle-open", line=dict(width=1.5)),
                                 name="flagged (no region)", customdata=cdata(nr), hovertemplate=htmpl))
    sizes = sc.loc[reg >= 0, "region_id"].value_counts()          # largest regions first
    ranked = list(sizes.index)
    top, other = ranked[:max_regions], ranked[max_regions:]
    for k, r in enumerate(top):
        g = sc[sc["region_id"] == r]
        fig.add_trace(go.Scatter(x=g["pc1"], y=g["pc2"], mode="markers",
                                 marker=dict(size=8, color=REGION_PALETTE[k % len(REGION_PALETTE)]),
                                 name=f"region {int(r)} (n={len(g)})", customdata=cdata(g), hovertemplate=htmpl))
    if other:
        g = sc[sc["region_id"].isin(other)]
        fig.add_trace(go.Scatter(x=g["pc1"], y=g["pc2"], mode="markers",
                                 marker=dict(size=6, color=TH["both"], symbol="diamond", opacity=0.7),
                                 name=f"other regions ({len(other)})", customdata=cdata(g), hovertemplate=htmpl))
    fig.update_xaxes(title="PC1", showgrid=True, gridcolor=TH["grid"], linecolor=TH["grid"], tickfont=dict(size=9, color=TH["fg"]))
    fig.update_yaxes(title="PC2", showgrid=True, gridcolor=TH["grid"], linecolor=TH["grid"], tickfont=dict(size=9, color=TH["fg"]))
    fig.update_layout(height=560, paper_bgcolor=TH["bg"], plot_bgcolor=TH["bg"], font=FONT,
                      legend=dict(font=dict(size=10)), title=dict(text=title, font=dict(size=12, color=TH["fg"])))
    return fig


def score_hist(sc, score_col, label):
    if score_col not in sc.columns:
        return None
    s = pd.to_numeric(sc[score_col], errors="coerce").to_numpy()
    flag = flag_mask(sc)
    thr = float(np.nanmin(s[flag.to_numpy()])) if flag.any() else None
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=s, nbinsx=60, marker_color=TH["normal"]))
    if thr is not None and np.isfinite(thr):
        fig.add_vline(x=thr, line_width=2, line_dash="dash", line_color=TH["kde"],
                      annotation_text="anomaly threshold", annotation_font_color=TH["fg"])
    fig.update_xaxes(title=label, showgrid=True, gridcolor=TH["grid"], linecolor=TH["grid"], tickfont=dict(size=9, color=TH["fg"]))
    fig.update_yaxes(title="count", showgrid=True, gridcolor=TH["grid"], linecolor=TH["grid"], tickfont=dict(size=9, color=TH["fg"]))
    fig.update_layout(height=320, paper_bgcolor=TH["bg"], plot_bgcolor=TH["bg"], font=FONT, showlegend=False,
                      bargap=0.02, title=dict(text=f"Score distribution ({label})", font=dict(size=12, color=TH["fg"])))
    return fig


@st.cache_data(show_spinner=False)
def read_csv_cached(path):
    return pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()


@st.cache_data(show_spinner=False)
def read_config(run_folder):
    f = os.path.join(run_folder, "run_config.json")
    try:
        return json.load(open(f)) if os.path.exists(f) else {}
    except Exception:
        return {}


def config_channels(cfg, default=("packet_count",)):
    ch = cfg.get("channels")
    if isinstance(ch, list):
        return [str(c) for c in ch]
    if isinstance(ch, str) and ch:
        return [c.strip() for c in ch.split(",") if c.strip()]
    return list(default)


def resolve_flow_parquet(cfg):
    """The correct per-flow signal parquet for this run: the exact file it embedded if run_config
    recorded it, else the aggregated file whose BIN matches (results depend on the data's bin)."""
    feat = cfg.get("features_file") or cfg.get("csv") or ""
    if feat:
        return feat if os.path.isabs(feat) else os.path.join(PROJECT_ROOT, feat.lstrip("./"))
    name = FLOW_PARQUET_BY_BIN.get(str(cfg.get("bin", "")).lower().strip())
    return os.path.join(PROJECT_ROOT, "data", "parquet", name) if name else DEF_FLOW_PARQUET


# ----------------------------------------------------------------------------
# Shared flow helpers (per-flow channel plots for KL / KDE)
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_flow_parquet(path):
    df = pd.read_parquet(path)
    df["datetime"] = pd.to_datetime(df["datetime"])
    pkt = "packet_count"
    if pkt in df.columns:
        df["total_packet_count"] = df[pkt]
    if all(c in df.columns for c in ["fwd_mean_packet_size", "fwd_packet_count", "bwd_mean_packet_size", "bwd_packet_count"]):
        df["total_mean_packet_size"] = np.where(
            df[pkt] > 0, (df["fwd_mean_packet_size"].fillna(0) * df["fwd_packet_count"].fillna(0) +
                          df["bwd_mean_packet_size"].fillna(0) * df["bwd_packet_count"].fillna(0)) / df[pkt], 0)
    if all(c in df.columns for c in ["fwd_mean_iat", "bwd_mean_iat"]):
        df["total_mean_iat"] = df[["fwd_mean_iat", "bwd_mean_iat"]].mean(axis=1, skipna=True).fillna(0)
    return df


def get_flow_frames(df, cids, channels, bin_freq="1min"):
    cids = [str(c) for c in cids]
    sub = df[df["conn_id"].astype(str).isin(cids)]
    flows = {}
    for cid, gg in sub.groupby("conn_id", sort=False):
        gg = gg.sort_values("datetime").drop_duplicates("datetime", keep="first").set_index("datetime")
        full = pd.date_range(gg.index.min(), gg.index.max(), freq=bin_freq)
        for ch in channels:
            if ch == "mean_iat" and ch not in gg.columns and "total_mean_iat" in gg.columns:
                gg["mean_iat"] = gg["total_mean_iat"]
            if ch == "mean_packet_size" and ch not in gg.columns and "total_mean_packet_size" in gg.columns:
                gg["mean_packet_size"] = gg["total_mean_packet_size"]
        cols = {ch: (pd.to_numeric(gg[ch], errors="coerce") if ch in gg.columns else pd.Series(0.0, index=gg.index))
                for ch in channels}
        frame = pd.DataFrame(cols).reindex(full).fillna(0.0)
        flows[str(cid)] = {"cid": str(cid), "frame": frame[channels].to_numpy(dtype=np.float32), "index": full,
                           "protocol": gg["protocol"].iloc[0] if "protocol" in gg.columns else "N/A",
                           "src_port": gg["src_port"].iloc[0] if "src_port" in gg.columns else "N/A",
                           "dst_port": gg["dst_port"].iloc[0] if "dst_port" in gg.columns else "N/A"}
    return flows


def plot_flow_detail(t, flow, channels, color, holdout=0, ncols=3):
    C = len(channels)
    nrows = math.ceil(C / ncols)
    fig = make_subplots(rows=nrows, cols=ncols, subplot_titles=channels, vertical_spacing=0.12, horizontal_spacing=0.05)
    x = flow["index"]
    h = len(x) - holdout
    for ci, ch in enumerate(channels):
        r, c = ci // ncols + 1, ci % ncols + 1
        fig.add_trace(go.Scatter(x=x, y=flow["frame"][:, ci], mode="lines+markers",
                                 line=dict(color=color, width=1.5), marker=dict(size=3),
                                 hovertemplate="%{x|%m-%d %H:%M}<br>%{y:.5g}<extra></extra>", showlegend=False), row=r, col=c)
        if holdout and h > 0:
            fig.add_vline(x=x[h], line_width=1.5, line_dash="dot", line_color=TH["vline"], row=r, col=c)
    fig.update_xaxes(showgrid=True, gridcolor=TH["grid"], linecolor=TH["grid"], tickformat="%m-%d %H:%M",
                     nticks=5, tickfont=dict(size=8, color=TH["fg"]))
    fig.update_yaxes(showgrid=True, gridcolor=TH["grid"], linecolor=TH["grid"], tickfont=dict(size=8, color=TH["fg"]))
    fig.update_annotations(font=dict(color=TH["fg"]))
    fig.update_layout(height=260 * nrows + 60, margin=dict(l=40, r=20, t=50, b=30),
                      paper_bgcolor=TH["bg"], plot_bgcolor=TH["bg"], font=FONT, hovermode="x unified",
                      title=dict(text=f"{t['tag']} | {flow['protocol']} {flow['src_port']}→{flow['dst_port']} "
                                      f"| score={t['d']:.3g}", font=dict(size=13, color=TH["fg"])))
    return fig


# ============================================================================
# PAGE 1 — OUI × Router (windows)   [oui_router_anomaly.py]
# ============================================================================
@st.cache_data(show_spinner=False)
def win_list_channels(windows_path):
    try:
        import pyarrow.parquet as pq
        names = set(pq.ParquetFile(windows_path).schema_arrow.names)
    except Exception:
        return DEFAULT_WIN_CHANNELS
    return [c for c in DEFAULT_WIN_CHANNELS if c in names] or \
           [c for c in names if c not in
            ("window_id", "series_id", "router", "oui", "direction", "start", "end", "n_active_bins")]


@st.cache_data(show_spinner=False)
def win_load_scores(run_folder):
    f = os.path.join(run_folder, "window_scores.csv")
    if not os.path.exists(f):
        return pd.DataFrame()
    df = pd.read_csv(f)
    df["start"] = pd.to_datetime(df["start"], errors="coerce")
    for c in ("kde_anom", "kl_anom"):
        if c in df.columns:
            df[c] = df[c].map(lambda v: str(v).strip().lower() in ("true", "1", "1.0")).astype(bool)
    for c in ("kde_score", "kl_D", "kde_rank_in_series", "kl_rank_in_series", "kl_sig_anom"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


@st.cache_data(show_spinner=False)
def win_load_summary(run_folder):
    f = os.path.join(run_folder, "series_summary.csv")
    return pd.read_csv(f) if os.path.exists(f) else pd.DataFrame()


@st.cache_data(show_spinner=False)
def win_load_frames(windows_path, run_folder, channels):
    scores = win_load_scores(run_folder)
    if scores.empty or not os.path.exists(windows_path):
        return {}
    wanted = set(scores["window_id"].astype(str))
    cols = ["window_id", "series_id", "start", "end"] + channels
    try:
        import pyarrow.parquet as pq
        avail = set(pq.ParquetFile(windows_path).schema_arrow.names)
        cols = [c for c in cols if c in avail]
    except Exception:
        pass
    wdf = pd.read_parquet(windows_path, columns=cols)
    wdf = wdf[wdf["window_id"].astype(str).isin(wanted)].copy()
    wdf["start"] = pd.to_datetime(wdf["start"], errors="coerce")
    if "end" in wdf.columns:
        wdf["end"] = pd.to_datetime(wdf["end"], errors="coerce")
    out = {}
    for sid, g in wdf.groupby("series_id", sort=False):
        g = g.sort_values("start").reset_index(drop=True)
        rec = {"window_id": g["window_id"].astype(str).to_numpy(), "start": g["start"].to_numpy()}
        for ch in channels:
            if ch in g.columns:
                rec[ch] = [np.asarray(v, dtype=np.float32) for v in g[ch]]
        step_s = None
        if "end" in g.columns and channels and channels[0] in rec and len(g):
            wl = len(rec[channels[0]][0])
            if wl > 1 and pd.notna(g["end"].iloc[0]) and pd.notna(g["start"].iloc[0]):
                span = (g["end"].iloc[0] - g["start"].iloc[0]) / np.timedelta64(1, "s")
                step_s = float(span) / (wl - 1) if span > 0 else None
        rec["step_s"] = step_s
        out[sid] = rec
    return out


def win_means(rec, channels):
    n = len(rec["start"])
    return {ch: (np.array([float(a.mean()) if a.size else 0.0 for a in rec[ch]]) if ch in rec else np.zeros(n))
            for ch in channels}


def win_series_label(row):
    nb = int(row.get("n_both_anom", 0))
    both_tag = f", ∩ {nb}" if nb else ""
    return (f"{row['oui']} | {row['router']} | {row['direction']}  "
            f"[{row.get('n_windows', '?')}w, KDE {int(row.get('n_kde_anom', 0))}, "
            f"KL {int(row.get('n_kl_anom', 0))}{both_tag}]")


def win_plot_profile(rec, ss, channels, row_h=150, col_w=1100):
    C = len(channels)
    x = pd.to_datetime(rec["start"])
    prof = win_means(rec, channels)
    sidx = ss.set_index(ss["window_id"].astype(str))
    wids = rec["window_id"]
    kde = np.array([bool(sidx["kde_anom"].get(w, False)) if "kde_anom" in sidx else False for w in wids])
    kl = np.array([bool(sidx["kl_anom"].get(w, False)) if "kl_anom" in sidx else False for w in wids])
    kde_sc = np.array([sidx["kde_score"].get(w, np.nan) if "kde_score" in sidx else np.nan for w in wids])
    kl_sc = np.array([sidx["kl_D"].get(w, np.nan) if "kl_D" in sidx else np.nan for w in wids])
    both, kde_only, kl_only = kde & kl, kde & ~kl, kl & ~kde
    fig = make_subplots(rows=C, cols=1, shared_xaxes=True, subplot_titles=channels, vertical_spacing=0.06)
    for ci, ch in enumerate(channels):
        y = prof[ch]
        fig.add_trace(go.Scatter(x=x, y=y, mode="lines+markers", line=dict(color=TH["normal"], width=1.2),
                                 marker=dict(size=3), customdata=np.column_stack([kde_sc, kl_sc]),
                                 hovertemplate="%{x|%m-%d %H:%M}<br>mean=%{y:.4g}"
                                 "<br>kde=%{customdata[0]:.3f} kl=%{customdata[1]:.3f}<extra></extra>",
                                 showlegend=False), row=ci + 1, col=1)
        for mask, color, sym, size, name, grp in [
            (kde_only, TH["kde"], "circle-open", 9, "KDE only", "kde"),
            (kl_only, TH["kl"], "x", 12, "KL only", "kl"),
            (both, TH["both"], "diamond", 11, "KDE ∩ KL", "both")]:
            if mask.any():
                fig.add_trace(go.Scatter(x=x[mask], y=y[mask], mode="markers",
                                         marker=dict(size=size, color=color, symbol=sym, line=dict(width=2)),
                                         name=name, legendgroup=grp, showlegend=(ci == 0),
                                         hovertemplate=name + "<br>%{x|%m-%d %H:%M}<extra></extra>"), row=ci + 1, col=1)
    fig.update_xaxes(showgrid=True, gridcolor=TH["grid"], linecolor=TH["grid"], tickfont=dict(size=8, color=TH["fg"]))
    fig.update_yaxes(showgrid=True, gridcolor=TH["grid"], linecolor=TH["grid"], tickfont=dict(size=8, color=TH["fg"]), zeroline=False)
    fig.update_annotations(font=dict(size=11, color=TH["fg"]))
    fig.update_layout(height=row_h * C + 90, width=col_w, margin=dict(l=55, r=15, t=40, b=30),
                      paper_bgcolor=TH["bg"], plot_bgcolor=TH["bg"], font=FONT, hovermode="x unified",
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, font=dict(size=10)))
    return fig


def win_plot_fullres(rec, ss, channels, col_w=1400, row_h=200):
    C = len(channels)
    step_s = rec.get("step_s") or 1.0
    starts = pd.to_datetime(rec["start"])
    sidx = ss.set_index(ss["window_id"].astype(str))
    wids = rec["window_id"]
    kde = np.array([bool(sidx["kde_anom"].get(w, False)) if "kde_anom" in sidx else False for w in wids])
    kl = np.array([bool(sidx["kl_anom"].get(w, False)) if "kl_anom" in sidx else False for w in wids])
    fig = make_subplots(rows=C, cols=1, shared_xaxes=True, subplot_titles=channels, vertical_spacing=0.07)
    done = {"kde": False, "kl": False, "both": False}

    def cat(j):
        if kde[j] and kl[j]:
            return "both", TH["both"], TH["shade_both"], "KDE ∩ KL"
        if kl[j]:
            return "kl", TH["kl"], TH["shade_kl"], "KL only"
        return "kde", TH["kde"], TH["shade_kde"], "KDE only"

    for ci, ch in enumerate(channels):
        if ch not in rec:
            continue
        tts, xs, ys = [], [], []
        for j, arr in enumerate(rec[ch]):
            tt = starts[j] + pd.to_timedelta(np.arange(arr.size) * step_s, unit="s")
            tts.append(tt.values); xs.append(tt.values); ys.append(arr)
        if not xs:
            continue
        fig.add_trace(go.Scatter(x=np.concatenate(xs), y=np.concatenate(ys), mode="lines",
                                 line=dict(color=TH["normal"], width=0.9),
                                 hovertemplate="%{x|%m-%d %H:%M}<br>%{y:.4g}<extra></extra>", showlegend=False), row=ci + 1, col=1)
        for j, arr in enumerate(rec[ch]):
            if not arr.size or not (kde[j] or kl[j]):
                continue
            tt = tts[j]; key, line, band, name = cat(j)
            fig.add_vrect(x0=tt[0], x1=tt[-1], fillcolor=band, line_width=0, layer="below", row=ci + 1, col=1)
            fig.add_trace(go.Scatter(x=tt, y=arr, mode="lines", line=dict(color=line, width=1.8), name=name,
                                     legendgroup=key, showlegend=not done[key],
                                     hovertemplate=name + " window<br>%{x|%m-%d %H:%M}<br>%{y:.4g}<extra></extra>"), row=ci + 1, col=1)
            done[key] = True
    fig.update_xaxes(showgrid=True, gridcolor=TH["grid"], linecolor=TH["grid"], tickfont=dict(size=8, color=TH["fg"]))
    fig.update_yaxes(showgrid=True, gridcolor=TH["grid"], linecolor=TH["grid"], tickfont=dict(size=8, color=TH["fg"]), zeroline=False)
    fig.update_annotations(font=dict(size=11, color=TH["fg"]))
    fig.update_layout(height=row_h * C + 90, width=col_w, margin=dict(l=55, r=15, t=55, b=30),
                      paper_bgcolor=TH["bg"], plot_bgcolor=TH["bg"], font=FONT, hovermode="x unified",
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, font=dict(size=10)),
                      title=dict(text="Full-resolution series — red = KDE only, orange = KL only, "
                                      "magenta = KDE ∩ KL (band + bold segment); zoom with the controls above",
                                 font=dict(size=12, color=TH["fg"])))
    return fig


def win_plot_window(rec, idx, ss, channels, ncols=3):
    C = len(channels)
    nrows = math.ceil(C / ncols)
    wid = rec["window_id"][idx]
    sset = set(ss["window_id"].astype(str))
    srow = ss.set_index(ss["window_id"].astype(str)).loc[wid] if wid in sset else None
    step_s = rec.get("step_s") or 1.0
    t0 = pd.to_datetime(rec["start"][idx])
    fig = make_subplots(rows=nrows, cols=ncols, subplot_titles=channels, vertical_spacing=0.14, horizontal_spacing=0.06)
    for ci, ch in enumerate(channels):
        r, c = ci // ncols + 1, ci % ncols + 1
        arr = rec[ch][idx] if ch in rec else np.zeros(1)
        tt = t0 + pd.to_timedelta(np.arange(arr.size) * step_s, unit="s")
        fig.add_trace(go.Scatter(x=tt, y=arr, mode="lines+markers", line=dict(color=TH["normal"], width=1.4),
                                 marker=dict(size=3), hovertemplate="%{x|%m-%d %H:%M}<br>%{y:.5g}<extra></extra>",
                                 showlegend=False), row=r, col=c)
    fig.update_xaxes(showgrid=True, gridcolor=TH["grid"], linecolor=TH["grid"], tickformat="%H:%M", nticks=5, tickfont=dict(size=8, color=TH["fg"]))
    fig.update_yaxes(showgrid=True, gridcolor=TH["grid"], linecolor=TH["grid"], tickfont=dict(size=8, color=TH["fg"]))
    fig.update_annotations(font=dict(color=TH["fg"]))
    sub = ""
    if srow is not None:
        flags = [m for m, c in (("KDE", "kde_anom"), ("KL", "kl_anom")) if bool(srow.get(c, False))]
        sub = (f" | kde={srow.get('kde_score', np.nan):.3f}  kl={srow.get('kl_D', np.nan):.3f}"
               f"{'  [' + '+'.join(flags) + ']' if flags else ''}")
    fig.update_layout(height=240 * nrows + 70, margin=dict(l=45, r=20, t=55, b=30),
                      paper_bgcolor=TH["bg"], plot_bgcolor=TH["bg"], font=FONT, hovermode="x unified",
                      title=dict(text=f"window {idx}  |  {pd.Timestamp(t0)}{sub}", font=dict(size=13, color=TH["fg"])))
    return fig


def page_oui_router():
    st.title("OUI × Router — Window Anomaly")
    st.caption("Per-series (router × OUI × direction) window anomalies from oui_router_anomaly.py. "
               "KDE = isolated atypical windows; KL = sustained change vs the rest of the series.")
    st.sidebar.header("Data sources")
    runs_dir = st.sidebar.text_input("Runs dir (results/oui_router)", DEF_OUI_RUNS)
    data_dir = st.sidebar.text_input("Data folder", DATA_DIR)
    windows_path = pick_parquet("Windows parquet", data_dir, DEF_WINDOWS_PARQUET)
    st.sidebar.caption("Per-window bin is read from each window's start/end — no bin assumption.")
    runs = list_runs(runs_dir)
    if not runs:
        st.warning(f"No run folders under `{runs_dir}`."); return
    run = os.path.join(runs_dir, st.sidebar.selectbox("Select run", runs))
    scores = win_load_scores(run)
    if scores.empty:
        st.warning("No window_scores.csv in this run."); return
    summary = win_load_summary(run)
    channels = win_list_channels(windows_path)
    if not os.path.exists(windows_path):
        st.warning(f"Windows parquet not found: `{windows_path}` — scores load, signal can't be drawn.")

    n_both = int((scores.get("kde_anom", False) & scores.get("kl_anom", False)).sum()) \
        if {"kde_anom", "kl_anom"} <= set(scores.columns) else 0
    c = st.columns(5)
    c[0].metric("Series", f"{scores['series_id'].nunique():,}")
    c[1].metric("Windows", f"{len(scores):,}")
    c[2].metric("KDE anomalies", f"{int(scores['kde_anom'].sum()) if 'kde_anom' in scores else 0:,}")
    c[3].metric("KL anomalies", f"{int(scores['kl_anom'].sum()) if 'kl_anom' in scores else 0:,}")
    c[4].metric("KDE ∩ KL", f"{n_both:,}")

    st.sidebar.header("Filters")
    base = summary.copy() if not summary.empty else (
        scores.groupby("series_id").agg(router=("router", "first"), oui=("oui", "first"),
            direction=("direction", "first"), n_windows=("window_id", "size"),
            n_kde_anom=("kde_anom", "sum"), n_kl_anom=("kl_anom", "sum")).reset_index())
    if {"kde_anom", "kl_anom"} <= set(scores.columns):
        nb = (scores.assign(_b=scores["kde_anom"] & scores["kl_anom"]).groupby("series_id")["_b"].sum().astype(int))
        base["n_both_anom"] = base["series_id"].map(nb).fillna(0).astype(int)
    else:
        base["n_both_anom"] = 0
    for col in ("n_kde_anom", "n_kl_anom", "n_windows", "n_both_anom"):
        if col in base.columns:
            base[col] = pd.to_numeric(base[col], errors="coerce").fillna(0).astype(int)
    sel_router = st.sidebar.selectbox("Router", ["All"] + sorted(base["router"].dropna().astype(str).unique()))
    sel_dir = st.sidebar.selectbox("Direction", ["All"] + sorted(base["direction"].dropna().astype(str).unique()))
    only_anom = st.sidebar.checkbox("Only series with anomalies", value=True)
    only_both = st.sidebar.checkbox("Only series with KDE ∩ KL", value=False)
    fb = base.copy()
    if sel_router != "All":
        fb = fb[fb["router"].astype(str) == sel_router]
    if sel_dir != "All":
        fb = fb[fb["direction"].astype(str) == sel_dir]
    if only_anom:
        fb = fb[(fb["n_kde_anom"] + fb["n_kl_anom"]) > 0]
    if only_both:
        fb = fb[fb["n_both_anom"] > 0]
    fb = fb.assign(_tot=fb["n_kde_anom"] + fb["n_kl_anom"]).sort_values(["_tot", "n_windows"], ascending=[False, False])
    if fb.empty:
        st.sidebar.write("No series match."); return
    st.sidebar.header("Series")
    labels = [win_series_label(r) for _, r in fb.iterrows()]
    sids = fb["series_id"].tolist()
    sel_sid = sids[labels.index(st.sidebar.radio("Select series", labels))]

    st.sidebar.header("Display")
    show_channels = st.sidebar.multiselect("Channels", channels, default=channels)
    show_fullres = st.sidebar.checkbox("Full-resolution reconstruction", value=True)
    show_profile = st.sidebar.checkbox("Per-window profile", value=True)
    if not show_channels:
        st.warning("Select at least one channel."); return

    srow = fb[fb["series_id"] == sel_sid].iloc[0]
    st.subheader(f"{srow['oui']}  ·  {srow['router']}  ·  {srow['direction']}")
    m = st.columns(4)
    m[0].metric("Windows", int(srow.get("n_windows", 0)))
    m[1].metric("KDE anomalies", int(srow.get("n_kde_anom", 0)))
    m[2].metric("KL anomalies", int(srow.get("n_kl_anom", 0)))
    m[3].metric("KDE ∩ KL", int(srow.get("n_both_anom", 0)))

    ss = scores[scores["series_id"] == sel_sid].sort_values("start").reset_index(drop=True)
    both_w = ss[(ss.get("kde_anom", False)) & (ss.get("kl_anom", False))] \
        if {"kde_anom", "kl_anom"} <= set(ss.columns) else ss.iloc[0:0]
    st.caption(("◆ **KDE ∩ KL** (magenta) at: " + ", ".join(pd.to_datetime(both_w["start"]).dt.strftime("%m-%d %H:%M")))
               if len(both_w) else "No window flagged by both KDE and KL in this series.")

    frames = win_load_frames(windows_path, run, show_channels) if os.path.exists(windows_path) else {}
    rec = frames.get(sel_sid)
    if rec is None:
        st.info("No window signal for this series — scores table only.")
        st.dataframe(ss, use_container_width=True, height=360); return

    if show_fullres:
        st.markdown("#### Full-resolution reconstruction")
        st.caption("Every window end to end at bin resolution. Anomalous windows highlighted "
                   "(band + bold segment): red = KDE, orange = KL, magenta = both.")
        show_scroll(win_plot_fullres(rec, ss, show_channels))
    if show_profile:
        st.markdown("#### Per-window profile")
        st.caption("One point per window (window-mean). ○ red = KDE, ✕ orange = KL, ◆ magenta = both.")
        show_scroll(win_plot_profile(rec, ss, show_channels))

    st.divider(); st.markdown("### 🔍 Inspect a window")
    sidx = ss.set_index(ss["window_id"].astype(str))
    opts = []
    for i in range(len(rec["window_id"])):
        wid = rec["window_id"][i]
        tags, extra = [], ""
        if wid in sidx.index:
            rr = sidx.loc[wid]
            if bool(rr.get("kde_anom", False)):
                tags.append("KDE")
            if bool(rr.get("kl_anom", False)):
                tags.append("KL")
            extra = f" | kde={rr.get('kde_score', np.nan):.2f} kl={rr.get('kl_D', np.nan):.2f}"
        tag = ("  [" + "+".join(tags) + "]") if tags else ""
        opts.append((f"win {i}  {pd.Timestamp(rec['start'][i])}{extra}{tag}", i, bool(tags)))
    opts.sort(key=lambda o: (not o[2], o[1]))
    choice = st.selectbox("Choose a window (anomalous first)", [o[0] for o in opts], index=0 if opts else None, placeholder="Select…")
    if choice is not None:
        idx = dict((o[0], o[1]) for o in opts)[choice]
        show(win_plot_window(rec, idx, ss, show_channels))
    with st.expander("Window scores table (this series)"):
        cols = [c for c in ["window_id", "start", "kde_score", "kde_rank_in_series", "kde_anom",
                            "kl_D", "kl_survivor", "kl_sig_anom", "kl_rank_in_series", "kl_anom"] if c in ss.columns]
        st.dataframe(ss[cols], use_container_width=True, height=360)


# ============================================================================
# PAGE 2 & 3 — flow experiments (anomaly_kl / kde), shared renderer
# ============================================================================
@st.cache_data(show_spinner=False)
def flow_load_scores(run_folder):
    f = os.path.join(run_folder, "scores.csv")
    if not os.path.exists(f):
        return pd.DataFrame()
    df = pd.read_csv(f)
    for c in ("D_i", "neg_log_density", "gpd_pvalue", "region_id", "pc1", "pc2", "chronos_anom_score"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if "cid" in df.columns:
        df["cid"] = df["cid"].astype(str)
    return df


@st.cache_data(show_spinner=False)
def load_nearest_kde(run_folder):
    """KDE: anomaly cid -> ordered list of nearest NORMAL cids (from nearest_normal.csv)."""
    f = os.path.join(run_folder, "nearest_normal.csv")
    if not os.path.exists(f):
        return {}
    df = pd.read_csv(f)
    out = {}
    for _, r in df.iterrows():
        cids = [c.strip() for c in str(r.get("nearest_normal_cids", "")).split(";") if c.strip()]
        out[str(r["anomaly_cid"])] = cids
    return out


@st.cache_data(show_spinner=False)
def load_nearest_kl(run_folder):
    """anomaly_kl: region_id -> ordered list of nearest reference (NORMAL) cids
    (from nearest_reference.csv)."""
    f = os.path.join(run_folder, "nearest_reference.csv")
    if not os.path.exists(f):
        return {}
    df = pd.read_csv(f)
    if "k" in df.columns:
        df = df.sort_values("k")
    out = {}
    for r, g in df.groupby("region_id"):
        out[int(r)] = [str(x) for x in g["ref_conn_id"].tolist()]
    return out


def nearest_normal_cids(kind, run, row, id_col, k):
    cid = str(row[id_col])
    if kind == "kde":
        return load_nearest_kde(run).get(cid, [])[:k]
    refs = load_nearest_kl(run)
    reg = int(row["region_id"]) if row.get("region_id", -1) >= 0 else -1
    lst = refs.get(reg, [])[:k]
    if not lst:
        nc = row.get("nearest_normal_cid")
        if isinstance(nc, str) and nc:
            lst = [nc]
    return lst


def plot_flows_grid(targets, flows, channels, row_h=95, col_w=240):
    """Small multiples: one row per flow (flagged flow first, in red; normal flows below, in blue),
    one column per channel. Modelled on dashboard.py so it reads the same way."""
    targets = [t for t in targets if t["cid"] in flows]
    if not targets:
        return None
    R, C = len(targets), len(channels)
    height = row_h * R + 75
    left, right = 150, 10
    width = left + right + col_w * C
    fig = make_subplots(rows=R, cols=C, column_titles=[pretty(c) for c in channels],
                        vertical_spacing=min(0.5 / R, 26 / height), horizontal_spacing=0.03)
    for ri, t in enumerate(targets):
        f = flows[t["cid"]]
        x = f["index"]
        for ci in range(C):
            fig.add_trace(go.Scatter(x=x, y=f["frame"][:, ci], mode="lines",
                                     line=dict(color=t["color"], width=1.2),
                                     hovertemplate="%{x|%m-%d %H:%M}<br>%{y:.4g}<extra></extra>",
                                     showlegend=False), row=ri + 1, col=ci + 1)
        fig.update_yaxes(title_text=t["label"], title_font=dict(size=9, color=t["color"]),
                         title_standoff=2, row=ri + 1, col=1)
    fig.update_xaxes(showticklabels=False, showgrid=True, gridcolor=TH["grid"], linecolor=TH["grid"])
    fig.update_xaxes(showticklabels=True, tickformat="%m-%d %H:%M", nticks=3, tickangle=0,
                     tickfont=dict(size=8, color=TH["fg"]), row=R)
    fig.update_yaxes(showgrid=True, gridcolor=TH["grid"], linecolor=TH["grid"], nticks=3,
                     tickfont=dict(size=8, color=TH["fg"]), zeroline=False)
    fig.update_annotations(font=dict(size=10, color=TH["fg"]))
    fig.update_layout(height=height, width=width, margin=dict(l=left, r=right, t=30, b=34),
                      paper_bgcolor=TH["bg"], plot_bgcolor=TH["bg"], font=FONT, hovermode="closest")
    return fig


def page_flow_experiment(title, subtitle, default_runs, runs_label, score_col, score_label, nearest_kind):
    st.title(title)
    st.caption(subtitle)
    st.sidebar.header("Data")
    runs_dir = st.sidebar.text_input(runs_label, default_runs)
    runs = list_runs(runs_dir)
    if not runs:
        st.warning(f"No run folders under `{runs_dir}`."); return
    run = os.path.join(runs_dir, st.sidebar.selectbox("Select run", runs))

    sc = flow_load_scores(run)
    if sc.empty:
        st.warning("No scores.csv in this run (it may be an incomplete run)."); return
    if score_col not in sc.columns:
        score_col = next((c for c in ("D_i", "neg_log_density") if c in sc.columns), score_col)
    cfg = read_config(run)
    anomdf = read_csv_cached(os.path.join(run, "anomalous_flows.csv"))
    flag = flag_mask(sc)
    n_reg = int(sc.loc[sc.get("region_id", -1) >= 0, "region_id"].nunique())

    # flow-data parquet: default to the file matching THIS run's data shape (bin)
    bin_freq = str(cfg.get("bin", "1min"))
    data_dir = st.sidebar.text_input("Data folder", DATA_DIR)
    flow_parquet = pick_parquet("Flow data parquet", data_dir, resolve_flow_parquet(cfg))
    src = "run_config.json (features_file)" if (cfg.get("features_file") or cfg.get("csv")) \
        else f"run_config.json (bin = {bin_freq})"
    st.sidebar.caption(f"Bin taken from {src}. The signal file must share that bin.")

    c = st.columns(3)
    c[0].metric("Flows analysed", f"{len(sc):,}")
    c[1].metric("Flagged as suspicious", f"{int(flag.sum()):,}")
    c[2].metric("Groups of similar flows", f"{n_reg:,}")

    # ---- suspicious flows table (the main artefact for a security analyst) ----
    st.markdown("#### Suspicious flows")
    st.caption("Each row is a flow the method flagged as unusual. Pick one below to see it next to the "
               "most similar **normal** flows and judge what stands out.")
    if anomdf.empty:
        anomdf = sc[flag].copy()
    front, seen = [], set()
    for col in ["conn_id", "cid", score_col, "D_i", "neg_log_density", "region_id", "oui", "router",
                "service", "protocol", "dst_port", "conn_direction", "temporal_pattern", "n_packets"]:
        if col in anomdf.columns and col not in seen:
            front.append(col); seen.add(col)
    rest = [col for col in anomdf.columns if col not in seen]
    st.dataframe(anomdf[front + rest], use_container_width=True, height=300)

    rsum = read_csv_cached(os.path.join(run, "region_summary.csv"))
    if not rsum.empty:
        with st.expander("What the flagged groups have in common (services, destinations, verdict)"):
            st.dataframe(rsum, use_container_width=True, height=280)

    # ---- the comparison: flagged flow vs its closest normal flows ----
    st.divider()
    st.markdown("### Compare a suspicious flow with its closest normal flows")
    if not os.path.exists(flow_parquet):
        st.info(f"Set **Flow data parquet** in the sidebar to the file this run used "
                f"(`{flow_parquet}` not found) to draw the comparison."); return
    id_col = "conn_id" if "conn_id" in anomdf.columns else ("cid" if "cid" in anomdf.columns else None)
    if id_col is None:
        st.info("No conn_id/cid column available to look up the flows."); return

    all_ch = config_channels(cfg, ("packet_count",))
    channels = st.multiselect("Channels to show", all_ch, default=all_ch)
    k = st.slider("How many normal flows to compare against", 1, 8, 5)
    if not channels:
        st.warning("Select at least one channel."); return

    sc_col = score_col if score_col in anomdf.columns else ("D_i" if "D_i" in anomdf.columns else None)
    adf = anomdf.copy()
    if sc_col:
        adf = adf.sort_values(sc_col, ascending=False)
    labs = []
    for _, r in adf.head(300).iterrows():
        s = r.get(sc_col, np.nan) if sc_col else np.nan
        labs.append(f"{str(r[id_col])[:24]} | {score_label}={s:.3g} | "
                    f"{r.get('oui', '')} {r.get('protocol', '')} dst={r.get('dst_port', '')} "
                    f"{r.get('service', '')}")
    sel = st.selectbox("Pick a suspicious flow to investigate", labs, index=None, placeholder="Select a flow…")
    if sel is None:
        return
    row = adf.head(300).iloc[labs.index(sel)]
    a_cid = str(row[id_col])
    normals = nearest_normal_cids(nearest_kind, run, row, id_col, k)

    pq_df = load_flow_parquet(flow_parquet)
    frames = get_flow_frames(pq_df, [a_cid] + normals, channels, bin_freq=bin_freq)
    if a_cid not in frames:
        st.warning(f"The flagged flow `{a_cid}` is not in the flow data parquet — can't draw it. "
                   f"Check that the parquet is the one this run used."); return

    targets = [{"cid": a_cid, "color": TH["kde"],
                "label": f"<b>⚠ FLAGGED</b><br>{frames[a_cid]['protocol']} "
                         f"{frames[a_cid]['src_port']}→{frames[a_cid]['dst_port']}<br>"
                         f"{score_label}={float(row.get(sc_col, np.nan)) if sc_col else float('nan'):.3g}"}]
    missing = 0
    for i, nc in enumerate(normals):
        if nc in frames:
            fdn = frames[nc]
            targets.append({"cid": nc, "color": TH["normal"],
                            "label": f"normal #{i+1}<br>{fdn['protocol']} {fdn['src_port']}→{fdn['dst_port']}"})
        else:
            missing += 1
    st.caption("**Top row (red) = the flagged flow.** The blue rows below are the most similar normal flows "
               "found by the method — compare the shapes to see what makes the flagged one stand out.")
    fig = plot_flows_grid(targets, frames, channels)
    if fig is not None:
        show_scroll(fig)
    if not normals:
        st.info("No pre-computed nearest-normal list for this flow in this run.")
    elif missing:
        st.caption(f"({missing} of the {len(normals)} nearest normal flows were not found in the parquet.)")

    with st.expander("See the flagged flow on its own (larger)"):
        t = {"cid": a_cid, "d": float(row.get(sc_col, np.nan)) if sc_col else 0.0, "tag": "⚠ FLAGGED flow"}
        show(plot_flow_detail(t, frames[a_cid], channels, TH["kde"], holdout=0))


def page_kl():
    page_flow_experiment(
        "KL anomaly — flows",
        "Flows whose behaviour differs most from the normal baseline (anomaly_kl.py). "
        "Pick a flagged flow to compare it against its closest normal matches.",
        DEF_KL_RUNS, "Runs dir (results/anomaly_kl)", "D_i", "difference score", nearest_kind="kl")


def page_kde():
    page_flow_experiment(
        "KDE anomaly — flows",
        "Flows that look unlike the bulk of normal traffic (kde_anomaly.py). "
        "Pick a flagged flow to compare it against its closest normal matches.",
        DEF_KDE_RUNS, "Runs dir (results/kde)", "neg_log_density", "unusualness score", nearest_kind="kde")


# ============================================================================
# PAGE — KL per-split flows   [original dashboard.py: results_dashboard/<run>/split*/scores_all.csv]
# ============================================================================
@st.cache_data(show_spinner=False)
def split_load_scores(run_folder):
    """Original anomaly_kl split layout: concat every split*/scores_all.csv with a 'split' column."""
    dfs = []
    for sf in sorted(glob.glob(os.path.join(run_folder, "split*"))):
        f = os.path.join(sf, "scores_all.csv")
        if os.path.exists(f):
            d = pd.read_csv(f)
            d["split"] = os.path.basename(sf)
            dfs.append(d)
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


def split_read_summary(run_folder, split):
    path = os.path.join(run_folder, split, "summary.txt")
    channels = ["packet_count", "mean_iat", "mean_packet_size"]
    holdout = 30
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                if line.startswith("channels:"):
                    s = line.split("channels:")[1].strip().strip("[]").replace("'", "")
                    channels = [c.strip() for c in s.split(",") if c.strip()]
                if "holdout" in line:
                    parts = line.split()
                    try:
                        holdout = int(parts[parts.index("holdout") + 1])
                    except Exception:
                        pass
    return channels, holdout


def plot_split_compact(targets, flows, channels, color, holdout, row_h=70, col_w=210):
    """Small multiples (one row per flow, one column per channel) with a holdout vline —
    the original dashboard.py view, drawn in our theme."""
    targets = [t for t in targets if t["cid"] in flows]
    if not targets:
        return None
    R, C = len(targets), len(channels)
    height = row_h * R + 70
    left, right = 120, 10
    width = left + right + col_w * C
    fig = make_subplots(rows=R, cols=C, column_titles=[pretty(c) for c in channels],
                        vertical_spacing=min(0.5 / R, 25 / height), horizontal_spacing=0.025)
    for ri, t in enumerate(targets):
        f = flows[t["cid"]]
        x = f["index"]
        h = len(x) - holdout
        for ci in range(C):
            fig.add_trace(go.Scatter(x=x, y=f["frame"][:, ci], mode="lines",
                                     line=dict(color=color, width=1),
                                     hovertemplate="%{x|%m-%d %H:%M}<br>%{y:.4g}<extra></extra>",
                                     showlegend=False), row=ri + 1, col=ci + 1)
            if holdout and h > 0:
                fig.add_vline(x=x[h], line_width=1, line_dash="dot", line_color=TH["vline"], row=ri + 1, col=ci + 1)
        fig.update_yaxes(title_text=f"<b>{t['tag']}</b><br>D={t['d']:.1e}<br>"
                                    f"{f['protocol']} {f['src_port']}→{f['dst_port']}",
                         title_font=dict(size=9, color=TH["fg"]), title_standoff=2, row=ri + 1, col=1)
    fig.update_xaxes(showticklabels=False, showgrid=True, gridcolor=TH["grid"], linecolor=TH["grid"])
    fig.update_xaxes(showticklabels=True, tickformat="%H:%M", nticks=3, tickangle=0,
                     tickfont=dict(size=8, color=TH["fg"]), row=R)
    fig.update_yaxes(showgrid=True, gridcolor=TH["grid"], linecolor=TH["grid"], nticks=3,
                     tickfont=dict(size=8, color=TH["fg"]), zeroline=False)
    fig.update_annotations(font=dict(size=10, color=TH["fg"]))
    fig.update_layout(height=height, width=width, margin=dict(l=left, r=right, t=30, b=30),
                      paper_bgcolor=TH["bg"], plot_bgcolor=TH["bg"], font=FONT, hovermode="closest")
    return fig


def page_kl_split():
    st.title("KL OUI anomaly — flows")
    st.caption("The original per-OUI KL view: flagged flows next to the most borderline normal flows, "
               "with the dotted line marking where the held-out window begins (reads results_dashboard/).")
    st.sidebar.header("Data")
    runs_dir = st.sidebar.text_input("Runs dir (results_dashboard)", DEF_SPLIT_RUNS)
    data_dir = st.sidebar.text_input("Data folder", DATA_DIR)
    flow_parquet = pick_parquet("Flow data parquet", data_dir, DEF_FLOW_PARQUET)
    runs = list_runs(runs_dir)
    if not runs:
        st.warning(f"No run folders under `{runs_dir}`."); return
    run = os.path.join(runs_dir, st.sidebar.selectbox("Select run", runs))

    scores = split_load_scores(run)
    if scores.empty:
        st.warning("No split*/scores_all.csv in this run."); return
    if "sig_anomaly" not in scores.columns:
        st.warning("scores_all.csv has no sig_anomaly column — unexpected format."); return

    have_pq = os.path.exists(flow_parquet)
    pq_df = load_flow_parquet(flow_parquet) if have_pq else None
    if have_pq:
        keep = [c for c in ["conn_id", "protocol", "src_port", "dst_port"] if c in pq_df.columns]
        meta = (pq_df[pq_df["conn_id"].astype(str).isin(scores["cid"].astype(str))][keep]
                .drop_duplicates("conn_id"))
        scores = scores.merge(meta, left_on=scores["cid"].astype(str),
                              right_on=meta["conn_id"].astype(str), how="left", suffixes=("", "_m"))
    else:
        st.warning(f"Flow data parquet not found: `{flow_parquet}` — table only.")

    anom = scores[scores["sig_anomaly"] == True]
    c = st.columns(3)
    c[0].metric("Flows scored", f"{len(scores):,}")
    c[1].metric("Flagged as suspicious", f"{len(anom):,}")
    c[2].metric("Splits", f"{scores['split'].nunique():,}")

    st.sidebar.header("Filters")
    prot = st.sidebar.selectbox("Protocol", ["All"] + sorted(anom["protocol"].dropna().astype(str).unique())) \
        if "protocol" in anom.columns else "All"
    ports = sorted(set(anom.get("src_port", pd.Series(dtype=str)).dropna().astype(str)) |
                   set(anom.get("dst_port", pd.Series(dtype=str)).dropna().astype(str)))
    port = st.sidebar.selectbox("Port", ["All"] + ports)
    fa = anom.copy()
    if prot != "All" and "protocol" in fa.columns:
        fa = fa[fa["protocol"].astype(str) == prot]
    if port != "All":
        fa = fa[(fa.get("src_port", pd.Series(index=fa.index)).astype(str) == port) |
                (fa.get("dst_port", pd.Series(index=fa.index)).astype(str) == port)]

    pairs = fa[["oui", "split"]].drop_duplicates().sort_values(["oui", "split"])
    opts = [f"{r['oui']} ({r['split']})" for _, r in pairs.iterrows()]
    st.sidebar.header("Flagged OUIs")
    if not opts:
        st.sidebar.write("No flagged OUIs match the filters."); return
    sel = st.sidebar.radio("Select OUI", opts)
    sel_oui, sel_split = sel.split(" (")[0], sel.split(" (")[1][:-1]
    st.subheader(f"OUI: {sel_oui}  |  Split: {sel_split}")

    channels_all, holdout = split_read_summary(run, sel_split)
    st.sidebar.header("Display")
    channels = st.sidebar.multiselect("Channels", channels_all, default=channels_all)
    max_flows = st.sidebar.slider("Max flows per section", 1, 20, 6)
    row_h = st.sidebar.slider("Row height (px)", 60, 220, 70, step=5)
    if not channels:
        st.warning("Select at least one channel."); return

    af = scores[(scores["sig_anomaly"] == True) & (scores["oui"] == sel_oui) & (scores["split"] == sel_split)]
    af = af.sort_values(["region_id", "D_i"], ascending=[True, False]).head(max_flows)
    na = scores[(scores["oui"] == sel_oui) & (scores["split"] == sel_split) &
                (scores["sig_anomaly"] == False) & (scores.get("set", "A") == "A")]
    na = na.sort_values("D_i", ascending=False).head(max_flows)
    t_anom = [{"cid": str(r["cid"]), "d": r["D_i"], "tag": f"R{r.get('region_id', '?')}"} for _, r in af.iterrows()]
    t_norm = [{"cid": str(r["cid"]), "d": r["D_i"], "tag": f"#{i+1}"} for i, (_, r) in enumerate(na.iterrows())]

    if not have_pq:
        st.dataframe(af, use_container_width=True, height=340); return

    flows = get_flow_frames(pq_df, [t["cid"] for t in t_anom + t_norm], channels)
    st.caption(f"Channels: {', '.join(channels)}  ·  dotted line = holdout start ({holdout} bins before end).  "
               f"Red = flagged, blue = the most borderline normal flows for comparison.")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"#### 🔴 Flagged flows ({len(t_anom)})")
        fig = plot_split_compact(t_anom, flows, channels, TH["kde"], holdout, row_h)
        show_scroll(fig) if fig else st.write("No flagged flows.")
    with col2:
        st.markdown(f"#### 🔵 Most borderline normal flows ({len(t_norm)})")
        fig = plot_split_compact(t_norm, flows, channels, TH["normal"], holdout, row_h)
        show_scroll(fig) if fig else st.write("No normal flows.")

    st.divider()
    st.markdown("### 🔍 Inspect a flow (all channels)")
    allt = ([("FLAGGED", t, TH["kde"]) for t in t_anom] + [("normal", t, TH["normal"]) for t in t_norm])
    allt = [x for x in allt if x[1]["cid"] in flows]
    if allt:
        labs = []
        for kind, t, _ in allt:
            f = flows[t["cid"]]
            labs.append(f"{kind} {t['tag']} | {f['protocol']} {f['src_port']}→{f['dst_port']} "
                        f"| D={t['d']:.2e} | {t['cid'][:18]}")
        ch = st.selectbox("Choose a flow", labs, index=None, placeholder="Select a flow…")
        if ch is not None:
            kind, t, color = allt[labs.index(ch)]
            t = {**t, "tag": f"{kind} {t['tag']}"}
            show(plot_flow_detail(t, flows[t["cid"]], channels, color, holdout))
    else:
        st.write("No flows available to inspect.")


# ============================================================================
# PAGE — Forecast sliding-window anomalies   [forecast_sliding_anomaly.py]
# ============================================================================
@st.cache_data(show_spinner=False)
def sliding_list(runs_dir):
    """Every series with a sliding_scores.csv -> (connection, target, path)."""
    rows = []
    if os.path.isdir(runs_dir):
        for conn in sorted(os.listdir(runs_dir)):
            cdir = os.path.join(runs_dir, conn)
            if not os.path.isdir(cdir):
                continue
            for tgt in sorted(os.listdir(cdir)):
                f = os.path.join(cdir, tgt, "sliding_scores.csv")
                if os.path.exists(f):
                    rows.append((conn, tgt, f))
    return pd.DataFrame(rows, columns=["connection", "target", "path"])


@st.cache_data(show_spinner=False)
def sliding_index(runs_dir):
    """Map (connection, target) -> {representation_tag: scores_path} across ALL representation
    folders. The same flow (same conn_id) appears under several reps (flow 1min / 5min), so this
    lets the dashboard show one flow's anomalies side by side across representations."""
    idx = {}
    for tag, tdir in sliding_reps(runs_dir):
        for conn in sorted(os.listdir(tdir)):
            cdir = os.path.join(tdir, conn)
            if not os.path.isdir(cdir):
                continue
            for tgt in sorted(os.listdir(cdir)):
                f = os.path.join(cdir, tgt, "sliding_scores.csv")
                if os.path.exists(f):
                    idx.setdefault((conn, tgt), {})[tag] = f
    return idx


@st.cache_data(show_spinner=False)
def sliding_reps(runs_dir):
    """Representation folders under results/sliding (one per run tag: mode + data + C/H).
    Falls back to a single '(all)' entry for the older flat layout."""
    reps = []
    if not os.path.isdir(runs_dir):
        return reps
    for tag in sorted(os.listdir(runs_dir)):
        tdir = os.path.join(runs_dir, tag)
        if os.path.isdir(tdir) and glob.glob(os.path.join(tdir, "*", "*", "sliding_scores.csv")):
            reps.append((tag, tdir))
    if not reps and glob.glob(os.path.join(runs_dir, "*", "*", "sliding_scores.csv")):
        reps.append(("(all)", runs_dir))
    return reps


@st.cache_data(show_spinner=False)
def sliding_summary(runs_dir):
    dfs = []
    for f in glob.glob(os.path.join(runs_dir, "summary*.csv")):   # summary.csv (per-tag) or summary_*.csv (old)
        try:
            dfs.append(pd.read_csv(f))
        except Exception:
            pass
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


@st.cache_data(show_spinner=False)
def sliding_load_scores(path):
    df = pd.read_csv(path)
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    for c in df.columns:
        if c.startswith("is_anom"):
            df[c] = df[c].map(lambda v: str(v).strip().lower() in ("true", "1", "1.0")).astype(bool)
    return df


def sliding_plot(df, flag_col, title, col_w=1400, row_h=440):
    fig = go.Figure()
    # prediction band
    fig.add_trace(go.Scatter(x=df["datetime"], y=df["upper"], mode="lines", line=dict(width=0),
                             showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=df["datetime"], y=df["lower"], mode="lines", line=dict(width=0),
                             fill="tonexty", fillcolor=TH["shade_kl"], name="prediction interval",
                             hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=df["datetime"], y=df["truth"], mode="lines",
                             line=dict(color=TH["normal"], width=1), name="actual",
                             hovertemplate="%{x|%m-%d %H:%M}<br>%{y:.4g}<extra></extra>"))
    fig.add_trace(go.Scatter(x=df["datetime"], y=df["pred"], mode="lines",
                             line=dict(color=TH["kl"], width=1), name="forecast (median)",
                             hovertemplate="%{x|%m-%d %H:%M}<br>%{y:.4g}<extra></extra>"))
    if flag_col in df.columns and df[flag_col].any():
        a = df[df[flag_col]]
        fig.add_trace(go.Scatter(x=a["datetime"], y=a["truth"], mode="markers",
                                 marker=dict(color=TH["kde"], size=9, symbol="circle-open", line=dict(width=2)),
                                 name=f"anomaly ({int(df[flag_col].sum())})",
                                 hovertemplate="anomaly<br>%{x|%m-%d %H:%M}<br>%{y:.4g}<extra></extra>"))
    fig.update_xaxes(showgrid=True, gridcolor=TH["grid"], linecolor=TH["grid"], tickfont=dict(size=9, color=TH["fg"]))
    fig.update_yaxes(showgrid=True, gridcolor=TH["grid"], linecolor=TH["grid"], tickfont=dict(size=9, color=TH["fg"]), zeroline=False)
    fig.update_layout(height=row_h, width=col_w, margin=dict(l=55, r=15, t=45, b=35),
                      paper_bgcolor=TH["bg"], plot_bgcolor=TH["bg"], font=FONT, hovermode="x unified",
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, font=dict(size=10)),
                      title=dict(text=title, font=dict(size=12, color=TH["fg"])))
    return fig


@st.cache_data(show_spinner=False)
def sliding_load_series(series_dir):
    f = os.path.join(series_dir, "sliding_series.csv")
    if not os.path.exists(f):
        return pd.DataFrame()
    d = pd.read_csv(f)
    d["datetime"] = pd.to_datetime(d["datetime"], errors="coerce")
    return d


@st.cache_data(show_spinner=False)
def sliding_load_meta(series_dir):
    f = os.path.join(series_dir, "sliding_meta.json")
    try:
        return json.load(open(f)) if os.path.exists(f) else {}
    except Exception:
        return {}


def sliding_windows_plot(series_df, stream, meta, flag_col, sel_win, show_bounds, title, col_w=1400, row_h=470):
    C = int(meta.get("context", 0))
    H = int(meta.get("horizon", 0))
    starts = meta.get("starts", [])
    t = series_df["datetime"].to_numpy()
    y = series_df["actual"].to_numpy()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=stream["datetime"], y=stream["upper"], mode="lines", line=dict(width=0),
                             showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=stream["datetime"], y=stream["lower"], mode="lines", line=dict(width=0),
                             fill="tonexty", fillcolor=TH["shade_kl"], name="prediction interval", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=t, y=y, mode="lines", line=dict(color=TH["normal"], width=1), name="actual",
                             hovertemplate="%{x|%m-%d %H:%M}<br>%{y:.4g}<extra></extra>"))
    fig.add_trace(go.Scatter(x=stream["datetime"], y=stream["pred"], mode="lines", line=dict(color=TH["kl"], width=1),
                             name="forecast (median)", hovertemplate="%{x|%m-%d %H:%M}<br>%{y:.4g}<extra></extra>"))
    if show_bounds and starts:
        for s in starts:
            if 0 <= s < len(t):
                fig.add_vline(x=t[s], line_width=0.5, line_dash="dot", line_color=TH["grid"])
    if sel_win is not None and starts and 0 <= sel_win < len(starts):
        s = starts[sel_win]
        fig.add_vrect(x0=t[max(0, s - C)], x1=t[max(0, s - 1)], fillcolor=TH["shade_both"], line_width=0,
                      layer="below", annotation_text="context", annotation_position="top left",
                      annotation_font=dict(size=10, color=TH["fg"]))
        fig.add_vrect(x0=t[min(len(t) - 1, s)], x1=t[min(len(t) - 1, s + H - 1)], fillcolor=TH["shade_kde"],
                      line_width=0, layer="below", annotation_text="horizon", annotation_position="top right",
                      annotation_font=dict(size=10, color=TH["fg"]))
    if flag_col in stream.columns and stream[flag_col].any():
        a = stream[stream[flag_col]]
        fig.add_trace(go.Scatter(x=a["datetime"], y=a["truth"], mode="markers",
                                 marker=dict(color=TH["kde"], size=9, symbol="circle-open", line=dict(width=2)),
                                 name=f"anomaly ({int(stream[flag_col].sum())})",
                                 hovertemplate="anomaly<br>%{x|%m-%d %H:%M}<br>%{y:.4g}<extra></extra>"))
    fig.update_xaxes(showgrid=True, gridcolor=TH["grid"], linecolor=TH["grid"], tickfont=dict(size=9, color=TH["fg"]))
    fig.update_yaxes(showgrid=True, gridcolor=TH["grid"], linecolor=TH["grid"], tickfont=dict(size=9, color=TH["fg"]), zeroline=False)
    fig.update_layout(height=row_h, width=col_w, margin=dict(l=55, r=15, t=45, b=35),
                      paper_bgcolor=TH["bg"], plot_bgcolor=TH["bg"], font=FONT, hovermode="x unified",
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, font=dict(size=10)),
                      title=dict(text=title, font=dict(size=12, color=TH["fg"])))
    return fig


def sliding_features_plot(series_df, stream, flag_col, channels, target, title, col_w=1400):
    """One stacked row per feature channel (shared x). Only the target row carries the forecast
    band + red ○ anomaly markers; the other channels are drawn as context (actual only)."""
    chans = [target] + [c for c in channels if c != target]
    n = len(chans)
    titles = [(f"★ {c}  —  forecast + anomalies" if c == target else c) for c in chans]
    fig = make_subplots(rows=n, cols=1, shared_xaxes=True, subplot_titles=titles,
                        vertical_spacing=min(0.07, 0.5 / max(1, n)))
    for i, ch in enumerate(chans, start=1):
        if ch == target:
            fig.add_trace(go.Scatter(x=stream["datetime"], y=stream["upper"], mode="lines",
                                     line=dict(width=0), showlegend=False, hoverinfo="skip"), row=i, col=1)
            fig.add_trace(go.Scatter(x=stream["datetime"], y=stream["lower"], mode="lines", line=dict(width=0),
                                     fill="tonexty", fillcolor=TH["shade_kl"], name="prediction interval",
                                     hoverinfo="skip"), row=i, col=1)
            yact = series_df[ch] if ch in series_df.columns else series_df.get("actual")
            fig.add_trace(go.Scatter(x=series_df["datetime"], y=yact, mode="lines",
                                     line=dict(color=TH["normal"], width=1), name="actual",
                                     hovertemplate="%{x|%m-%d %H:%M}<br>%{y:.4g}<extra></extra>"), row=i, col=1)
            fig.add_trace(go.Scatter(x=stream["datetime"], y=stream["pred"], mode="lines",
                                     line=dict(color=TH["kl"], width=1), name="forecast (median)",
                                     hovertemplate="%{x|%m-%d %H:%M}<br>%{y:.4g}<extra></extra>"), row=i, col=1)
            if flag_col and flag_col in stream.columns and stream[flag_col].any():
                a = stream[stream[flag_col]]
                fig.add_trace(go.Scatter(x=a["datetime"], y=a["truth"], mode="markers",
                                         marker=dict(color=TH["kde"], size=8, symbol="circle-open", line=dict(width=2)),
                                         name=f"anomaly ({int(stream[flag_col].sum())})",
                                         hovertemplate="anomaly<br>%{x|%m-%d %H:%M}<br>%{y:.4g}<extra></extra>"),
                              row=i, col=1)
        else:
            fig.add_trace(go.Scatter(x=series_df["datetime"], y=series_df[ch], mode="lines",
                                     line=dict(color=TH["fg"], width=1), name=ch, showlegend=False, opacity=0.85,
                                     hovertemplate="%{x|%m-%d %H:%M}<br>%{y:.4g}<extra></extra>"), row=i, col=1)
        fig.update_yaxes(showgrid=True, gridcolor=TH["grid"], linecolor=TH["grid"],
                         tickfont=dict(size=9, color=TH["fg"]), zeroline=False, row=i, col=1)
    fig.update_xaxes(showgrid=True, gridcolor=TH["grid"], linecolor=TH["grid"],
                     tickfont=dict(size=9, color=TH["fg"]))
    fig.update_layout(height=max(320, 185 * n), width=col_w, margin=dict(l=55, r=15, t=55, b=35),
                      paper_bgcolor=TH["bg"], plot_bgcolor=TH["bg"], font=FONT, hovermode="x unified",
                      legend=dict(orientation="h", yanchor="bottom", y=1.015, x=0, font=dict(size=10)),
                      title=dict(text=title, font=dict(size=12, color=TH["fg"])))
    for ann in fig.layout.annotations:          # subplot titles -> theme color, smaller
        ann.font = dict(size=11, color=TH["fg"])
    return fig


def sliding_live_flags(df, live):
    """Recompute an is_anom boolean array from the STORED raw columns, live in-app, no re-run.
       live = {"mode": "error", "val": tau}       -> normalized forecast error > tau
              {"mode": "interval", "val": k}       -> truth outside the interval widened/narrowed by k
    Methods 1/3/4 use expanding baselines that are not stored, so they can't be retuned exactly;
    these two knobs are fully reconstructable from truth/pred/lower/upper (+ the saved score)."""
    if live["mode"] == "error":
        sc = next((df[c] for c in df.columns if c.startswith("score_m")), None)
        if sc is None:
            d = (df["truth"] - df["pred"]).abs()
            sc = d / (d.max() + 1e-9)
        return (sc.to_numpy() > live["val"])
    k = live["val"]
    up = df["pred"] + k * (df["upper"] - df["pred"])
    lo = df["pred"] - k * (df["pred"] - df["lower"])
    return ((df["truth"] > up) | (df["truth"] < lo)).to_numpy()


def sliding_render_series(scores_path, sel_flag, label, key_prefix="", allow_windows=True, live=None):
    """Metrics row + whole-series/windows plot for ONE series. Reused by both the
    single-representation view and the per-flow cross-representation comparison.
    If `live` is given, anomalies are recomputed in-app at that threshold (no re-run)."""
    df = sliding_load_scores(scores_path)
    series_dir = os.path.dirname(scores_path)
    meta = sliding_load_meta(series_dir)
    sdf_full = sliding_load_series(series_dir)
    target = meta.get("target", "packet_count")
    # ONLY the channels that were fed to the multivariate model (recorded in meta["channels"]).
    # Never fall back to "every column in the file" — if the model-input list isn't recorded
    # (older runs), show just the target rather than unrelated features.
    chans = [c for c in (meta.get("channels") or []) if c in sdf_full.columns]
    if not chans and not sdf_full.empty and (target in sdf_full.columns or "actual" in sdf_full.columns):
        chans = [target]

    flags_present = [c for c in df.columns if c.startswith("is_anom")]
    if live is not None:
        df = df.copy()
        df["is_anom_live"] = sliding_live_flags(df, live)
        flag_col = "is_anom_live"
    else:
        flag_col = "is_anom_consensus" if sel_flag == "consensus" else f"is_anom_{sel_flag}"

    cols = st.columns(2 + len(flags_present))
    cols[0].metric("Evaluated points", f"{len(df):,}")
    cols[1].metric("live" if live is not None else sel_flag,
                   f"{int(df[flag_col].sum()):,}" if flag_col in df.columns else "—",
                   help="recomputed in-app at the live threshold" if live is not None else None)
    for i, c in enumerate(flags_present):
        lbl = "consensus" if c == "is_anom_consensus" else c.replace("is_anom_", "")
        cols[i + 2].metric(lbl, f"{int(df[c].sum()):,}")
    if flag_col not in df.columns:
        st.info(f"`{flag_col}` not in this run; showing method 0 instead.")
        flag_col = "is_anom_m0" if "is_anom_m0" in df.columns else (flags_present[0] if flags_present else None)

    view = ("Whole series (stream)" if not allow_windows else
            st.radio("View", ["Whole series (stream)", "Windows (context + horizon)"],
                     horizontal=True, key=key_prefix + "view"))
    if view.startswith("Whole"):
        show_all = len(chans) > 1 and st.checkbox("Show all model-input channels", value=True,
                                                  key=key_prefix + "allfeat")
        if show_all:
            st.caption(f"One row per **model-input channel** ({', '.join(chans)}). The forecast band + "
                       f"red ○ anomalies are on the target (**{target}**) only; the other channels were fed "
                       f"to the multivariate model and are shown as context (actual only).")
            show_scroll(sliding_features_plot(sdf_full, df, flag_col, chans, target, label))
        else:
            st.caption("Blue = actual, orange = forecast (median), shaded = prediction interval, "
                       "red ○ = flagged by the selected method.")
            show_scroll(sliding_plot(df, flag_col, label))
    else:
        if sdf_full.empty or not meta.get("starts"):
            st.info("Per-window view needs sliding_series.csv + sliding_meta.json in the series folder — "
                    "re-run forecast_sliding_anomaly.py to generate them. Showing the whole-series stream instead.")
            show_scroll(sliding_plot(df, flag_col, label))
        else:
            starts = meta["starts"]
            show_bounds = st.checkbox("Show every window boundary", value=True, key=key_prefix + "bounds")
            sel_win = st.slider("Window (context + horizon highlighted)", 0, max(0, len(starts) - 1), 0,
                                key=key_prefix + "win")
            s = starts[sel_win]
            st.caption(f"Window {sel_win} of {len(starts) - 1}: context = {meta['context']} bins ending at index {s}; "
                       f"horizon = next {meta['horizon']} bins.  Purple band = context the model saw, "
                       f"red band = predicted horizon.  Orange = forecast, red ○ = anomaly.")
            show_scroll(sliding_windows_plot(sdf_full, df, meta, flag_col, sel_win, show_bounds,
                                             f"{label} · window {sel_win}/{len(starts)-1}"))

    # table of the flagged points (uses the active flag, live or saved)
    if flag_col and flag_col in df.columns and df[flag_col].any():
        with st.expander(f"Flagged points ({int(df[flag_col].sum())})", expanded=False):
            sc_col = next((c for c in df.columns if c.startswith("score_m")), None)
            keep = ["datetime", "truth", "pred", "lower", "upper"] + ([sc_col] if sc_col else []) + \
                   (["votes"] if "votes" in df.columns else [])
            fdf = df[df[flag_col]][keep]
            st.dataframe(fdf.sort_values(sc_col or "datetime", ascending=False) if sc_col else fdf,
                         use_container_width=True, height=280, hide_index=True)


def page_sliding():
    st.title("Forecast — sliding window")
    st.caption("Causal sliding-window forecast anomalies (forecast_sliding_anomaly.py): the model predicts "
               "each window from its recent history; a point far from the forecast / outside its interval is flagged.")
    st.sidebar.header("Data")
    runs_dir = st.sidebar.text_input("Runs dir (results/sliding)", DEF_SLIDING_RUNS)
    reps = sliding_reps(runs_dir)
    if not reps:
        st.warning(f"No sliding results under `{runs_dir}`."); return

    # which flag to highlight (shared by both modes)
    flag_opts = ["consensus", "m0", "m1", "m2", "m3", "m4", "m5"]
    sel_flag = st.sidebar.selectbox("Highlight", flag_opts, index=0)
    anom_col = "anom_consensus" if sel_flag == "consensus" else f"anom_{sel_flag}"

    # live threshold tuning (recomputed in-app from stored errors/interval — no re-run)
    live = None
    tune = st.sidebar.checkbox("Tune threshold live (no re-run)", value=False,
                               help="Recompute anomalies in the app from the saved forecast errors / "
                                    "prediction interval. Overrides the Highlight method above.")
    if tune:
        basis = st.sidebar.radio("Live basis", ["Normalized error", "Interval width"],
                                 help="Normalized error = |actual − forecast| / global range. "
                                      "Interval width = widen/narrow the prediction band.")
        if basis.startswith("Normalized"):
            val = st.sidebar.slider("Error threshold τ  (lower = more anomalies)", 0.0, 1.0, 0.30, 0.01)
            live = {"mode": "error", "val": float(val)}
        else:
            val = st.sidebar.slider("Interval × k  (lower = more anomalies)", 0.2, 3.0, 1.0, 0.05)
            live = {"mode": "interval", "val": float(val)}
        st.sidebar.caption("Note: methods m1/m3/m4 use expanding baselines not stored on disk, so they "
                           "can't be retuned exactly; these two knobs are exact from the saved columns.")

    mode = st.sidebar.radio("Browse", ["By representation", "Compare one flow across representations"])

    # ---- Compare one flow across representations (same conn_id in flow_1min / flow_5min / ...) ----
    if mode.startswith("Compare"):
        idx = sliding_index(runs_dir)
        if not idx:
            st.warning("No series found."); return
        flows = sorted(idx.keys(), key=lambda k: (-len(idx[k]), k[0], k[1]))  # most-represented first
        st.sidebar.header("Flow")
        only_multi = st.sidebar.checkbox("Only flows present in ≥2 representations", value=True)
        view_flows = [k for k in flows if (not only_multi or len(idx[k]) >= 2)]
        if not view_flows:
            st.info("No flow is present in more than one representation yet. Re-run the other representations "
                    "(they share conn_ids) so the same flow can be compared here. Showing all flows for now.")
            view_flows = flows
        view_labels = [f"{c}  /  {t}   —  {len(idx[(c, t)])} rep: {', '.join(sorted(idx[(c, t)]))}"
                       for (c, t) in view_flows]
        sel = st.sidebar.selectbox("Pick a flow", view_labels)
        conn, tgt = view_flows[view_labels.index(sel)]
        reps_here = idx[(conn, tgt)]

        st.subheader(f"{conn}  ·  {tgt}")
        st.caption("Same flow, one block per representation — compare which points each binning (and its C/H) "
                   f"flags as anomalous. Highlight method: **{sel_flag}**.")
        chosen = st.multiselect("Representations to show", sorted(reps_here), default=sorted(reps_here))
        if not chosen:
            st.info("Select at least one representation."); return

        # compact side-by-side anomaly counts for the picked flow
        crows = []
        for tag in sorted(reps_here):
            d = sliding_load_scores(reps_here[tag])
            m = sliding_load_meta(os.path.dirname(reps_here[tag]))
            row = {"representation": tag, "C": m.get("context", ""), "H": m.get("horizon", ""),
                   "eval_points": len(d)}
            if live is not None:
                row["live"] = int(sliding_live_flags(d, live).sum())
            for c in d.columns:
                if c.startswith("is_anom"):
                    row[("consensus" if c == "is_anom_consensus" else c.replace("is_anom_", ""))] = int(d[c].sum())
            crows.append(row)
        st.dataframe(pd.DataFrame(crows), use_container_width=True, hide_index=True)

        for tag in chosen:
            meta = sliding_load_meta(os.path.dirname(reps_here[tag]))
            ch = f" · C{meta.get('context', '?')} H{meta.get('horizon', '?')}" if meta else ""
            st.markdown(f"### `{tag}`{ch}")
            sliding_render_series(reps_here[tag], sel_flag, f"{tag}  ·  {conn} · {tgt} · highlight: {sel_flag}",
                                  key_prefix=f"cmp_{tag}_", allow_windows=True, live=live)
            st.divider()
        return

    # ---- By representation (default) ----
    rep_label = st.sidebar.selectbox("Representation", [r[0] for r in reps],
                                     help="one entry per run: per-flow 1min / 5min / router×oui, each with its C and H")
    base = dict(reps)[rep_label]
    st.markdown(f"**Representation:** `{rep_label}`")
    lst = sliding_list(base)
    if lst.empty:
        st.warning("No series in this representation."); return
    summ = sliding_summary(base)

    # run-level overview (from summary files if present)
    if not summ.empty:
        st.markdown("#### Overview")
        count_cols = [c for c in summ.columns if c.startswith("anom_")]
        show_cols = [c for c in ["connection", "target", "n_bins", "n_windows", "n_eval"] if c in summ.columns] + count_cols
        st.dataframe(summ[show_cols].sort_values(
            next((c for c in summ.columns if c.startswith("anom_consensus")), count_cols[0] if count_cols else "connection"),
            ascending=False) if count_cols else summ[show_cols],
            use_container_width=True, height=240)

    # rank the series for the picker by the selected flag's count (from summary), else alphabetical
    lst = lst.copy()
    if not summ.empty:
        key = None
        for c in summ.columns:
            if c == anom_col or (sel_flag == "consensus" and c.startswith("anom_consensus")):
                key = c; break
        if key:
            m = summ.set_index(["connection", "target"])[key].to_dict()
            lst["rank"] = [m.get((r.connection, r.target), -1) for r in lst.itertuples(index=False)]
            lst = lst.sort_values("rank", ascending=False)

    st.sidebar.header("Series")
    labels = [f"{r.connection}  /  {r.target}" + (f"   [{int(r.rank)}]" if "rank" in lst.columns and r.rank >= 0 else "")
              for r in lst.itertuples(index=False)]
    sel = st.sidebar.radio("Select series", labels)
    row = lst.iloc[labels.index(sel)]
    st.subheader(f"{row['connection']}  ·  {row['target']}")
    sliding_render_series(row["path"], sel_flag,
                          f"{row['connection']} · {row['target']} · highlight: {sel_flag}",
                          key_prefix="single_", allow_windows=True, live=live)


# ============================================================================
# Dispatch
# ============================================================================
st.sidebar.title("Anomaly dashboards")
EXPERIMENTS = {
    "OUI × Router — windows": page_oui_router,
    "KL anomaly — flows": page_kl,
    "KL OUI anomaly — flows": page_kl_split,
    "KDE anomaly — flows": page_kde,
    "Forecast — sliding window": page_sliding,
}
choice = st.sidebar.radio("Experiment", list(EXPERIMENTS.keys()), index=0)
st.sidebar.divider()
EXPERIMENTS[choice]()