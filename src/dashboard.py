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
  • KL OUI anomaly — flows           : per-split KL-OUI permutation experiment
        results/anomaly_kl_oui/<run>/split*/scores_all.csv (+ summary.txt per split;
        known_manufacturer/ and randomized_mac/ are sub-category folders — randomized_mac is empty)

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

# WebGL scatter: renders the FULL series (no downsampling) far faster in the browser than SVG.
# Used for the forecast plots (few subplots each, well under the browser's WebGL-context limit).
GL = go.Scattergl

# st.fragment: a decorated block re-runs ONLY itself when its own widgets change, instead of re-running
# the whole page. Big win for sliders/toggles. Fall back to a no-op decorator on older Streamlit.
_FRAGMENT = getattr(st, "fragment", None) or getattr(st, "experimental_fragment", None)


def fragment(fn):
    return _FRAGMENT(fn) if _FRAGMENT else fn

st.set_page_config(layout="wide")

# ----------------------------------------------------------------------------
# Project paths (all overridable in the sidebar)
# ----------------------------------------------------------------------------
PROJECT_ROOT = os.environ.get("ANOMALY_ROOT", "/mnt/anomaly_detection")  # override on deploy, no code edit
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "parquet")   # holds the parquet files below
DEF_OUI_RUNS = os.path.join(PROJECT_ROOT, "results", "oui_router")
DEF_WINDOWS_PARQUET = os.path.join(DATA_DIR, "oui_windows.parquet")     # OUI window signal
DEF_SERIES_PARQUET = os.path.join(DATA_DIR, "oui_series.parquet")       # (pre-window OUI series)
DEF_KL_RUNS = os.path.join(PROJECT_ROOT, "results", "anomaly_kl")
DEF_SPLIT_RUNS = os.path.join(PROJECT_ROOT, "results", "anomaly_kl_oui")   # split KL-OUI runs (each run has split*/scores_all.csv)
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
    # Native Streamlit plotly chart (no iframe). plotly.js is loaded ONCE in the app bundle and the
    # chart is updated in place on rerun — far faster on interaction than rebuilding an HTML iframe
    # each time. Zoom/pan via the toolbar and mouse-wheel/pinch (scrollZoom); double-click resets.
    try:
        fig.update_layout(autosize=True, width=None)
    except Exception:
        pass
    cfg = {"scrollZoom": True, "displaylogo": False, "responsive": True}
    try:
        st.plotly_chart(fig, use_container_width=True, theme=None, config=cfg)
    except TypeError:
        st.plotly_chart(fig, use_container_width=True, config=cfg)


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
def flow_meta_map(path):
    """conn_id -> protocol/src_port/dst_port (one row per conn_id). Reads only those columns, so it is
    cheap even on the big flow parquet; used to annotate the forecast overview."""
    want = ["conn_id", "protocol", "src_port", "dst_port"]
    try:
        import pyarrow.parquet as pq
        cols = [c for c in want if c in set(pq.ParquetFile(path).schema.names)]
    except Exception:
        return pd.DataFrame()
    if "conn_id" not in cols:
        return pd.DataFrame()
    d = pd.read_parquet(path, columns=cols)
    return d.drop_duplicates("conn_id").astype({"conn_id": str}).set_index("conn_id")


@st.cache_data(show_spinner=False)
def load_flow_parquet(path):
    # read ONLY the columns the flow plots use (drops bytes/router/oui/is_iot/conn_direction/has_ech):
    # less I/O + RAM, no change to anything shown. Falls back to all columns if the schema can't be read.
    want = ["conn_id", "datetime", "packet_count", "fwd_packet_count", "bwd_packet_count",
            "fwd_mean_packet_size", "bwd_mean_packet_size", "fwd_mean_iat", "bwd_mean_iat",
            "protocol", "src_port", "dst_port"]
    try:
        import pyarrow.parquet as pq
        have = set(pq.ParquetFile(path).schema.names)
        cols = [c for c in want if c in have]
    except Exception:
        cols = None
    df = pd.read_parquet(path, columns=cols) if cols else pd.read_parquet(path)
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
    # downcast floats to shrink RAM + speed ops (displayed values unchanged). conn_id is left as-is
    # (object) because it is used in groupby/merge and category dtype changes that behavior.
    for c in df.select_dtypes("float64").columns:
        df[c] = df[c].astype("float32")
    return df


@st.cache_data(show_spinner=False)
def get_flow_frames(parquet_path, cids, channels, bin_freq="1min"):
    """Cached: build per-flow frames for the given conn_ids. Keyed on path + cids + channels + bin
    (all cheap to hash), so it is NOT recomputed on every rerun the way passing the 295 MB DataFrame
    would force. This was the main reason the flow pages reloaded slowly."""
    return _flow_frames(load_flow_parquet(parquet_path), list(cids), list(channels), bin_freq)


def _flow_frames(df, cids, channels, bin_freq):
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
    fig.update_layout(height=row_h * C + 110, width=col_w, margin=dict(l=55, r=15, t=96, b=30),
                      paper_bgcolor=TH["bg"], plot_bgcolor=TH["bg"], font=FONT, hovermode="x unified",
                      legend=dict(orientation="h", yanchor="bottom", y=1.015, x=0, xanchor="left",
                                  font=dict(size=10), bgcolor="rgba(0,0,0,0)"),
                      title=dict(text="Full-resolution series — coloured bands & bold segments mark anomalous "
                                      "windows (zoom with the controls above)",
                                 x=0, xanchor="left", xref="paper",
                                 y=1, yanchor="top", yref="container",
                                 font=dict(size=13, color=TH["fg"])),
                      title_pad=dict(t=6, l=2))
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


# ============================================================================
# Plain-language help shown at the top of each page (for first-time users)
# ============================================================================
HELP = {
    "oui_router": (
        "**What this page shows.** Traffic is grouped into *series* = one **router × device-maker (OUI) × "
        "direction**. Each series is cut into time **windows**, and two independent detectors score every window:\n\n"
        "- **KDE** — flags a window that looks **isolated / atypical** compared with the other windows of the same "
        "series (an unusual moment).\n"
        "- **KL** — flags a window where the behaviour **shifted and stayed different** from the rest of the series "
        "(a sustained change).\n"
        "- **KDE ∩ KL** — windows both detectors agree on (strongest signal).\n\n"
        "**How to read it.** Pick a series on the left; the plot shows its channels over time with the flagged "
        "windows highlighted. The embedding map places windows that behave alike near each other — outliers sit apart."
    ),
    "kl_flows": (
        "**What this page shows.** Whole **flows** whose overall behaviour differs most from the normal baseline, "
        "measured by **KL divergence** (a “how different is this distribution” score — higher = more different).\n\n"
        "**How to read it.** The table lists the flagged flows (most different first). Pick one and the app draws it "
        "(**red**) next to its **closest normal flows** (**blue**) — if the red curve looks clearly different from the "
        "blue ones, that's the anomaly."
    ),
    "kde_flows": (
        "**What this page shows.** Whole **flows** that look **unlike the bulk of normal traffic**, measured by a "
        "**density** score (KDE): normal traffic sits in crowded regions; unusual flows sit where few others do "
        "(higher “unusualness” = more alone).\n\n"
        "**How to read it.** Pick a flagged flow (**red**) and compare it with its nearest normal flows (**blue**)."
    ),
    "kl_split": (
        "**What this page shows.** The per-**OUI** KL view, computed on data **split** into groups (e.g. "
        "known-manufacturer vs randomized-MAC). For a chosen OUI it shows the **flagged flows** on top and the "
        "**most borderline normal flows** below (normals that came closest to being flagged); the dotted line marks "
        "where the held-out window begins.\n\n"
        "**How to read it.** Compare the red (flagged) rows against the blue (borderline-normal) rows — the flagged "
        "ones should deviate most near the held-out region."
    ),
    "sliding": (
        "**What this page shows.** A **forecasting** detector: a time-series model predicts what each bin *should* be "
        "from its recent history, and a bin is flagged when the **actual** value is far from the **forecast** / "
        "outside its predicted range.\n\n"
        "**Representations (windowing)** — how each series is cut for the model:\n"
        "- **causal (sliding)** — a context window slides along; the model forecasts a short horizon just after it, "
        "again and again. Best coverage of the series.\n"
        "- **chunks (no sliding)** — the series is cut into disjoint long windows; the tail of each is forecast from "
        "the rest.\n"
        "- **full (no sliding)** — one window per series: the whole series minus the last few bins is the context, and "
        "only those last bins are forecast.\n\n"
        "Pick the data too: **flow 1-min / 5-min** (per connection) or **oui×router** (aggregated).\n\n"
        "**Methods (m0–m5)** — six ways of deciding “too far” from the forecast:\n"
        "- **m0** outside the prediction band · **m2** normalized error too big · **m1** error unusually large vs its "
        "history (z-score) · **m3** error ≫ the typical error (MAE) · **m4** same on percentage error (MAPE) · "
        "**m5** = m0 **and** m2.\n"
        "- **consensus** = a bin flagged only where **all** chosen methods agree (intersection).\n\n"
        "**Views.** *Whole series* = the full timeline with anomalies marked. *Windows* = zoom into one window: "
        "**blue** = actual (context + horizon), **orange** = forecast over the horizon, **red ○** = anomalies.\n\n"
        "**Live thresholds.** Turn on *Live thresholds* to retune every method in the app and watch all counts update "
        "instantly — open *What do these thresholds mean?* for each knob."
    ),
}


def page_help(key):
    md = HELP.get(key)
    if md:
        with st.expander("ℹ️ How to read this page", expanded=False):
            st.markdown(md)


def page_oui_router():
    st.title("OUI × Router — Window Anomaly")
    st.caption("Per-series (router × OUI × direction) window anomalies from oui_router_anomaly.py. "
               "KDE = isolated atypical windows; KL = sustained change vs the rest of the series.")
    page_help("oui_router")
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
    sel_sid = sids[labels.index(st.sidebar.selectbox("Select series", labels))]  # selectbox: fast with many series

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
    page_help("kl_flows" if nearest_kind == "kl" else "kde_flows")
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

    # ---- suspicious flows table ----
    st.markdown("#### Suspicious flows")
    st.caption("Each row is a flow the method flagged as unusual. Pick one below to see it next to the "
               "most similar **normal** flows and judge what stands out.")
    if anomdf.empty:
        anomdf = sc[flag].copy()
    # show ONLY flow-related columns (drop method/score columns such as D_i, neg_log_density, silhouette, scores)
    id_cols = [c for c in ["conn_id", "cid"] if c in anomdf.columns]
    flow_cols = ["region_id", "oui", "router", "service", "protocol", "src_port", "dst_port",
                 "conn_direction", "is_iot"]
    cols_show = id_cols + [c for c in flow_cols if c in anomdf.columns and c not in id_cols]
    st.dataframe(anomdf[cols_show] if cols_show else anomdf, use_container_width=True, height=300)

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

    @fragment      # channel / router / flow pickers re-render only THIS block, not the whole page
    def _compare_flows():
        channels = st.multiselect("Channels to show", all_ch, default=all_ch)
        k = st.slider("How many normal flows to compare against", 1, 8, 5)
        if not channels:
            st.warning("Select at least one channel."); return
        sc_col = score_col if score_col in anomdf.columns else ("D_i" if "D_i" in anomdf.columns else None)
        adf = anomdf.copy()
        if sc_col:
            adf = adf.sort_values(sc_col, ascending=False)
        # filter flagged flows by router (router = prefix of conn_id/cid, or the 'router' column if present)
        adf["_router"] = (adf["router"].astype(str) if "router" in adf.columns
                          else adf[id_col].astype(str).str.split("_", n=1).str[0])
        routers = ["All"] + sorted(adf["_router"].dropna().unique())
        sel_router = st.selectbox("Router", routers, help="show only flows from one router")
        if sel_router != "All":
            adf = adf[adf["_router"] == sel_router]
        if adf.empty:
            st.info("No flagged flows for this router."); return
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
        frames = get_flow_frames(flow_parquet, [a_cid] + normals, channels, bin_freq=bin_freq)
        if a_cid not in frames:
            st.warning(f"The flagged flow `{a_cid}` is not in the flow data parquet — can't draw it. "
                       f"Check that the parquet is the one this run used."); return
        targets = [{"cid": a_cid, "color": TH["kde"],
                    "label": f"<b>⚠ FLAGGED</b><br>{frames[a_cid]['protocol']} "
                             f"{frames[a_cid]['src_port']}→{frames[a_cid]['dst_port']}<br>"
                             f"{score_label}={float(row.get(sc_col, np.nan)) if sc_col else float('nan'):.3g}"}]
        for i, nc in enumerate(normals):
            if nc in frames:
                fdn = frames[nc]
                targets.append({"cid": nc, "color": TH["normal"],
                                "label": f"normal #{i+1}<br>{fdn['protocol']} {fdn['src_port']}→{fdn['dst_port']}"})
        st.caption("**Top row (red) = the flagged flow.** The blue rows below are the most similar normal flows "
                   "found by the method — compare the shapes to see what makes the flagged one stand out.")
        fig = plot_flows_grid(targets, frames, channels)
        if fig is not None:
            show_scroll(fig)

        if not normals:
            st.info("No pre-computed nearest-normal list for this flow in this run.")
        else:
            missing = sum(1 for nc in normals if nc not in frames)
            if missing:
                st.caption(f"({missing} of the {len(normals)} nearest normal flows were not found in the parquet.)")

        with st.expander("See the flagged flow on its own (larger)"):
            t = {"cid": a_cid, "d": float(row.get(sc_col, np.nan)) if sc_col else 0.0, "tag": "⚠ FLAGGED flow"}
            show(plot_flow_detail(t, frames[a_cid], channels, TH["kde"], holdout=0))

    _compare_flows()


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
# PAGE — KL per-split flows   [results/anomaly_kl_oui/<run>/split*/scores_all.csv]
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
               "with the dotted line marking where the held-out window begins "
               "(reads results/anomaly_kl_oui/<run>/split*/scores_all.csv).")
    page_help("kl_split")
    st.sidebar.header("Data")
    runs_dir = st.sidebar.text_input("Runs dir (results/anomaly_kl_oui)", DEF_SPLIT_RUNS)
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
    sel = st.sidebar.selectbox("Select OUI", opts)
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

    flows = get_flow_frames(flow_parquet, [t["cid"] for t in t_anom + t_norm], channels)
    st.caption(f"Channels: {', '.join(channels)}  ·  dotted line = holdout start ({holdout} bins before end).  "
               f"Red = flagged, blue = the most borderline normal flows for comparison.")
    # stacked one below the other (not side by side)
    st.markdown(f"#### 🔴 Flagged flows ({len(t_anom)})")
    fig = plot_split_compact(t_anom, flows, channels, TH["kde"], holdout, row_h)
    show_scroll(fig) if fig else st.write("No flagged flows.")
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
    fig.add_trace(GL(x=df["datetime"], y=df["upper"], mode="lines", line=dict(width=0),
                     showlegend=False, hoverinfo="skip"))
    fig.add_trace(GL(x=df["datetime"], y=df["lower"], mode="lines", line=dict(width=0),
                     fill="tonexty", fillcolor=TH["shade_kl"], name="prediction interval",
                     hoverinfo="skip"))
    fig.add_trace(GL(x=df["datetime"], y=df["truth"], mode="lines",
                     line=dict(color=TH["normal"], width=1), name="actual",
                     hovertemplate="%{x|%m-%d %H:%M}<br>%{y:.4g}<extra></extra>"))
    fig.add_trace(GL(x=df["datetime"], y=df["pred"], mode="lines",
                     line=dict(color=TH["kl"], width=1), name="forecast (median)",
                     hovertemplate="%{x|%m-%d %H:%M}<br>%{y:.4g}<extra></extra>"))
    if flag_col in df.columns and df[flag_col].any():
        a = df[df[flag_col]]
        fig.add_trace(GL(x=a["datetime"], y=a["truth"], mode="markers",
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


def sliding_windows_plot(series_df, stream, meta, flag_col, sel_win, title, col_w=1400, row_h=470):
    """Zoom to ONE window only: context shown as the blue actual line; horizon shown with the blue
    actual line + the orange forecast (+ prediction band) + anomaly markers. Nothing outside this
    window is drawn, so the forecast no longer appears to 'leave' the horizon."""
    C = int(meta.get("context", 0))
    H = int(meta.get("horizon", 0))
    starts = meta.get("starts", [])
    t = series_df["datetime"].to_numpy()
    y = series_df["actual"].to_numpy()
    fig = go.Figure()
    if not starts or sel_win is None or not (0 <= sel_win < len(starts)) or len(t) == 0:
        fig.update_layout(height=row_h, width=col_w, paper_bgcolor=TH["bg"], plot_bgcolor=TH["bg"],
                          title=dict(text="No window to show", font=dict(size=12, color=TH["fg"])))
        return fig
    s = starts[sel_win]
    c0, c1 = max(0, s - C), s                       # context [c0, c1)
    h0, h1 = s, min(len(t), s + H)                   # horizon [h0, h1)

    # horizon rows of THIS window only (window column is reliable for non-overlapping windows)
    if "window" in stream.columns:
        hz = stream[stream["window"] == sel_win]
    else:
        hz = stream[(stream["datetime"] >= t[h0]) & (stream["datetime"] <= t[min(len(t) - 1, h1 - 1)])]

    # prediction interval band — horizon only
    if len(hz):
        fig.add_trace(GL(x=hz["datetime"], y=hz["upper"], mode="lines", line=dict(width=0),
                         showlegend=False, hoverinfo="skip"))
        fig.add_trace(GL(x=hz["datetime"], y=hz["lower"], mode="lines", line=dict(width=0),
                         fill="tonexty", fillcolor=TH["shade_kl"], name="prediction interval", hoverinfo="skip"))
    # actual (blue) over context + horizon
    fig.add_trace(GL(x=t[c0:h1], y=y[c0:h1], mode="lines+markers", line=dict(color=TH["normal"], width=1.5),
                     marker=dict(size=4), name="actual",
                     hovertemplate="%{x|%m-%d %H:%M}<br>actual %{y:.4g}<extra></extra>"))
    # forecast (orange) + anomalies — horizon only
    if len(hz):
        fig.add_trace(GL(x=hz["datetime"], y=hz["pred"], mode="lines+markers", line=dict(color=TH["kl"], width=1.5),
                         marker=dict(size=4), name="forecast (median)",
                         hovertemplate="%{x|%m-%d %H:%M}<br>forecast %{y:.4g}<extra></extra>"))
        if flag_col and flag_col in hz.columns and hz[flag_col].any():
            a = hz[hz[flag_col]]
            fig.add_trace(GL(x=a["datetime"], y=a["truth"], mode="markers",
                             marker=dict(color=TH["kde"], size=11, symbol="circle-open", line=dict(width=2)),
                             name=f"anomaly ({int(hz[flag_col].sum())})",
                             hovertemplate="anomaly<br>%{x|%m-%d %H:%M}<br>%{y:.4g}<extra></extra>"))
    # light shading to mark the two regions
    fig.add_vrect(x0=t[c0], x1=t[max(c0, c1 - 1)], fillcolor=TH["shade_both"], line_width=0, layer="below",
                  annotation_text="context", annotation_position="top left", annotation_font=dict(size=10, color=TH["fg"]))
    fig.add_vrect(x0=t[h0], x1=t[min(len(t) - 1, h1 - 1)], fillcolor=TH["shade_kde"], line_width=0, layer="below",
                  annotation_text="horizon", annotation_position="top right", annotation_font=dict(size=10, color=TH["fg"]))
    fig.update_xaxes(range=[t[c0], t[min(len(t) - 1, h1 - 1)]], showgrid=True, gridcolor=TH["grid"],
                     linecolor=TH["grid"], tickfont=dict(size=9, color=TH["fg"]))
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
            fig.add_trace(GL(x=stream["datetime"], y=stream["upper"], mode="lines",
                             line=dict(width=0), showlegend=False, hoverinfo="skip"), row=i, col=1)
            fig.add_trace(GL(x=stream["datetime"], y=stream["lower"], mode="lines", line=dict(width=0),
                             fill="tonexty", fillcolor=TH["shade_kl"], name="prediction interval",
                             hoverinfo="skip"), row=i, col=1)
            yact = series_df[ch] if ch in series_df.columns else series_df.get("actual")
            fig.add_trace(GL(x=series_df["datetime"], y=yact, mode="lines",
                             line=dict(color=TH["normal"], width=1), name="actual",
                             hovertemplate="%{x|%m-%d %H:%M}<br>%{y:.4g}<extra></extra>"), row=i, col=1)
            fig.add_trace(GL(x=stream["datetime"], y=stream["pred"], mode="lines",
                             line=dict(color=TH["kl"], width=1), name="forecast (median)",
                             hovertemplate="%{x|%m-%d %H:%M}<br>%{y:.4g}<extra></extra>"), row=i, col=1)
            if flag_col and flag_col in stream.columns and stream[flag_col].any():
                a = stream[stream[flag_col]]
                fig.add_trace(GL(x=a["datetime"], y=a["truth"], mode="markers",
                                 marker=dict(color=TH["kde"], size=8, symbol="circle-open", line=dict(width=2)),
                                 name=f"anomaly ({int(stream[flag_col].sum())})",
                                 hovertemplate="anomaly<br>%{x|%m-%d %H:%M}<br>%{y:.4g}<extra></extra>"),
                              row=i, col=1)
        else:
            fig.add_trace(GL(x=series_df["datetime"], y=series_df[ch], mode="lines",
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


def sliding_consensus_flags(df, methods, min_votes):
    """Recompute consensus live: flag where at least `min_votes` of the SELECTED methods
    (restricted to those present in df) agree. Returns (bool array, methods_used, k) or None."""
    present = [m for m in methods if f"is_anom_{m}" in df.columns]
    if not present:
        return None
    votes = sum(df[f"is_anom_{m}"].astype(int) for m in present)
    k = min(min_votes, len(present))
    return (votes >= k).to_numpy(), present, k


def sliding_cap(is_anom, score, frac):
    """Keep at most frac*N flags, the highest-scoring ones (port of cap_by_fraction)."""
    if not frac or frac <= 0:
        return is_anom
    cap = max(1, int(np.ceil(frac * len(is_anom))))
    idx = np.flatnonzero(is_anom)
    if len(idx) <= cap:
        return is_anom
    keep = idx[np.argsort(score[idx])[::-1][:cap]]
    out = np.zeros_like(is_anom)
    out[keep] = True
    return out


def sliding_detect_all(truth, pred, lower, upper, y_range, th):
    """Vectorized, faithful port of forecast_sliding_anomaly.detect() for ALL methods m0..m5,
    recomputed live from the stored truth/pred/lower/upper. O(n) (the script's loop is O(n^2)),
    so it is fast enough to update on every slider move. th keys: interval, normdiff, z, mae_k,
    mape_k, mape_min."""
    truth = np.asarray(truth, float); pred = np.asarray(pred, float)
    lower = np.asarray(lower, float); upper = np.asarray(upper, float)
    n = len(truth)
    if n == 0:
        return {m: np.zeros(0, bool) for m in ["m0", "m1", "m2", "m3", "m4", "m5"]}
    diff = np.abs(truth - pred)
    eps = 1e-2
    yr = y_range + 1e-5
    f = float(th.get("interval", 1.0))                       # widen/narrow the interval for m0/m5
    up = pred + f * (upper - pred)
    lo = pred - f * (pred - lower)
    m0 = (truth > up + eps) | (truth < lo - eps)
    m2 = (diff >= eps) & (diff / yr > th["normdiff"])
    m5 = m0 & m2
    cnt = np.arange(n, dtype=float)                          # #previous points seen at index i

    def prev_sum(v):
        c = np.cumsum(v); ps = np.empty(n); ps[0] = 0.0; ps[1:] = c[:-1]
        return ps

    # m1: expanding z-score of squared error (baseline over all previous points, ddof=1)
    se = diff * diff
    ps = prev_sum(se); ps2 = prev_sum(se * se)
    mean_se = np.divide(ps, cnt, out=np.zeros(n), where=cnt > 0)
    var = np.divide(ps2 - np.divide(ps ** 2, np.where(cnt > 0, cnt, 1)),
                    np.where(cnt > 1, cnt - 1, 1), out=np.zeros(n), where=cnt > 1)
    std_se = np.sqrt(np.clip(var, 0, None))
    with np.errstate(divide="ignore", invalid="ignore"):
        zsc = np.where(std_se > 0, (se - mean_se) / np.where(std_se > 0, std_se, 1), 0.0)
    base = (diff >= eps) & (cnt >= 2)
    m1 = base & (((std_se > 0) & (zsc > th["z"])) | ((std_se <= 0) & (se > mean_se)))
    # m3: expanding MAE ratio
    mae_prev = np.divide(prev_sum(diff), cnt, out=np.zeros(n), where=cnt > 0)
    m3 = np.zeros(n, bool)
    e1 = (diff >= eps) & (cnt == 1); m3[e1] = diff[e1] > th["mae_k"] * mae_prev[e1]
    g2 = (diff >= eps) & (cnt >= 2); m3[g2] = (mae_prev[g2] > 0) & (diff[g2] > th["mae_k"] * mae_prev[g2])
    # m4: expanding MAPE
    ape = diff / (np.abs(truth) + 1e-5)
    mape_prev = np.divide(prev_sum(ape), cnt, out=np.zeros(n), where=cnt > 0)
    m4 = np.zeros(n, bool)
    e1 = (diff >= eps) & (cnt == 1); m4[e1] = (ape[e1] > th["mape_k"] * mape_prev[e1]) & (ape[e1] > th["mape_min"])
    g2 = (diff >= eps) & (cnt >= 2); m4[g2] = (mape_prev[g2] > 0) & (ape[g2] > th["mape_k"] * mape_prev[g2]) & (ape[g2] > th["mape_min"])
    return {"m0": m0, "m1": m1, "m2": m2, "m3": m3, "m4": m4, "m5": m5}


def sliding_apply_live(df, sdf_full, live_th):
    """Rewrite df's is_anom_m0..m5 + is_anom_consensus in place from a live threshold set."""
    if not {"truth", "pred", "lower", "upper"}.issubset(df.columns):
        return df
    if "actual" in sdf_full.columns and not sdf_full.empty:
        y_range = float(sdf_full["actual"].max() - sdf_full["actual"].min())
    else:
        y_range = float(df["truth"].max() - df["truth"].min())
    flags = sliding_detect_all(df["truth"], df["pred"], df["lower"], df["upper"], y_range, live_th)
    diff = (df["truth"] - df["pred"]).abs().to_numpy()
    for mname, arr in flags.items():
        df[f"is_anom_{mname}"] = sliding_cap(arr, diff, live_th.get("cap_frac", 0.0))
    cm = [m for m in live_th.get("cons_methods", ["m0", "m1", "m2", "m3", "m4", "m5"]) if f"is_anom_{m}" in df.columns]
    if cm:
        votes = sum(df[f"is_anom_{m}"].astype(int) for m in cm)
        k = min(live_th.get("cons_min", 2), len(cm))
        df["is_anom_consensus"] = (votes >= k)
    return df


@fragment
def sliding_render_series(scores_path, sel_flag, label, key_prefix="", allow_windows=True,
                          live=None, cons=None, live_th=None):
    """Metrics row + whole-series/windows plot for ONE series. Reused by both the
    single-representation view and the per-flow cross-representation comparison.
    If `live_th` is given, EVERY method (m0..m5) and the consensus are recomputed in-app at those
    thresholds — all metrics/plots update live. `live`/`cons` are the older single-flag knobs."""
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

    df = df.copy()
    live_on = live_th is not None
    if live_on:
        df = sliding_apply_live(df, sdf_full, live_th)       # recompute m0..m5 live (consensus set below)

    method_cols = [c for c in ["is_anom_m0", "is_anom_m1", "is_anom_m2", "is_anom_m3", "is_anom_m4", "is_anom_m5"]
                   if c in df.columns]
    # ONE consensus = INTERSECTION of the methods chosen in "Consensus methods" (all present if none chosen)
    if live_on and live_th.get("cons_methods"):
        sel_methods = live_th["cons_methods"]
    elif cons is not None:
        sel_methods = cons["methods"]
    else:
        sel_methods = [c.replace("is_anom_", "") for c in method_cols]
    present = [m for m in sel_methods if f"is_anom_{m}" in df.columns]
    if present:
        votes = sum(df[f"is_anom_{m}"].astype(int) for m in present)
        df["is_anom_consensus"] = (votes >= len(present))     # len(present) = all must agree = intersection
    elif "is_anom_consensus" not in df.columns:
        df["is_anom_consensus"] = False

    flag_col = "is_anom_consensus" if sel_flag == "consensus" else f"is_anom_{sel_flag}"
    if flag_col not in df.columns:
        flag_col = method_cols[0] if method_cols else None

    cols = st.columns(1 + len(method_cols) + 1)
    cols[0].metric("Evaluated points", f"{len(df):,}")
    for i, c in enumerate(method_cols):
        cols[i + 1].metric(c.replace("is_anom_", "") + (" ✎" if live_on else ""), f"{int(df[c].sum()):,}")
    cols[-1].metric("consensus ∩ " + ",".join(present) if present else "consensus",
                    f"{int(df['is_anom_consensus'].sum()):,}",
                    help="intersection: bins flagged by ALL the selected consensus methods")

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
            sel_win = st.slider("Window (context + horizon)", 0, max(0, len(starts) - 1), 0,
                                key=key_prefix + "win")
            st.caption(f"Window {sel_win} of {len(starts) - 1} — only this window is shown.  "
                       f"Blue = actual (context {meta.get('context', '?')} bins + horizon {meta.get('horizon', '?')} bins), "
                       f"orange = forecast over the horizon, shaded = prediction interval, red ○ = anomaly.")
            show_scroll(sliding_windows_plot(sdf_full, df, meta, flag_col, sel_win,
                                             f"{label} · window {sel_win}/{len(starts)-1}"))

    # per-method verdicts for this series: for every point flagged by >=1 method (or the active
    # flag), show each method's decision (m0..m5), the consensus, votes and score.
    method_cols = [c for c in df.columns if c.startswith("is_anom_m")]
    if method_cols or (flag_col in df.columns):
        import numpy as _np
        any_anom = df[method_cols].any(axis=1) if method_cols else _np.zeros(len(df), dtype=bool)
        mask = any_anom | (df[flag_col] if flag_col in df.columns else False)
        n_hit = int(_np.asarray(mask).sum())

        # per-method summary line (count + % of evaluated points)
        if method_cols:
            summ = " · ".join(f"**{c.replace('is_anom_', '')}**: {int(df[c].sum())}" for c in method_cols)
            st.caption(f"What each method declared for this series (of {len(df)} evaluated points) — {summ}")

        with st.expander(f"Per-method verdicts — {n_hit} point(s) flagged by ≥1 method", expanded=False):
            if n_hit == 0:
                st.write("No method flagged any point in this series.")
            else:
                sc_col = next((c for c in df.columns if c.startswith("score_m")), None)
                base = [c for c in ["datetime", "truth", "pred", "lower", "upper"] if c in df.columns]
                extra = (["is_anom_consensus"] if "is_anom_consensus" in df.columns else []) + \
                        (["votes"] if "votes" in df.columns else []) + ([sc_col] if sc_col else [])
                tbl = df[mask][base + method_cols + extra].copy()
                tbl = tbl.rename(columns={c: c.replace("is_anom_", "") for c in method_cols})
                if "is_anom_consensus" in tbl.columns:
                    tbl = tbl.rename(columns={"is_anom_consensus": "consensus"})
                if sc_col:
                    tbl = tbl.sort_values(sc_col, ascending=False)
                st.dataframe(tbl, use_container_width=True, height=320, hide_index=True)
                st.caption("Each m0–m5 column = that method's verdict for the point (checked = anomaly). "
                           "`consensus` = current consensus rule, `votes` = how many methods agreed, "
                           "`score_m*` = normalized forecast error.")


@st.cache_data(show_spinner=False)
def sliding_rep_info(tag_dir):
    """Peek one series' meta in a representation folder to get its windowing + C/H + mode."""
    for f in sorted(glob.glob(os.path.join(tag_dir, "*", "*", "sliding_meta.json")))[:1]:
        try:
            m = json.load(open(f))
            return {"windowing": m.get("windowing", "causal"), "context": m.get("context"),
                    "horizon": m.get("horizon"), "mode": m.get("mode", ""), "data": m.get("data")}
        except Exception:
            pass
    return {"windowing": "causal", "context": None, "horizon": None, "mode": "", "data": None}


# windowing mode -> short human label for the UI
WIN_LABEL = {"causal": "causal (sliding)", "chunks": "chunks (no sliding)", "full": "full series (no sliding)"}


def page_sliding():
    st.title("Forecast anomalies")
    st.caption("Forecast-based anomalies (forecast_sliding_anomaly.py). Windowing per run: "
               "**causal** = sliding overlapping context; **chunks** = disjoint long windows (no sliding); "
               "**full** = one window per series, context = whole series − horizon (no sliding). "
               "A point far from the forecast / outside its interval is flagged.")
    page_help("sliding")
    st.sidebar.header("Data")
    runs_dir = st.sidebar.text_input("Runs dir (results/sliding)", DEF_SLIDING_RUNS)
    reps = sliding_reps(runs_dir)
    if not reps:
        st.warning(f"No results under `{runs_dir}`."); return

    # filter representations by windowing type (causal = sliding; chunks/full = the no-sliding experiments)
    rep_info = {tag: sliding_rep_info(d) for tag, d in reps}
    wins_present = sorted({rep_info[t]["windowing"] for t, _ in reps})
    if len(wins_present) > 1:
        win_filter = st.sidebar.radio("Windowing", ["all"] + wins_present, horizontal=True,
                                      format_func=lambda w: "all" if w == "all" else WIN_LABEL.get(w, w),
                                      help="causal = sliding; chunks / full = the non-sliding experiments")
        reps = [(t, d) for (t, d) in reps if win_filter == "all" or rep_info[t]["windowing"] == win_filter]
        if not reps:
            st.warning("No representations match this windowing filter."); return

    # which flag to highlight (shared by both modes)
    flag_opts = ["consensus", "m0", "m1", "m2", "m3", "m4", "m5"]
    sel_flag = st.sidebar.selectbox("Highlight", flag_opts, index=0)
    anom_col = "anom_consensus" if sel_flag == "consensus" else f"anom_{sel_flag}"

    # live thresholds: recompute EVERY method (m0..m5) + consensus in-app at the thresholds below.
    # Everything (all metrics, plots, verdicts) updates live when you move a slider — no re-run.
    live = None
    live_th = None
    cons = None
    method_opts = ["m0", "m1", "m2", "m3", "m4", "m5"]
    full_live = st.sidebar.checkbox("Live thresholds (recompute all methods)", value=False,
                                    help="Recompute m0–m5 and the consensus in the app at the thresholds below. "
                                         "Every metric and plot updates live; nothing is re-run offline.")
    if full_live:
        with st.sidebar.expander("Method thresholds", expanded=True):
            interval = st.slider("m0 / m5 · interval × k  (lower = more)", 0.2, 3.0, 1.0, 0.05)
            normdiff = st.slider("m2 / m5 · normalized-error τ  (lower = more)", 0.0, 1.0, 0.30, 0.01)
            z = st.slider("m1 · z-score  (lower = more)", 1.0, 10.0, 4.0, 0.1)
            mae_k = st.slider("m3 · MAE × k  (lower = more)", 1.0, 20.0, 5.0, 0.5)
            mape_k = st.slider("m4 · MAPE × k  (lower = more)", 1.0, 20.0, 5.0, 0.5)
            mape_min = st.slider("m4 · MAPE floor", 0.0, 2.0, 0.5, 0.05)
            cap_frac = st.slider("Max anomalies per method (fraction, 0 = off)", 0.0, 0.5, 0.0, 0.01)
        with st.sidebar.expander("What do these thresholds mean?", expanded=False):
            st.markdown(
                "Each method flags a bin where the **actual** value is too far from the model's **forecast**.\n\n"
                "- **m0 · interval × k** — the model predicts a band it expects the value to stay in; m0 flags "
                "values **outside** the band. `k` widens (k>1) or narrows (k<1) the band.\n"
                "- **m2 · normalized-error τ** — flags when |actual − forecast| exceeds **τ × the series' value range** "
                "(τ=0.3 ⇒ error bigger than 30% of the range).\n"
                "- **m1 · z-score** — flags when the squared error is more than **z standard deviations** above its "
                "running average (z=4 ⇒ very unusual).\n"
                "- **m3 · MAE × k** — flags when the error is more than **k× the running average error**.\n"
                "- **m4 · MAPE × k (+ floor)** — same as m3 but on **percentage** error (|error|/|actual|); the floor "
                "ignores tiny values so near-zero bins don't blow up.\n"
                "- **Max anomalies per method** — a **ceiling**: keep at most this fraction of bins (the highest-scoring) "
                "per method; 0 = no cap.\n"
                "- **Consensus** — a bin is flagged only where **all** selected methods agree (intersection).\n\n"
                "**Rule of thumb: lower threshold = more sensitive = more anomalies.**")
        cons_methods = st.sidebar.multiselect("Consensus methods (intersection)", method_opts, default=method_opts,
                                              help="Consensus = a point flagged only where ALL selected methods agree.")
        live_th = {"interval": interval, "normdiff": normdiff, "z": z, "mae_k": mae_k,
                   "mape_k": mape_k, "mape_min": mape_min, "cap_frac": cap_frac,
                   "cons_methods": cons_methods, "cons_min": len(cons_methods)}   # len = intersection
    elif sel_flag == "consensus":
        # retune the consensus from the SAVED per-method flags = intersection of the selected methods
        cons_methods = st.sidebar.multiselect("Consensus methods (intersection)", method_opts, default=method_opts,
                                              help="Consensus = flagged only where ALL selected methods agree.")
        cons = {"methods": cons_methods, "min": len(cons_methods)}   # len = intersection

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

        # compact side-by-side counts for the picked flow: m0..m5 + ONE consensus = intersection
        sel_methods = (live_th.get("cons_methods") if live_th else (cons["methods"] if cons else None)) \
            or ["m0", "m1", "m2", "m3", "m4", "m5"]
        crows = []
        for tag in sorted(reps_here):
            d = sliding_load_scores(reps_here[tag]).copy()
            m = sliding_load_meta(os.path.dirname(reps_here[tag]))
            if live_th is not None:
                d = sliding_apply_live(d, sliding_load_series(os.path.dirname(reps_here[tag])), live_th)
            row = {"representation": tag, "C": m.get("context", ""), "H": m.get("horizon", ""),
                   "win": m.get("windowing", "causal"), "eval_points": len(d)}
            for mm in ["m0", "m1", "m2", "m3", "m4", "m5"]:
                if f"is_anom_{mm}" in d.columns:
                    row[mm] = int(d[f"is_anom_{mm}"].sum())
            present = [mm for mm in sel_methods if f"is_anom_{mm}" in d.columns]
            if present:
                votes = sum(d[f"is_anom_{mm}"].astype(int) for mm in present)
                row["consensus"] = int((votes >= len(present)).sum())     # intersection
            crows.append(row)
        st.dataframe(pd.DataFrame(crows), use_container_width=True, hide_index=True)

        for tag in chosen:
            meta = sliding_load_meta(os.path.dirname(reps_here[tag]))
            ch = (f" · C{meta.get('context', '?')} H{meta.get('horizon', '?')} · {meta.get('windowing', 'causal')}"
                  if meta else "")
            st.markdown(f"### `{tag}`{ch}")
            sliding_render_series(reps_here[tag], sel_flag, f"{tag}  ·  {conn} · {tgt} · highlight: {sel_flag}",
                                  key_prefix=f"cmp_{tag}_", allow_windows=True, live=live, cons=cons, live_th=live_th)
            st.divider()
        return

    # ---- By representation (default) ----
    def rep_fmt(tag):
        i = rep_info.get(tag, {})
        ch = f" C{i.get('context')} H{i.get('horizon')}" if i.get("context") else ""
        return f"{tag}  ·  {WIN_LABEL.get(i.get('windowing', 'causal'), i.get('windowing'))}{ch}"
    rep_tags = [r[0] for r in reps]
    rep_label = st.sidebar.selectbox("Representation", rep_tags, format_func=rep_fmt,
                                     help="each run: windowing (causal=sliding, chunks/full=non-sliding) + C/H")
    base = dict(reps)[rep_label]
    _i = rep_info.get(rep_label, {})
    st.markdown(f"**Representation:** `{rep_label}`  ·  windowing: **{_i.get('windowing', 'causal')}**"
                + (f"  ·  C{_i.get('context')} H{_i.get('horizon')}" if _i.get("context") else ""))
    lst = sliding_list(base)
    if lst.empty:
        st.warning("No series in this representation."); return
    summ = sliding_summary(base)

    # run-level overview (from summary files if present): per-method + consensus counts, plus protocol/ports
    if not summ.empty:
        st.markdown("#### Overview")
        over = summ.copy()
        # attach protocol + ports (per conn_id) from this representation's flow parquet, when it has them
        datafile = rep_info.get(rep_label, {}).get("data")
        if datafile:
            ppath = os.path.join(DATA_DIR, datafile)
            mmap = flow_meta_map(ppath) if os.path.exists(ppath) else pd.DataFrame()
            if not mmap.empty:
                over["connection"] = over["connection"].astype(str)
                over = over.merge(mmap, left_on="connection", right_index=True, how="left")
        count_cols = [c for c in over.columns if c.startswith("anom_")]
        meta_cols = [c for c in ["protocol", "src_port", "dst_port"] if c in over.columns]
        show_cols = [c for c in ["connection", "target", "n_bins", "n_windows", "n_eval"] if c in over.columns] \
            + meta_cols + count_cols
        disp = over[show_cols].rename(columns={c: ("consensus" if "consensus" in c else c.replace("anom_", ""))
                                               for c in count_cols})
        sort_c = "consensus" if "consensus" in disp.columns else (
            disp.columns[len(disp.columns) - 1] if count_cols else "connection")
        st.dataframe(disp.sort_values(sort_c, ascending=False) if count_cols else disp,
                     use_container_width=True, height=280)
        st.caption("Anomaly counts are the saved offline values (per method m0–m5 and the consensus). "
                   "Protocol/ports are looked up per conn_id from the flow parquet.")

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
    # filter flows by router (router = conn_id prefix, e.g. "22101113e3ce_...")
    lst["router"] = lst["connection"].astype(str).str.split("_", n=1).str[0]
    routers = ["All"] + sorted(lst["router"].unique())
    sel_router = st.sidebar.selectbox("Router", routers,
                                      help="show only flows from one router (parsed from the conn_id prefix)")
    if sel_router != "All":
        lst = lst[lst["router"] == sel_router]
    if lst.empty:
        st.warning("No series for this router."); return
    labels = [f"{r.connection}  /  {r.target}" + (f"   [{int(r.rank)}]" if "rank" in lst.columns and r.rank >= 0 else "")
              for r in lst.itertuples(index=False)]
    sel = st.sidebar.selectbox("Select series", labels)   # selectbox: fast + searchable with thousands of series
    row = lst.iloc[labels.index(sel)]
    st.subheader(f"{row['connection']}  ·  {row['target']}")
    sliding_render_series(row["path"], sel_flag,
                          f"{row['connection']} · {row['target']} · highlight: {sel_flag}",
                          key_prefix="single_", allow_windows=True, live=live, cons=cons, live_th=live_th)


# ============================================================================
# Dispatch
# ============================================================================
st.sidebar.title("Anomaly dashboards")
with st.sidebar.expander("ℹ️ About this dashboard", expanded=False):
    st.markdown(
        "This app looks for **unusual network traffic** on home routers. Each page is one detection "
        "**experiment** — a different way of deciding what counts as unusual. Pick one below; each page has its "
        "own *How to read this page* panel at the top.\n\n"
        "- **OUI × Router — windows** — scans each router × device-maker series window by window (KDE = odd moments, "
        "KL = lasting changes).\n"
        "- **KL anomaly — flows** — whole connections most different from normal (distribution distance).\n"
        "- **KL OUI anomaly — flows** — the per-device-maker KL view, split by group.\n"
        "- **KDE anomaly — flows** — whole connections that sit far from the crowd (density).\n"
        "- **Forecast — sliding window** — predicts each bin from its history and flags what the model did not expect.\n\n"
        "Everywhere: **red = flagged / anomalous**, **blue = normal / actual**, **orange = the model's forecast**.")

# (3) preload the flow parquets once (cached), so the first flow page doesn't wait for the big read.
#     After the cache is warm this is instant; the spinner shows only on the first cold start.
_preload = [p for p in dict.fromkeys([DEF_FLOW_PARQUET, FLOW_PARQUET_BY_BIN.get("1min"),
                                      FLOW_PARQUET_BY_BIN.get("5min")]) if p and os.path.exists(p)]
if _preload:
    with st.spinner("Preloading flow data… (first start only)"):
        for _pq in _preload:
            try:
                load_flow_parquet(_pq)
            except Exception:
                pass

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