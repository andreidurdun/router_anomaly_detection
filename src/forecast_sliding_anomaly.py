#!/usr/bin/env python3
"""
forecast_sliding_anomaly.py  —  variant 1: causal SLIDING-WINDOW forecast anomaly detection

Fixed context C (a multiple of 16, the Chronos-2 patch size) and fixed horizon H slide
across each series with step = H (non-overlapping horizons), so the whole timeline is
covered once and the anomaly scores are comparable across series and positions.

Design choices that matter for cost/correctness:
  * the Chronos-2 model is loaded ONCE for the whole run;
  * all windows of a series are forecast in ONE batched predict_df call (one `id` per
    window) instead of one call per window;
  * the model is called MULTIVARIATE (all numeric channels as a joint target) and we keep
    ONLY the `packet_count` forecast — the other channels' outputs are dropped;
  * anomaly methods 1..5 are all applied to the SAME forecast (no re-forecasting), and the
    expanding-baseline methods (1=Z-score, 3=MAE, 4=MAPE) accumulate over the full per-series
    stream (not reset per window), so their baseline is stable.

Representations (same data files as the rest of the pipeline):
  1. per-flow, 1-min   --data new_data_aggregated_60s.parquet --mode flow        --context 128 --horizon 12
  2. per-flow, 5-min   --data new_data_aggregated_5min.parquet --mode flow        --context 48  --horizon 6
  3. router x oui 5-min --data oui_series.parquet              --mode oui-router   --context 48  --horizon 6

Outputs (per series):   <out-dir>/<connection>/<target>/
    sliding_scores.csv                 datetime, truth, pred, lower, upper, is_anom_m{1..5}, score_m{1..5}
    sliding_C{C}_H{H}_method{m}.png    series + forecast stream + anomalies for method m

NOTE (version sensitivity): _select_target() handles the common shapes returned by a
multivariate predict_df; adjust it there if your chronos version differs.

Run with --dry-run first to see which series qualify and the window counts.
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_OUT = os.path.join(ROOT, "results", "sliding")
EXCLUDED = ["datetime", "id", "src_ip", "dst_ip", "src_port", "dst_port", "router", "oui", "direction"]
FOURTUPLE = ["src_ip", "dst_ip", "src_port", "dst_port"]
# default channels fed to the multivariate model for per-flow runs (override with --channels)
DEFAULT_FLOW_CHANNELS = ["packet_count", "fwd_mean_packet_size", "bwd_mean_packet_size",
                         "fwd_mean_iat", "bwd_mean_iat"]


# ----------------------------------------------------------------------------
# IO / helpers
# ----------------------------------------------------------------------------
def read_table(path, columns=None):
    if str(path).endswith(".parquet"):
        return pd.read_parquet(path, columns=columns)
    return pd.read_csv(path, usecols=columns)


def read_head(path, n=5):
    """Cheap column/dtype sniff without reading the whole file."""
    if str(path).endswith(".parquet"):
        import pyarrow.parquet as pq
        try:
            return next(pq.ParquetFile(path).iter_batches(batch_size=n)).to_pandas()
        except StopIteration:
            return pd.read_parquet(path)
    return pd.read_csv(path, nrows=n)


def _is_listlike(v):
    return isinstance(v, (list, tuple, np.ndarray))


def detect_list_channels(sample):
    """Columns that hold per-series arrays (oui_series.parquet stores channels as lists)."""
    return [c for c in sample.columns if sample[c].map(_is_listlike).any()]


def expand_list_series(path, bin_seconds, id_cols, channels, min_len):
    """Turn a LIST-format parquet (one row per series, channels = arrays) into a long frame
    (one row per time step) with datetime + scalar channels + id columns + series_id.
    Series shorter than min_len are skipped. Bin step: from 'end'/'start' if present, else bin_seconds."""
    df = pd.read_parquet(path)
    start_col = "start" if "start" in df.columns else next(
        (c for c in df.columns if np.issubdtype(df[c].dtype, np.datetime64)), None)
    if start_col is None:
        raise SystemExit("list-format parquet has no 'start'/datetime column to anchor the series.")
    end_col = "end" if "end" in df.columns else None
    df[start_col] = pd.to_datetime(df[start_col], errors="coerce")
    if end_col:
        df[end_col] = pd.to_datetime(df[end_col], errors="coerce")
    parts = []
    for idx, row in df.iterrows():
        arrs = {c: np.asarray(row[c], dtype="float64") for c in channels if _is_listlike(row[c])}
        if not arrs:
            continue
        n = min(len(a) for a in arrs.values())
        if n < min_len:
            continue
        start = row[start_col]
        if pd.isna(start):
            continue
        if end_col and pd.notna(row[end_col]) and n > 1 and row[end_col] > start:
            step_s = (row[end_col] - start).total_seconds() / (n - 1)
        else:
            step_s = float(bin_seconds)
        dt = start + pd.to_timedelta(np.arange(n) * step_s, unit="s")
        sub = pd.DataFrame({"datetime": dt})
        for c in channels:
            sub[c] = arrs[c][:n]
        for ic in id_cols:
            sub[ic] = row[ic]
        sub["series_id"] = sanitize("_".join(str(row[ic]) for ic in id_cols))
        parts.append(sub)
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


def sanitize(s):
    return "".join(ch if (ch.isalnum() or ch in "._-") else "_" for ch in str(s))


def numeric_channels(df):
    return [c for c in df.columns if c not in EXCLUDED and pd.api.types.is_numeric_dtype(df[c])]


def round_mult16(c):
    """Nearest positive multiple of 16 (the Chronos-2 patch size)."""
    return max(16, int(round(c / 16.0)) * 16)


def group_keys(df, mode):
    if mode == "flow":
        if all(c in df.columns for c in FOURTUPLE):
            return FOURTUPLE, "fourtuple"
        if "conn_id" in df.columns:
            return ["conn_id"], "conn_id"
        raise SystemExit("flow mode needs src_ip/dst_ip/src_port/dst_port or conn_id.")
    keys = [c for c in ["router", "oui", "direction"] if c in df.columns]
    if "router" not in keys or "oui" not in keys:
        raise SystemExit("oui-router mode needs 'router' and 'oui'.")
    return keys, "oui-router"


def conn_folder(kind, key_vals, keys):
    if kind == "oui-router":
        d = {k: v for k, v in zip(keys, key_vals)}
        name = f"{d['router']}_{d['oui']}" + (f"_{d.get('direction')}" if "direction" in keys else "")
    elif kind == "conn_id":
        name = str(key_vals[0])
    else:  # fourtuple
        d = {k: v for k, v in zip(keys, key_vals)}
        name = f"{d['src_ip']}_to_{d['dst_ip']}_{d['src_port']}_{d['dst_port']}"
    return sanitize(name)


def clean_series(g, channels):
    """Sort, aggregate duplicate timestamps, resample to the inferred bin filling 0, force a
    strictly uniform frequency. Returns a df with 'datetime','id' + channels (uniform grid)."""
    g = g.sort_values("datetime").copy()
    agg = {c: "sum" for c in channels}
    for c in ("mean_packet_size", "std_packet_size", "mean_iat", "mean_tcp_window",
              "inbound_outbound_byte_ratio", "inbound_outbound_packet_ratio"):
        if c in agg:
            agg[c] = "mean"
    if "min_tcp_window" in agg:
        agg["min_tcp_window"] = "min"
    if "max_tcp_window" in agg:
        agg["max_tcp_window"] = "max"
    g = g.groupby("datetime").agg(agg).reset_index().sort_values("datetime")
    if len(g) < 2:
        return None
    freq = g["datetime"].diff().mode()[0]
    if freq < pd.Timedelta(seconds=1):
        freq = pd.Timedelta(minutes=5)
    g = g.set_index("datetime").resample(freq).asfreq()
    g[channels] = g[channels].fillna(0.0)
    g = g.reset_index()
    # force strictly uniform timeline
    g["datetime"] = pd.date_range(start=g["datetime"].iloc[0], periods=len(g), freq=freq)
    g["id"] = "s"
    for c in channels:
        g[c] = g[c].astype("float64")
    return g


def build_windows(n, c, h, step):
    """Start indices so that each window uses rows [s-c:s] as context and [s:s+h] as horizon."""
    return list(range(c, n - h + 1, step))


# ----------------------------------------------------------------------------
# Multivariate forecast, keep only the target channel
# ----------------------------------------------------------------------------
def _qcol(cols, q):
    """Find the column whose name parses to the quantile value closest to q."""
    best, bd = None, 1e18
    for c in cols:
        if c in ("id", "datetime", "target", "variable", "item_id", "column", "series", "channel"):
            continue
        try:
            v = float(c)
        except (TypeError, ValueError):
            continue
        if abs(v - q) < bd:
            bd, best = abs(v - q), c
    return best


def _select_target(raw, target_col):
    for key in ("target", "variable", "item_id", "column", "series", "channel"):
        if key in raw.columns and raw[key].astype(str).eq(str(target_col)).any():
            return raw[raw[key].astype(str) == str(target_col)].drop(columns=[key]).copy()
    pref = f"{target_col}/"
    if any(str(col).startswith(pref) for col in raw.columns):
        out = raw[[c for c in ("id", "datetime") if c in raw.columns]].copy()
        for c in raw.columns:
            if str(c).startswith(pref):
                out[str(c)[len(pref):]] = raw[c].values
        return out
    return raw


def forecast_series(pipeline, sdf, starts, c, h, channels, target, quantiles):
    """ONE batched predict_df for all windows of a series (one id per window).
    Returns {window_index: (median[h], lower[h], upper[h])} for `target`."""
    variate = [target] + [ch for ch in channels if ch != target]
    ctx_parts, fut_parts = [], []
    for i, s in enumerate(starts):
        wid = f"w{i}"
        ctx = sdf.iloc[s - c:s][["datetime"] + channels].copy()
        ctx["id"] = wid
        ctx_parts.append(ctx)
        fut = sdf.iloc[s:s + h][["datetime"]].copy()
        fut["id"] = wid
        fut_parts.append(fut)
    big = pd.concat(ctx_parts, ignore_index=True)
    bigfut = pd.concat(fut_parts, ignore_index=True)
    try:
        raw = pipeline.predict_df(big, future_df=bigfut, prediction_length=h,
                                  quantile_levels=quantiles, id_column="id",
                                  timestamp_column="datetime", target=variate)
    except TypeError:
        raw = pipeline.predict_df(big, prediction_length=h, quantile_levels=quantiles,
                                  id_column="id", timestamp_column="datetime", target=variate)
    sel = _select_target(raw, target)
    med_c = _qcol(sel.columns, 0.5)
    lo_c = _qcol(sel.columns, quantiles[0])
    hi_c = _qcol(sel.columns, quantiles[-1])
    out = {}
    for wid, gg in sel.groupby("id"):
        gg = gg.sort_values("datetime")
        med = gg[med_c].to_numpy()
        lo = gg[lo_c].to_numpy() if lo_c else med
        hi = gg[hi_c].to_numpy() if hi_c else med
        out[int(str(wid)[1:])] = (med, lo, hi)
    return out


# ----------------------------------------------------------------------------
# Anomaly methods 1..5 over the per-series stream (faithful to forecast_covariate_absolute.py)
# ----------------------------------------------------------------------------
def detect(method, truth, pred, lower, upper, y_range, global_y_range, th):
    """th: dict with keys z, mae_k, mape_k, mape_min, normdiff (all tunable)."""
    n = len(truth)
    is_anom = np.zeros(n, dtype=bool)
    score = np.zeros(n)
    se_list, ae_list, ape_list = [], [], []
    for i in range(n):
        t, p = truth[i], pred[i]
        lo, up = lower[i], upper[i]
        diff = abs(t - p)
        flag = False
        if method == 0:
            flag = (t > up + 1e-2) or (t < lo - 1e-2)
        elif method == 2:
            flag = diff >= 1e-2 and diff / (y_range + 1e-5) > th["normdiff"]
        elif method == 5:
            m2 = diff >= 1e-2 and diff / (y_range + 1e-5) > th["normdiff"]
            m0 = (t > up + 1e-2) or (t < lo - 1e-2)
            flag = m2 and m0
        elif method == 1:
            se = diff ** 2
            if diff >= 1e-2 and len(se_list) >= 2:
                mean_se, std_se = np.mean(se_list), np.std(se_list, ddof=1)
                if std_se > 0:
                    flag = (se - mean_se) / std_se > th["z"]
                elif se > mean_se:
                    flag = True
            se_list.append(se)
        elif method == 3:
            if diff >= 1e-2:
                if len(ae_list) >= 2:
                    mae = np.mean(ae_list)
                    flag = mae > 0 and diff > th["mae_k"] * mae
                elif len(ae_list) == 1:
                    flag = diff > th["mae_k"] * ae_list[0]
            ae_list.append(diff)
        elif method == 4:
            ape = diff / (abs(t) + 1e-5)
            if diff >= 1e-2:
                if len(ape_list) >= 2:
                    mape = np.mean(ape_list)
                    flag = mape > 0 and ape > th["mape_k"] * mape and ape > th["mape_min"]
                elif len(ape_list) == 1:
                    flag = ape > th["mape_k"] * ape_list[0] and ape > th["mape_min"]
            ape_list.append(ape)
        is_anom[i] = flag
        score[i] = min(diff / (global_y_range + 1e-5), 1.0)
    return is_anom, score


def cap_by_fraction(is_anom, score, frac):
    """Keep at most frac*N flags, the highest-scoring ones (expected-contamination ceiling)."""
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


# ----------------------------------------------------------------------------
# Plot one method
# ----------------------------------------------------------------------------
def plot_method(sdf, target, stream, method, is_anom, score, out_png, title):
    from matplotlib.dates import DateFormatter
    plt.figure(figsize=(12, 5))
    plt.plot(sdf["datetime"], sdf[target], color="black", lw=0.8, label="actual")
    plt.plot(stream["datetime"], stream["pred"], color="blue", lw=1.0, label="forecast (median)")
    plt.fill_between(stream["datetime"], stream["lower"], stream["upper"], color="blue",
                     alpha=0.15, label="prediction interval")
    if is_anom.any():
        xa = stream["datetime"].to_numpy()[is_anom]
        ya = stream["truth"].to_numpy()[is_anom]
        sc = score[is_anom]
        plt.scatter(xa, ya, color="red", s=35, zorder=5, label=f"anomaly ({int(is_anom.sum())})")
        # annotate only the most severe few — annotating hundreds is slow and unreadable
        for j in np.argsort(sc)[::-1][:25]:
            plt.annotate(f"{sc[j]:.2f}", (xa[j], ya[j]), textcoords="offset points", xytext=(0, 5),
                         ha="center", fontsize=7, color="black")
    plt.title(title)
    plt.xlabel("time")
    plt.ylabel(target)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8, loc="upper right")   # explicit loc: "best" is slow on long series
    plt.gca().xaxis.set_major_formatter(DateFormatter("%d-%m %H:%M"))
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()


# ----------------------------------------------------------------------------
# Pipeline loading (Chronos-2 multivariate)
# ----------------------------------------------------------------------------
def load_pipeline(model_type, device="cpu"):
    if model_type != "chronos-2":
        raise SystemExit("This sliding script currently supports --model chronos-2 only.")
    os.environ["TORCH_COMPILE_DISABLE"] = "1"
    import torch
    from chronos import Chronos2Pipeline
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device_map = device   # default cpu: GTX 1080 Ti (sm_61) is incompatible with sm_70+ torch builds
    print(f"  device = {device_map}" + ("  (CUDA not available -> cpu)" if device == "cpu" and not torch.cuda.is_available() else ""))
    last = None
    for kw in ({"dtype": torch.bfloat16}, {"torch_dtype": torch.bfloat16}, {}):
        try:
            return Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map=device_map, **kw)
        except TypeError as e:
            last = e
    raise last


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Causal sliding-window forecast anomaly detection (methods 1..5)")
    ap.add_argument("--data", required=True)
    ap.add_argument("--mode", required=True, choices=["flow", "oui-router"])
    ap.add_argument("--context", type=int, default=128, help="context length; rounded to a multiple of 16")
    ap.add_argument("--horizon", type=int, default=12, help="forecast horizon = evaluation window per step")
    ap.add_argument("--step", type=int, default=0, help="slide step (default = horizon, non-overlapping)")
    ap.add_argument("--methods", default="0,1,2,3,4,5",
                    help="which anomaly methods to run (0=bounds, 1=Z, 2=norm-dist, 3=MAE, 4=MAPE, 5=ensemble 0+2)")
    ap.add_argument("--target", default="packet_count", help="channel to forecast/evaluate (must be among the channels)")
    ap.add_argument("--channels", default="",
                    help="comma list of channels for the multivariate model. Default for --mode flow is "
                         f"{','.join(DEFAULT_FLOW_CHANNELS)}; for other modes it is all numeric columns. "
                         "Pass this to override.")
    ap.add_argument("--bin-seconds", type=int, default=300,
                    help="bin size for LIST-format inputs (oui_series.parquet) when 'end' is absent (default 300 = 5min)")
    # --- anomaly sensitivity (defaults chosen to avoid over-flagging) ---
    ap.add_argument("--interval", type=float, default=0.98,
                    help="central prediction interval for bounds methods 0/5. Max 0.98: Chronos-2 is "
                         "trained only on q0.01..q0.99, so the tails are clamped to [0.01, 0.99].")
    ap.add_argument("--max-anomaly-frac", type=float, default=0.02,
                    help="cap EACH method to at most this fraction of points (highest score kept); 0 disables")
    ap.add_argument("--z", type=float, default=4.0, help="method 1: Z-score threshold on squared errors")
    ap.add_argument("--mae-k", type=float, default=5.0, help="method 3: error > k * expanding MAE")
    ap.add_argument("--mape-k", type=float, default=5.0, help="method 4: APE > k * expanding MAPE")
    ap.add_argument("--mape-min", type=float, default=0.5, help="method 4: also require APE above this")
    ap.add_argument("--normdiff", type=float, default=0.3, help="methods 2/5: |err|/range threshold")
    ap.add_argument("--consensus", type=int, default=2,
                    help="also mark is_anom_consensus where at least this many methods agree (0 disables)")
    ap.add_argument("--model", default="chronos-2", choices=["chronos-2"])
    ap.add_argument("--device", default="auto", choices=["cpu", "cuda", "auto"],
                    help="auto (default: GPU if available, e.g. RTX 4090), cuda, or cpu")
    ap.add_argument("--min-bins", type=int, default=0, help="extra length floor; must be >= context+horizon (auto)")
    ap.add_argument("--limit", type=int, default=0, help="cap number of series (0 = all), ranked by total target")
    ap.add_argument("--max-windows", type=int, default=0, help="cap windows per series (0 = all)")
    ap.add_argument("--out-dir", default=DEFAULT_OUT)
    ap.add_argument("--tag", default="",
                    help="representation subfolder under out-dir (default: <mode>_<datafile>_C<C>_H<H>). "
                         "Keeps per-flow 1min / 5min / oui-router results separate instead of overwriting.")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not os.path.exists(args.data):
        raise SystemExit(f"--data not found: {args.data}")
    C = round_mult16(args.context)
    H = args.horizon
    S = args.step if args.step > 0 else H
    if C != args.context:
        print(f"context {args.context} -> {C} (multiple of 16)")
    min_len = max(C + H, args.min_bins)
    methods = [int(m) for m in str(args.methods).split(",") if m.strip() != ""]
    # Chronos-2 is trained on q0.01..q0.99 -> clamp the tails to that range
    lo_q = max(0.01, round((1.0 - args.interval) / 2.0, 5))
    hi_q = min(0.99, round(1.0 - (1.0 - args.interval) / 2.0, 5))
    quantiles = [lo_q, 0.5, hi_q]
    eff = round(hi_q - lo_q, 5)
    note = "" if abs(eff - args.interval) < 1e-9 else f"  (requested {args.interval}, clamped to Chronos-2's q0.01..q0.99)"
    th = dict(z=args.z, mae_k=args.mae_k, mape_k=args.mape_k, mape_min=args.mape_min, normdiff=args.normdiff)
    print(f"interval = q{lo_q}..q{hi_q} ({eff} central){note} | max_anomaly_frac={args.max_anomaly_frac} "
          f"| thresholds {th}")
    data_stem = os.path.splitext(os.path.basename(args.data))[0]
    tag = args.tag or f"{args.mode}_{data_stem}_C{C}_H{H}"
    out_base = os.path.join(args.out_dir, tag)
    print(f"representation tag: {tag}  ->  {out_base}")

    # sniff columns; handle two input shapes: long (per-flow) or list/wide (oui_series.parquet)
    sample = read_head(args.data, 5)
    listch = detect_list_channels(sample)
    if listch:
        # LIST-format: each row is one series with channel arrays -> expand to long
        id_cols = [c for c in ["router", "oui", "direction"] if c in sample.columns] or \
                  (["series_id"] if "series_id" in sample.columns else [])
        if not id_cols:
            raise SystemExit("list-format parquet: need router/oui/direction or series_id to identify series.")
        channels = [c.strip() for c in args.channels.split(",") if c.strip()] if args.channels else list(listch)
        channels = [c for c in channels if c in listch]
        if args.target not in channels:
            raise SystemExit(f"target '{args.target}' not among list channels {list(listch)}; pass --target one of them.")
        print(f"[list-format] expanding series rows -> long | id={id_cols} | channels={channels} "
              f"| bin_seconds={args.bin_seconds} (used only if 'end' absent)")
        df = expand_list_series(args.data, args.bin_seconds, id_cols, channels, min_len)
        if df.empty:
            print(f"No series with >= {min_len} bins after expansion."); return
        keys, kind = id_cols, "oui-router"
    else:
        keys, kind = group_keys(sample, args.mode)
        avail = numeric_channels(sample)
        if args.channels:
            channels = [c.strip() for c in args.channels.split(",") if c.strip()]
            missing = [c for c in channels if c not in avail]
            if missing:
                raise SystemExit(f"--channels not in data: {missing}. Available numeric channels: {avail}")
        elif args.mode == "flow":
            # per-flow default: only the curated flow channels (not every numeric column)
            channels = [c for c in DEFAULT_FLOW_CHANNELS if c in avail]
            skipped = [c for c in DEFAULT_FLOW_CHANNELS if c not in avail]
            if skipped:
                print(f"NOTE: default flow channels not found in data (skipped): {skipped}")
            if not channels:
                print("WARNING: none of the default flow channels present; falling back to all numeric channels.")
                channels = avail
            elif args.target in avail and args.target not in channels:
                channels = [args.target] + channels      # keep a non-default --target usable
        else:
            channels = avail
        if args.target not in channels:
            raise SystemExit(f"target '{args.target}' not among channels {channels}. "
                             f"Pass --target <one of them>. Available numeric channels: {avail}")
        read_cols = list(dict.fromkeys(keys + ["datetime"] + channels))
        df = read_table(args.data, columns=read_cols)
    if "datetime" not in df.columns:
        raise SystemExit("input has no 'datetime' column.")
    df["datetime"] = pd.to_datetime(df["datetime"])
    print(f"[{args.mode}] group by {keys} | channels={channels} | C={C} H={H} step={S} "
          f"| methods={methods} | min_len={min_len}")

    # ---- fast, vectorized candidate scan: NO per-series resample, groups not materialized ----
    gb = df.groupby(keys, sort=False)
    agg = gb.agg(n=("datetime", "nunique"), vol=(args.target, "sum"),
                 tmin=("datetime", "min"), tmax=("datetime", "max"))
    top_key = agg["n"].idxmax()
    _gkey = (lambda k: (k,) if len(keys) == 1 else k)   # pandas wants a tuple for 1-key groupby
    fd = gb.get_group(_gkey(top_key)).sort_values("datetime")["datetime"].diff().mode()
    freq = fd.iloc[0] if len(fd) else pd.Timedelta(minutes=5)
    if freq < pd.Timedelta(seconds=1):
        freq = pd.Timedelta(minutes=5)
    agg["span_bins"] = ((agg["tmax"] - agg["tmin"]) / freq).round().astype("int64") + 1
    qual = agg[agg["span_bins"] >= min_len].sort_values("vol", ascending=False)
    if args.limit and args.limit > 0:
        qual = qual.head(args.limit)
    print(f"inferred bin = {freq} | {len(qual)} series qualify (span >= {min_len} bins)"
          + (f", capped to {args.limit}" if args.limit else ""))
    if len(qual) == 0:
        return

    if args.dry_run:
        for key in qual.index:
            kv = key if isinstance(key, tuple) else (key,)
            sb = int(qual.loc[key, "span_bins"])
            nwin = len(build_windows(sb, C, H, S))
            print(f"  {conn_folder(kind, kv, keys)}: ~{sb} bins, ~{nwin} windows, vol={float(qual.loc[key, 'vol']):.0f}")
        print("(dry-run: model not loaded)")
        return

    print("Loading Chronos-2 (once)...")
    pipeline = load_pipeline(args.model, args.device)

    summary = []
    for ci, key in enumerate(qual.index):
        kv = key if isinstance(key, tuple) else (key,)
        folder = conn_folder(kind, kv, keys)
        sdf = clean_series(gb.get_group(_gkey(key)), channels)
        if sdf is None:
            continue
        starts = build_windows(len(sdf), C, H, S)
        if args.max_windows and args.max_windows > 0:
            starts = starts[: args.max_windows]
        if not starts:
            continue
        print(f"[{ci+1}/{len(qual)}] {folder}: {len(sdf)} bins, {len(starts)} windows")

        try:
            fc = forecast_series(pipeline, sdf, starts, C, H, channels, args.target, quantiles)
        except Exception as e:
            print(f"   forecast failed: {e}")
            continue

        # assemble the per-series stream in time order
        recs = []
        for i, s in enumerate(starts):
            if i not in fc:
                continue
            med, lo, up = fc[i]
            for k in range(min(H, len(med))):
                row_idx = s + k
                recs.append((i, sdf["datetime"].iloc[row_idx], float(sdf[args.target].iloc[row_idx]),
                             float(med[k]), float(lo[k]), float(up[k])))
        if not recs:
            continue
        stream = pd.DataFrame(recs, columns=["window", "datetime", "truth", "pred", "lower", "upper"])
        stream = stream.drop_duplicates("datetime").sort_values("datetime").reset_index(drop=True)

        vals = sdf[args.target].to_numpy()
        y_range = float(vals.max() - vals.min())
        global_y_range = y_range if y_range > 0 else 1.0
        truth = stream["truth"].to_numpy()
        pred = stream["pred"].to_numpy()
        lower = stream["lower"].to_numpy()
        upper = stream["upper"].to_numpy()

        out_feat = os.path.join(out_base, folder, args.target)
        os.makedirs(out_feat, exist_ok=True)
        counts = {}
        for m in methods:
            is_anom, score = detect(m, truth, pred, lower, upper, y_range, global_y_range, th)
            is_anom = cap_by_fraction(is_anom, score, args.max_anomaly_frac)
            stream[f"is_anom_m{m}"] = is_anom
            stream[f"score_m{m}"] = score
            counts[m] = int(is_anom.sum())
            png = os.path.join(out_feat, f"sliding_C{C}_H{H}_method{m}.png")
            plot_method(sdf, args.target, stream, m, is_anom, score, png,
                        f"{folder} | {args.target} | sliding C={C} H={H} | method {m} "
                        f"({counts[m]} anomalies)")
        # consensus: a point is an anomaly only if at least --consensus methods agree
        cons_thr = 0
        if args.consensus and args.consensus > 0 and methods:
            cons_thr = min(args.consensus, len(methods))
            votes = sum(stream[f"is_anom_m{m}"].astype(int) for m in methods)
            cons = votes >= cons_thr
            stream["votes"] = votes
            stream["is_anom_consensus"] = cons
            counts["consensus"] = int(cons.sum())
            png = os.path.join(out_feat, f"sliding_C{C}_H{H}_consensus_min{cons_thr}.png")
            plot_method(sdf, args.target, stream, "consensus", cons.to_numpy(),
                        stream[f"score_m{methods[0]}"].to_numpy(), png,
                        f"{folder} | {args.target} | sliding C={C} H={H} | "
                        f"consensus (>= {cons_thr} of {len(methods)} methods): {counts['consensus']} anomalies")

        stream.to_csv(os.path.join(out_feat, "sliding_scores.csv"), index=False)
        # full actual series for EVERY channel (incl. the first context) + window geometry, for the
        # dashboard: all features are plotted even though only the target is forecast/flagged.
        series_out = sdf[["datetime"] + channels].copy()
        series_out["actual"] = sdf[args.target].values      # backward-compat alias for the target channel
        series_out.to_csv(os.path.join(out_feat, "sliding_series.csv"), index=False)
        import json
        json.dump({"context": int(C), "horizon": int(H), "step": int(S), "interval": float(eff),
                   "target": args.target, "channels": list(channels),
                   "starts": [int(s) for s in starts], "n": int(len(sdf)),
                   "mode": args.mode, "tag": tag, "data": os.path.basename(args.data)},
                  open(os.path.join(out_feat, "sliding_meta.json"), "w"))
        msg = "   anomalies per method: " + ", ".join(f"m{m}={counts[m]}" for m in methods)
        if cons_thr:
            msg += f"  | consensus(>={cons_thr})={counts['consensus']}"
        print(msg)
        summary.append(dict(connection=folder, target=args.target, n_bins=len(sdf),
                            n_windows=len(starts), n_eval=len(stream),
                            **{f"anom_m{m}": counts[m] for m in methods},
                            **({f"anom_consensus_min{cons_thr}": counts["consensus"]} if cons_thr else {})))

    if summary:
        os.makedirs(out_base, exist_ok=True)
        sp = os.path.join(out_base, "summary.csv")
        pd.DataFrame(summary).to_csv(sp, index=False)
        print(f"\nSummary ({len(summary)} series) -> {sp}")


if __name__ == "__main__":
    main()