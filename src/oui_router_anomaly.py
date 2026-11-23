#!/usr/bin/env python3
"""
oui_router_anomaly.py  (src/)

Finds ANOMALOUS WINDOWS inside each series (router, OUI, direction), independently per
series. Every window gets a score saying how atypical it is versus the rest of that
device-direction's own behaviour. Two methods, both scoring PER WINDOW:

  KDE per series:
    score each window by the (neg log) density of the series' windows -- a window in a
    low-density region is atypical versus the rest of the series. No groups.

  KL on consecutive groups:
    split the series' windows into consecutive groups of --group-size. Run the KL method
    --n_groups times: group k in A, the other groups in B. KL gives a D_i per window in A,
    so every window gets a D_i (from the run where its group was A). A window with a high
    D_i is dense in its own group but rare in the rest of the series -> a sustained change
    localised to that window. The group must be big enough that the local density in A is
    estimable (default 8); windows at a group boundary depend a little on where the cut is.

Both run on the WINDOW parquet from series_to_windows.py (one row per window, channel lists
of length W, plus series_id / router / oui / direction). Windows are embedded once with
Chronos-2 and PCA-projected; the two scorers then work per series on those embeddings.

Output (results/oui_router/<tag>/):
  window_scores.csv   per window: series_id, router, oui, direction, start, w_index,
                      kde_score, kl_score (NaN if its series was too short for that method),
                      kde_rank_in_series, kl_rank_in_series
  series_summary.csv  per series: n_windows, n_kde_anom, n_kl_anom, top window times
  anomalous_windows_in_series.pdf   each series with its anomalous windows marked
                      (by whichever method, selectable)

USAGE
  python src/oui_router_anomaly.py --windows data/parquet/oui_windows.parquet \
      --reg concat --pca-dim 16 --group-size 8 --min-windows 16 \
      --kde-frac 0.1 --kl-frac 0.1 --device cuda
"""
import argparse
import importlib.util
import os

import numpy as np
import pandas as pd

_spec = importlib.util.spec_from_file_location(
    "anomaly_kl", os.path.join(os.path.dirname(os.path.abspath(__file__)), "anomaly_kl.py"))
AK = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(AK)
_kspec = importlib.util.spec_from_file_location(
    "kde_density", os.path.join(os.path.dirname(os.path.abspath(__file__)), "kde_density.py"))
KDE = importlib.util.module_from_spec(_kspec)
_kspec.loader.exec_module(KDE)

CHANNELS = ["packet_count", "bytes", "n_conns"]


def load_windows(path, channels, max_windows=0, seed=0):
    import pyarrow.parquet as pq
    cols = ["window_id", "series_id", "router", "oui", "direction", "start"] + channels
    avail = set(pq.ParquetFile(path).schema_arrow.names)
    cols = [c for c in cols if c in avail]
    df = pd.read_parquet(path, columns=cols)
    if max_windows and len(df) > max_windows:
        df = df.sample(max_windows, random_state=seed).reset_index(drop=True)
    AK._log(f"{len(df):,} windows, {df['series_id'].nunique()} series")
    frames = [np.stack([np.asarray(getattr(r, c), np.float32) for c in channels], axis=1)
              for r in df.itertuples(index=False)]
    return df.reset_index(drop=True), frames


def kl_group_scores(P_series, group_size, smooth_k, perms, alpha, sig_alpha, seed):
    """Full anomaly_kl pipeline per consecutive group. P_series: (m, d) embeddings of one
    series' windows IN TIME ORDER. For each group k: A = group k, B = the rest, then run
    EXACTLY the anomaly_kl steps -- kl_contributions, permutation_null (threshold +
    cluster-mass null), survivors (D_i > thr), cluster_regions, rank_regions -- with the
    SAME smooth_k for contributions and null (required for calibration).

    smooth_k is reduced per group when A is small: _smooth averages each D_i with its k
    nearest A-neighbours, so k must be < |A|; we use min(smooth_k, |A|-1).

    Returns, per window (aligned to P_series rows):
      D        : D_i (raw, un-smoothed, from kl_contributions with smooth_k=0) -- for ranking
      D_smooth : the smoothed D_i actually tested against the permutation threshold
      survivor : D_smooth > permutation threshold
      sig      : survivor AND in a region with cluster-mass p <= sig_alpha
      region   : region id within the group (-1 = none)
    NaN where a group was too small to run."""
    m = len(P_series)
    out = {k: np.full(m, np.nan) for k in ("D", "D_smooth", "survivor", "sig", "region")}
    n_groups = int(np.ceil(m / group_size))
    if n_groups < 2:
        return out
    bounds = [(g * group_size, min((g + 1) * group_size, m)) for g in range(n_groups)]
    region_offset = 0
    for gi, (a0, a1) in enumerate(bounds):
        A = P_series[a0:a1]
        B = np.delete(P_series, np.arange(a0, a1), axis=0)
        nA = len(A)
        if nA < 2 or len(B) < 2:
            continue
        sk = min(smooth_k, nA - 1) if smooth_k > 0 else 0       # k must be < |A|
        # --- exactly the anomaly_kl steps ---
        D_raw = AK.kl_contributions(A, B, smooth_k=0)
        D_sm = AK.kl_contributions(A, B, smooth_k=sk)
        rng = np.random.default_rng(seed + 1000 * gi)
        thr, null_region_max = AK.permutation_null(A, B, perms, alpha, rng, smooth_k=sk)
        survivors = D_sm > thr
        region = AK.cluster_regions(A, D_sm, survivors)         # -1 where not a region
        reg_df = AK.rank_regions(region, D_sm, D_raw, float(D_raw.sum()), null_region_max,
                                 marked=None, XA=A)
        sig_ids = set(reg_df.loc[reg_df["p_value"] <= sig_alpha, "region_id"]) if len(reg_df) else set()
        sig = np.isin(region, list(sig_ids)) if sig_ids else np.zeros(nA, bool)
        # write back
        out["D"][a0:a1] = D_raw
        out["D_smooth"][a0:a1] = D_sm
        out["survivor"][a0:a1] = survivors.astype(float)
        out["sig"][a0:a1] = sig.astype(float)
        reg_global = np.where(region >= 0, region + region_offset, -1)
        out["region"][a0:a1] = reg_global
        region_offset += (int(region.max()) + 1) if region.max() >= 0 else 0
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--windows", required=True, help="window parquet from series_to_windows.py")
    ap.add_argument("--channels", default=",".join(CHANNELS))
    ap.add_argument("--reg", default="concat")
    ap.add_argument("--pca-dim", type=int, default=16)
    ap.add_argument("--group-size", type=int, default=8,
                    help="windows per consecutive group for KL (default 8; must keep A estimable)")
    ap.add_argument("--min-windows", type=int, default=16,
                    help="skip series with fewer windows than this (need >= 2 groups for KL)")
    ap.add_argument("--smooth-k", type=int, default=5,
                    help="paper smoothing of D_i over k nearest A-neighbours; auto-reduced to "
                         "|A|-1 per group when A is small")
    ap.add_argument("--perms", type=int, default=20, help="KL permutations per group (null)")
    ap.add_argument("--alpha", type=float, default=0.01, help="survivor threshold quantile (KL)")
    ap.add_argument("--sig-alpha", type=float, default=0.05, help="region p-value cutoff (KL)")
    ap.add_argument("--bandwidth", default="cv", help="KDE bandwidth: cv|rips|scott|float")
    ap.add_argument("--kde-frac", type=float, default=0.1,
                    help="flag the top fraction of windows per series by KDE score")
    ap.add_argument("--kl-frac", type=float, default=0.1,
                    help="flag the top fraction of windows per series by KL score")
    ap.add_argument("--mark-method", choices=["kde", "kl", "both"], default="both",
                    help="which score marks windows in the PDF")
    ap.add_argument("--embed-batch", type=int, default=64)
    ap.add_argument("--embed-cap", type=int, default=0)
    ap.add_argument("--max-windows", type=int, default=0)
    ap.add_argument("--max-series-plots", type=int, default=40)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--root", default=AK.DEFAULT_ROOT)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()
    channels = [c.strip() for c in args.channels.split(",") if c.strip()]
    reg = args.reg
    if reg not in ("mean", "concat", "target"):
        reg = channels.index(reg) if reg in channels else (int(reg) if reg.lstrip("-").isdigit() else "concat")

    df, frames = load_windows(args.windows, channels, args.max_windows, args.seed)
    if len(df) < 10:
        print("too few windows."); return

    # embed all windows once, project once
    ch = AK.Chronos(device=args.device)
    AK._log(f"embedding {len(frames):,} windows (reg={reg}) ...")
    emb = ch.embed(frames, channels, args.embed_batch, cap=args.embed_cap, reg_mode=reg)
    P = KDE._project(emb, args.pca_dim)                 # whitened projection (good for KDE & KL)
    AK._log(f"projected to {P.shape[1]}-D")

    # per series: order windows by time, run both scorers
    df["_row"] = np.arange(len(df))
    kde_score = np.full(len(df), np.nan)
    kl_D = np.full(len(df), np.nan)           # raw D_i (for ranking)
    kl_surv = np.full(len(df), np.nan)        # survivor (D_smooth > perm threshold)
    kl_sig = np.full(len(df), np.nan)         # survivor AND in a significant region
    series_rows = []
    n_series = n_kde_ok = n_kl_ok = 0
    for sid, g in df.groupby("series_id", sort=False):
        g = g.sort_values("start")
        rows = g["_row"].to_numpy()
        m = len(rows)
        n_series += 1
        if m < args.min_windows:
            continue
        Ps = P[rows]
        # KDE per series
        try:
            s, _ = KDE.kde_scores(Ps, bandwidth=args.bandwidth)
            kde_score[rows] = s; n_kde_ok += 1
        except Exception as e:
            AK._log(f"KDE failed for {sid}: {type(e).__name__}")
        # KL on consecutive groups -- FULL anomaly_kl pipeline per group
        kl = kl_group_scores(Ps, args.group_size, args.smooth_k, args.perms, args.alpha,
                             args.sig_alpha, args.seed)
        kl_D[rows] = kl["D"]; kl_surv[rows] = kl["survivor"]; kl_sig[rows] = kl["sig"]
        if np.isfinite(kl["D"]).any():
            n_kl_ok += 1
    AK._log(f"{n_series} series; KDE scored {n_kde_ok}, KL scored {n_kl_ok} "
            f"(>= {args.min_windows} windows)")

    df["kde_score"] = kde_score
    df["kl_D"] = kl_D
    # kl_surv is float with NaN where a group was too small; keep NaN, mark 1.0 as survivor
    df["kl_survivor"] = pd.array([True if v == 1.0 else (False if v == 0.0 else pd.NA)
                                  for v in kl_surv], dtype="boolean")
    df["kl_sig_anom"] = kl_sig                                  # 1.0 = significant KL anomaly
    df["kde_rank_in_series"] = df.groupby("series_id")["kde_score"].rank(ascending=False, method="first")
    df["kl_rank_in_series"] = df.groupby("series_id")["kl_D"].rank(ascending=False, method="first")
    df["kde_anom"] = _flag_top(df, "kde_score", "series_id", args.kde_frac)
    # KL anomaly = significant region membership (the FULL pipeline verdict), not top-frac
    df["kl_anom"] = (df["kl_sig_anom"] == 1.0).fillna(False)

    out_dir = args.out_dir or _out_dir(args)
    os.makedirs(out_dir, exist_ok=True)
    keep = ["window_id", "series_id", "router", "oui", "direction", "start",
            "kde_score", "kde_rank_in_series", "kde_anom",
            "kl_D", "kl_survivor", "kl_sig_anom", "kl_rank_in_series", "kl_anom"]
    df[[c for c in keep if c in df.columns]].to_csv(
        os.path.join(out_dir, "window_scores.csv"), index=False)
    _series_summary(out_dir, df)
    _plot_series(out_dir, df, channels, frames, args.mark_method, args.max_series_plots)
    AK._log(f"results -> {out_dir}")


def _flag_top(df, score_col, by, frac):
    out = np.zeros(len(df), bool)
    for _, g in df.groupby(by):
        s = g[score_col]
        valid = s.notna()
        if valid.sum() == 0:
            continue
        k = max(1, int(round(frac * valid.sum())))
        thr = s[valid].nlargest(k).min()
        out[g.index[valid & (s >= thr)]] = True
    return out


def _series_summary(out_dir, df):
    rows = []
    for sid, g in df.groupby("series_id"):
        if g["kde_score"].notna().sum() == 0 and g["kl_D"].notna().sum() == 0:
            continue
        rows.append({"series_id": sid, "router": g["router"].iloc[0], "oui": g["oui"].iloc[0],
                     "direction": g["direction"].iloc[0], "n_windows": len(g),
                     "n_kde_anom": int(g["kde_anom"].sum()), "n_kl_anom": int(g["kl_anom"].sum())})
    pd.DataFrame(rows).sort_values("n_kde_anom", ascending=False).to_csv(
        os.path.join(out_dir, "series_summary.csv"), index=False)


def _plot_series(out_dir, df, channels, frames, mark_method, cap):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    pc_idx = channels.index("packet_count") if "packet_count" in channels else 0
    # series with any anomaly, most first
    flagcol = {"kde": ["kde_anom"], "kl": ["kl_anom"], "both": ["kde_anom", "kl_anom"]}[mark_method]
    anom_any = df[flagcol].any(axis=1)
    order = (df[anom_any].groupby("series_id").size().sort_values(ascending=False).index.tolist())
    if cap:
        order = order[:cap]
    if not order:
        return
    with PdfPages(os.path.join(out_dir, "anomalous_windows_in_series.pdf")) as pdf:
        for sid in order:
            g = df[df.series_id == sid].sort_values("start")
            fig, ax = plt.subplots(figsize=(14, 3.2))
            x = np.arange(len(g))
            # draw the per-window packet_count mean as the series profile
            prof = [frames[r][:, pc_idx].mean() for r in g["_row"]]
            ax.plot(x, prof, "-o", ms=3, lw=0.8, color="#1f77b4", label="window mean packet_count")
            for col, color, lab in [("kde_anom", "#d62728", "KDE anomaly"),
                                    ("kl_anom", "#ff7f0e", "KL anomaly")]:
                if col in flagcol:
                    m = g[col].to_numpy()
                    ax.scatter(x[m], np.array(prof)[m], s=60, color=color, zorder=3, label=lab)
            r = g.iloc[0]
            ax.set_title(f"{r.oui} on {r.router} ({r.direction}) -- {len(g)} windows", fontsize=9)
            ax.set_xlabel("window index (time order)"); ax.set_ylabel("mean packet_count")
            ax.legend(fontsize=7)
            fig.tight_layout(); pdf.savefig(fig); plt.close(fig)


def _out_dir(args):
    base = os.path.join(args.root, "results", "oui_router")
    name = f"reg-{args.reg}_pca{args.pca_dim}_grp{args.group_size}_seed{args.seed}"
    os.makedirs(base, exist_ok=True)
    run = 1
    while os.path.exists(os.path.join(base, f"{name}_run{run}")):
        run += 1
    return os.path.join(base, f"{name}_run{run}")


if __name__ == "__main__":
    main()