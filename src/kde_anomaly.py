#!/usr/bin/env python3
"""
kde_anomaly.py  (src/)  -- standalone KDE anomaly detection with the same results and plots
as anomaly_kl, but single-sample (density-based), NOT the KL method.

Pipeline: load flows -> embed with Chronos-2 (directly, or from a cached .npz) -> PCA ->
KDE anomaly score (neg log-density; low density = anomalous) -> threshold the top fraction
-> cluster the anomalies into regions (HDBSCAN) -> write scores, regions, summary and the
anomaly_kl-style plots.

This is an INDEPENDENT detector: it never uses an A/B split or the KL contribution. A flow
is anomalous simply if it sits in a low-density region of the whole embedding.

Outputs (results/kde/<tag>/):
    scores.csv            per flow: cid, neg_log_density, anomaly (top-frac), region_id,
                          sig_anomaly, pc1, pc2, oui/router/... when present
    regions.csv           per anomaly region: size, mean score, dominant oui/service
    anomalous_flows.csv   the flagged flows with metadata
    clustering.png        embedding 2-D: all flows, anomalies coloured by region
    score_distribution.png  neg-log-density histogram with the anomaly threshold
    regions_map.png       significant regions coloured + temporal pattern panel
    validation.pdf        score distribution, embedding map, per-region overlays
    embeddings_proj.npz   projected embeddings + cid + scores (cache for re-plotting)
    run_config.json, summary.txt

USAGE
  # embed directly with Chronos-2:
  python src/kde_anomaly.py --csv data/.../features.parquet --device cuda \
      --channels packet_count,fwd_mean_iat,fwd_mean_packet_size,bwd_mean_iat,bwd_mean_packet_size \
      --reg concat --pca-dim 32 --anomaly-frac 0.05

  # reuse cached embeddings (skip Chronos):
  python src/kde_anomaly.py --emb results/kde/<run>/embeddings_proj.npz --anomaly-frac 0.05
"""
import argparse
import importlib.util
import json
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

CHANNELS = "packet_count,fwd_mean_iat,fwd_mean_packet_size,bwd_mean_iat,bwd_mean_packet_size"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default=None, help="features parquet (embed with Chronos-2)")
    ap.add_argument("--only", default=None)
    ap.add_argument("--features-dir", default=None)
    ap.add_argument("--emb", default=None, help="cached embeddings_proj.npz (skip Chronos)")
    ap.add_argument("--root", default=AK.DEFAULT_ROOT)
    ap.add_argument("--bin", default="5min")
    ap.add_argument("--channels", default=CHANNELS)
    ap.add_argument("--min-bins", type=int, default=1,
                    help="keep only flows spanning >= this many bins (default 1 = no filter)")
    ap.add_argument("--min-packets", type=int, default=1,
                    help="keep only flows with >= this many packets (default 1 = no filter)")
    ap.add_argument("--min-minutes", type=float, default=0,
                    help="keep only flows lasting >= this many minutes; overrides --min-bins "
                         "and is bin-size-independent (default 0 = use --min-bins)")
    ap.add_argument("--reg", default="concat", help="REG reduction: mean|concat|<channel>|<idx>")
    ap.add_argument("--pca-dim", type=int, default=32, help="KDE needs low-D (16..64)")
    ap.add_argument("--bandwidth", default="cv",
                    help="cv (default) | rips (lookout: Rips death-diameter quantile, consistent) "
                         "| scott | silverman | float")
    ap.add_argument("--rips-gamma", type=float, default=0.98,
                    help="quantile of Rips death diameters for --bandwidth rips (default 0.98)")
    ap.add_argument("--robust-scale", action="store_true",
                    help="lookout-style robust standardization (median + robust whitening) before "
                         "PCA, instead of mean/std; more resistant to outliers")
    ap.add_argument("--leave-one-out", action="store_true", help="robust LOO scoring (slower)")
    ap.add_argument("--anomaly-frac", type=float, default=0.05,
                    help="top fraction flagged (used when --threshold frac)")
    ap.add_argument("--threshold", choices=["frac", "gpd"], default="frac",
                    help="frac: flag the top --anomaly-frac by score (always flags that fraction). "
                         "gpd: lookout-style GPD on the surprisal tail -> flags by probability, and "
                         "can flag ZERO when the data is clean.")
    ap.add_argument("--alpha", type=float, default=0.001,
                    help="GPD anomaly probability threshold for --threshold gpd (default 0.001)")
    ap.add_argument("--beta", type=float, default=0.90,
                    help="GPD tail proportion for --threshold gpd (fit to the top 1-beta; default 0.90)")
    ap.add_argument("--min-cluster", type=int, default=5, help="HDBSCAN min_cluster_size for regions")
    ap.add_argument("--max-flows", type=int, default=0)
    ap.add_argument("--embed-batch", type=int, default=64)
    ap.add_argument("--embed-cap", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--no-save-embeddings", action="store_true")
    ap.add_argument("--nearest-k", type=int, default=5,
                    help="how many nearest NORMAL flows to draw behind each anomaly (default 5)")
    ap.add_argument("--max-anom-plots", type=int, default=40,
                    help="cap the anomalies drawn in anomalies_vs_nearest.pdf (default 40; 0=all)")
    args = ap.parse_args()
    channels = [c.strip() for c in args.channels.split(",") if c.strip()]
    reg = args.reg
    if reg not in ("mean", "concat", "target"):
        reg = channels.index(reg) if reg in channels else (int(reg) if reg.lstrip("-").isdigit() else "concat")

    # ---- get projected embeddings: from cache, or embed with Chronos-2 ----
    meta_df = None
    flows = None
    emb_raw = None
    if args.emb:
        z = np.load(args.emb, allow_pickle=True)
        cids = np.array([str(c) for c in z["cid"]]) if "cid" in z else None
        if "emb_raw" in z:                      # new cache: raw embeddings -> project fresh
            P = KDE._project(z["emb_raw"], args.pca_dim, robust=args.robust_scale)
            _log(f"loaded {len(P):,} raw embeddings from {args.emb}; re-projected (pca={args.pca_dim}, "
                 f"whitened)")
        else:                                   # old cache: only projected P -> whiten it now
            P = z["P"] if "P" in z else (z["emb"] if "emb" in z else z["XA"])
            P = (P - P.mean(0)) / (P.std(0) + 1e-9)
            _log(f"loaded {len(P):,} projected embeddings from {args.emb} (whitened on load; old "
                 f"cache had un-whitened components)")
        if cids is None:
            cids = np.array([str(i) for i in range(len(P))])
        path = args.emb
    else:
        path = _resolve(args)
        if path is None:
            print("no features file found (give --csv or --only/--features-dir)."); return
        _log(f"features: {path}")
        _filt = (f"duration >= {args.min_minutes} min" if args.min_minutes
                 else f"span >= {args.min_bins} bins") + f" & packets >= {args.min_packets}"
        _log(f"flow filter: {_filt}")
        flows = AK.load_flows(path, channels, args.bin, args.min_bins, args.min_packets,
                              "packet_count", max_flows=args.max_flows, seed=args.seed,
                              min_minutes=args.min_minutes)
        _log(f"kept {len(flows):,} flows after filtering")
        if len(flows) < 20:
            print("too few flows after filtering."); return
        cids = np.array([f["cid"] for f in flows])
        ch = AK.Chronos(device=args.device)
        _log(f"embedding {len(flows):,} flows with Chronos-2 (reg={reg}) ...")
        emb = ch.embed([f["frame"] for f in flows], channels, args.embed_batch,
                       cap=args.embed_cap, reg_mode=reg)
        emb_raw = np.asarray(emb, np.float32)     # keep raw for cache re-projection
        P = KDE._project(emb_raw, args.pca_dim, robust=args.robust_scale)
        _log(f"projected to {P.shape[1]}-D")
        meta_df = _load_meta(path, cids)

    out_dir = args.out_dir or _out_dir(args, path)
    os.makedirs(out_dir, exist_ok=True)

    # ---- KDE scoring ----
    _log(f"KDE scoring (bandwidth={args.bandwidth}"
         + (", leave-one-out" if args.leave_one_out else "") + ") ...")
    scores, bw = KDE.kde_scores(P, args.bandwidth, leave_one_out=args.leave_one_out,
                                rips_gamma=args.rips_gamma)
    _log(f"bandwidth={bw:.4g}; neg-log-density median {np.median(scores):.3f}, max {scores.max():.3f}")

    # threshold: fixed fraction, or lookout GPD on the surprisal tail
    if args.threshold == "gpd":
        pvals = KDE.gpd_anomaly_prob(scores, beta=args.beta)
        anomaly = pvals < args.alpha
        thr = float(scores[anomaly].min()) if anomaly.any() else float(scores.max())
        _log(f"flagged {int(anomaly.sum())} / {len(scores)} flows (GPD, p<{args.alpha}); "
             f"GPD can flag zero on clean data")
        if not anomaly.any():
            _log(f"  -> ZERO anomalies at alpha={args.alpha}. The tail is not heavy enough at "
                 f"this threshold. Try a looser --alpha (e.g. 0.01 or 0.05), or --threshold frac.")
    else:
        k = max(1, int(round(args.anomaly_frac * len(scores))))
        thr = np.partition(scores, -k)[-k]
        anomaly = scores >= thr
        pvals = None
        _log(f"flagged {int(anomaly.sum())} / {len(scores)} flows (top {args.anomaly_frac:.0%}, thr={thr:.3f})")

    # cluster the anomalies into regions
    region = np.full(len(P), -1, dtype=int)
    aidx = np.flatnonzero(anomaly)
    if len(aidx) >= args.min_cluster:
        region[aidx] = _cluster(P[aidx], args.min_cluster)
    sig_ids = sorted(int(r) for r in set(region[aidx]) if r >= 0)
    _log(f"anomaly regions: {len(sig_ids)}")

    # ---- write results ----
    sc = pd.DataFrame({"cid": cids, "neg_log_density": scores,
                       "gpd_pvalue": (pvals if pvals is not None else np.nan), "anomaly": anomaly,
                       "region_id": region, "sig_anomaly": anomaly & (region >= 0),
                       "pc1": P[:, 0], "pc2": P[:, 1]})
    if meta_df is not None:
        sc = sc.merge(meta_df, on="cid", how="left")
    sc = sc.sort_values("neg_log_density", ascending=False)
    sc.to_csv(os.path.join(out_dir, "scores.csv"), index=False)
    sc[sc.anomaly].to_csv(os.path.join(out_dir, "anomalous_flows.csv"), index=False)
    _regions_csv(out_dir, sc, region, scores, sig_ids, meta_df)

    if not args.no_save_embeddings:
        save_kw = dict(P=P.astype(np.float32), cid=cids, scores=scores.astype(np.float32),
                       region=region)
        if emb_raw is not None:                   # only when we embedded fresh
            save_kw["emb_raw"] = emb_raw
        np.savez_compressed(os.path.join(out_dir, "embeddings_proj.npz"), **save_kw)
    _config(out_dir, args, len(P), bw, int(anomaly.sum()), len(sig_ids))

    # ---- plots (anomaly_kl style) ----
    oui_P = (meta_df.set_index("cid")["oui"].reindex([str(c) for c in cids]).to_numpy()
             if meta_df is not None and "oui" in meta_df.columns else None)
    _plot_clustering(out_dir, P, anomaly, region, sig_ids)
    _plot_score_dist(out_dir, scores, thr, args.anomaly_frac)
    _plot_regions_map(out_dir, P, region, sig_ids, scores, oui_P)
    _plot_validation(out_dir, P, scores, anomaly, region, sig_ids)
    if flows is not None:
        _plot_anomalies_vs_nearest(out_dir, P, scores, anomaly, flows, channels,
                                   int(args.nearest_k), int(args.max_anom_plots))
    _summary(out_dir, args, len(P), scores, anomaly, sig_ids, bw, sc, meta_df)
    _log(f"results -> {out_dir}")


# --------------------------------------------------------------------------- #
def _log(m):
    AK._log(m)


def _resolve(args):
    import glob
    if args.csv:
        return args.csv
    fdir = args.features_dir or os.path.join(args.root, "data", "features")
    cand = sorted(glob.glob(os.path.join(fdir, "*.parquet")))
    if args.only:
        cand = [p for p in cand if os.path.basename(p).startswith(args.only) or args.only in os.path.basename(p)]
    return cand[0] if cand else None


def _load_meta(path, cids):
    import pyarrow.dataset as ds, pyarrow as pa, pyarrow.compute as pc
    cols = ["conn_id", "oui", "router", "protocol", "dst_port", "conn_direction", "is_iot"]
    avail = set(ds.dataset(path, format="parquet").schema.names)
    cols = [c for c in cols if c in avail]
    if "conn_id" not in cols:
        return None
    t = ds.dataset(path, format="parquet").to_table(
        columns=cols, filter=pc.field("conn_id").isin(pa.array([str(c) for c in cids])))
    d = t.to_pandas().drop_duplicates("conn_id")
    d = d.rename(columns={"conn_id": "cid"}); d["cid"] = d["cid"].astype(str)
    return d


def _cluster(X, min_cluster):
    try:
        import hdbscan
        lab = hdbscan.HDBSCAN(min_cluster_size=max(2, min_cluster)).fit_predict(X)
        if (lab < 0).all():
            lab = hdbscan.HDBSCAN(min_cluster_size=max(2, min_cluster),
                                  allow_single_cluster=True).fit_predict(X)
        return lab
    except Exception:
        from sklearn.cluster import DBSCAN
        return DBSCAN(eps=float(np.median(np.std(X, 0))) + 1e-9, min_samples=min_cluster).fit_predict(X)


def _regions_csv(out_dir, sc, region, scores, sig_ids, meta_df):
    from collections import Counter
    rows = []
    for r in sig_ids:
        m = region == r
        row = {"region_id": r, "size": int(m.sum()), "mean_neg_log_density": float(scores[m].mean())}
        if meta_df is not None and "oui" in sc.columns:
            sub = sc[sc.region_id == r]
            if sub["oui"].notna().any():
                row["dominant_oui"] = Counter(sub["oui"].dropna()).most_common(1)[0][0]
            if "dst_port" in sub and sub["dst_port"].notna().any():
                row["dominant_dst_port"] = Counter(sub["dst_port"].dropna()).most_common(1)[0][0]
        rows.append(row)
    df = pd.DataFrame(rows)
    if len(df):
        df = df.sort_values("size", ascending=False)
    df.to_csv(os.path.join(out_dir, "regions.csv"), index=False)


def _config(out_dir, args, n, bw, n_anom, n_reg):
    json.dump({"method": "KDE", "n_flows": n, "bandwidth": bw, "anomaly_frac": args.anomaly_frac,
               "n_anomalies": n_anom, "n_regions": n_reg, "reg": args.reg, "pca_dim": args.pca_dim,
               "channels": args.channels, "bin": args.bin, "min_bins": args.min_bins,
               "min_packets": args.min_packets, "leave_one_out": args.leave_one_out, "seed": args.seed},
              open(os.path.join(out_dir, "run_config.json"), "w"), indent=2)


def _out_dir(args, path):
    tag = os.path.splitext(os.path.basename(path))[0][:8]
    base = os.path.join(args.root, "results", "kde")
    d = os.path.join(base, f"{tag}_reg-{args.reg}_pca{args.pca_dim}_af{int(args.anomaly_frac*100)}_seed{args.seed}")
    os.makedirs(base, exist_ok=True)
    run = 1
    while os.path.exists(f"{d}_run{run}"):
        run += 1
    return f"{d}_run{run}"


def _safe_bins(x, want=30):
    """Histogram bin edges valid even for constant/tiny-range data."""
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if x.size == 0:
        return np.array([0.0, 1.0])
    lo, hi = float(x.min()), float(x.max())
    if hi <= lo:
        pad = (abs(lo) * 1e-6) or 1e-6
        return np.array([lo - pad, lo + pad])
    n = int(min(want, max(1, x.size)))
    e = np.linspace(lo, hi, n + 1)
    return e if np.all(np.diff(e) > 0) else np.array([lo, hi])


def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _plot_clustering(out_dir, P, anomaly, region, sig_ids):
    plt = _mpl()
    xy = P[:, :2]
    fig, ax = plt.subplots(1, 2, figsize=(15, 7))
    ax[0].scatter(xy[~anomaly, 0], xy[~anomaly, 1], s=4, c="#e4e4e4", label="normal", zorder=1)
    ax[0].scatter(xy[anomaly, 0], xy[anomaly, 1], s=14, c="#d62728",
                  label=f"anomaly (n={int(anomaly.sum())})", zorder=3)
    ax[0].set_title("KDE anomalies (low density)"); ax[0].legend(fontsize=8)
    cmap = plt.get_cmap("tab20")
    ax[1].scatter(xy[:, 0], xy[:, 1], s=4, c="#e4e4e4", zorder=1)
    for gi, r in enumerate(sig_ids):
        m = region == r
        ax[1].scatter(xy[m, 0], xy[m, 1], s=16, color=cmap(gi % 20), label=f"R{r} (n={int(m.sum())})", zorder=3)
    ax[1].set_title(f"anomaly regions ({len(sig_ids)})")
    if sig_ids:
        ax[1].legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    for a in ax:
        a.set_xlabel("PC1"); a.set_ylabel("PC2")
    fig.suptitle("flows in the Chronos-2 embedding (PCA 2-D) -- KDE density anomalies", fontsize=12)
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "clustering.png"), dpi=140, bbox_inches="tight")
    plt.close(fig)


def _plot_score_dist(out_dir, scores, thr, frac):
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.hist(scores, bins=60, color="#888", alpha=0.8)
    ax.axvline(thr, color="#d62728", ls="--", label="anomaly threshold")
    ax.set_xlabel("neg log-density (higher = rarer = more anomalous)"); ax.set_ylabel("flows")
    ax.set_title("KDE anomaly score distribution"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "score_distribution.png"), dpi=140); plt.close(fig)


def _plot_regions_map(out_dir, P, region, sig_ids, scores, oui_P):
    from collections import Counter
    plt = _mpl()
    xy = P[:, :2]
    fig, ax = plt.subplots(1, 2, figsize=(15, 7))
    insig = ~np.isin(region, sig_ids) if sig_ids else np.ones(len(xy), bool)
    cmap = plt.get_cmap("tab20")
    ax[0].scatter(xy[insig, 0], xy[insig, 1], s=4, c="#e4e4e4", zorder=1)
    for gi, r in enumerate(sig_ids):
        m = region == r
        lab = f"R{r}"
        if oui_P is not None:
            vals = [o for o in oui_P[m] if isinstance(o, str) and o]
            if vals:
                lab += f" {Counter(vals).most_common(1)[0][0]}"
        ax[0].scatter(xy[m, 0], xy[m, 1], s=16, color=cmap(gi % 20), label=f"{lab} (n={int(m.sum())})", zorder=3)
    if not sig_ids:
        ax[0].text(0.5, 0.5, "no anomaly regions\n(no anomalies, or they did not cluster)",
                   transform=ax[0].transAxes, ha="center", va="center", fontsize=10, color="#888")
    ax[0].set_title("anomaly regions"); ax[0].set_xlabel("PC1"); ax[0].set_ylabel("PC2")
    if sig_ids:
        ax[0].legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    s2 = ax[1].scatter(xy[:, 0], xy[:, 1], s=6, c=scores, cmap="viridis")
    fig.colorbar(s2, ax=ax[1], label="neg log-density")
    ax[1].set_title("density score over the embedding"); ax[1].set_xlabel("PC1"); ax[1].set_ylabel("PC2")
    fig.suptitle("KDE regions map", fontsize=12)
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "regions_map.png"), dpi=140, bbox_inches="tight")
    plt.close(fig)


def _plot_anomalies_vs_nearest(out_dir, P, scores, anomaly, flows, channels, k, cap):
    """For each anomalous flow, draw its channels (red) over its k nearest NORMAL flows
    (green) in the embedding -- so you see what the anomaly does differently. Also writes
    nearest_normal.csv (anomaly cid -> nearest normal cids and distances)."""
    from sklearn.neighbors import NearestNeighbors
    import pandas as pd
    plt = _mpl()
    from matplotlib.backends.backend_pdf import PdfPages
    a_idx = np.flatnonzero(anomaly)
    n_idx = np.flatnonzero(~anomaly)
    if len(a_idx) == 0 or len(n_idx) == 0:
        return
    kk = min(k, len(n_idx))
    nn = NearestNeighbors(n_neighbors=kk).fit(P[n_idx])
    # order anomalies by score (most anomalous first); cap how many we draw
    order = a_idx[np.argsort(-scores[a_idx])]
    if cap and len(order) > cap:
        order = order[:cap]
    dist, nbr = nn.kneighbors(P[order])          # nbr indexes into n_idx
    C = len(channels)
    rows = []
    # one block per anomaly = 1 row for the anomaly (red) + kk rows for its nearest
    # normals (green), each flow on its OWN axes (not overlaid). One anomaly block per page.
    with PdfPages(os.path.join(out_dir, "anomalies_vs_nearest.pdf")) as pdf:
        for oi, ai in enumerate(order):
            af = flows[ai]
            near = [n_idx[j] for j in nbr[oi]]
            nrows = 1 + len(near)
            fig, axes = plt.subplots(nrows, C, figsize=(3.3 * C, 1.6 * nrows + 0.8), squeeze=False)
            # anomaly row
            for c, chn in enumerate(channels):
                ax = axes[0][c]
                v = af["frame"][:, c]
                ax.plot(np.arange(len(v)), v, lw=1.0, color="#d62728")
                ax.tick_params(labelsize=5); ax.set_title(chn, fontsize=8)
            axes[0][0].set_ylabel(f"ANOMALY\n{af['cid']}\nscore {scores[ai]:.2f}", fontsize=6,
                                  color="#d62728")
            # one nearest-normal per row
            for r, nj in enumerate(near, start=1):
                for c in range(C):
                    ax = axes[r][c]
                    v = flows[nj]["frame"][:, c]
                    ax.plot(np.arange(len(v)), v, lw=0.9, color="#2ca02c")
                    ax.tick_params(labelsize=5)
                axes[r][0].set_ylabel(f"normal #{r}\n{flows[nj]['cid']}\nd={dist[oi][r-1]:.2f}",
                                      fontsize=6, color="#2ca02c")
            fig.suptitle(f"anomaly {af['cid']} (red) and its {len(near)} nearest NORMAL flows "
                         f"(green), each on its own axes", fontsize=10)
            fig.tight_layout(rect=[0, 0, 1, 0.97]); pdf.savefig(fig); plt.close(fig)
            rows.append({"anomaly_cid": af["cid"], "anomaly_score": float(scores[ai]),
                         "nearest_normal_cids": "; ".join(flows[nj]["cid"] for nj in near),
                         "nearest_dists": "; ".join(f"{d:.3f}" for d in dist[oi])})
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, "nearest_normal.csv"), index=False)
    _log(f"anomalies_vs_nearest.pdf: {len(order)} anomalies, {kk} nearest normals each "
         f"(one block per anomaly, flows drawn separately)")


def _plot_validation(out_dir, P, scores, anomaly, region, sig_ids):
    plt = _mpl()
    from matplotlib.backends.backend_pdf import PdfPages
    xy = P[:, :2]
    with PdfPages(os.path.join(out_dir, "validation.pdf")) as pdf:
        fig, ax = plt.subplots(1, 2, figsize=(13, 5))
        ax[0].hist(scores[~anomaly], bins=_safe_bins(scores[~anomaly], 50), color="#ccc",
                   alpha=0.7, density=True, label="normal")
        if anomaly.any():
            ax[0].hist(scores[anomaly], bins=_safe_bins(scores[anomaly], 30), color="#d62728",
                       alpha=0.7, density=True, label=f"anomaly (n={int(anomaly.sum())})")
        else:
            ax[0].text(0.5, 0.5, "0 anomalies flagged\n(loosen --alpha or use --threshold frac)",
                       transform=ax[0].transAxes, ha="center", va="center", fontsize=10, color="#d62728")
        ax[0].set_xlabel("neg log-density"); ax[0].set_ylabel("density"); ax[0].legend(fontsize=8)
        ax[0].set_title("score: anomalies sit in the right tail")
        sc2 = ax[1].scatter(xy[:, 0], xy[:, 1], s=6, c=scores, cmap="viridis")
        fig.colorbar(sc2, ax=ax[1], label="neg log-density")
        ax[1].set_title("density over the embedding"); ax[1].set_xlabel("PC1"); ax[1].set_ylabel("PC2")
        fig.tight_layout(); pdf.savefig(fig); plt.close(fig)
        # per-region embedding location
        if sig_ids:
            cmap = plt.get_cmap("tab20")
            fig, ax = plt.subplots(figsize=(8, 7))
            ax.scatter(xy[:, 0], xy[:, 1], s=4, c="#e4e4e4", zorder=1)
            for gi, r in enumerate(sig_ids):
                m = region == r
                ax.scatter(xy[m, 0], xy[m, 1], s=16, color=cmap(gi % 20), label=f"R{r}", zorder=3)
            ax.legend(fontsize=7); ax.set_title("anomaly regions"); ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
            fig.tight_layout(); pdf.savefig(fig); plt.close(fig)


def _summary(out_dir, args, n, scores, anomaly, sig_ids, bw, sc, meta_df):
    lines = [f"KDE anomaly detection (independent, single-sample density)",
             f"flows: {n:,}   reg={args.reg}   pca-dim={args.pca_dim}   bandwidth={bw:.4g}"
             + ("  (leave-one-out)" if args.leave_one_out else ""),
             f"anomalies: {int(anomaly.sum())} (top {args.anomaly_frac:.0%})   regions: {len(sig_ids)}",
             f"neg-log-density: median {np.median(scores):.3f}, p99 {np.percentile(scores,99):.3f}, "
             f"max {scores.max():.3f}", ""]
    top = sc.head(15)
    lines.append("top flows by anomaly score:")
    for r in top.itertuples():
        o = f"  [{getattr(r,'oui','')}]" if "oui" in sc.columns and pd.notna(getattr(r, "oui", None)) else ""
        lines.append(f"  {r.cid}{o}: {r.neg_log_density:.3f}")
    with open(os.path.join(out_dir, "summary.txt"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()