#!/usr/bin/env python3
"""
sweep_report.py

Run the four improvement experiments for the per-router device-clustering task in ONE
pass and render a single PDF with tables and plots. Reuses the clustering functions from
oui_cluster_experiment.py (same reduction, same e-means/HDBSCAN, same per-router scoring).

The four experiments, as a grid:
  1. representation : pass one cache per pool (e.g. the dur16 ctx AND reg caches) -> the
                      report compares them side by side.
  2. reduction      : PCA dim in --pca-dims, plus a UMAP arm per --umap-dims.
  3. k-select       : how e-means picks k -- silhouette (unsupervised) vs vmeasure/ami
                      (balanced oracle) vs oui (device-count oracle).
  4. variant        : pure (embedding only) vs scalars (embedding + mu/sigma magnitude).
HDBSCAN is added as a reference (it needs no k) so e-means is judged against it.

Every e-means grid point is run per router and aggregated (mean AMI/homogeneity/
completeness/coverage/estimated-k over routers), exactly like --per-router in the main
script. Results also go to a CSV next to the PDF.

USAGE
  # build the caches once (GPU), then sweep from them (CPU, no GPU):
  python src/sweep_report.py --caches emb_..._dur16_ctx.npz emb_..._dur16_reg.npz \
      --out report.pdf
  # trim the grid:
  python src/sweep_report.py --caches C.npz --pca-dims 16,32 --k-selects silhouette,oui
"""
import argparse
import itertools
import os
import time

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from oui_cluster_experiment import load_cache, evaluate_per_router_clustering


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def emeans_cfg(k_select, k_min, k_max, epochs):
    return {"clusterer": "emeans", "k": 0, "k_mult": 1.0, "k_select": k_select,
            "k_min": k_min, "k_max": k_max, "emeans_epochs": epochs, "emeans_spherical": None}


def build_grid(args):
    """Yield (label_dict, reduction_kwargs, clus_cfg) for every experiment cell."""
    pca_dims = [int(x) for x in args.pca_dims.split(",") if x]
    umap_dims = [int(x) for x in args.umap_dims.split(",") if x]
    variants = [v for v in args.variants.split(",") if v]
    k_selects = [k for k in args.k_selects.split(",") if k]
    reductions = [("pca", d, 0) for d in pca_dims] + [("umap", args.umap_before, d) for d in umap_dims]
    cells = []
    for variant, (rname, pca_dim, umap_dim) in itertools.product(variants, reductions):
        red = dict(pca_dim=pca_dim, umap_dim=umap_dim)
        reduction_label = f"pca{pca_dim}" if umap_dim == 0 else f"umap{umap_dim}"
        for k_select in k_selects:                            # e-means arms
            cells.append((dict(variant=variant, reduction=reduction_label,
                               clusterer="emeans", k_select=k_select), red,
                          emeans_cfg(k_select, args.k_min, args.k_max, args.emeans_epochs)))
        if args.include_hdbscan:                              # one HDBSCAN reference per reduction
            cells.append((dict(variant=variant, reduction=reduction_label,
                               clusterer="hdbscan", k_select="-"), red,
                          {"clusterer": "hdbscan"}))
    return cells


def sweep(args):
    rows = []
    for cache_path in args.caches:
        cache = load_cache(cache_path)
        pool = cache.get("pool", "?")
        label = os.path.splitext(os.path.basename(cache_path))[0]
        log(f"cache {label} (pool={pool}): {cache['emb'].shape[0]:,} flows, "
            f"{len(set(cache['oui']))} OUIs, {len(set(cache['router']))} routers")
        cells = build_grid(args)
        for i, (lab, red, clus_cfg) in enumerate(cells, 1):
            variant = lab["variant"]
            _, agg = evaluate_per_router_clustering(
                cache, variant, red["pca_dim"], red["umap_dim"], args.umap_neighbors,
                args.umap_min_dist, args.min_cluster_size, args.min_samples, args.selection,
                args.seed, None, True, args.min_oui_per_router, clus_cfg)
            row = {"cache": label, "pool": pool, **lab}
            row.update({k: agg.get(k) for k in ("n_routers", "mean_AMI", "std_AMI", "mean_ARI",
                        "mean_homogeneity", "mean_completeness", "mean_coverage",
                        "mean_n_clusters", "mean_k_abs_error")})
            rows.append(row)
            log(f"  [{i}/{len(cells)}] {label} {variant}/{lab['reduction']}/"
                f"{lab['clusterer']}/{lab['k_select']}: AMI="
                f"{(agg.get('mean_AMI') or float('nan')):.3f}")
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ #
# PDF report                                                         #
# ------------------------------------------------------------------ #
def _table_page(pdf, df, title):
    show = df.copy()
    for c in ("mean_AMI", "std_AMI", "mean_ARI", "mean_homogeneity", "mean_completeness",
              "mean_coverage", "mean_n_clusters", "mean_k_abs_error"):
        if c in show:
            show[c] = show[c].map(lambda v: "" if pd.isna(v) else f"{v:.3f}")
    cols = ["cache", "variant", "reduction", "clusterer", "k_select", "mean_AMI", "std_AMI",
            "mean_ARI", "mean_homogeneity", "mean_completeness", "mean_coverage",
            "mean_n_clusters"]
    cols = [c for c in cols if c in show.columns]
    rows_per_page = 28
    for start in range(0, len(show), rows_per_page):
        chunk = show.iloc[start:start + rows_per_page][cols]
        fig, ax = plt.subplots(figsize=(11.7, 8.3))          # A4 landscape
        ax.axis("off")
        ax.set_title(title + (f"  (rows {start+1}-{start+len(chunk)})" if len(show) > rows_per_page else ""),
                     fontsize=12, pad=14)
        t = ax.table(cellText=chunk.values, colLabels=cols, loc="center", cellLoc="center")
        t.auto_set_font_size(False); t.set_fontsize(7); t.scale(1, 1.3)
        pdf.savefig(fig); plt.close(fig)


def _bar(pdf, title, labels, values, errs=None, ylabel="mean AMI", rotate=60):
    fig, ax = plt.subplots(figsize=(11.7, 6))
    x = np.arange(len(labels))
    ax.bar(x, values, yerr=errs, capsize=3, color="#4C72B0")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=rotate, ha="right", fontsize=7)
    ax.set_ylabel(ylabel); ax.set_title(title, fontsize=12)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); pdf.savefig(fig); plt.close(fig)


def _grouped_bar(pdf, title, df, group_col, series_col, value_col="mean_AMI", ylabel="mean AMI"):
    groups = sorted(df[group_col].dropna().unique())
    series = sorted(df[series_col].dropna().unique())
    fig, ax = plt.subplots(figsize=(11.7, 6))
    w = 0.8 / max(1, len(series))
    for i, s in enumerate(series):
        vals = [df[(df[group_col] == g) & (df[series_col] == s)][value_col].max() for g in groups]
        ax.bar(np.arange(len(groups)) + i * w, vals, w, label=str(s))
    ax.set_xticks(np.arange(len(groups)) + w * (len(series) - 1) / 2)
    ax.set_xticklabels(groups, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel(ylabel); ax.set_title(title, fontsize=12); ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); pdf.savefig(fig); plt.close(fig)


def make_report(df, out_pdf, args):
    with PdfPages(out_pdf) as pdf:
        # cover
        fig = plt.figure(figsize=(11.7, 8.3)); fig.clf()
        best = df.loc[df["mean_AMI"].idxmax()] if df["mean_AMI"].notna().any() else None
        txt = ["Per-router device clustering -- experiment sweep",
               time.strftime("%Y-%m-%d %H:%M"), "",
               f"caches: {', '.join(os.path.basename(c) for c in args.caches)}",
               f"variants: {args.variants}", f"pca-dims: {args.pca_dims}  umap-dims: {args.umap_dims}",
               f"k-selects: {args.k_selects}", f"configs evaluated: {len(df)}", ""]
        if best is not None:
            txt += [f"best: AMI={best['mean_AMI']:.3f}  ({best['cache']} / {best['variant']} / "
                    f"{best['reduction']} / {best['clusterer']} / k={best['k_select']})"]
        fig.text(0.07, 0.92, "\n".join(txt), va="top", fontsize=12, family="monospace")
        pdf.savefig(fig); plt.close(fig)

        # full results table, sorted by AMI
        ranked = df.sort_values("mean_AMI", ascending=False).reset_index(drop=True)
        _table_page(pdf, ranked, "All configurations (ranked by mean AMI)")

        # top configs bar
        top = ranked.head(min(20, len(ranked)))
        lab = [f"{r.cache[:10]}/{r.variant}/{r.reduction}/{r.clusterer}/{r.k_select}"
               for r in top.itertuples()]
        _bar(pdf, "Top configurations by mean AMI", lab, top["mean_AMI"].values,
             top["std_AMI"].values)

        em = df[df["clusterer"] == "emeans"]
        # exp 1: representation (cache/pool)
        if df["cache"].nunique() > 1:
            _grouped_bar(pdf, "Exp 1 - representation: best AMI per cache x variant",
                         df, "cache", "variant")
        # exp 2: reduction
        _grouped_bar(pdf, "Exp 2 - reduction: best AMI per reduction x variant",
                     df, "reduction", "variant")
        # exp 3: k-select (e-means only)
        if not em.empty:
            _grouped_bar(pdf, "Exp 3 - k-select: best AMI per k-select x cache (e-means)",
                         em, "k_select", "cache")
        # exp 4: pure vs scalars
        _grouped_bar(pdf, "Exp 4 - variant: best AMI per variant x cache", df, "variant", "cache")

        # homogeneity vs completeness tradeoff (all configs)
        fig, ax = plt.subplots(figsize=(8.3, 8.3))
        for clus, mk in (("emeans", "o"), ("hdbscan", "s")):
            sub = df[df["clusterer"] == clus]
            if not sub.empty:
                ax.scatter(sub["mean_completeness"], sub["mean_homogeneity"],
                           label=clus, marker=mk, alpha=0.7)
        ax.set_xlabel("mean completeness"); ax.set_ylabel("mean homogeneity")
        ax.set_title("Fragmentation trade-off (upper-right is best)"); ax.legend()
        ax.grid(alpha=0.3); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        fig.tight_layout(); pdf.savefig(fig); plt.close(fig)
    log(f"wrote {out_pdf}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--caches", nargs="+", required=True, help="one or more embedding caches (.npz)")
    ap.add_argument("--out", default="sweep_report.pdf")
    ap.add_argument("--variants", default="pure,scalars")
    ap.add_argument("--pca-dims", default="16,32,64")
    ap.add_argument("--umap-dims", default="10", help="UMAP target dims (comma); '' to skip UMAP")
    ap.add_argument("--umap-before", type=int, default=64, help="PCA dim before UMAP")
    ap.add_argument("--k-selects", default="silhouette,vmeasure,oui")
    ap.add_argument("--include-hdbscan", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--k-min", type=int, default=2)
    ap.add_argument("--k-max", type=int, default=12)
    ap.add_argument("--emeans-epochs", type=int, default=5)
    ap.add_argument("--umap-neighbors", type=int, default=30)
    ap.add_argument("--umap-min-dist", type=float, default=0.0)
    ap.add_argument("--min-cluster-size", type=int, default=25)
    ap.add_argument("--min-samples", type=int, default=0)
    ap.add_argument("--selection", default="eom")
    ap.add_argument("--min-oui-per-router", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    df = sweep(args)
    if df.empty:
        raise SystemExit("no results produced")
    csv = os.path.splitext(args.out)[0] + ".csv"
    df.to_csv(csv, index=False); log(f"wrote {csv}")
    make_report(df, args.out, args)


if __name__ == "__main__":
    main()
    import sys
    sys.stdout.flush(); os._exit(0)