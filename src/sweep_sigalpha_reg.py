#!/usr/bin/env python3
"""
sweep_sigalpha_reg.py  (src/)

Sweep over (reg x sig-alpha) and draw ALL clustering views in ONE grid PNG.
  rows    = reg representations: each channel separately (target = that channel's REG)
  columns = sig-alpha thresholds (0.01, 0.025, 0.05, 0.1, 0.15, 0.2 by default)

Key efficiency: embedding, KL contributions, clustering and region p-values depend
only on `reg`, NOT on sig-alpha -- sig-alpha only decides which regions count as
SIGNIFICANT. So we compute the heavy part ONCE per channel, then for each sig-alpha
just recolour which regions pass. The filtering (ground truth) is computed once and
reused for every reg (it is independent of reg).

USAGE
    python src/sweep_sigalpha_reg.py --only 2eadb48f --device cuda \
        --min-bins 48 --min-packets 10 --ref-frac 0.7 --smooth-k 5 \
        --filter-method method5 --anomaly-frac 0.02 \
        --channels packet_count,fwd_mean_iat,fwd_mean_packet_size,bwd_mean_iat,bwd_mean_packet_size

Output: results/anomaly_kl/sweep_<tag>_<filter>_seed<seed>/sweep_sigalpha_reg.png
        (+ a small CSV of #significant-regions per cell)
"""
import argparse
import os
import numpy as np
import pandas as pd

import importlib.util
_spec = importlib.util.spec_from_file_location("anomaly_kl",
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "anomaly_kl.py"))
AK = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(AK)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default=None)
    ap.add_argument("--only", default=None)
    ap.add_argument("--features-dir", default=None)
    ap.add_argument("--root", default=AK.DEFAULT_ROOT)
    ap.add_argument("--bin", default="5min")
    ap.add_argument("--channels", default="packet_count,fwd_mean_iat,fwd_mean_packet_size,"
                                          "bwd_mean_iat,bwd_mean_packet_size")
    ap.add_argument("--min-bins", type=int, default=48)
    ap.add_argument("--min-packets", type=int, default=10)
    ap.add_argument("--ref-frac", type=float, default=0.7)
    ap.add_argument("--holdout", type=int, default=24)
    ap.add_argument("--band-lo", type=float, default=0.1)
    ap.add_argument("--band-hi", type=float, default=0.9)
    ap.add_argument("--norm-thr", type=float, default=0.1)
    ap.add_argument("--anomaly-frac", type=float, default=0.02)
    ap.add_argument("--filter-method", choices=["method5", "lof"], default="method5")
    ap.add_argument("--lof-neighbors", type=int, default=20)
    ap.add_argument("--pca-dim", type=int, default=64)
    ap.add_argument("--perms", type=int, default=20)
    ap.add_argument("--alpha", type=float, default=0.01)
    ap.add_argument("--smooth-k", type=int, default=5)
    ap.add_argument("--sig-alphas", default="0.01,0.025,0.05,0.1,0.15,0.2")
    ap.add_argument("--embed-batch", type=int, default=64)
    ap.add_argument("--embed-cap", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--max-flows", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-filter-cache", action="store_true")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    channels = [c.strip() for c in args.channels.split(",") if c.strip()]
    sig_alphas = [float(x) for x in args.sig_alphas.replace(" ", "").split(",") if x]
    pkt_col = "packet_count"

    # resolve features file
    import glob
    if args.csv:
        path = args.csv
    else:
        fdir = args.features_dir or os.path.join(args.root, "data", "features")
        files = glob.glob(os.path.join(fdir, "*.parquet"))
        parts_here = glob.glob(os.path.join(fdir, "[0-9][0-9][0-9][0-9][0-9][0-9].parquet"))
        if parts_here and len(parts_here) == len(files):
            cand = [fdir]          # --features-dir IS a directory of parts: read all of them
        else:
            cand = sorted(files) + sorted(d for d in glob.glob(os.path.join(fdir, "*.parts"))
                                          if os.path.isdir(d))
        if args.only:
            cand = [p for p in cand if os.path.basename(p).startswith(args.only)
                    or args.only in os.path.basename(p)]
        if not cand:
            print("no features file found"); return
        path = cand[0]
    AK._log(f"features: {path}")
    AK._log(f"channels (rows): {channels}")
    AK._log(f"sig-alphas (cols): {sig_alphas}")

    # ---- load flows + split ----
    flows = AK.load_flows(path, channels, args.bin, args.min_bins, args.min_packets,
                          pkt_col, max_flows=args.max_flows, seed=args.seed)
    AK._log(f"kept {len(flows)} flows")
    if len(flows) < 20:
        print("too few flows."); return
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(flows))
    nB = int(round(args.ref_frac * len(flows)))
    idxB = set(order[:nB].tolist())
    setB = [flows[i] for i in range(len(flows)) if i in idxB]
    setA = [flows[i] for i in range(len(flows)) if i not in idxB]
    AK._log(f"split: B={len(setB)} A={len(setA)}")

    ch = AK.Chronos(device=args.device)

    # ---- filter B ONCE (independent of reg) -> marked anomalies, moved to A ----
    scores = _filter_scores(ch, setB, channels, args, path)
    n_mark = int(round(args.anomaly_frac * len(setB)))
    mk = set(int(i) for i in np.argsort(-scores)[:n_mark] if scores[i] > 0)
    movedA = [dict(setB[k], marked=True) for k in range(len(setB)) if k in mk]
    B = [dict(setB[k], marked=False) for k in range(len(setB)) if k not in mk]
    for f in setA:
        f["marked"] = False
    A = setA + movedA
    marked = np.array([f["marked"] for f in A], dtype=bool)
    AK._log(f"filter={args.filter_method}: marked {len(movedA)}; A={len(A)} B={len(B)}")
    if len(B) < 10 or len(A) < 10:
        print("a set too small after filtering."); return

    # ---- embed per channel ONCE each; compute regions + p-values per reg ----
    framesA = [f["frame"] for f in A]; framesB = [f["frame"] for f in B]
    per_reg = {}                    # channel -> dict(region, D, reg_df)
    for c, chn in enumerate(channels):
        AK._log(f"[row {c}] reg={chn}: embedding + KL ...")
        eA = ch.embed(framesA, channels, args.embed_batch, cap=args.embed_cap, reg_mode=c)
        eB = ch.embed(framesB, channels, args.embed_batch, cap=args.embed_cap, reg_mode=c)
        P = AK.project(np.vstack([eA, eB]), args.pca_dim)
        XA, XB = P[:len(eA)], P[len(eA):]
        D = AK.kl_contributions(XA, XB, smooth_k=args.smooth_k)
        D_raw = AK.kl_contributions(XA, XB, smooth_k=0)
        dkl = float(D_raw.sum())
        thr, nrm = AK.permutation_null(XA, XB, args.perms, args.alpha,
                                       np.random.default_rng(args.seed + c), smooth_k=args.smooth_k)
        survivors = D > thr
        region = AK.cluster_regions(XA, D, survivors)
        reg_df = AK.rank_regions(region, D, D_raw, dkl, nrm, marked=marked, XA=XA)
        per_reg[chn] = dict(XA=XA, D=D, region=region, reg_df=reg_df)
        AK._log(f"[row {c}] regions={len(reg_df)} "
                f"(sig@0.05={int((reg_df['p_value']<=0.05).sum()) if len(reg_df) else 0})")

    # ---- draw the grid: rows = reg (channel), cols = sig-alpha ----
    out_dir = args.out_dir or _sweep_dir(args, path, channels)
    os.makedirs(out_dir, exist_ok=True)
    _draw_grid(out_dir, per_reg, channels, sig_alphas, marked)
    _write_counts(out_dir, per_reg, channels, sig_alphas)
    AK._log(f"wrote {out_dir}/sweep_sigalpha_reg.png and counts.csv")


def _filter_scores(ch, setB, channels, args, path):
    """Return the per-B-flow anomaly score from the chosen filter (cached)."""
    fkey = AK._filter_cache_key(args, path, channels)
    cache = AK._load_filter_cache(args, fkey) if not args.no_filter_cache else None
    if cache is not None:
        AK._log(f"filter CACHE HIT ({fkey[:12]})")
        return np.array([cache["score"].get(f["cid"], 0.0) for f in setB], dtype=np.float64)
    if args.filter_method == "lof":
        feats = np.vstack([AK._flow_stats(f["frame"]) for f in setB])
        scores = AK._lof_scores(feats, n_neighbors=args.lof_neighbors)
        out = [[0] * len(channels) for _ in setB]
    else:
        lo, med, hi = ch.forecast_band([f["frame"] for f in setB], channels,
                                       args.holdout, args.band_lo, args.band_hi, args.batch_size)
        scores = np.zeros(len(setB)); out = []
        for k, f in enumerate(setB):
            fr = f["frame"]; T = fr.shape[0]; actual = fr[T - args.holdout:, :]
            yr = fr.max(0) - fr.min(0)
            cc = np.zeros(fr.shape[1], bool)
            for ci in range(fr.shape[1]):
                nz = fr[:, ci][np.abs(fr[:, ci]) > 1e-6]
                if nz.size == 0 or (nz.max() - nz.min()) <= 1e-6:
                    cc[ci] = True
            diff = np.abs(actual - med[k]); norm = diff / (yr[None, :] + 1e-5)
            m0 = (actual > hi[k] + 1e-2) | (actual < lo[k] - 1e-2)
            valid = m0 & (diff >= 1e-2) & (~cc[None, :])
            scores[k] = float(np.where(valid, np.maximum(0.0, norm - args.norm_thr), 0.0).sum())
            out.append((valid & (norm > args.norm_thr)).sum(0).tolist())
    if not args.no_filter_cache:
        AK._save_filter_cache(args, fkey, setB, scores, out)
    return scores


def _draw_grid(out_dir, per_reg, channels, sig_alphas, marked):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    R, Cc = len(channels), len(sig_alphas)
    fig, axes = plt.subplots(R, Cc, figsize=(3.2 * Cc, 3.0 * R), squeeze=False)
    cmap = plt.get_cmap("tab10")
    for r, chn in enumerate(channels):
        pr = per_reg[chn]; XA = pr["XA"]; region = pr["region"]; reg_df = pr["reg_df"]
        xy = XA[:, :2]
        for cc, sa in enumerate(sig_alphas):
            ax = axes[r][cc]
            sig_ids = set(reg_df.loc[reg_df["p_value"] <= sa, "region_id"].tolist()) \
                if len(reg_df) else set()
            in_sig = np.isin(region, list(sig_ids)) if sig_ids else np.zeros(len(region), bool)
            ax.scatter(xy[~in_sig, 0], xy[~in_sig, 1], s=3, c="#e6e6e6", zorder=1)
            for gi, rid in enumerate(sorted(sig_ids, key=lambda x: -(region == x).sum())):
                m = region == rid
                ax.scatter(xy[m, 0], xy[m, 1], s=10, c=[cmap(gi % 10)],
                           edgecolors="none", zorder=3)
            ax.set_xticks([]); ax.set_yticks([])
            n_anom = int(in_sig.sum())
            ax.set_title(f"{len(sig_ids)} reg / {n_anom} anom", fontsize=7)
            if r == 0:
                ax.annotate(f"sig-alpha = {sa}", xy=(0.5, 1.28), xycoords="axes fraction",
                            ha="center", fontsize=10, weight="bold")
            if cc == 0:
                ax.set_ylabel(chn, fontsize=9, weight="bold")
    fig.suptitle("Clustering across reg (rows) x sig-alpha (cols) -- coloured = significant regions",
                 fontsize=13, y=0.998)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(os.path.join(out_dir, "sweep_sigalpha_reg.png"), dpi=130)
    plt.close(fig)


def _write_counts(out_dir, per_reg, channels, sig_alphas):
    rows = []
    for chn in channels:
        reg_df = per_reg[chn]["reg_df"]
        row = {"reg": chn}
        for sa in sig_alphas:
            n = int((reg_df["p_value"] <= sa).sum()) if len(reg_df) else 0
            row[f"sig@{sa}"] = n
        rows.append(row)
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, "counts.csv"), index=False)


def _sweep_dir(args, path, channels):
    fname = os.path.splitext(os.path.basename(path))[0]
    tag = ("host_" + fname[len("host_"):][:4]) if fname.startswith("host_") else fname[:4]
    base = os.path.join(args.root, "results", "anomaly_kl")
    d = os.path.join(base, f"sweep_{tag}_filt-{args.filter_method}"
                     f"_af{int(args.anomaly_frac*100)}_seed{args.seed}")
    run = 1
    while os.path.exists(f"{d}_run{run}"):
        run += 1
    return f"{d}_run{run}"


if __name__ == "__main__":
    main()