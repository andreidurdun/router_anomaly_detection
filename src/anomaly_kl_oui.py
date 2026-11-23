#!/usr/bin/env python3
"""
anomaly_kl_oui.py  (lives in  Anomaly_detection/src/, next to anomaly_kl.py)

PER-DEVICE-TYPE (OUI) anomaly study: do the anomalies of a device type differ per router?

For EVERY OUI (device type = first 3 bytes of the MAC), independently:
  1. pool ALL its flows from ALL routers;
  2. METHOD 5 (Chronos forecast of the last --holdout bins, all channels given to Chronos)
     scores every flow; the top --anomaly-frac of the OUI (score > 0) are MARKED;
  3. random split of the pool (--ref-frac) as in the original script:
        B (reference) = unmarked flows of the reference part
        A (target)    = the rest of the pool + ALL marked flows
     With --split-type random there is NO Method-5 forecast: the pool is split into
     --n-splits folds and the experiment runs once per fold (fold = A, the other folds
     = B), writing split1..splitN/ subfolders; Chronos is used ONLY for the embedding.
  4. Chronos-2 REG embedding -> PCA on A u B -> local KL contribution D_i ->
     permutation null -> HDBSCAN regions -> cluster-mass p-values;
  5. VALIDATION: AUROC / precision / recall of D_i vs the Method-5 marks;
  6. ROUTER BREAKDOWN: for every router of the OUI, how many flows are Method-5 marked,
     how many fall in significant KL regions, Fisher test (router vs the other routers),
     chi-square test of homogeneity across routers, router composition of every region.

Reuses the building blocks of anomaly_kl_mac.py (load_flows, Chronos, project,
kl_contributions, permutation_null, cluster_regions, rank_regions, validate,
method5_scores, normalize_oui, _is_local_oui, _bh, _reg_tag).

OUTPUTS (results/anomaly_kl/<run>/)
    oui_pdfs/<OUI>.pdf    ONE PDF PER OUI: validation page, embedding maps (KL regions / Method 5 /
                          by router), router breakdown bars, then the plain TIME SERIES of the
                          chosen channel (default packet_count), ONE PANEL PER ROUTER, no markers
    oui_summary.csv       one row per OUI
    router_breakdown.csv  one row per (OUI, router)
    regions_all.csv       every region of every OUI, with its router composition
    scores_all.csv        every flow: OUI, router, set A/B, marked, D_i, region ...
    oui_router_heatmap.png  Method-5 marked fraction and KL-flagged fraction per OUI x router
    summary.txt

EXAMPLE
    python src/anomaly_kl_oui.py --only <hash> --device cuda --bin 1min \
        --min-minutes 60 --holdout 30 --context-bins 30 \
        --channels packet_count,mean_iat,mean_packet_size --router-regex '^([^_]+)_'
"""
import argparse
import glob
import os
import re
import sys
import time
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import anomaly_kl_mac as K                       # noqa: E402
from anomaly_kl_mac import _log                  # noqa: E402

ORANGE, RED, GREY = "#ff9f1c", "#d62728", "#7f8c9a"
RANDOM_KEY = "RANDOMIZED"
KNOWN_DIR, RANDOM_DIR = "known_manufacturer", "randomized_mac"

def _mac_type(o):
    return "randomized" if (str(o).startswith("RND_") or K._is_local_oui(o)) else "known"


# --------------------------------------------------------------------------- #
# Grouping                                                                     #
# --------------------------------------------------------------------------- #
def group_flows(flows, args, rng):
    """-> {oui: {router: [flow idx]}} after dropping unusable / random-MAC flows,
    --only-oui, --min-oui-flows, --min-routers and --max-oui-flows."""
    by = defaultdict(lambda: defaultdict(list))
    n_miss = n_rand = 0
    for i, f in enumerate(flows):
        o, r = f.get("oui"), f.get("router")
        if not o or r is None or str(r) == "":
            n_miss += 1
            continue
        if K._is_local_oui(o):
            n_rand += 1
            if args.drop_random_mac:
                continue
            if args.random_group_by == "mac":          # one group per full MAC
                m = re.sub(r"[^0-9A-Fa-f]", "", str(f.get("mac") or "")).upper()
                if len(m) >= 12:
                    o = "RND_" + m[:12]
        by[o][str(r)].append(i)
    _log(f"flows without OUI or router: {n_miss}/{len(flows)}; randomised MACs: {n_rand} "
         f"({'dropped' if args.drop_random_mac else 'grouped by ' + args.random_group_by})")
    if args.only_oui:
        keep = {x.strip().upper() if x.strip().upper().startswith("RND_") else K.normalize_oui(x)
                for x in args.only_oui.split(",") if x.strip()}
        by = {o: d for o, d in by.items() if o in keep}
    out, small = {}, 0
    for o, d in by.items():
        n = sum(len(v) for v in d.values())
        min_n = args.min_random_flows if _mac_type(o) == "randomized" else args.min_oui_flows
        if n < min_n or len(d) < args.min_routers:
            small += 1
            continue
        if args.max_oui_flows and n > args.max_oui_flows:       # random cap, keeps router mix
            allx = [i for v in d.values() for i in v]
            keep = set(rng.choice(allx, args.max_oui_flows, replace=False).tolist())
            d = {r: [i for i in v if i in keep] for r, v in d.items()}
            d = {r: v for r, v in d.items() if v}
        out[o] = dict(d)
    _log(f"OUIs: {len(by)}; analysed: {len(out)} (skipped {small} with < {args.min_oui_flows} flows "f"or < {args.min_routers} routers)")
    return out


# --------------------------------------------------------------------------- #
# KL analysis of one OUI                                                       #
# --------------------------------------------------------------------------- #
def analyze_oui(EA, EB, markedA, args, rng):
    P = K.project(np.vstack([EA, EB]), args.pca_dim)
    XA, XB = P[:len(EA)], P[len(EA):]
    D = K.kl_contributions(XA, XB, smooth_k=args.smooth_k)
    D_raw = K.kl_contributions(XA, XB, smooth_k=0)
    dkl = float(D_raw.sum())
    thr, null_max = K.permutation_null(XA, XB, args.perms, args.alpha, rng, smooth_k=args.smooth_k)
    surv = D > thr
    region = K.cluster_regions(XA, D, surv)
    reg_df = K.rank_regions(region, D, D_raw, dkl, null_max, marked=markedA, XA=XA)
    if len(reg_df):
        sig_ids = set(reg_df.loc[reg_df["p_value"] <= args.sig_alpha, "region_id"].tolist())
        top_p = float(reg_df["p_value"].min())
    else:
        sig_ids, top_p = set(), 1.0
    sig = np.isin(region, list(sig_ids)) if sig_ids else np.zeros(len(XA), dtype=bool)
    val = K.validate(markedA, D, sig)
    return {"D": D, "dkl": dkl, "thr": float(thr), "survivors": surv, "region": region,
            "reg_df": reg_df, "sig_ids": sig_ids, "sig": sig, "top_p": top_p, "val": val,
            "xyA": XA[:, :2], "xyB": XB[:, :2]}


def _chi2_p(rows):
    t = np.array(rows, dtype=float)
    t = t[t.sum(1) > 0]
    if len(t) < 2 or t[:, 0].sum() == 0 or t[:, 1].sum() == 0:
        return float("nan")
    try:
        from scipy.stats import chi2_contingency
        return float(chi2_contingency(t)[1])
    except Exception:
        return float("nan")


def _fisher_p(k1, n1, k2, n2):
    if n1 == 0 or n2 == 0:
        return float("nan")
    try:
        from scipy.stats import fisher_exact
        return float(fisher_exact([[k1, n1 - k1], [k2, n2 - k2]])[1])
    except Exception:
        return float("nan")


# --------------------------------------------------------------------------- #
# Plots                                                                        #
# --------------------------------------------------------------------------- #
def _router_colors(routers):
    import matplotlib.pyplot as plt
    cm = plt.get_cmap("tab20")
    return {r: cm(k % 20) for k, r in enumerate(sorted(routers))}


def _edges(values, nbins):
    """Histogram bin edges that are always valid, even if all values are (nearly) identical."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return None
    lo, hi = float(v.min()), float(v.max())
    pad = max(1e-6, 1e-6 * max(abs(lo), abs(hi)))
    if hi - lo < pad:                       # degenerate range -> widen it
        lo, hi = lo - pad, hi + pad
    return np.linspace(lo, hi, nbins + 1)


def _hist(ax, x, edges, **kw):
    """Histogram on pre-computed edges; silently skips empty input."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0 or edges is None:
        return
    ax.hist(x, bins=edges, **kw)


def _page_validation(pdf, plt, o, res, markedA, header):
    D, sig = np.asarray(res["D"], dtype=float), np.asarray(res["sig"], dtype=bool)
    fin = np.isfinite(D)
    D, sig = D[fin], sig[fin]
    edges = _edges(D, 40)
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    if sig.any():
        _hist(ax, D[~sig], edges, alpha=0.55, color="#888", density=True, label="not in a sig. region")
        _hist(ax, D[sig], edges, alpha=0.8, color="#2ca02c", density=True,
              label=f"significant region (n={int(sig.sum())})")
        ax.set_title("D_i: significant regions vs rest"); ax.legend(); ax.set_xlabel("D_i")
    else:
        ax.text(0.5, 0.5, "no significant regions", ha="center", va="center", transform=ax.transAxes)
        ax.axis("off")
    fig.suptitle(header, fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.90]); pdf.savefig(fig); plt.close(fig)


def _page_maps(pdf, plt, o, res, top10_non_anom):
    xy, xyB = res["xyA"], res["xyB"]
    reg_df, region = res["reg_df"], res["region"]
    surv, sig_ids = res["survivors"], res["sig_ids"]
    pmap = dict(zip(reg_df["region_id"], reg_df["p_value"])) if len(reg_df) else {}
    sig_mask = np.isin(region, list(sig_ids)) if sig_ids else np.zeros(len(xy), dtype=bool)
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))

    # (a) significant KL regions
    a = ax[0]
    a.scatter(xyB[:, 0], xyB[:, 1], s=5, c="#e6e6e6", zorder=1, label="B (reference)")
    a.scatter(xy[~surv, 0], xy[~surv, 1], s=8, c="#cccccc", zorder=2, label="A not anomalous")
    non = surv & ~sig_mask
    if non.any():
        a.scatter(xy[non, 0], xy[non, 1], s=12, c="#999999", alpha=0.6, zorder=3,
                  label=f"survivors non-sig (n={int(non.sum())})")
    cm = plt.get_cmap("tab10")
    for gi, r in enumerate(sorted(sig_ids)):
        m = region == r
        a.scatter(xy[m, 0], xy[m, 1], s=40, c=[cm(gi % 10)], edgecolors="k", linewidths=0.5,
                  zorder=5, label=f"region {r} (n={int(m.sum())}, p={pmap.get(r, float('nan')):.3f})")
    a.set_title(f"KL: {len(sig_ids)} significant region(s)"); a.legend(fontsize=6, loc="best")

    # (b) the 10 highest-D_i flows that are NOT in a significant region
    b = ax[1]
    top = np.asarray(top10_non_anom, dtype=int)
    b.scatter(xyB[:, 0], xyB[:, 1], s=5, c="#e6e6e6", zorder=1, label="B (reference)")
    b.scatter(xy[:, 0], xy[:, 1], s=8, c="#cccccc", zorder=2, label="A")
    if sig_mask.any():
        b.scatter(xy[sig_mask, 0], xy[sig_mask, 1], s=14, c="#2ca02c", alpha=0.5, zorder=3,
                  label="declared anomalous (sig. regions)")
    if len(top):
        sc = b.scatter(xy[top, 0], xy[top, 1], s=70, c=res["D"][top], cmap="autumn_r",
                       edgecolors="k", linewidths=0.8, zorder=5, label="top-10 D_i (not anomalous)")
        for rank, j in enumerate(top, 1):
            b.annotate(str(rank), (xy[j, 0], xy[j, 1]), fontsize=7, xytext=(3, 3),
                       textcoords="offset points", zorder=6)
        fig.colorbar(sc, ax=b, label="D_i")
    b.set_title("10 highest D_i flows after the declared anomalies"); b.legend(fontsize=6, loc="best")

    for x in ax:
        x.set_xlabel("PC1"); x.set_ylabel("PC2")
    fig.suptitle(f"OUI {o}: embedding space (PCA 2-D)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95]); pdf.savefig(fig); plt.close(fig)


def _page_router_bars(pdf, plt, o, rstats):
    df = pd.DataFrame(rstats).sort_values("n_flows", ascending=False)
    x = np.arange(len(df)); w = 0.38
    fig, ax = plt.subplots(1, 2, figsize=(14, 4.8))
    ax[0].bar(x, df["n_flows"], color=GREY)
    ax[0].set_xticks(x); ax[0].set_xticklabels(df["router"], rotation=45, ha="right", fontsize=7)
    ax[0].set_title("flows of this OUI per router"); ax[0].set_ylabel("flows")
    ax[1].bar(x - w / 2, df["frac_m5_marked"], w, color=RED, label="Method-5 marked")
    ax[1].bar(x + w / 2, df["frac_kl_flagged"], w, color=ORANGE, label="KL significant regions")
    ax[1].set_xticks(x); ax[1].set_xticklabels(df["router"], rotation=45, ha="right", fontsize=7)
    ax[1].set_title("fraction of the router's flows flagged"); ax[1].legend(fontsize=8)
    for xi, (_, r) in zip(x, df.iterrows()):
        if not np.isnan(r["fisher_p_m5"]) and r["fisher_p_m5"] <= 0.05:
            ax[1].text(xi - w / 2, r["frac_m5_marked"], "*", ha="center", va="bottom", color=RED)
        if not np.isnan(r["fisher_p_kl"]) and r["fisher_p_kl"] <= 0.05:
            ax[1].text(xi + w / 2, r["frac_kl_flagged"], "*", ha="center", va="bottom", color=ORANGE)
    fig.suptitle(f"OUI {o}: router breakdown (* = Fisher p<=0.05, router vs the other routers)", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.94]); pdf.savefig(fig); plt.close(fig)


def _ts_sum(ax, idx, flows, ci, delta_ns):
    """Plain line: the channel summed over all flows of the router at each timestamp.
    The line is broken where there is no data (no interpolation across gaps)."""
    ts, v = [], []
    for i in idx:
        fl = flows[i]; fr = fl["frame"]; T = fr.shape[0]
        ts.append(fl["t0"].value + np.arange(T, dtype=np.int64) * delta_ns)
        v.append(fr[:, ci].astype(np.float64))
    g = pd.DataFrame({"ts": np.concatenate(ts), "v": np.concatenate(v)}).groupby("ts")["v"].sum().sort_index()
    x = g.index.to_numpy().astype(np.int64)
    y = g.to_numpy(dtype=np.float64)
    gap = np.where(np.diff(x) > 1.5 * delta_ns)[0] + 1
    if len(gap):
        x = np.insert(x, gap, x[gap - 1] + delta_ns)
        y = np.insert(y, gap, np.nan)
    ax.plot(pd.to_datetime(x), y, lw=0.8, color=GREY)


def _ts_flows(ax, idx, flows, ci, delta_ns, max_flows):
    """Plain lines: one thin line per flow (first max_flows flows of the router)."""
    for i in list(idx)[:max_flows]:
        fl = flows[i]; fr = fl["frame"]; T = fr.shape[0]
        x = pd.to_datetime(fl["t0"].value + np.arange(T, dtype=np.int64) * delta_ns)
        ax.plot(x, fr[:, ci], lw=0.7, color=GREY, alpha=0.6)

def _pages_flow_matrix(pdf, plt, o, flows, args, res, A, idxs, channels, kind):
    """Pages of per-flow panels (rows = flows, cols = channels).
    idxs : positions inside A.   kind: 'anom' (red, labelled by region) or 'top10' (black)."""
    delta_ns = int(pd.Timedelta(pd.tseries.frequencies.to_offset(args.bin)).value)

    targets = []
    for j in idxs:
        j = int(j)
        tag = (f"ANOMALY (Reg {res['region'][j]})" if kind == "anom"
               else f"TOP-10 NON-ANOM (rank {len(targets) + 1})")
        targets.append((A[j], res["D"][j], tag))
    if not targets:
        return False

    C, per_page = len(channels), 10
    color = "#d62728" if kind == "anom" else "#000000"
    title = ("Anomalous flows (significant KL regions)" if kind == "anom"
             else "Top-10 D_i flows outside significant regions")

    for p0 in range(0, len(targets), per_page):
        chunk = targets[p0:p0 + per_page]
        fig, axes = plt.subplots(len(chunk), C, figsize=(3.8 * C, 2.2 * len(chunk)), squeeze=False)
        for ri, (flow_idx, d_i, tag) in enumerate(chunk):
            f = flows[flow_idx]
            T = f["frame"].shape[0]
            x = (np.arange(T) * delta_ns + f["t0"].value) / 1e9 / 3600
            for c_idx, chn in enumerate(channels):
                ax = axes[ri, c_idx]
                ax.plot(x, f["frame"][:, c_idx], color=color, lw=1.0)
                h = T - args.holdout
                if h > 0:
                    ax.axvline((h * delta_ns + f["t0"].value) / 1e9 / 3600, color="k", ls=":", lw=0.8)
                if ri == 0:
                    ax.set_title(chn, fontsize=9)
                if c_idx == 0:
                    proto = f.get('protocol', 'N/A')
                    sport = f.get('src_port', 'N/A')
                    dport = f.get('dst_port', 'N/A')
                    ax.set_ylabel(f"{tag}\n{str(f['cid'])[:18]}\nD_i={d_i:.2e}\nProto: {proto} Port: {sport}->{dport}", fontsize=7)
                ax.tick_params(labelsize=6)
                if c_idx == C - 1:
                    ax.set_xlabel("hours since 1970", fontsize=6)
        fig.suptitle(f"OUI {o}: {title}", fontsize=11)
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        pdf.savefig(fig)
        plt.close(fig)
    return True

def _plot_heatmap(out_dir, rdf, max_rows=60):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    if rdf.empty:
        return
    order = rdf.groupby("oui")["n_flows"].sum().sort_values(ascending=False).index[:max_rows]
    sub = rdf[rdf["oui"].isin(order)]
    fig, axes = plt.subplots(1, 2, figsize=(max(8, 1.6 * sub["router"].nunique() + 6),
                                            max(3, 0.35 * len(order) + 2)))
    for ax, col, title in ((axes[0], "frac_m5_marked", "Method-5 marked fraction"),
                           (axes[1], "frac_kl_flagged", "KL-flagged fraction (significant regions)")):
        M = sub.pivot(index="oui", columns="router", values=col).loc[list(order)]
        im = ax.imshow(M.to_numpy(dtype=float), aspect="auto", cmap="magma_r")
        ax.set_xticks(range(M.shape[1])); ax.set_xticklabels(M.columns, rotation=45, ha="right", fontsize=7)
        ax.set_yticks(range(M.shape[0])); ax.set_yticklabels(M.index, fontsize=7)
        fig.colorbar(im, ax=ax); ax.set_title(title, fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "oui_router_heatmap.png"), dpi=140)
    plt.close(fig)


def make_out_dir(args, path, channels):
    if args.out_dir:
        d = args.out_dir
    else:
        base = os.path.join(args.root, "results", "anomaly_kl")
        fname = os.path.splitext(os.path.basename(path))[0]
        tag = ("host_" + fname[len("host_"):][:4]) if fname.startswith("host_") else fname[:4]
        stem = (f"{tag}_oui_f{len(channels)}_m{int(args.min_minutes)}_h{args.holdout}"
                f"_reg-{K._reg_tag(args.reg, channels)}_seed{args.seed}")
        os.makedirs(base, exist_ok=True)
        run = 1
        while os.path.exists(os.path.join(base, f"{stem}_run{run}")):
            run += 1
        d = os.path.join(base, f"{stem}_run{run}")
    os.makedirs(os.path.join(d, KNOWN_DIR), exist_ok=True)
    os.makedirs(os.path.join(d, RANDOM_DIR), exist_ok=True)
    _log(f"results -> {d}")
    return d


def run(flows, channels, args, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    rng = np.random.default_rng(args.seed)
    groups = group_flows(flows, args, rng)
    if not groups:
        print("no OUI passes the filters; lower --min-oui-flows / --min-routers."); return
    ts_name = args.ts_channel or "packet_count"
    if ts_name not in channels:
        print(f"--ts-channel '{ts_name}' is not in --channels {channels}"); return
    ci = channels.index(ts_name)

    need = sorted({i for d in groups.values() for v in d.values() for i in v})
    ch = K.Chronos(device=args.device)
    if args.split_type == "random":
        m5_score, n_short = {i: 0.0 for i in need}, 0
        _log(f"RANDOM split: no Method-5 forecast; Chronos used only for embedding")
    else:
        _log(f"Method 5 on {len(need)} flows (holdout {args.holdout}, context >= {args.context_bins}, "
             f"band {args.band_lo}-{args.band_hi}) ...")
        m5_score, m5_flags, n_short = K.method5_scores(ch, flows, need, channels, args)
        if n_short:
            _log(f"Method 5: {n_short}/{len(need)} flows too short to forecast -> treated as NORMAL")
    _log(f"embedding {len(need)} flows (reg={K._reg_tag(args.reg, channels)}) ...")
    emb = ch.embed([flows[i]["frame"] for i in need], channels, args.embed_batch,
                   cap=args.embed_cap, reg_mode=args.reg)
    row = {i: k for k, i in enumerate(need)}
    del ch

    out_dir = make_out_dir(args, path, channels)
    if args.split_type == "random":
        n_splits = max(2, int(args.n_splits))
        for k in range(n_splits):
            split_dir = os.path.join(out_dir, f"split{k+1}")
            _log(f"=== split {k+1}/{n_splits} -> {split_dir} ===")
            def assign(idx, o, k=k, ns=n_splits):
                perm = rng.permutation(idx).tolist()
                target = set(perm[k::ns])
                A = [i for i in idx if i in target]
                B = [i for i in idx if i not in target]
                return A, B, set()
            _run_ouis(flows, channels, args, groups, emb, row, m5_score, assign,
                      split_dir, rng, tag=f"split{k+1}/{n_splits} ")
        _log(f"all {n_splits} splits done -> {out_dir}")
        return

    def assign_method5(idx, o):
        n = len(idx)
        n_mark = int(round(args.anomaly_frac * n))
        top = sorted(idx, key=lambda i: -m5_score[i])[:n_mark]
        marked = {i for i in top if m5_score[i] > 0}
        perm = rng.permutation(idx).tolist()
        B0 = set(perm[:int(round(args.ref_frac * n))])
        A = [i for i in idx if (i not in B0) or (i in marked)]
        B = [i for i in idx if (i in B0) and (i not in marked)]
        return A, B, marked
    _run_ouis(flows, channels, args, groups, emb, row, m5_score, assign_method5,
              out_dir, rng)


def _run_ouis(flows, channels, args, groups, emb, row, m5_score, assign, out_dir, rng,
              tag=""):
    """Run the per-OUI KL study for ONE split assignment and write all outputs into
    out_dir. assign(idx, oui) -> (A, B, marked): target flows, reference flows and the
    set of Method-5-marked flows (empty in random mode)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    os.makedirs(os.path.join(out_dir, KNOWN_DIR), exist_ok=True)
    os.makedirs(os.path.join(out_dir, RANDOM_DIR), exist_ok=True)
    oui_rows, router_rows, region_dfs, score_dfs = [], [], [], []
    ouis = sorted(groups, key=lambda o: -sum(len(v) for v in groups[o].values()))
    for k, o in enumerate(ouis):
        rd = groups[o]
        mac_type = _mac_type(o)
        idx = sorted(i for v in rd.values() for i in v)
        n = len(idx)
        # --- split of the pooled OUI (Method-5 marks + ref-frac, or k-fold random) ---
        A, B, marked = assign(idx, o)
        if len(A) < 10 or len(B) < 10:
            _log(f"{tag}[{k+1}/{len(ouis)}] OUI {o}: A={len(A)} B={len(B)} too small -> skipped")
            continue
        markedA = np.array([i in marked for i in A], dtype=bool)
        res = analyze_oui(emb[[row[i] for i in A]], emb[[row[i] for i in B]], markedA, args,
                          np.random.default_rng(args.seed + 1000 + k))
        kl_sig = {A[j] for j in np.where(res["sig"])[0]}
        A_pos = {i: j for j, i in enumerate(A)}
        A_routers = [flows[i]["router"] for i in A]

        # --- router breakdown ---
        rstats = []
        for r in sorted(rd):
            ids = rd[r]
            rest = [i for rr in rd if rr != r for i in rd[rr]]
            nA_r = [i for i in ids if i in A_pos]
            restA = [i for i in rest if i in A_pos]
            n_m5 = sum(i in marked for i in ids); n_m5_rest = sum(i in marked for i in rest)
            n_kl = len(kl_sig & set(ids)); n_kl_rest = len(kl_sig & set(rest))
            rstats.append({
                "oui": o, "mac_type": mac_type, "router": r, "n_flows": len(ids), "n_A": len(nA_r),
                "n_m5_marked": n_m5, "frac_m5_marked": n_m5 / len(ids),
                "n_kl_flagged": n_kl, "frac_kl_flagged": n_kl / len(ids),
                "frac_kl_of_A": (n_kl / len(nA_r)) if nA_r else float("nan"),
                "mean_D_i_A": float(np.mean([res["D"][A_pos[i]] for i in nA_r])) if nA_r else float("nan"),
                "fisher_p_m5": _fisher_p(n_m5, len(ids), n_m5_rest, len(rest)),
                "fisher_p_kl": _fisher_p(n_kl, len(nA_r), n_kl_rest, len(restA))})
        router_rows += rstats
        p_chi_m5 = _chi2_p([(r["n_m5_marked"], r["n_flows"] - r["n_m5_marked"]) for r in rstats])
        p_chi_kl = _chi2_p([(r["n_kl_flagged"], r["n_A"] - r["n_kl_flagged"]) for r in rstats])

        reg_df = res["reg_df"].copy()
        if len(reg_df):
            comp = []
            for r_id in reg_df["region_id"]:
                c = Counter(A_routers[j] for j in np.where(res["region"] == r_id)[0])
                comp.append(";".join(f"{r}:{c[r]}" for r in sorted(c, key=lambda x: -c[x])))
            reg_df["routers"] = comp
            reg_df["n_routers"] = [len(s.split(";")) for s in comp]
            reg_df.insert(0, "oui", o)
            reg_df.insert(1, "mac_type", mac_type)
            region_dfs.append(reg_df)

        # --- per-flow scores ---
        Dfull = {A[j]: j for j in range(len(A))}
        score_dfs.append(pd.DataFrame({
            "oui": o, "mac_type": mac_type, "oui_raw": [flows[i]["oui"] for i in idx], "router": [flows[i]["router"] for i in idx], "cid": [flows[i]["cid"] for i in idx],
            "set": ["A" if i in A_pos else "B" if i in set(B) else "unused" for i in idx],
            "m5_score": [m5_score[i] for i in idx], "m5_marked": [i in marked for i in idx],
            "D_i": [res["D"][Dfull[i]] if i in Dfull else np.nan for i in idx],
            "survivor": [bool(res["survivors"][Dfull[i]]) if i in Dfull else False for i in idx],
            "region_id": [int(res["region"][Dfull[i]]) if i in Dfull else -1 for i in idx],
            "sig_anomaly": [i in kl_sig for i in idx]}))

        v = res["val"]
        oui_rows.append({
            "oui": o, "mac_type": mac_type, "n_flows": n, "n_routers": len(rd), "n_A": len(A), "n_B": len(B),
            "n_m5_marked": len(marked), "D_KL": res["dkl"], "n_regions": int(len(res["reg_df"])),
            "n_sig_regions": len(res["sig_ids"]), "n_kl_flagged": len(kl_sig), "top_p": res["top_p"],
            "auroc_m5": v["auroc"], "precision_m5": v["precision"], "recall_m5": v["recall"],
            "chi2_p_m5_by_router": p_chi_m5, "chi2_p_kl_by_router": p_chi_kl,
            "router_most_m5": max(rstats, key=lambda r: r["frac_m5_marked"])["router"],
            "router_most_kl": max(rstats, key=lambda r: r["frac_kl_flagged"])["router"]})

        header = (f"OUI {o}: {n} flows at {len(rd)} routers | A={len(A)} B={len(B)} | "
                  f"M5-marked {len(marked)} | D_KL={res['dkl']:.3g}, top-p={res['top_p']:.3f}, "
                  f"{len(res['sig_ids'])} sig. regions | AUROC={v['auroc']:.3f} "
                  f"prec={v['precision']:.2f} rec={v['recall']:.2f}")
        oui_name = re.sub(r"[^\w\-]", "_", str(o))
        oui_dir = os.path.join(out_dir, RANDOM_DIR if mac_type == "randomized" else KNOWN_DIR, oui_name)
        os.makedirs(oui_dir, exist_ok=True)

        non_anom_mask = (~np.isin(res["region"], list(res["sig_ids"]))
                         if res["sig_ids"] else np.ones(len(A), dtype=bool))
        non_anom_idx = np.where(non_anom_mask)[0]
        top10_non_anom = (non_anom_idx[np.argsort(-res["D"][non_anom_idx])[:10]]
                          if len(non_anom_idx) else np.array([], dtype=int))

        # anomalous flows: ordered by region, then D_i descending
        anom_idx = np.where(res["sig"])[0].tolist()
        anom_idx.sort(key=lambda j: (int(res["region"][j]), -res["D"][j]))

        # PDF 1: validation + maps + router breakdown + anomalous flows
        with PdfPages(os.path.join(oui_dir, f"{oui_name}_anomalous.pdf")) as pdf:
            _page_validation(pdf, plt, o, res, markedA, header)
            _page_maps(pdf, plt, o, res, top10_non_anom)
            _page_router_bars(pdf, plt, o, rstats)
            _pages_flow_matrix(pdf, plt, o, flows, args, res, A, anom_idx, channels, "anom")

        # PDF 2: the top-10 D_i flows that are not anomalous
        if len(top10_non_anom):
            with PdfPages(os.path.join(oui_dir, f"{oui_name}_top10_d_i.pdf")) as pdf:
                _pages_flow_matrix(pdf, plt, o, flows, args, res, A, top10_non_anom,
                                   channels, "top10")
        _log(f"{tag}[{k+1}/{len(ouis)}] OUI {o}: {n} flows/{len(rd)} routers A={len(A)} B={len(B)} "
            f"marked={len(marked)} D_KL={res['dkl']:.3g} sig={len(res['sig_ids'])} "
            f"top-p={res['top_p']:.3f} AUROC={v['auroc']:.3f}")

    if not oui_rows:
        print("no OUI was analysed."); return
    oui_df = pd.DataFrame(oui_rows)
    oui_df["top_p_bh"] = oui_df.groupby("mac_type")["top_p"].transform(lambda s: pd.Series(K._bh(s.to_numpy()), index=s.index))
    oui_df["significant"] = oui_df["top_p"] <= args.sig_alpha
    oui_df["significant_bh"] = oui_df["top_p_bh"] <= args.sig_alpha
    oui_df = oui_df.sort_values("D_KL", ascending=False).reset_index(drop=True)
    rdf = pd.DataFrame(router_rows)
    oui_df.to_csv(os.path.join(out_dir, "oui_summary.csv"), index=False)
    rdf.to_csv(os.path.join(out_dir, "router_breakdown.csv"), index=False)
    (pd.concat(region_dfs, ignore_index=True) if region_dfs else pd.DataFrame()).to_csv(
        os.path.join(out_dir, "regions_all.csv"), index=False)
    pd.concat(score_dfs, ignore_index=True).to_csv(os.path.join(out_dir, "scores_all.csv"), index=False)
    _plot_heatmap(out_dir, rdf)
    with open(os.path.join(out_dir, "summary.txt"), "w") as fh:
        if args.split_type == "random":
            fh.write(f"PER-OUI anomaly study ({tag.strip()}): random k-fold split -- "
                     "A = one fold of the OUI pool, B = the other folds; no Method-5 marks\n")
        else:
            fh.write("PER-OUI anomaly study: B = unmarked pooled flows of the OUI (all routers), "
                     "A = random part + Method-5-marked flows\n")
        fh.write(f"channels: {channels}\n")
        fh.write(f"bin {args.bin} min-minutes {args.min_minutes} holdout {args.holdout} "
                 f"context-bins {args.context_bins} anomaly-frac {args.anomaly_frac} "
                 f"ref-frac {args.ref_frac} split-type {args.split_type} "
                 f"reg={K._reg_tag(args.reg, channels)} pca-dim {args.pca_dim} "
                 f"perms {args.perms} sig-alpha {args.sig_alpha}\n")
        fh.write(f"OUIs analysed: {len(oui_df)}; with a significant KL region: "
                 f"{int(oui_df['significant'].sum())} (BH: {int(oui_df['significant_bh'].sum())})\n")
        fh.write(f"OUIs whose Method-5 marks differ across routers (chi2 p<=0.05): "
                 f"{int((oui_df['chi2_p_m5_by_router'] <= 0.05).sum())}\n")
        fh.write(f"OUIs whose KL flags differ across routers (chi2 p<=0.05): "
                 f"{int((oui_df['chi2_p_kl_by_router'] <= 0.05).sum())}\n\nPER OUI:\n"
                 + oui_df.to_string(index=False) + "\n")
    _log(f"wrote outputs to {out_dir}")


def find_features(args):
    if args.csv:
        return args.csv
    fdir = args.features_dir or os.path.join(args.root, "data", "features")
    files = glob.glob(os.path.join(fdir, "*.parquet"))
    parts = glob.glob(os.path.join(fdir, "[0-9][0-9][0-9][0-9][0-9][0-9].parquet"))
    if parts and len(parts) == len(files):
        cand = [fdir]
    else:
        cand = sorted(files) + sorted(d for d in glob.glob(os.path.join(fdir, "*.parts")) if os.path.isdir(d))
    if args.only:
        cand = [p for p in cand if args.only in os.path.basename(p)]
    return cand[0] if cand else None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default=None)
    ap.add_argument("--only", default=None)
    ap.add_argument("--features-dir", default=None)
    ap.add_argument("--root", default=K.DEFAULT_ROOT)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--bin", default="1min", help="bin size of the features file (default 1min)")
    ap.add_argument("--channels", default="packet_count,mean_iat,mean_packet_size",
                    help="all are given to Chronos; the first is the target")
    ap.add_argument("--min-minutes", type=float, default=60,
                    help="minimum flow duration in minutes (default 60; overrides --min-bins)")
    ap.add_argument("--min-bins", type=int, default=60)
    ap.add_argument("--min-packets", type=int, default=10)
    ap.add_argument("--only-conns", default=None)
    ap.add_argument("--max-flows", type=int, default=0, help="smoke test: random subset of N flows")
    # Method 5
    ap.add_argument("--holdout", type=int, default=30, help="bins forecast per flow (default 30 = 30 min)")
    ap.add_argument("--context-bins", type=int, default=30,
                    help="minimum history before the holdout (default 30 = 30 min)")
    ap.add_argument("--band-lo", type=float, default=0.1)
    ap.add_argument("--band-hi", type=float, default=0.9)
    ap.add_argument("--norm-thr", type=float, default=0.1)
    ap.add_argument("--anomaly-frac", type=float, default=0.1,
                    help="top fraction of each OUI's flows (by Method-5 score > 0) marked anomalous")
    ap.add_argument("--batch-size", type=int, default=256)
    # split / KL
    ap.add_argument("--split-type", choices=["method5", "random"], default="method5",
                    help="method5 = current behaviour (Method-5 marks + random --ref-frac "
                         "split of each OUI pool). random = no Method-5 forecast at all: "
                         "each OUI's filtered flows are split into --n-splits folds and "
                         "the experiment runs --n-splits times, each fold taking its turn "
                         "as target A while the other folds are the reference B; outputs "
                         "go to split1..splitN subfolders. Chronos is then used ONLY for "
                         "the embedding.")
    ap.add_argument("--n-splits", type=int, default=5,
                    help="number of folds for --split-type random (default 5)")
    ap.add_argument("--ref-frac", type=float, default=0.7)
    ap.add_argument("--pca-dim", type=int, default=64)
    ap.add_argument("--perms", type=int, default=20)
    ap.add_argument("--alpha", type=float, default=0.01)
    ap.add_argument("--sig-alpha", type=float, default=0.05)
    ap.add_argument("--smooth-k", type=int, default=5)
    ap.add_argument("--reg", default="target")
    ap.add_argument("--embed-batch", type=int, default=64)
    ap.add_argument("--embed-cap", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    # OUI / router
    ap.add_argument("--oui-col", default="oui")
    ap.add_argument("--mac-col", default="mac",
                help="column with the full MAC (needed to group randomized MACs individually)")    
    ap.add_argument("--router-col", default="router")
    ap.add_argument("--router-regex", default=None)
    ap.add_argument("--only-oui", default=None, help="comma-separated OUIs (default ALL)")
    ap.add_argument("--min-oui-flows", type=int, default=20,
                    help="minimum flows of an OUI (all routers pooled) to analyse it (default 50)")
    ap.add_argument("--min-routers", type=int, default=1,
                    help="minimum number of routers an OUI must appear at (default 1; use 2 to "
                         "study only OUIs that can be compared across routers)")
    ap.add_argument("--max-oui-flows", type=int, default=0,
                    help="randomly cap the flows of each OUI (0 = no cap)")
    # time-series pages (plain series, no markers)
    ap.add_argument("--ts-channel", default=None, help="channel drawn (default packet_count)")
    ap.add_argument("--ts-mode", choices=["sum", "flows"], default="sum",
                    help="'sum' = summed channel per router (default); 'flows' = individual flows")
    ap.add_argument("--ts-max-flows", type=int, default=40, help="--ts-mode flows: flows per panel")
    ap.add_argument("--ts-routers-per-page", type=int, default=6)
    ap.add_argument("--drop-random-mac", "--drop-local-mac", dest="drop_random_mac",
                    action=argparse.BooleanOptionalAction, default=False,
                    help="drop randomised MACs (default OFF: they are kept and analysed)")
    ap.add_argument("--random-group-by", choices=["oui", "mac"], default="mac",
                    help="randomized MACs: one group per random OUI (default) or per full MAC "
                        "(needs --mac-col)")
    ap.add_argument("--min-set-size", type=int, default=10,
                    help="minimum flows in both A and B to run the KL analysis (default 10)")
    ap.add_argument("--min-random-flows", type=int, default=20,
                help="minimum flows of ONE randomized MAC to analyse it")
    args = ap.parse_args()

    K.MIN_CONTEXT_BINS = args.context_bins
    channels = [c.strip() for c in args.channels.split(",") if c.strip()]
    if "packet_count" not in channels:
        print("--channels must include packet_count"); return
    if str(args.reg).lstrip("-").isdigit():
        args.reg = int(args.reg)
    elif args.reg in channels:
        args.reg = channels.index(args.reg)
    elif args.reg not in ("target", "mean", "concat"):
        print(f"--reg '{args.reg}' invalid"); return
    # consistency of min duration vs holdout + context
    bin_min = pd.Timedelta(pd.tseries.frequencies.to_offset(args.bin)).total_seconds() / 60.0
    eff = int(np.ceil(args.min_minutes / bin_min)) + 1 if args.min_minutes else args.min_bins
    need = args.holdout + args.context_bins
    if eff < need:
        _log(f"WARNING: shortest kept flows have ~{eff} bins < holdout+context = {need}: they cannot "
             f"be forecast and are treated as NORMAL. Use --holdout <= {max(1, eff - args.context_bins)}.")
    else:
        _log(f"check OK: flows have >= ~{eff} bins >= holdout {args.holdout} + context {args.context_bins}")

    path = find_features(args)
    if not path:
        print("no features file found"); return
    _log(f"features: {path}; channels: {channels}")
    only_conns = None
    if args.only_conns:
        oc = pd.read_csv(args.only_conns)
        col = "conn_id" if "conn_id" in oc.columns else oc.columns[0]
        only_conns = pd.Index(oc[col].astype(str).unique())
    flows = K.load_flows(path, channels, args.bin, args.min_bins, args.min_packets, "packet_count",
                         max_flows=args.max_flows, seed=args.seed, only_conns=only_conns,
                         min_minutes=args.min_minutes, oui_col=args.oui_col, mac_col=args.mac_col,
                         router_col=args.router_col, router_regex=args.router_regex)
    _log(f"kept {len(flows)} flows")
    if len(flows) < 20:
        print("too few flows."); return
    run(flows, channels, args, path)


if __name__ == "__main__":
    main()