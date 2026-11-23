#!/usr/bin/env python3
"""
anomaly_kl.py  (lives in  Anomaly_detection/src/)

===============================================================================
Anomaly detection by LOCALISING THE RELATIVE ENTROPY between two sets of flows,
following "Explaining Dataset Differences from Local Contributions to Relative
Entropy" (WACV 2027 #2475), with Chronos-2 as the encoder instead of DINOv2.

Each flow (connection time series) is ONE unit -- the analogue of an image in
the paper. We:
  1. keep flows with >= --min-bins REAL bins and >= --min-packets packets;
  2. split them randomly into B (--ref-frac, e.g. 80%) and A (the rest, 20%);
  3. FILTER B with Chronos-2 forecast intervals: forecast packet_count (with
     mean_iat + mean_packet_size as past covariates); a flow whose holdout has
     >= --band-bins points OUTSIDE the [q_lo, q_hi] band is anomalous and is
     MOVED into A and marked as a known anomaly (ground truth for validation);
  4. EMBED every flow with Chronos-2's REG token (embed over the 3 channels,
     whole series) -> one vector per flow;
  5. project A u B to --pca-dim principal components (fitted on A u B, as in the
     paper -- this keeps the direction a sparse anomalous sub-population adds);
  6. for each flow in A, compute its LOCAL KL CONTRIBUTION D_i (count-based,
     Eq. 3 of the paper) against the density of B around it;
  7. calibrate D_i on a permutation null (relabel A u B, T times), keep the
     survivors, cluster them (HDBSCAN) into regions with a cluster-mass p-value;
  8. VALIDATE: how well D_i (and the regions) recover the flows marked anomalous
     by the interval filter -- AUROC, precision/recall at the survivor threshold.

The interval filter (step 3) provides the "ground truth"; the KL method (steps
4-7) is the method under test. Agreement between the two is the result.

USAGE
    python src/anomaly_kl.py --only 2eadb48f --device cuda \
        --min-bins 48 --min-packets 15 --ref-frac 0.8 \
        --channels packet_count,mean_iat,mean_packet_size

KEY OPTIONS
    --only <hash>        session (name or prefix); default: first features file
    --csv <path>         explicit features parquet/csv
    --channels a,b,c     target first, then covariates (default: packet_count,
                         mean_iat,mean_packet_size). NOTE: needs these columns in
                         the features file (5-tuple aggregation has fwd_/bwd_*;
                         pass e.g. packet_count,fwd_mean_iat,fwd_mean_packet_size).
    --min-bins N         keep flows with SPAN >= N bins (continuous grid length, default 48)
    --min-packets N      keep flows with >= N packets (default 15)
    --ref-frac F         fraction of flows put in B (reference) (default 0.8)
    --split-type T       method5 (default: the behaviour above) or random: no Method-5
                         forecast; the filtered flows are split into --n-splits folds and
                         the experiment runs once per fold (fold = target A, the rest =
                         reference B), writing split1..splitN/ subfolders. Chronos is then
                         used ONLY for the embedding (no marks, validation skipped).
    --n-splits N         number of folds for --split-type random (default 5)
    --holdout N          forecast horizon for the interval filter (default 24)
    --band-lo/--band-hi  Chronos quantiles of the interval (default 0.1/0.9)
    --band-bins N        a flow is anomalous if >= N holdout points fall outside
                         the band (default 1 -- a single anomalous bin)
    --pca-dim N          projection dim, fitted on A u B (default 64)
    --perms T            permutations for the null (default 20)
    --alpha A            survivor quantile is 1-A (default 0.01)
    --device auto|cuda|cpu
    --seed S             RNG seed for the split (default 0)
    --out-dir PATH       results dir (default results/anomaly_kl/<run>)

OUTPUT (results/anomaly_kl/<run>/)
    scores.csv           per flow: set (A/B), marked_anomaly, D_i, survivor,
                         region_id, projected coords omitted
    regions.csv          per region: size, score, share of divergence, p-value
    summary.txt          filter counts + validation (AUROC, precision/recall)
    validation.png       D_i distribution for marked vs rest, ROC, embedding 2-D
===============================================================================
"""

import argparse
import os
import sys
import time
import re
import math
import numpy as np
import pandas as pd

DEFAULT_ROOT = os.path.dirname(os.path.abspath(__file__))


def _log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


MIN_CONTEXT_BINS = 30  # overwritten by anomaly_kl_oui.py (--context-bins)


def normalize_oui(x):
    """'08:7c:39', 'aa-bb-cc', 'aabbcc..' -> '087C39' (None if unusable)."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    s = re.sub(r"[^0-9A-Fa-f]", "", str(x)).upper()
    return s[:6] if len(s) >= 6 else None


def _is_local_oui(oui):
    """Locally-administered (randomised) MAC: bit 0x02 of the first byte."""
    try:
        return bool(int(str(oui)[:2], 16) & 0x02)
    except Exception:
        return False


def _bh(p):
    """Benjamini-Hochberg adjusted p-values."""
    p = np.asarray(p, dtype=float)
    n = len(p)
    if n == 0:
        return p
    order = np.argsort(p)
    ranked = p[order] * n / (np.arange(n) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(ranked, 0, 1)
    return out


def method5_scores(ch, flows, idxs, channels, args):
    """-> (score {idx: float}, flags {idx: bool[holdout, C]}, n_short).
    Flows shorter than holdout + MIN_CONTEXT_BINS get score 0 (treated as normal)."""
    H, C = args.holdout, len(channels)
    score = {i: 0.0 for i in idxs}
    flags = {}
    ok = [i for i in idxs if flows[i]["frame"].shape[0] >= H + MIN_CONTEXT_BINS]
    CONST_EPS = 1e-6
    CH = 2000
    for s in range(0, len(ok), CH):
        part = ok[s:s + CH]
        lo, med, hi = ch.forecast_band([flows[i]["frame"] for i in part], channels, H,
                                       args.band_lo, args.band_hi, args.batch_size)
        for k, i in enumerate(part):
            fr = flows[i]["frame"]; T = fr.shape[0]
            actual = fr[T - H:, :]
            yr = fr.max(axis=0) - fr.min(axis=0)
            const_ch = np.zeros(C, dtype=bool)
            for ci in range(C):
                nz = fr[:, ci][np.abs(fr[:, ci]) > CONST_EPS]
                if nz.size == 0 or (nz.max() - nz.min()) <= CONST_EPS:
                    const_ch[ci] = True
            diff = np.abs(actual - med[k])
            norm = diff / (yr[None, :] + 1e-5)
            out = (actual > hi[k] + 1e-2) | (actual < lo[k] - 1e-2)
            valid = out & (diff >= 1e-2) & (~const_ch[None, :])
            excess = np.where(valid, np.maximum(0.0, norm - args.norm_thr), 0.0)
            score[i] = float(excess.sum())
            flags[i] = valid & (norm > args.norm_thr)
        _log(f"  Method 5: {min(s + CH, len(ok))}/{len(ok)} flows forecast")
    return score, flags, len(idxs) - len(ok)


# --------------------------------------------------------------------------- #
# Data loading: one padded series per flow, plus real-bin / packet counts     #
# --------------------------------------------------------------------------- #
def load_flows(path, channels, bin_freq, min_bins, min_packets, pkt_col, max_flows=0, seed=0,
               only_conns=None, min_minutes=None, oui_col=None, mac_col=None,
               router_col=None, router_regex=None):
    """Return a list of dicts {cid, frame[T,C], n_real, span, n_pkts} for flows
    passing the filters (SPAN >= min_bins AND packets >= min_packets).
    FAST: eligibility is decided with a single vectorised groupby BEFORE any
    padding, so the expensive per-flow sort+reindex runs only on flows that pass
    (and only up to max_flows of them)."""
    
    # We might need raw fwd/bwd columns to compute total_mean_iat etc
    base_cols = {"datetime", "conn_id", pkt_col, 'fwd_packet_count', 'bwd_packet_count', 
                 'fwd_mean_packet_size', 'bwd_mean_packet_size', 'fwd_mean_iat', 'bwd_mean_iat',
                 'protocol', 'src_port', 'dst_port'}
    if oui_col: base_cols.add(oui_col)
    if mac_col: base_cols.add(mac_col)
    if router_col: base_cols.add(router_col)
    need = list(base_cols.union(set(channels)))
    
    import pyarrow.parquet as pq
    schema = pq.read_schema(path)
    avail = set(schema.names)
    need = [c for c in need if c in avail]
    
    df = pd.read_parquet(path, columns=need) if str(path).endswith(".parquet") \
        else pd.read_csv(path, usecols=need)
        
    df["datetime"] = pd.to_datetime(df["datetime"])
    
    # Compute derived 'total_' features exactly like anomaly_set_difference.py
    if pkt_col in df.columns:
        df['total_packet_count'] = df[pkt_col]
        
    if all(c in df.columns for c in ['fwd_mean_packet_size', 'fwd_packet_count', 'bwd_mean_packet_size', 'bwd_packet_count']):
        df['total_mean_packet_size'] = np.where(
            df[pkt_col] > 0,
            (df['fwd_mean_packet_size'].fillna(0) * df['fwd_packet_count'].fillna(0) + 
             df['bwd_mean_packet_size'].fillna(0) * df['bwd_packet_count'].fillna(0)) / df[pkt_col],
            0
        )
    if all(c in df.columns for c in ['fwd_mean_iat', 'bwd_mean_iat']):
        df['total_mean_iat'] = df[['fwd_mean_iat', 'bwd_mean_iat']].mean(axis=1, skipna=True).fillna(0)
    
    # Also map mean_iat to total_mean_iat if requested and missing
    if "mean_iat" in channels and "mean_iat" not in df.columns and "total_mean_iat" in df.columns:
        df["mean_iat"] = df["total_mean_iat"]
    if "mean_packet_size" in channels and "mean_packet_size" not in df.columns and "total_mean_packet_size" in df.columns:
        df["mean_packet_size"] = df["total_mean_packet_size"]
    
    delta = pd.Timedelta(pd.tseries.frequencies.to_offset(bin_freq))
    # --- cheap vectorised eligibility (no padding) ---
    g = df.groupby("conn_id", sort=False)
    agg = g["datetime"].agg(["min", "max"])
    agg["span"] = ((agg["max"] - agg["min"]) / delta).astype("int64") + 1
    agg["pkts"] = g[pkt_col].sum()
    mask = (agg["span"] >= min_bins) & (agg["pkts"] >= min_packets)
    if min_minutes:
        mask &= ((agg["max"] - agg["min"]).dt.total_seconds() / 60.0 >= min_minutes)
    if only_conns is not None:
        mask &= agg.index.astype(str).isin(set(map(str, only_conns)))
    eligible = agg.index[mask]
    _log(f"eligible flows (span>={min_bins} & pkts>={min_packets}" + 
         (f" & mins>={min_minutes}" if min_minutes else "") + 
         f"): {len(eligible)} / {g.ngroups}")
    if max_flows and len(eligible) > max_flows:
        pick = np.random.default_rng(seed).choice(len(eligible), max_flows, replace=False)
        eligible = eligible[np.sort(pick)]
        _log(f">>> SMOKE TEST: padding only {len(eligible)} flows (--max-flows {max_flows}) <<<")
    keep = set(eligible.tolist())
    # --- pad ONLY the eligible flows ---
    flows = []
    sub = df[df["conn_id"].isin(keep)]
    for cid, gg in sub.groupby("conn_id", sort=False):
        gg = gg.sort_values("datetime").drop_duplicates("datetime", keep="first").set_index("datetime")
        n_real = len(gg)
        span = int((gg.index.max() - gg.index.min()) / delta) + 1
        n_pkts = float(gg[pkt_col].sum())
        full = pd.date_range(gg.index.min(), gg.index.max(), freq=bin_freq)
        frame = gg[channels].apply(pd.to_numeric, errors="coerce").reindex(full).fillna(0.0)
        d = {"cid": str(cid), "frame": frame[channels].to_numpy(dtype=np.float32),
             "n_real": int(n_real), "span": int(span), "n_pkts": n_pkts,
             "t0": full[0]}

        # OUI: normalize to 'AABBCC' from the OUI column, or fall back to MAC.
        o_raw = None
        if oui_col and oui_col in gg.columns:
            pv = gg[oui_col].dropna()
            o_raw = pv.iloc[0] if len(pv) else None
        if o_raw is None and mac_col and mac_col in gg.columns:
            pv = gg[mac_col].dropna()
            o_raw = pv.iloc[0] if len(pv) else None
        d["oui"] = normalize_oui(o_raw)

        if mac_col and mac_col in gg.columns:
            pv = gg[mac_col].dropna()
            d["mac"] = pv.iloc[0] if len(pv) else None

        # Router: use the router column if populated; otherwise derive it from conn_id.
        r = None
        if router_col and router_col in gg.columns:
            pv = gg[router_col].dropna()
            r = str(pv.iloc[0]) if len(pv) else None
        if r is None or r.lower() in ("", "nan", "none"):
            r = str(cid)
        if router_regex:
            m = re.search(router_regex, r)
            if m:
                r = m.group(1) if m.groups() else m.group(0)
        d["router"] = r

        # Protocol and ports
        for k in ['protocol', 'src_port', 'dst_port']:
            if k in gg.columns:
                pv = gg[k].dropna()
                d[k] = pv.iloc[0] if len(pv) else None

        flows.append(d)
    return flows


# --------------------------------------------------------------------------- #
# Chronos-2 wrapper: forecast intervals (filter B) and REG embeddings         #
# --------------------------------------------------------------------------- #
class Chronos:
    def __init__(self, device="cuda", model="amazon/chronos-2"):
        import torch
        from chronos import Chronos2Pipeline
        dev = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
        self.dev = dev
        dt = torch.bfloat16 if dev == "cuda" else torch.float32
        last = None
        for kw in (dict(device_map=dev, torch_dtype=dt),
                   dict(device_map=dev, dtype=dt),
                   dict(device_map=dev),
                   dict()):
            try:
                self.pipe = Chronos2Pipeline.from_pretrained(model, **kw)
                break
            except TypeError as e:
                last = e
        else:
            raise RuntimeError(f"could not load Chronos-2: {last}")
        # move to device if the loader didn't
        try:
            if hasattr(self.pipe, "model"):
                self.pipe.model = self.pipe.model.to(dev)
        except Exception:
            pass
        _log(f"chronos-2 loaded on {dev}")

    def forecast_band(self, frames, channels, holdout, q_lo, q_hi, batch=256):
        """MULTIVARIATE: forecast ALL channels jointly over the last `holdout`
        bins (each channel is a target in one group). Return lo/med/hi of shape
        [N, holdout, C] -- one band per channel. Leakage-free: future_df has only
        timestamps, no values."""
        import pandas as pd
        C = len(channels)
        lo = np.zeros((len(frames), holdout, C), dtype=np.float32)
        md = np.zeros_like(lo); hi = np.zeros_like(lo)
        step = pd.Timedelta(minutes=5)
        for s in range(0, len(frames), batch):
            chunk = frames[s:s + batch]
            hist_rows, fut_rows = [], []
            for j, fr in enumerate(chunk):
                T = fr.shape[0]
                ctx = fr[:T - holdout]
                idx = pd.date_range("2020-01-01", periods=len(ctx), freq="5min")
                d = {"id": str(j), "datetime": idx}
                for ci, ch in enumerate(channels):
                    d[ch] = ctx[:, ci]
                hist_rows.append(pd.DataFrame(d))
                fidx = pd.date_range(idx[-1] + step, periods=holdout, freq="5min")
                fut_rows.append(pd.DataFrame({"id": str(j), "datetime": fidx}))
            hist = pd.concat(hist_rows, ignore_index=True)
            fut = pd.concat(fut_rows, ignore_index=True)
            pred = self.pipe.predict_df(
                hist, future_df=fut, prediction_length=holdout,
                quantile_levels=[q_lo, 0.5, q_hi], id_column="id",
                timestamp_column="datetime", target=list(channels),  # ALL channels
                batch_size=512, validate_inputs=False)
            for j in range(len(chunk)):
                p = pred[pred["id"] == str(j)]
                for ci, ch in enumerate(channels):
                    loc, mdc, hic = _pick_quantiles(p, ch, q_lo, q_hi, holdout)
                    lo[s + j, :, ci] = loc; md[s + j, :, ci] = mdc; hi[s + j, :, ci] = hic
        return lo, md, hi

    def embed(self, frames, channels, batch=64, cap=0, max_batch_tokens=200000,
              reg_mode="target"):
        """Return ONE reduced vector per flow (N, d'), reducing each flow's per-channel
        REG tokens ON THE FLY (via reg_mode) so we never hold the full (N, C, d) tensor
        -- this keeps memory at the old, low level. reg_mode: 'target' (channel 0),
        'mean', 'concat', or an int channel index.
        Memory-safe batching: flows have very different lengths; a batch pads to the
        longest in it, so we form DYNAMIC batches with a token budget (long-flow
        batches hold few flows), cap series length, and free the GPU cache per batch."""
        import numpy as np
        N = len(frames)

        def prep(fr):
            a = fr if (not cap or fr.shape[0] <= cap) else fr[-cap:]
            return np.ascontiguousarray(a.T, dtype=np.float32)      # (C, T)

        def reduce_one(a):                                          # a: (C, tokens, d)
            if a.ndim == 3:
                reg = a[:, -1, :]                                   # (C, d) per-channel REG
            elif a.ndim == 2:
                reg = a[-1][None, :]
            else:
                reg = a[None, :]
            if isinstance(reg_mode, int):
                return reg[reg_mode].astype(np.float32)
            if reg_mode == "mean":
                return reg.mean(axis=0).astype(np.float32)
            if reg_mode == "concat":
                return reg.reshape(-1).astype(np.float32)
            return reg[0].astype(np.float32)                       # 'target' -> channel 0

        lens = [(cap if cap and frames[i].shape[0] > cap else frames[i].shape[0]) for i in range(N)]
        order = sorted(range(N), key=lambda i: lens[i])
        out = [None] * N
        i = 0
        while i < N:
            j = i; maxlen = lens[order[i]]
            while j < N and (j - i) < batch:
                cand = max(maxlen, lens[order[j]])
                if (j - i + 1) * cand > max_batch_tokens and j > i:
                    break
                maxlen = cand; j += 1
            idxs = order[i:j]
            chunk = [prep(frames[k]) for k in idxs]
            o = self.pipe.embed(chunk, batch_size=len(chunk))
            emb_list = o[0] if isinstance(o, tuple) else o
            for k, e in zip(idxs, emb_list):
                out[k] = reduce_one(_to_np(e))                     # reduce NOW -> one vector
            del chunk, o, emb_list
            try:
                import torch; torch.cuda.empty_cache()
            except Exception:
                pass
            i = j
        return np.stack(out, axis=0)                               # (N, d')

    def embed_bins(self, frames, channels, holdout=24, L=12, batch=256, reg_mode="target"):
        """Extract sliding windows of length L for the holdout period of each flow,
        embed them, and return ONE vector per BIN (shape [N_flows * holdout, d'])."""
        import numpy as np
        
        windows = []
        for fr in frames:
            T = fr.shape[0]
            for t in range(T - holdout, T):
                win = fr[t - L + 1 : t + 1]
                windows.append(np.ascontiguousarray(win.T, dtype=np.float32))
                
        out = []
        
        def reduce_one(a):
            if a.ndim == 3:
                reg = a[:, -1, :]
            elif a.ndim == 2:
                reg = a[-1][None, :]
            else:
                reg = a[None, :]
            if isinstance(reg_mode, int):
                return reg[reg_mode].astype(np.float32)
            if reg_mode == "mean":
                return reg.mean(axis=0).astype(np.float32)
            if reg_mode == "concat":
                return reg.reshape(-1).astype(np.float32)
            return reg[0].astype(np.float32)

        for s in range(0, len(windows), batch):
            chunk = windows[s : s + batch]
            o = self.pipe.embed(chunk, batch_size=len(chunk))
            emb_list = o[0] if isinstance(o, tuple) else o
            for e in emb_list:
                out.append(reduce_one(_to_np(e)))
            del chunk, o, emb_list
            try:
                import torch; torch.cuda.empty_cache()
            except Exception:
                pass
        if not out:
            return np.zeros((0, 0), dtype=np.float32)
        return np.stack(out, axis=0)

    def _embed_df(self, dfb, channels):   # kept for compatibility (unused now)
        return _to_2d(self.pipe.embed_df(dfb, id_column="id", timestamp_column="datetime"))


def _to_2d(out):
    import numpy as np
    try:
        import torch
        if isinstance(out, torch.Tensor):
            out = out.detach().float().cpu().numpy()
    except Exception:
        pass
    if isinstance(out, tuple):
        out = out[0]
    out = np.asarray(out)
    if out.ndim == 3:            # [B, tokens, d] -> take REG (last) or mean
        out = out[:, -1, :]
    return out.astype(np.float32)


# --------------------------------------------------------------------------- #
# Local KL contribution (count-based, Eq. 3) + permutation null + regions     #
# --------------------------------------------------------------------------- #
def _pick_quantiles(p, ch, q_lo, q_hi, holdout):
    """Extract lo/med/hi for target `ch` from a predict_df result. Multivariate
    predict_df returns rows for ALL targets with a 'target_name' column and plain
    quantile columns ('0.1','0.5','0.9'); we must select this channel's rows.
    Falls back to prefixed columns or plain columns for other API shapes."""
    import numpy as np
    if "target_name" in p.columns:                     # multivariate long format
        pc = p[p["target_name"] == ch]
        return (pc[str(q_lo)].to_numpy()[:holdout],
                pc["0.5"].to_numpy()[:holdout],
                pc[str(q_hi)].to_numpy()[:holdout])
    cols = p.columns
    def col(q):
        for name in (f"{ch}/{q}", f"{ch}_{q}", f"{q}_{ch}", str(q)):
            if name in cols:
                return p[name].to_numpy()[:holdout]
        return p[str(q)].to_numpy()[:holdout]
    return col(q_lo), col(0.5), col(q_hi)


def _to_np(e):
    import numpy as np
    try:
        import torch
        if isinstance(e, torch.Tensor):
            e = e.detach().float().cpu().numpy()
    except Exception:
        pass
    return np.asarray(e, dtype=np.float32)


def reduce_regs(regs, mode="target"):
    """Reduce per-channel REG tokens (N, C, d) to one vector per flow (N, d').
      'target'  -> channel 0 (the target, e.g. packet_count)
      'mean'    -> mean over channels
      'concat'  -> channels concatenated (d' = C*d)
      int k     -> channel k
    All channels were given to the encoder, so even 'target' carries cross-channel
    information via group attention."""
    import numpy as np
    regs = np.asarray(regs)
    if regs.ndim == 2:            # already (N, d)
        return regs
    if isinstance(mode, int):
        return regs[:, mode, :]
    if mode == "mean":
        return regs.mean(axis=1)
    if mode == "concat":
        return regs.reshape(len(regs), -1)
    return regs[:, 0, :]          # 'target' -> channel 0


def project(emb, dim):
    """PCA to `dim` components, fitted on ALL rows (A u B) -- keeps the sparse
    anomaly direction (paper Sec. 3.2/4). Uses the RANDOMIZED solver, which needs
    far less memory than the exact SVD when the input has many dimensions (e.g.
    --reg concat gives C*768 dims) -- this is what caused the OOM before."""
    from sklearn.decomposition import PCA
    X = np.ascontiguousarray(emb, dtype=np.float32)
    X = X - X.mean(0, keepdims=True)
    dim = min(dim, X.shape[1], max(2, X.shape[0] - 1))
    # randomized SVD: memory ~ O(n * dim) instead of O(n * D) intermediates
    solver = "randomized" if X.shape[1] > 2304 else "auto"
    return PCA(n_components=dim, svd_solver=solver,
               random_state=0).fit_transform(X).astype(np.float32)


def _smooth(D, XA, k=5):
    """Average each contribution with its k nearest target neighbours (paper
    Sec. 3.2): raises the region-level signal-to-noise ratio. Returns a smoothed
    copy of D."""
    if k <= 0 or len(XA) <= 1:
        return D
    from sklearn.neighbors import NearestNeighbors
    kk = min(k + 1, len(XA))
    nn = NearestNeighbors(n_neighbors=kk).fit(XA)
    idx = nn.kneighbors(XA, return_distance=False)   # includes self at col 0
    return D[idx].mean(axis=1)


def kl_contributions(XA, XB, smooth_k=0):
    """Count-based local contribution D_i for each row of XA against XB (Eq. 3).
    eps_i = max(dist to nearest A, dist to nearest B); a_i,b_i = #A,#B within eps.
    D_i = (1/n)(psi(a_i) - psi(b_i) + ln(m/(n-1))).
    If smooth_k>0, each D_i is averaged with its k nearest A-neighbours (paper
    Sec. 3.2). The permutation null must use the SAME smooth_k for calibration."""
    from scipy.special import digamma
    from sklearn.neighbors import NearestNeighbors, BallTree
    n, m = len(XA), len(XB)
    nnA = NearestNeighbors(n_neighbors=2).fit(XA)   # self + 1 -> nearest other A
    nnB = NearestNeighbors(n_neighbors=1).fit(XB)
    rhoA = nnA.kneighbors(XA)[0][:, 1]              # nearest A (excluding self)
    nuB = nnB.kneighbors(XA)[0][:, 0]              # nearest B
    eps = np.maximum(rhoA, nuB) * (1 + 1e-6)
    ba = BallTree(XA); bb = BallTree(XB)
    a = ba.query_radius(XA, r=eps, count_only=True).astype(float) - 1.0   # minus self
    b = bb.query_radius(XA, r=eps, count_only=True).astype(float)
    a = np.clip(a, 1, None); b = np.clip(b, 1, None)
    D = (digamma(a) - digamma(b) + np.log(m / max(n - 1, 1))) / n
    if smooth_k > 0:
        D = _smooth(D, XA, smooth_k)
    return D


def permutation_threshold(XA, XB, T, alpha, rng, smooth_k=0):
    """Null: relabel A u B, recompute D on the A-sized part, T times; threshold
    is the (1-alpha) quantile of pooled null contributions. Smoothing is applied
    IDENTICALLY to the null (paper Sec. 3.2) so the calibration stays valid."""
    X = np.vstack([XA, XB]); n = len(XA)
    null = []
    for _ in range(T):
        perm = rng.permutation(len(X))
        A2, B2 = X[perm[:n]], X[perm[n:]]
        null.append(kl_contributions(A2, B2, smooth_k=smooth_k))
    null = np.concatenate(null)
    return np.quantile(null, 1 - alpha), null


def _cluster_labels(X):
    """HDBSCAN labels for points X (>=5 needed), fallback to DBSCAN. -1 = noise."""
    if len(X) < 5:
        return np.zeros(len(X), dtype=int) if len(X) else np.array([], dtype=int)
    try:
        import hdbscan
        return hdbscan.HDBSCAN(min_cluster_size=5, metric="euclidean").fit_predict(X)
    except Exception:
        from sklearn.cluster import DBSCAN
        return DBSCAN(eps=np.median(np.std(X, 0)) + 1e-9, min_samples=5).fit_predict(X)


def permutation_null(XA, XB, T, alpha, rng, smooth_k=0):
    """One pass over T relabellings that returns BOTH calibration pieces:
      - thr: the (1-alpha) quantile of pooled null contributions (survivor threshold)
      - null_region_max: for each permutation, the largest region score s(R) formed
        under the null (using the SAME threshold+clustering) -> for cluster-mass
        p-values. Region score is the sum of (smoothed) contributions of a cluster."""
    X = np.vstack([XA, XB]); n = len(X); nA = len(XA)
    # first collect pooled null contributions to set the threshold
    perms = [rng.permutation(n) for _ in range(T)]
    null_contribs = []
    per_perm_D = []
    for perm in perms:
        A2, B2 = X[perm[:nA]], X[perm[nA:]]
        d = kl_contributions(A2, B2, smooth_k=smooth_k)
        per_perm_D.append((perm, d))
        null_contribs.append(d)
    thr = float(np.quantile(np.concatenate(null_contribs), 1 - alpha))
    # now, with the threshold fixed, find the max region score under each permutation
    null_region_max = []
    for perm, d in per_perm_D:
        A2 = X[perm[:nA]]
        surv = d > thr
        idx = np.where(surv)[0]
        best = 0.0
        if len(idx) >= 5:
            lab = _cluster_labels(A2[idx])
            for r in set(lab.tolist()):
                if r < 0:
                    continue
                best = max(best, float(d[idx][lab == r].sum()))
        null_region_max.append(best)
    return thr, np.array(null_region_max)


def rank_regions(region, D, D_raw, dkl, null_region_max, marked=None, XA=None):
    """Build the region table: for each region (id>=0) its size, score s(R)=sum D_i,
    share of the divergence (sum of RAW contributions / D_KL), cluster-mass p-value
    from the null region maxima, silhouette (cluster tightness/separation, needs XA),
    and (if given) how many members are Method-5 marked. Sorted by score desc."""
    import pandas as pd
    T = len(null_region_max)
    # per-point silhouette over the clustered survivors (region>=0), once
    sil_by_point = None
    if XA is not None:
        clustered = region >= 0
        labs = region[clustered]
        if clustered.sum() >= 3 and np.unique(labs).size >= 2:
            try:
                from sklearn.metrics import silhouette_samples
                s = silhouette_samples(XA[clustered], labs)
                sil_by_point = np.full(len(region), np.nan)
                sil_by_point[np.where(clustered)[0]] = s
            except Exception:
                sil_by_point = None
    rows = []
    for r in sorted(set(region.tolist())):
        if r < 0:
            continue
        m = region == r
        s = float(D[m].sum())
        share = float(D_raw[m].sum() / dkl) if dkl not in (0, None) else float("nan")
        p = (1 + int(np.sum(null_region_max >= s))) / (T + 1)
        row = {"region_id": int(r), "size": int(m.sum()), "score": s,
               "share_of_divergence": share, "p_value": p}
        if sil_by_point is not None:
            row["silhouette"] = float(np.nanmean(sil_by_point[m]))
        if marked is not None:
            row["n_marked_method5"] = int(marked[m].sum())
            row["frac_marked_method5"] = float(marked[m].mean())
        rows.append(row)
    if not rows:
        return pd.DataFrame(columns=["rank", "region_id", "size", "score",
                                     "share_of_divergence", "p_value", "silhouette"])
    df = pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)
    df.insert(0, "rank", np.arange(1, len(df) + 1))
    return df


def cluster_regions(XA, D, survivors):
    """HDBSCAN on the surviving rows of A; return region_id per A-row (-1 = none)."""
    region = np.full(len(XA), -1, dtype=int)
    idx = np.where(survivors)[0]
    if len(idx) < 5:
        if len(idx) > 0:
            region[idx] = 0
        return region
    region[idx] = _cluster_labels(XA[idx])
    return region


# --------------------------------------------------------------------------- #
# Validation                                                                   #
# --------------------------------------------------------------------------- #
def validate(marked, D, survivors):
    """marked: bool array over A (interval-filter anomalies). D: contributions.
    Report AUROC of D for marked-vs-rest, and precision/recall at the survivor
    threshold."""
    from sklearn.metrics import roc_auc_score
    out = {}
    if marked.sum() > 0 and marked.sum() < len(marked):
        out["auroc"] = float(roc_auc_score(marked.astype(int), D))
    else:
        out["auroc"] = float("nan")
    tp = int((survivors & marked).sum()); fp = int((survivors & ~marked).sum())
    fn = int((~survivors & marked).sum())
    out["precision"] = tp / max(tp + fp, 1)
    out["recall"] = tp / max(tp + fn, 1)
    out["n_marked"] = int(marked.sum()); out["n_survivors"] = int(survivors.sum())
    out["tp"] = tp; out["fp"] = fp; out["fn"] = fn
    return out


# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #
def _plot_null(out_dir, top_ps, fpr, ks_p, sig_alpha):
    """Histogram of top-region p-values under the null. Should be ~uniform; a spike
    near 0 would mean the test invents anomalies."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    ax[0].hist(top_ps, bins=min(21, max(5, len(top_ps)//2)),
               range=(0, 1), color="#1f77b4", alpha=0.8, edgecolor="k")
    ax[0].axvline(sig_alpha, color="#d62728", ls="--",
                  label=f"sig_alpha={sig_alpha}")
    ax[0].set_title(f"Null top-region p-values (should be uniform)\n"
                    f"FPR={fpr:.3f} (nominal {sig_alpha}), KS p={ks_p:.3f}")
    ax[0].set_xlabel("top-region p-value"); ax[0].set_ylabel("count"); ax[0].legend()
    # QQ vs uniform
    s = np.sort(top_ps); q = (np.arange(1, len(s)+1) - 0.5) / len(s)
    ax[1].plot(q, s, "o-", ms=4, color="#1f77b4")
    ax[1].plot([0, 1], [0, 1], "--", color="#bbb")
    ax[1].set_title("Q-Q plot vs uniform (on the diagonal = calibrated)")
    ax[1].set_xlabel("theoretical uniform quantile"); ax[1].set_ylabel("observed p-value")
    fig.suptitle("NULL TEST: is the region test calibrated on data with no real difference?",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(out_dir, "null_test.png"), dpi=150)
    plt.close(fig)


def _plot_null_placeholder():
    pass


def discover(XA, XB, perms, alpha, rng, smooth_k):
    """Full KL discovery on projected sets: contributions, permutation threshold +
    cluster-mass null, survivors, regions, top region p-value. Returns a dict."""
    D = kl_contributions(XA, XB, smooth_k=smooth_k)
    thr, null_region_max = permutation_null(XA, XB, perms, alpha, rng, smooth_k=smooth_k)
    survivors = D > thr
    region = cluster_regions(XA, D, survivors)
    # score + p of each region; top region p-value (the family-wise verdict)
    top_p = 1.0; scores = []
    for r in sorted(set(region[region >= 0].tolist())):
        s = float(D[region == r].sum())
        scores.append(s)
    if scores:
        smax = max(scores)
        top_p = (1 + int(np.sum(null_region_max >= smax))) / (len(null_region_max) + 1)
    n_reg = len(scores)
    return {"D": D, "thr": thr, "survivors": survivors, "region": region,
            "n_regions": n_reg, "top_p": top_p, "null_region_max": null_region_max}


def run_null_test(embeddings, pca_dim, perms, alpha, sig_alpha, smooth_k, n_trials, seed):
    """SPECIFICITY test. Repeatedly split the SAME homogeneous pool of embeddings
    randomly into A/B (no real difference exists) and run discovery. Under a valid
    test, the top-region p-value should be UNIFORM on [0,1] and the fraction of
    trials with a 'significant' region (p<=sig_alpha) should be ~sig_alpha, not more.
    Returns (top_p_values, false_positive_rate, ks_p_uniform)."""
    rng = np.random.default_rng(seed)
    E = embeddings
    n = len(E); nA = n // 2
    top_ps = []
    for t in range(n_trials):
        perm = rng.permutation(n)
        P = project(E, pca_dim)                       # project the pooled set
        XA, XB = P[perm[:nA]], P[perm[nA:]]
        res = discover(XA, XB, perms, alpha, np.random.default_rng(seed + 100 + t), smooth_k)
        top_ps.append(res["top_p"])
        _log(f"  null trial {t+1}/{n_trials}: {res['n_regions']} regions, top-p={res['top_p']:.3f}")
    top_ps = np.array(top_ps)
    fpr = float(np.mean(top_ps <= sig_alpha))
    # KS test of uniformity of the top-region p-values
    try:
        from scipy.stats import kstest
        ks_p = float(kstest(top_ps, "uniform").pvalue)
    except Exception:
        ks_p = float("nan")
    return top_ps, fpr, ks_p


def _filter_cache_key(args, path, channels):
    """Hash of everything that determines the Method-5 filtering result (NOT --reg,
    which only affects the embedding afterwards)."""
    import hashlib
    key = "|".join(str(x) for x in [
        os.path.abspath(path), args.bin, ",".join(channels),
        args.min_bins, args.min_packets, args.ref_frac, args.seed,
        args.holdout, args.band_lo, args.band_hi, args.norm_thr,
        getattr(args, "max_flows", 0)])
    return hashlib.md5(key.encode()).hexdigest()


def _filter_cache_path(args, fkey):
    d = os.path.join(args.root, "results", "anomaly_kl", "_filter_cache")
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, f"{fkey}.pkl")


def _load_filter_cache(args, fkey):
    import pickle
    p = _filter_cache_path(args, fkey)
    if not os.path.exists(p):
        return None
    try:
        with open(p, "rb") as fh:
            return pickle.load(fh)
    except Exception:
        return None


def _save_filter_cache(args, fkey, setB, scores, out_per_ch_all, per_bin_scores_all):
    import pickle
    obj = {"score": {setB[k]["cid"]: float(scores[k]) for k in range(len(setB))},
           "out": {setB[k]["cid"]: out_per_ch_all[k] for k in range(len(setB))},
           "per_bin_score": {setB[k]["cid"]: per_bin_scores_all[k] for k in range(len(setB))}}
    try:
        with open(_filter_cache_path(args, fkey), "wb") as fh:
            pickle.dump(obj, fh)
    except Exception as e:
        _log(f"(could not write filter cache: {e})")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default=None)
    ap.add_argument("--only", default=None)
    ap.add_argument("--features-dir", default=None)
    ap.add_argument("--root", default=DEFAULT_ROOT)
    ap.add_argument("--bin", default="5min")
    ap.add_argument("--channels", default="packet_count,mean_iat,mean_packet_size")
    ap.add_argument("--min-bins", type=int, default=48)
    ap.add_argument("--min-packets", type=int, default=10)
    ap.add_argument("--ref-frac", type=float, default=0.7)
    ap.add_argument("--holdout", type=int, default=24)
    ap.add_argument("--band-lo", type=float, default=0.1)
    ap.add_argument("--band-hi", type=float, default=0.9)
    ap.add_argument("--anomaly-frac", type=float, default=0.1,
                    help="fraction of B marked anomalous = the top-scoring flows by the "
                         "continuous Method-5 score (default 0.10). B keeps the rest, so B "
                         "stays large by construction. Flows with score 0 are never marked.")
    ap.add_argument("--band-bins", type=int, default=1, help="(unused with score mode; kept for compat)")
    ap.add_argument("--norm-thr", type=float, default=0.1,
                    help="Method 5 normalized-distance threshold: a bin counts only if "
                         "|truth-median|/y_range > this AND truth is outside the band (default 0.1)")
    ap.add_argument("--pca-dim", type=int, default=64)
    ap.add_argument("--perms", type=int, default=20)
    ap.add_argument("--alpha", type=float, default=0.01)
    ap.add_argument("--sig-alpha", type=float, default=0.05,
                    help="a region is SIGNIFICANT (KL's final anomaly call) if its "
                         "cluster-mass p-value <= this (default 0.05). Metrics and the "
                         "clustering plot use significant regions.")
    ap.add_argument("--smooth-k", type=int, default=5,
                    help="average each contribution with its k nearest A-neighbours "
                         "(paper Sec. 3.2; raises region signal-to-noise). 0 = off. "
                         "The permutation null is smoothed identically.")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-flows", type=int, default=0,
                    help="SMOKE TEST: use at most N flows (0 = all). A random subset is "
                         "taken BEFORE the split, so the whole pipeline runs on N flows -- "
                         "use e.g. 400 to validate the run in seconds instead of the full set.")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--reg", default="target",
                    help="flow embedding = which channel's REG token (all channels are "
                         "given to the encoder, so group attention links them): "
                         "'target' (channel 0, default) or a CHANNEL NAME / int index "
                         "to pick the target channel [METHOD 1], "
                         "'concat' (all channels' REG concatenated) [METHOD 2], "
                         "'mean' (average of channels' REG) [METHOD 3].")
    ap.add_argument("--null-test", action="store_true",
                    help="SPECIFICITY test: split the SAME pool randomly A/B (no real "
                         "difference) --null-trials times and report the false-positive "
                         "rate and p-value uniformity. Does NOT run the normal pipeline.")
    ap.add_argument("--null-trials", type=int, default=39,
                    help="number of random A/B splits for the null test (default 39)")
    ap.add_argument("--embed-batch", type=int, default=64,
                    help="flows per Chronos embed call (default 64; lower if embedding OOMs). "
                         "Flows are length-sorted so each batch pads to a similar length.")
    ap.add_argument("--embed-cap", type=int, default=0,
                    help="cap each series to its LAST N bins before embedding (0 = whole "
                         "series). Use e.g. 1024 if a few very long flows cause embed OOM.")
    ap.add_argument("--no-filter-cache", action="store_true",
                    help="recompute the Method-5 filtering (Chronos forecast) even if a "
                         "cached result exists for these filtering params. By default the "
                         "filtering result is cached and reused across --reg variants.")
    ap.add_argument("--split-type", choices=["method5", "random"], default="method5",
                    help="method5 = current behaviour (random --ref-frac split, Chronos "
                         "forecast marks anomalies of B, they are moved to A). random = "
                         "no Method-5 forecast at all: the filtered flows are split into "
                         "--n-splits folds and the experiment is run --n-splits times, "
                         "each fold taking its turn as target A while the other folds are "
                         "the reference B; outputs go to split1..splitN subfolders. "
                         "Chronos is then used ONLY for the embedding.")
    ap.add_argument("--n-splits", type=int, default=5,
                    help="number of folds for --split-type random (default 5)")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()
    channels = [c.strip() for c in args.channels.split(",") if c.strip()]
    # --reg can be: 'target'/'mean'/'concat', an integer index, or a CHANNEL NAME.
    if str(args.reg).lstrip("-").isdigit():
        args.reg = int(args.reg)                          # channel index
    elif args.reg in channels:
        args.reg = channels.index(args.reg)              # channel name -> index
        _log(f"--reg resolved to channel index {args.reg} ({channels[args.reg]})")
    elif args.reg not in ("target", "mean", "concat"):
        print(f"--reg '{args.reg}' is not target/mean/concat, an index, or one of "
              f"{channels}"); return

    pkt_col = "packet_count"
    import glob
    if args.csv:
        path = args.csv
    else:
        fdir = args.features_dir or os.path.join(args.root, "data", "parquet")
        cand = sorted(glob.glob(os.path.join(fdir, "*aggregated.parquet")))
        if not cand:
            cand = sorted(glob.glob(os.path.join(fdir, "*.parquet")))
        if args.only:
            cand = [p for p in cand if os.path.basename(p).startswith(args.only)
                    or args.only in os.path.basename(p)]
        if not cand:
            print(f"no features file found in {fdir}"); return
        path = cand[0]
    _log(f"features: {path}")
    
    # Handle naming parity with anomaly_set_difference
    channels = [c.replace("total_", "").replace("packet_count", "packet_count") for c in channels]
    if "total_packet_count" in args.channels:
        pkt_col = "packet_count"
        
    _log(f"channels: {channels}  (target={channels[0]}, covariates={channels[1:]})")

    flows = load_flows(path, channels, args.bin, args.min_bins, args.min_packets, pkt_col,
                       max_flows=args.max_flows, seed=args.seed)
    _log(f"kept {len(flows)} flows (span >= {args.min_bins} bins & >= {args.min_packets} pkts)")
    if len(flows) < 20:
        print("too few flows for a meaningful run."); return

    # 80/20 split
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(flows))
    nB = int(round(args.ref_frac * len(flows)))
    idxB = set(order[:nB].tolist());
    setB = [flows[i] for i in range(len(flows)) if i in idxB]
    setA = [flows[i] for i in range(len(flows)) if i not in idxB]
    _log(f"split: B(ref)={len(setB)}  A(target)={len(setA)}")

    ch = Chronos(device=args.device)

    # ---- RANDOM k-fold mode: no Method-5 forecast, no marks. The filtered flows
    # are split into --n-splits folds; the experiment runs once per fold with that
    # fold as target A and the other folds as reference B. Chronos only embeds. ----
    if args.split_type == "random":
        K = max(2, int(args.n_splits))
        folds = [order[k::K].tolist() for k in range(K)]
        base_out = _make_out_dir(args, path, flows, channels)
        _log(f"RANDOM split: {len(flows)} flows -> {K} folds of "
             f"{[len(f) for f in folds]} (Chronos used only for embedding)")
        for k in range(K):
            idxA = set(folds[k])
            setA_k = [f for j, f in enumerate(flows) if j in idxA]
            setB_k = [f for j, f in enumerate(flows) if j not in idxA]
            for f in setA_k + setB_k:
                f["marked"] = False
                f["anom_score"] = float("nan")
                f["out_per_ch"] = [0] * len(channels)
                f["per_bin_score"] = np.zeros(args.holdout, dtype=np.float32)
            _log(f"[split {k+1}/{K}] embedding A={len(setA_k)} B={len(setB_k)} "
                 f"(reg={_reg_tag(args.reg, channels)}) ...")
            embA_k = ch.embed([f["frame"] for f in setA_k], channels, args.embed_batch,
                              cap=args.embed_cap, reg_mode=args.reg)
            embB_k = ch.embed([f["frame"] for f in setB_k], channels, args.embed_batch,
                              cap=args.embed_cap, reg_mode=args.reg)
            split_dir = os.path.join(base_out, f"split{k+1}")
            os.makedirs(split_dir, exist_ok=True)
            ctx = dict(A=setA_k, B=setB_k, movedA=[],
                       marked=np.zeros(len(setA_k), dtype=bool),
                       anom_score=np.full(len(setA_k), np.nan),
                       channels=channels, flows=flows, setA=setA_k, setB=setB_k,
                       args=args, path=path,
                       rng=np.random.default_rng(args.seed + 100 + k))
            analyze_and_write(embA_k, embB_k, split_dir,
                              tag=f"split{k+1}/{K} reg={_reg_tag(args.reg, channels)}", **ctx)
        _log(f"all {K} splits done -> {base_out}")
        return


    # ---- filter B with forecast intervals (Method 5). This forecast is identical
    # across runs that share the filtering params (only --reg differs), so cache the
    # per-flow anomaly SCORE keyed on those params and skip the forecast on a hit. ----
    fkey = _filter_cache_key(args, path, channels)
    cache = _load_filter_cache(args, fkey) if not args.no_filter_cache else None
    if cache is not None and "per_bin_score" in cache:
        _log(f"filtering: CACHE HIT ({fkey[:12]}) -> skipping Chronos forecast")
        scores = np.array([cache["score"].get(f["cid"], 0.0) for f in setB], dtype=np.float64)
        out_per_ch_all = [cache["out"].get(f["cid"], [0] * len(channels)) for f in setB]
        per_bin_scores_all = [cache["per_bin_score"].get(f["cid"], np.zeros(args.holdout, dtype=np.float32)) for f in setB]
        band_by_cid = {}                                   # bands not cached (plots skip overlay)
    else:
        _log("filtering B with Chronos intervals ...")
        framesB = [f["frame"] for f in setB]
        lo, med, hi = ch.forecast_band(framesB, channels, args.holdout,
                                       args.band_lo, args.band_hi, args.batch_size)
        scores = np.zeros(len(setB), dtype=np.float64)
        out_per_ch_all = []
        per_bin_scores_all = []
        band_by_cid = {}
        CONST_EPS = 1e-6
        for k, f in enumerate(setB):
            fr = f["frame"]; T = fr.shape[0]
            actual = fr[T - args.holdout:, :]
            yr = fr.max(axis=0) - fr.min(axis=0)
            const_ch = np.zeros(fr.shape[1], dtype=bool)
            for ci in range(fr.shape[1]):
                nz = fr[:, ci][np.abs(fr[:, ci]) > CONST_EPS]
                if nz.size == 0 or (nz.max() - nz.min()) <= CONST_EPS:
                    const_ch[ci] = True
            diff = np.abs(actual - med[k])
            norm = diff / (yr[None, :] + 1e-5)
            m0 = (actual > hi[k] + 1e-2) | (actual < lo[k] - 1e-2)
            valid = m0 & (diff >= 1e-2) & (~const_ch[None, :])
            excess = np.where(valid, np.maximum(0.0, norm - args.norm_thr), 0.0)
            scores[k] = float(excess.sum())
            out_per_ch_all.append((valid & (norm > args.norm_thr)).sum(axis=0).tolist())
            per_bin_scores_all.append(excess.sum(axis=1).astype(np.float32))
            band_by_cid[f["cid"]] = (lo[k], med[k], hi[k])
        if not args.no_filter_cache:
            _save_filter_cache(args, fkey, setB, scores, out_per_ch_all, per_bin_scores_all)
            _log(f"filtering: cached result ({fkey[:12]})")

    # mark the top --anomaly-frac by score as anomalies (moved to A); rest stay in B.
    n_mark = int(round(args.anomaly_frac * len(setB)))
    order = np.argsort(-scores)
    mark_idx = set(int(i) for i in order[:n_mark] if scores[i] > 0)

    keepB, movedA = [], []
    for k, f in enumerate(setB):
        rec = dict(f)
        rec["anom_score"] = float(scores[k])
        rec["out_per_ch"] = out_per_ch_all[k]
        rec["per_bin_score"] = per_bin_scores_all[k]
        if k in mark_idx:
            rec["marked"] = True
            if f["cid"] in band_by_cid:
                rec["band"] = band_by_cid[f["cid"]]        # only present when not cached
            movedA.append(rec)
        else:
            rec["marked"] = False
            keepB.append(rec)
    for f in setA:
        f["marked"] = False
        f["anom_score"] = float("nan")
        f["out_per_ch"] = [0] * len(channels)
        f["per_bin_score"] = np.zeros(args.holdout, dtype=np.float32)
    A = setA + movedA
    B = keepB
    _log(f"anomaly score: marked top {args.anomaly_frac:.0%} = {len(movedA)} flows "
         f"(score>0); B now {len(B)}, A now {len(A)} ({sum(f['marked'] for f in A)} marked)")
    if scores.max() > 0:
        _log(f"  score range: max {scores.max():.3g}, "
             f"median(nonzero) {np.median(scores[scores>0]):.3g}, "
             f"zeros {int((scores==0).sum())}/{len(scores)}")
    if len(B) < 10 or len(A) < 10:
        print("after filtering, a set is too small."); return

    # ---- embed holdout bins: ONE reduced vector per BIN ----
    L_context = 12
    _log(f"embedding holdout bins (L={L_context}, reg={_reg_tag(args.reg, channels)}) ...")
    embA = ch.embed_bins([f["frame"] for f in A], channels, holdout=args.holdout, L=L_context,
                         batch=args.embed_batch, reg_mode=args.reg)          # (NA * holdout, d')
    embB = ch.embed_bins([f["frame"] for f in B], channels, holdout=args.holdout, L=L_context,
                         batch=args.embed_batch, reg_mode=args.reg)          # (NB * holdout, d')
    _log(f"bin embeddings: A {embA.shape}  B {embB.shape}")

    # ---- Unroll A and marked labels into individual bins ----
    unrolled_A = []
    marked_bins = []
    anom_score_bins = []
    
    for f in A:
        pbs = f.get("per_bin_score", np.zeros(args.holdout, dtype=np.float32))
        for t in range(args.holdout):
            sc = float(pbs[t])
            # A bin is marked if the flow was marked AND this specific bin has score > 0
            is_m = bool(f["marked"] and sc > 0)
            
            rec = {
                "cid": f"{f['cid']}_t{t}",
                "n_real": 1,
                "n_pkts": 0,
                "frame": f["frame"], # Needed by plotting functions? Let's leave out unless needed.
                "marked": is_m,
                "anom_score": sc if f["marked"] else float("nan")
            }
            unrolled_A.append(rec)
            marked_bins.append(is_m)
            anom_score_bins.append(rec["anom_score"])
            
    marked = np.array(marked_bins, dtype=bool)
    anom_score = np.array(anom_score_bins, dtype=float)

    # ---- NULL TEST (specificity): homogeneous pool, random A/B, no real difference ----
    if args.null_test:
        _log(f">>> NULL TEST (reg={_reg_tag(args.reg, channels)}): {args.null_trials} random "
             f"A/B splits of the pooled {len(embA)+len(embB)} flows (no real difference) <<<")
        pool = np.vstack([embA, embB])
        top_ps, fpr, ks_p = run_null_test(
            pool, args.pca_dim, args.perms, args.alpha, args.sig_alpha,
            args.smooth_k, args.null_trials, args.seed)
        _log(f"NULL RESULT: false-positive rate = {fpr:.3f} "
             f"(nominal {args.sig_alpha}); {int(np.sum(top_ps<=args.sig_alpha))}/{args.null_trials} "
             f"trials 'significant'")
        _log(f"  p-value uniformity (KS test): p={ks_p:.3f} "
             f"({'uniform -> well calibrated' if ks_p>0.05 else 'NOT uniform -> miscalibrated'})")
        # write a small report + histogram
        _fn = os.path.splitext(os.path.basename(path))[0]
        _tag = ("host_" + _fn[len("host_"):][:4]) if _fn.startswith("host_") else _fn[:4]
        nd = args.out_dir or os.path.join(args.root, "results", "anomaly_kl",
                                          f"{_tag}_NULLTEST")
        os.makedirs(nd, exist_ok=True)
        pd.DataFrame({"trial": range(1, len(top_ps)+1), "top_region_p": top_ps}).to_csv(
            os.path.join(nd, "null_pvalues.csv"), index=False)
        _plot_null(nd, top_ps, fpr, ks_p, args.sig_alpha)
        with open(os.path.join(nd, "null_summary.txt"), "w") as fh:
            fh.write(f"NULL TEST ({args.null_trials} random A/B splits of {len(pool)} flows)\n")
            fh.write(f"false-positive rate: {fpr:.4f} (nominal {args.sig_alpha})\n")
            fh.write(f"KS uniformity p-value: {ks_p:.4f} "
                     f"({'PASS uniform' if ks_p>0.05 else 'FAIL not uniform'})\n")
            fh.write(f"interpretation: FPR near {args.sig_alpha} and KS p>0.05 mean the "
                     f"permutation test is well calibrated (does not invent anomalies).\n")
        _log(f"wrote {nd}/null_summary.txt, null_pvalues.csv, null_test.png")
        return

    # ---- reduce per-channel REGs and run the analysis ----
    base_out = _make_out_dir(args, path, flows, channels)
    ctx = dict(A=unrolled_A, B=B, movedA=movedA, marked=marked, anom_score=anom_score,
               channels=channels, flows=flows, setA=setA, setB=setB, args=args,
               path=path, rng=rng)

    # one analysis on the chosen representation. For per-channel analysis, run this
    # script once per channel with --reg <channel_name> (same multivariate embed,
    # group attention on all channels, target = that channel).
    _log(f"reg = {_reg_tag(args.reg, channels)}  -> embedding dim {embA.shape[1]}")
    analyze_and_write(embA, embB, base_out, tag=f"reg={_reg_tag(args.reg, channels)}", **ctx)


def _reg_tag(reg, channels):
    """Human-readable name of the reg mode for filenames/logs: a channel index ->
    that channel's name; 'target' -> the first channel's name; else the keyword."""
    if isinstance(reg, int):
        return channels[reg] if 0 <= reg < len(channels) else f"ch{reg}"
    if reg == "target":
        return channels[0] if channels else "target"
    return reg                                            # 'mean' / 'concat'


def _make_out_dir(args, path, flows, channels):
    if args.out_dir:
        d = args.out_dir
    else:
        base = os.path.join(args.root, "results", "anomaly_kl")
        fname = os.path.splitext(os.path.basename(path))[0]
        # per-IP (host) feature files are named host_<hash>; carry that into the
        # result dir name so host runs are clearly separated from 5-tuple runs.
        if fname.startswith("host_"):
            tag = "host_" + fname[len("host_"):][:4]
        else:
            tag = fname[:4]
        stem = (f"{tag}_f{len(channels)}"
                f"_b{args.min_bins}_p{args.min_packets}_ref{int(args.ref_frac*100)}"
                f"_af{int(args.anomaly_frac*100)}"
                f"_reg-{_reg_tag(args.reg, channels)}_seed{args.seed}")
        os.makedirs(base, exist_ok=True)
        run = 1
        while os.path.exists(os.path.join(base, f"{stem}_run{run}")):
            run += 1
        d = os.path.join(base, f"{stem}_run{run}")
    os.makedirs(d, exist_ok=True)
    _log(f"results -> {d}")
    return d


def analyze_and_write(embA, embB, out_dir, A, B, movedA, marked, anom_score,
                      channels, flows, setA, setB, args, path, rng, tag=""):
    """Project -> KL contributions -> permutation calibration -> regions -> ranking
    -> validation -> write scores/regions/summary + all plots, into out_dir. Used
    once (single reg mode) or once per channel (per-channel mode)."""
    allE = np.vstack([embA, embB])
    P = project(allE, args.pca_dim)
    XA, XB = P[:len(embA)], P[len(embA):]
    _log(f"  [{tag}] projected to {P.shape[1]} dims; computing D_i ...")
    D = kl_contributions(XA, XB, smooth_k=args.smooth_k)
    D_raw = kl_contributions(XA, XB, smooth_k=0)
    dkl = float(D_raw.sum())

    thr, null_region_max = permutation_null(XA, XB, args.perms, args.alpha, rng, smooth_k=args.smooth_k)
    survivors = D > thr
    region = cluster_regions(XA, D, survivors)
    n_regions = len(set(region[region >= 0].tolist()))
    reg_df = rank_regions(region, D, D_raw, dkl, null_region_max, marked=marked, XA=XA)
    n_sig = int((reg_df["p_value"] <= args.sig_alpha).sum()) if len(reg_df) else 0
    sig_ids = set(reg_df.loc[reg_df["p_value"] <= args.sig_alpha, "region_id"].tolist()) \
        if len(reg_df) else set()
    sig_anomaly = np.array([r in sig_ids for r in region], dtype=bool)
    _log(f"  [{tag}] survivors {int(survivors.sum())}/{len(XA)}; regions {n_regions} "
         f"({n_sig} significant); D_KL={dkl:.4g}")

    val = validate(marked, D, sig_anomaly)
    have = ~np.isnan(anom_score)
    if have.sum() >= 5 and np.unique(anom_score[have]).size > 1:
        from scipy.stats import spearmanr
        val["spearman"] = float(spearmanr(anom_score[have], D[have]).correlation)
    else:
        val["spearman"] = float("nan")
    _log(f"  [{tag}] AUROC={val['auroc']:.3f} precision={val['precision']:.3f} "
         f"recall={val['recall']:.3f} spearman={val['spearman']:.3f}")

    pd.DataFrame({
        "cid": [f["cid"] for f in A], "set": "A",
        "marked_anomaly": marked, "D_i": D, "chronos_anom_score": anom_score,
        "survivor": survivors, "region_id": region, "sig_anomaly": sig_anomaly,
        "n_real": [f["n_real"] for f in A], "n_pkts": [f["n_pkts"] for f in A],
    }).sort_values("D_i", ascending=False).to_csv(os.path.join(out_dir, "scores.csv"), index=False)
    reg_df.to_csv(os.path.join(out_dir, "regions.csv"), index=False)
    with open(os.path.join(out_dir, "summary.txt"), "w") as fh:
        fh.write(f"tag: {tag}\nfeatures: {path}\nchannels: {channels}\n")
        fh.write(f"kept flows: {len(flows)}; split B={len(setB)} A={len(setA)}; "
                 f"anomaly-frac {args.anomaly_frac}; final A={len(A)} bins (marked {marked.sum()}) B={len(B)} flows\n")
        fh.write(f"reg={_reg_tag(args.reg, channels)} pca-dim {P.shape[1]} "
                 f"perms {args.perms} alpha {args.alpha} smooth-k {args.smooth_k}\n")
        fh.write(f"survivors {int(survivors.sum())}, regions {n_regions} "
                 f"({n_sig} significant p<={args.sig_alpha}), D_KL={dkl:.6g}\n")
        fh.write(f"VALIDATION AUROC={val['auroc']:.4f} precision={val['precision']:.4f} "
                 f"recall={val['recall']:.4f} spearman={val['spearman']:.4f}\n")
        cols = [c for c in ["rank", "region_id", "size", "score", "share_of_divergence",
                            "p_value", "silhouette", "frac_marked_method5"] if c in reg_df.columns]
        fh.write("\nTOP 15 REGIONS:\n" + reg_df[cols].head(15).to_string(index=False) + "\n")

    _make_plots(out_dir, D, marked, XA, survivors, sig_anomaly, region, sig_ids)
    _plot_excluded(out_dir, movedA, channels, args.holdout, args.band_lo, args.band_hi)
    _plot_clustering(out_dir, XA, D, survivors, region, marked, sig_ids, reg_df)
    _plot_kl_anomalies(out_dir, A, region, D, reg_df, sig_ids, channels, args.holdout)
    _log(f"  [{tag}] wrote outputs to {out_dir}")


def _plot_kl_anomalies(out_dir, A, region, D, reg_df, sig_ids, channels, holdout,
                       per_region=6):
    """(Disabled for bin-level analysis) Plot the actual time series of the flows KL flags..."""
    _log("Skipping kl_anomaly_flows.png (incompatible with bin-level analysis).")
    return



def _plot_clustering(out_dir, XA, D, survivors, region, marked, sig_ids=None, reg_df=None):
    """Two side-by-side 2-D views of A in embedding space:
      LEFT  = anomalies found by the KL method (survivors, coloured by region)
      RIGHT = anomalies found by Chronos + Method 5 (the interval-marked flows)
    So you can eyeball how the two detectors agree in the same space."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    # a 2-D layout: PCA is already applied; use first 2 dims (or t-SNE if wanted)
    fig, ax = plt.subplots(1, 2, figsize=(16, 6.6))

    # LEFT: KL method -- highlight the SIGNIFICANT regions (p<=sig_alpha) in
    # distinct colours; non-significant survivors are muted grey, background
    # light grey. These significant regions are the method's real anomalies.
    xy = XA[:, :2]
    sig = set(sig_ids) if sig_ids else set()
    # p-value lookup for the legend
    pmap = {}
    if reg_df is not None and len(reg_df):
        pmap = dict(zip(reg_df["region_id"], reg_df["p_value"]))
    ax[0].scatter(xy[~survivors, 0], xy[~survivors, 1], s=6, c="#eeeeee",
                  label="not anomalous", zorder=1)
    nonsig = survivors & ~np.isin(region, list(sig)) if sig else survivors
    if nonsig.any():
        ax[0].scatter(xy[nonsig, 0], xy[nonsig, 1], s=10, c="#bbbbbb",
                      edgecolors="none", alpha=0.6, zorder=2,
                      label=f"survivors, non-significant (n={int(nonsig.sum())})")
    # significant regions, sorted by size, distinct colours
    sig_sorted = sorted(sig, key=lambda r: -(int((region == r).sum())))
    cmap = plt.get_cmap("tab10")
    for gi, r in enumerate(sig_sorted):
        m = region == r
        pv = pmap.get(r, float("nan"))
        ax[0].scatter(xy[m, 0], xy[m, 1], s=42, c=[cmap(gi % 10)],
                      edgecolors="k", linewidths=0.5, zorder=5,
                      label=f"region {r}  (n={int(m.sum())}, p={pv:.3f})")
    ax[0].set_title(f"KL method: {len(sig)} SIGNIFICANT region(s) "
                    f"(p<=sig_alpha) of {len(set(region[region>=0]))} total")
    ax[0].set_xlabel("PC1"); ax[0].set_ylabel("PC2")
    ax[0].legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.005, 1.0),
                 framealpha=0.9, borderaxespad=0.)

    # RIGHT: Chronos + Method 5 -- marked flows in red
    ax[1].scatter(xy[~marked, 0], xy[~marked, 1], s=8, c="#e2e2e2",
                  label="normal", zorder=1)
    ax[1].scatter(xy[marked, 0], xy[marked, 1], s=16, c="#d62728",
                  edgecolors="none", alpha=0.55, zorder=3,
                  label=f"Method-5 anomaly (n={int(marked.sum())})")
    ax[1].set_title(f"Chronos + Method 5: {int(marked.sum())} anomalies")
    ax[1].set_xlabel("PC1"); ax[1].set_ylabel("PC2")
    ax[1].legend(fontsize=8, loc="upper left")

    fig.suptitle("A in Chronos-2 REG embedding space (PCA 2-D)  --  "
                 "left: KL significant regions   right: Method-5 labels", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(out_dir, "clustering.png"), bbox_inches="tight", dpi=150)
    plt.close(fig)
    _log(f"wrote {os.path.join(out_dir, 'clustering.png')}")


def _plot_excluded(out_dir, movedA, channels, holdout, q_lo, q_hi, max_pages=60):
    """One page per excluded (anomalous) flow: for each channel, history +
    forecast band + actual, with the channels that broke the band flagged red."""
    if not movedA:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    C = len(channels)
    ncols = min(C, 3); nrows = int(np.ceil(C / ncols))
    for i, f in enumerate(movedA[:max_pages]):
            fr = f["frame"]; T = fr.shape[0]; hshow = min(T - holdout, 40)
            lo, med, hi = f.get("band", (None, None, None))
            fig, axes = plt.subplots(nrows, ncols, figsize=(4.6 * ncols, 2.6 * nrows))
            axes = np.atleast_1d(axes).flatten()
            for ci, chn in enumerate(channels):
                ax = axes[ci]
                hx = np.arange(T - holdout - hshow, T - holdout)
                fx = np.arange(T - holdout, T)
                ax.plot(hx, fr[T - holdout - hshow:T - holdout, ci], color="#444", lw=0.7)
                if lo is not None:
                    ax.plot(fx, med[:, ci], color="blue", lw=0.9)
                    ax.fill_between(fx, lo[:, ci], hi[:, ci], color="blue", alpha=0.15)
                ax.plot(fx, fr[T - holdout:, ci], color="green", lw=0.9, ls="--")
                broke = f.get("out_per_ch", [0] * C)[ci]
                col = "#d62728" if broke else "#2ca02c"
                ax.set_title(f"{chn}  ({broke} bins out)", fontsize=8, color=col)
                ax.tick_params(labelsize=6)
            for j in range(C, len(axes)):
                axes[j].axis("off")
            fig.suptitle(f"EXCLUDED {f['cid']}   span={f.get('span','?')}  "
                         f"pkts={int(f.get('n_pkts',0))}   "
                         f"[grey=history blue=forecast+band green=actual; red title = broke band]",
                         fontsize=9)
            fig.tight_layout(rect=[0, 0, 1, 0.96])
            plt.savefig(os.path.join(out_dir, f"excluded_flow_{i}.png"), bbox_inches="tight", dpi=150)
            plt.close(fig)
    _log(f"wrote {min(len(movedA), max_pages)} excluded flows as PNGs")


def _make_plots(out_dir, D, marked, XA, survivors, sig_anomaly=None, region=None, sig_ids=None):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.8))

    # (1) D_i distribution: Method-5-marked vs rest (visualises AUROC)
    if marked.any() and (~marked).any():
        ax[0].hist(D[~marked], bins=50, alpha=0.6, label="rest", color="#888",
                   density=True)
        ax[0].hist(D[marked], bins=50, alpha=0.7, label="Method-5 anomaly",
                   color="#d62728", density=True)
        ax[0].set_title("D_i: Method-5-marked vs rest (density)")
        ax[0].legend(); ax[0].set_xlabel("D_i")

    # (2) ROC curve of D_i for Method-5 labels -> the AUROC value made visual
    if marked.any() and (~marked).any():
        from sklearn.metrics import roc_curve, roc_auc_score
        fpr, tpr, _ = roc_curve(marked.astype(int), D)
        auc = roc_auc_score(marked.astype(int), D)
        ax[1].plot(fpr, tpr, color="#1f77b4", lw=1.6, label=f"AUROC = {auc:.3f}")
        ax[1].plot([0, 1], [0, 1], color="#bbb", ls="--", lw=1, label="chance")
        ax[1].set_xlabel("false-positive rate"); ax[1].set_ylabel("true-positive rate")
        ax[1].set_title("ROC: D_i as detector of Method-5 anomalies")
        ax[1].legend(loc="lower right")
    else:
        ax[1].axis("off")

    # (3) D_i in SIGNIFICANT regions vs everything else -- shows the significant
    # regions really are the high-D_i tail (why they survive the test).
    if sig_anomaly is not None and sig_anomaly.any():
        ax[2].hist(D[~sig_anomaly], bins=50, alpha=0.55, label="not in a sig. region",
                   color="#888", density=True)
        ax[2].hist(D[sig_anomaly], bins=30, alpha=0.8,
                   label=f"in a significant region (n={int(sig_anomaly.sum())})",
                   color="#2ca02c", density=True)
        ax[2].set_title("D_i: significant regions vs rest")
        ax[2].legend(); ax[2].set_xlabel("D_i")
    else:
        ax[2].text(0.5, 0.5, "no significant regions", ha="center", va="center",
                   transform=ax[2].transAxes); ax[2].axis("off")

    fig.suptitle("Validation of the KL method", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(out_dir, "validation.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()