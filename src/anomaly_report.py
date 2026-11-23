#!/usr/bin/env python3
"""
anomaly_report.py  (src/)  -- called at the end of every anomaly_kl.py run.

Turns the significant regions of a run into reviewable artefacts. Everything is
written into the run directory:

  regions_map.png          (1) embedding map: significant regions coloured and labelled
                               (rank, service, size); right panel = anomalous flows
                               coloured by temporal pattern
  anomalous_flows.csv      (2) one row per flow in a SIGNIFICANT region: region, conn id,
                               protocol, endpoints, service, timing, temporal pattern,
                               per-channel stats, D_i, typicality, nearest normal flow ...
  regions/region_XX_R<id>.pdf (3) one PDF per significant region with ALL its flows:
                               p1 overview (summary, time-strip, overlay), p2 contrast vs
                               reference + nearest normal flows, p3.. every member flow
  region_summary.csv       per significant region: service/protocol mix, endpoints,
                               temporal-pattern mix, synchrony, contrast highlights
  region_contrast.csv      per region x feature: region vs reference (Cohen's d, KS)
  nearest_reference.csv    for each region, the k closest NORMAL (reference) flows
  review_sample.csv / review_key.csv / review_sample.pdf
                           blinded sample (flagged + random normal flows, shuffled) for
                           manual labelling -> precision estimate without labels
  embeddings_proj.npz      projected embeddings + D_i + regions (re-analysis w/o re-embed)
  run_config.json          every argument + counts (reproducibility)
  README_outputs.txt       description of all files / columns

Endpoint convention: a connection id is "ip:port<->ip:port" (canonical order). The
DESTINATION is taken as the endpoint with the LOWER port (the service side); the
other endpoint is the source. Per-IP (host) runs have no ports -> endpoints empty.
"""
import json
import math
import re
import os
from collections import Counter

import numpy as np
import pandas as pd

SERVICES = {20: "FTP-data", 21: "FTP", 22: "SSH", 23: "Telnet", 25: "SMTP", 53: "DNS",
            67: "DHCP", 68: "DHCP", 80: "HTTP", 110: "POP3", 123: "NTP", 137: "NetBIOS",
            138: "NetBIOS", 139: "NetBIOS", 143: "IMAP", 161: "SNMP", 443: "HTTPS/QUIC",
            445: "SMB", 465: "SMTPS", 514: "Syslog", 587: "SMTP", 853: "DNS-over-TLS",
            993: "IMAPS", 995: "POP3S", 1194: "OpenVPN", 1900: "SSDP", 3053: "svc:3053",
            3389: "RDP", 5222: "XMPP", 5223: "XMPP-TLS", 5228: "GCM/FCM", 5353: "mDNS",
            5355: "LLMNR", 6881: "BitTorrent", 7000: "cast", 8009: "cast", 8080: "HTTP-alt",
            8443: "HTTPS-alt", 51820: "WireGuard",
            # IoT-typical
            1883: "MQTT", 8883: "MQTT-TLS", 5683: "CoAP", 5684: "CoAP-DTLS", 554: "RTSP",
            8554: "RTSP-alt", 3478: "STUN/TURN", 5349: "STUN/TURN-TLS", 19302: "STUN(Google)",
            49152: "UPnP", 6667: "IRC", 1080: "SOCKS", 9100: "printer", 631: "IPP"}


def _log(msg):
    print(f"[report] {msg}", flush=True)


# --------------------------------------------------------------------------- #
# per-flow descriptors                                                         #
# --------------------------------------------------------------------------- #
def service_name(port):
    if port is None:
        return "host"
    if port in SERVICES:
        return SERVICES[port]
    return f"port:{port}" if port < 1024 else f"hi:{port}"


def endpoints(cid):
    """-> (src_ip, src_port, dst_ip, dst_port). dst = side with the lower port."""
    cid = str(cid)
    if "<->" not in cid:
        return cid, None, "", None                          # host-mode id = IP
    eps = []
    for p in cid.split("<->"):
        p = p.rsplit("/", 1)[0]
        ip, _, port = p.rpartition(":")
        try:
            eps.append((ip, int(port)))
        except ValueError:
            eps.append((p, None))
    (a_ip, a_p), (b_ip, b_p) = eps[0], eps[1]
    if a_p is not None and b_p is not None and a_p < b_p:
        return b_ip, b_p, a_ip, a_p                          # a is the service side
    return a_ip, a_p, b_ip, b_p


def _router_from_cid(cid):
    """Router prefix from ids written by new_data_feature_extractor.py (<router12>_<uid>_<ms>)."""
    m = re.match(r"^([0-9a-fA-F]{12})_\d+_\d+$", str(cid))
    return m.group(1) if m else ""


def flow_endpoints(f):
    """Endpoints of a flow: parsed from the id when it is "ip:port<->ip:port", else from
    the metadata columns of the features file (router logs: device + dst_port)."""
    cid = str(f["cid"])
    if "<->" in cid:
        return endpoints(cid)
    m = f.get("meta") or {}
    src = m.get("src_ip") or m.get("device") or m.get("oui") or cid
    sp = m.get("src_port")
    sp = int(sp) if sp is not None and sp == sp and int(sp) >= 0 else None
    dp = m.get("dst_port")
    dp = int(dp) if dp is not None and dp == dp and int(dp) > 0 else None
    return str(src), sp, str(m.get("dst_ip") or ""), dp


def temporal_profile(frame, t0, step, pc_idx):
    """Describe WHEN a flow is active. Activity = any channel non-zero in the bin
    (padding bins are all-zero). Returns a dict incl. a coarse pattern label."""
    act = np.abs(frame).sum(axis=1) > 0
    pc = frame[:, pc_idx]
    T = len(act)
    n_act = int(act.sum())
    idx = np.flatnonzero(act)
    # bursts = runs of consecutive active bins
    if n_act:
        breaks = np.flatnonzero(np.diff(idx) > 1)
        starts = np.r_[idx[0], idx[breaks + 1]]
        ends = np.r_[idx[breaks], idx[-1]]
    else:
        starts = ends = np.array([], dtype=int)
    lens = ends - starts + 1
    gaps = starts[1:] - ends[:-1] - 1 if len(starts) > 1 else np.array([])
    period_bins, strength = np.nan, np.nan
    x = act.astype(float) - act.mean()
    if T >= 8 and x.std() > 0:
        n = int(2 ** math.ceil(math.log2(2 * T)))
        f = np.fft.rfft(x, n)
        acf = np.fft.irfft(f * np.conj(f), n)[:T]
        acf = acf / acf[0]
        lo, hi = 2, max(3, T // 2)
        if hi > lo:
            lag = int(np.argmax(acf[lo:hi])) + lo
            period_bins, strength = float(lag), float(acf[lag])
    starts_iv = np.diff(starts) if len(starts) > 2 else np.array([])
    gap_cv = float(np.std(gaps) / np.mean(gaps)) if len(gaps) >= 2 and np.mean(gaps) > 0 else np.nan
    step_min = step.total_seconds() / 60.0
    frac = n_act / T if T else 0.0
    if frac >= 0.8:
        label = "continuous"
    elif len(starts) == 1:
        label = "single-burst"
    elif (strength == strength and strength >= 0.5 and len(starts) >= 3):
        label = f"periodic (~{period_bins * step_min:.0f} min)"
    elif len(starts_iv) >= 3 and np.std(starts_iv) / max(np.mean(starts_iv), 1e-9) < 0.25:
        label = f"periodic (~{np.median(starts_iv) * step_min:.0f} min)"
    elif frac < 0.3:
        label = "sporadic"
    else:
        label = "intermittent"
    peak = int(np.argmax(pc)) if T else 0
    hours = [(t0 + i * step).hour for i in idx] if n_act else []
    return {
        "start_time": t0, "end_time": t0 + (T - 1) * step,
        "duration_h": (T - 1) * step_min / 60.0, "span_bins": T, "active_bins": n_act,
        "active_frac": frac, "n_bursts": int(len(starts)),
        "max_burst_bins": int(lens.max()) if len(lens) else 0,
        "median_gap_bins": float(np.median(gaps)) if len(gaps) else np.nan,
        "gap_cv": gap_cv,
        "period_min": period_bins * step_min if period_bins == period_bins else np.nan,
        "period_strength": strength,
        "peak_time": t0 + peak * step, "peak_packets": float(pc[peak]) if T else 0.0,
        "dominant_hour": Counter(hours).most_common(1)[0][0] if hours else -1,
        "temporal_pattern": label,
        "_active_idx": idx,
    }


def channel_stats(frame, channels):
    act = np.abs(frame).sum(axis=1) > 0
    out = {}
    for c, ch in enumerate(channels):
        col = frame[:, c]
        out[f"{ch}_mean_active"] = float(col[act].mean()) if act.any() else 0.0
        out[f"{ch}_max"] = float(col.max()) if len(col) else 0.0
    return out


def _concentration(counter):
    tot = sum(counter.values())
    if tot <= 0 or len(counter) <= 1:
        return 1.0
    H = -sum((c / tot) * math.log(c / tot) for c in counter.values() if c > 0)
    return float(1.0 - H / math.log(len(counter)))


def _frac(s):
    """Fraction of 1s in a 0/1 metadata column; NaN when the column is absent/empty."""
    v = pd.to_numeric(pd.Series(s), errors="coerce")
    return float(v.mean()) if v.notna().any() else np.nan


def _mix(values, k=None):
    """All values (or the top k) sorted by frequency: "a: 12 (60%); b: 8 (40%)"."""
    c = Counter(v for v in values if v is not None and str(v) != "" and str(v) != "nan")
    tot = sum(c.values()) or 1
    return "; ".join(f"{v}: {n} ({100 * n / tot:.0f}%)" for v, n in c.most_common(k))


def _short(mix, k=5):
    """First k entries of a _mix string + '(+N more)' -- for the PDF header only."""
    parts = [p for p in str(mix).split("; ") if p]
    return "; ".join(parts[:k]) + (f"  (+{len(parts) - k} more, see region_composition.csv)"
                                    if len(parts) > k else "")


# --------------------------------------------------------------------------- #
# main entry                                                                   #
# --------------------------------------------------------------------------- #
def write_report(out_dir, A, B, XA, XB, D, region, sig_ids, reg_df, marked, anom_score,
                 channels, args, path):
    step = pd.Timedelta(pd.tseries.frequencies.to_offset(args.bin))
    pc_idx = channels.index("packet_count") if "packet_count" in channels else 0
    sig_ids = set(int(r) for r in sig_ids)
    _write_config(out_dir, args, path, channels, A, B, region, sig_ids)
    if not getattr(args, "no_save_embeddings", False):
        np.savez_compressed(os.path.join(out_dir, "embeddings_proj.npz"),
                            XA=XA.astype(np.float32), XB=XB.astype(np.float32),
                            cidA=np.array([f["cid"] for f in A]),
                            cidB=np.array([f["cid"] for f in B]),
                            D=D.astype(np.float32), region=region.astype(np.int32))
    _sigalpha_sweep(out_dir, XA, region, reg_df, channels, args)
    if not sig_ids:
        _log("no significant regions -> only config/embeddings/sweep written")
        _write_readme(out_dir, channels)
        return

    rank_of = {int(r): int(k) for r, k in zip(reg_df["region_id"], reg_df["rank"])}
    info = reg_df.set_index("region_id")
    order = sorted(sig_ids, key=lambda r: rank_of.get(r, 10 ** 6))
    members = {r: np.flatnonzero(region == r) for r in order}

    # per-point silhouette over clustered survivors (if affordable)
    psil = np.full(len(XA), np.nan)
    clustered = np.flatnonzero(region >= 0)
    if 3 <= len(clustered) <= 20000 and len(set(region[clustered])) >= 2:
        from sklearn.metrics import silhouette_samples
        psil[clustered] = silhouette_samples(XA[clustered], region[clustered])

    from sklearn.neighbors import NearestNeighbors
    nnB = NearestNeighbors(n_neighbors=1).fit(XB)

    # ---------- (2) anomalous_flows.csv ----------
    rows, prof_cache = [], {}
    for r in order:
        idx = members[r]
        cen = XA[idx].mean(0)
        cd = np.linalg.norm(XA[idx] - cen, axis=1)
        med = float(np.median(cd)) or 1.0
        dists, nbr = nnB.kneighbors(XA[idx])
        d_rank = (-D[idx]).argsort().argsort() + 1
        for j, i in enumerate(idx):
            f = A[i]
            prof = temporal_profile(f["frame"], f["t0"], step, pc_idx)
            prof_cache[i] = prof
            s_ip, s_p, d_ip, d_p = flow_endpoints(f)
            m = f.get("meta") or {}
            row = {
                "region_rank": rank_of.get(r), "region_id": r,
                "region_p_value": float(info.loc[r, "p_value"]),
                "region_share_of_divergence": float(info.loc[r, "share_of_divergence"]),
                "conn_id": f["cid"], "protocol": f.get("proto", ""),
                "router": m.get("router") or _router_from_cid(f["cid"]), "device": m.get("device", ""),
                "oui": m.get("oui", ""), "is_iot": m.get("is_iot", ""),
                "conn_direction": m.get("conn_direction", ""), "has_ech": m.get("has_ech", ""),
                "src_ip": s_ip, "src_port": s_p, "dst_ip": d_ip, "dst_port": d_p,
                "service": service_name(d_p),
                "n_packets": f["n_pkts"],
            }
            row.update({k: v for k, v in prof.items() if not k.startswith("_")})
            row.update(channel_stats(f["frame"], channels))
            row.update({
                "D_i": float(D[i]), "D_rank_in_region": int(d_rank[j]),
                "typicality": float(cd[j] / med),             # ~1 typical, >>1 region outlier
                "point_silhouette": float(psil[i]),
                "filter_marked": bool(marked[i]),
                "filter_score": float(anom_score[i]) if anom_score[i] == anom_score[i] else np.nan,
                "nearest_normal_cid": B[int(nbr[j, 0])]["cid"],
                "nearest_normal_dist": float(dists[j, 0]),
            })
            rows.append(row)
    flows_df = pd.DataFrame(rows).sort_values(["region_rank", "D_i"], ascending=[True, False])
    svc_by_region = flows_df.groupby("region_id")["service"].agg(lambda s: Counter(s).most_common(1)[0][0])
    flows_df.insert(4, "region_service", flows_df["region_id"].map(svc_by_region))
    flows_df.to_csv(os.path.join(out_dir, "anomalous_flows.csv"), index=False)
    _log(f"anomalous_flows.csv: {len(flows_df)} flows in {len(order)} significant regions")

    # ---------- contrast vs reference, nearest references ----------
    rng = np.random.default_rng(getattr(args, "seed", 0))
    bsel = rng.choice(len(B), min(len(B), 20000), replace=False)
    ref_feats = pd.DataFrame([_feature_vec(B[k], channels, step, pc_idx) for k in bsel])
    contrast_rows, near_rows, summary_rows = [], [], []
    nnBk = NearestNeighbors(n_neighbors=min(10, len(XB))).fit(XB)
    for r in order:
        idx = members[r]
        reg_feats = pd.DataFrame([_feature_vec(A[i], channels, step, pc_idx, prof_cache.get(i))
                                  for i in idx])
        contrast_rows += _contrast(r, reg_feats, ref_feats)
        dist, nb = nnBk.kneighbors(XA[idx].mean(0, keepdims=True))
        for k, (dd, bi) in enumerate(zip(dist[0], nb[0])):
            pb = temporal_profile(B[bi]["frame"], B[bi]["t0"], step, pc_idx)
            near_rows.append({"region_id": r, "region_rank": rank_of.get(r), "k": k + 1,
                              "_b_index": int(bi),
                              "ref_conn_id": B[bi]["cid"], "dist_to_region_centroid": float(dd),
                              "ref_temporal_pattern": pb["temporal_pattern"],
                              "ref_n_packets": B[bi]["n_pkts"]})
        summary_rows.append(_region_summary(r, rank_of, info, flows_df, A, idx, prof_cache, step,
                                            contrast_rows))
    contrast_df = pd.DataFrame(contrast_rows)
    contrast_df.to_csv(os.path.join(out_dir, "region_contrast.csv"), index=False)
    pd.DataFrame(near_rows).drop(columns=["_b_index"]).to_csv(
        os.path.join(out_dir, "nearest_reference.csv"), index=False)
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(out_dir, "region_summary.csv"), index=False)
    summary_df = _add_verdict_and_global(summary_df, flows_df, A, step, pc_idx)
    summary_df.to_csv(os.path.join(out_dir, "region_summary.csv"), index=False)
    _write_composition(out_dir, flows_df, rank_of)
    _write_profiles_txt(out_dir, summary_df)
    _plot_region_heatmap(out_dir, summary_df)
    _plot_Di_by_service(out_dir, D, region, order, flows_df, rank_of)
    _plot_region_similarity(out_dir, flows_df, order, rank_of)
    _log("region_summary.csv, region_contrast.csv, nearest_reference.csv written")

    # ---------- (1) region map ----------
    _plot_region_map(out_dir, XA, region, order, rank_of, svc_by_region, flows_df, A)

    # ---------- (3) one PDF per significant region with ALL its flows ----------
    if not getattr(args, "no_region_pdfs", False):
        rdir = os.path.join(out_dir, "regions")
        os.makedirs(rdir, exist_ok=True)
        cap = int(getattr(args, "max_flows_per_region", 0) or 0)
        for r in order:
            _region_pdf(rdir, r, rank_of, A, B, members[r], D, prof_cache, channels, step,
                        pc_idx, summary_df, contrast_df, near_rows, cap)
        _log(f"wrote {len(order)} region PDFs to {rdir}")

    # ---------- blinded review sample ----------
    _review_sample(out_dir, A, flows_df, region, sig_ids, channels, step, pc_idx,
                   int(getattr(args, "review_n", 30)), rng)
    if not getattr(args, "no_case_files", False):
        try:
            import case_files
            case_files.write_case_files(out_dir, A, B, flows_df, summary_df, contrast_df, near_rows,
                                        order, rank_of, channels, step, pc_idx, args, path)
        except Exception as e:
            _log(f"case files skipped: {type(e).__name__}: {e}")
    _write_readme(out_dir, channels)


# --------------------------------------------------------------------------- #
# helpers                                                                      #
# --------------------------------------------------------------------------- #
def _feature_vec(f, channels, step, pc_idx, prof=None):
    prof = prof or temporal_profile(f["frame"], f["t0"], step, pc_idx)
    v = channel_stats(f["frame"], channels)
    v.update({"active_frac": prof["active_frac"], "span_bins": prof["span_bins"],
              "n_bursts": prof["n_bursts"], "log_n_packets": math.log1p(f["n_pkts"])})
    return v


def _contrast(r, reg, ref):
    from scipy.stats import ks_2samp
    out = []
    for col in reg.columns:
        a, b = reg[col].to_numpy(float), ref[col].to_numpy(float)
        sp = math.sqrt((np.var(a) + np.var(b)) / 2) or 1e-12
        try:
            ks = ks_2samp(a, b)
            kst, ksp = float(ks.statistic), float(ks.pvalue)
        except Exception:
            kst, ksp = np.nan, np.nan
        out.append({"region_id": r, "feature": col,
                    "region_median": float(np.median(a)), "reference_median": float(np.median(b)),
                    "cohens_d": float((a.mean() - b.mean()) / sp),
                    "ks_stat": kst, "ks_p": ksp})
    return out


def _write_composition(out_dir, flows_df, rank_of):
    """region_composition.csv: EVERY value of each descriptor in each significant region,
    one row per (region, field, value) with the number and share of the region's flows."""
    f = flows_df.copy()
    f["destination"] = [f"{a}:{b}" if a else f"{p}/{b}" for a, b, p in
                        zip(f["dst_ip"], f["dst_port"], f["protocol"])]
    fields = ["service", "protocol", "destination", "src_ip", "router", "oui", "conn_direction",
              "is_iot", "has_ech", "temporal_pattern"]
    rows = []
    for rid, g in f.groupby("region_id"):
        n = len(g)
        for fld in fields:
            if fld not in g.columns:
                continue
            vc = g[fld].astype(str).replace({"": None, "nan": None, "None": None}).dropna().value_counts()
            for val, cnt in vc.items():
                rows.append({"region_rank": rank_of.get(rid), "region_id": rid,
                             "field": "source" if fld == "src_ip" else fld, "value": val,
                             "n_flows": int(cnt), "pct_of_region": round(100.0 * cnt / n, 1)})
    pd.DataFrame(rows).sort_values(["region_rank", "field", "n_flows"], ascending=[True, True, False]) \
        .to_csv(os.path.join(out_dir, "region_composition.csv"), index=False)



def _verdict(port_c, ip_c, thr=0.6):
    """Same rule as analyze_regions.py: a region is coherent when most of its flows share
    a service port or a source/destination."""
    if port_c == port_c and port_c >= thr:
        return "COHERENT (dominant service)"
    if ip_c == ip_c and ip_c >= thr:
        return "COHERENT (dominant source/destination)"
    return "mixed / possibly noise"


def _add_verdict_and_global(summary_df, flows_df, A, step, pc_idx):
    """Verdict + how each region compares with ALL flows of A (packets, duration)."""
    s = summary_df.copy()
    g_pk = float(np.median([f["n_pkts"] for f in A]))
    g_span = float(np.median([len(f["frame"]) for f in A]))
    spans = flows_df.groupby("region_id")["span_bins"].median()
    s["verdict"] = [_verdict(a, b) for a, b in zip(s["port_concentration"], s["ip_concentration"])]
    s["global_median_n_packets"] = g_pk
    s["packets_vs_global"] = s["median_n_packets"] / (g_pk or 1)
    s["median_span_bins"] = s["region_id"].map(spans)
    s["global_median_span_bins"] = g_span
    s["span_vs_global"] = s["median_span_bins"] / (g_span or 1)
    return s


def _write_profiles_txt(out_dir, s):
    """Human-readable profile of every significant region with ALL values (no top-N)."""
    def g(row, k, fmt="{}"):
        v = row.get(k, "")
        return fmt.format(v) if v == v and v != "" else "-"
    lines = [f"PROFILES OF THE {len(s)} SIGNIFICANT REGIONS (all values, sorted by frequency)",
             "verdict: COHERENT = >=60% concentration on one service or one source/destination", ""]
    for _, row in s.sort_values("region_rank").iterrows():
        lines += [
            "=" * 90,
            f"REGION #{int(row.region_rank)} (id {int(row.region_id)})   flows={int(row.n_flows)}   "
            f"p={row.p_value:.3f}   share of divergence={row.share_of_divergence:.1%}   "
            f"silhouette={g(row, 'silhouette', '{:.2f}')}",
            f"  verdict: {row.verdict}   (port concentration {g(row, 'port_concentration', '{:.2f}')}, "
            f"ip concentration {g(row, 'ip_concentration', '{:.2f}')})",
            f"  vs all flows: packets {row.median_n_packets:.0f} vs {row.global_median_n_packets:.0f} "
            f"(x{row.packets_vs_global:.1f}),  span {row.median_span_bins:.0f} vs "
            f"{row.global_median_span_bins:.0f} bins (x{row.span_vs_global:.1f})",
            f"  seen {row.first_seen} -> {row.last_seen};  filter-marked {row.frac_filter_marked:.0%}"
            + (f";  IoT {row.frac_iot:.0%};  ECH {row.frac_ech:.0%}" if row.get("frac_iot") == row.get("frac_iot") else ""),
            f"  synchrony {g(row, 'sync_jaccard', '{:.2f}')};  peak co-activity "
            f"{g(row, 'peak_coactivity_frac', '{:.0%}')} at {g(row, 'peak_coactivity_time')}",
            f"  most different from reference: {g(row, 'most_different_features')}",
            f"  services:      {g(row, 'service_mix')}",
            f"  protocols:     {g(row, 'protocol_mix')}",
            f"  destinations:  {g(row, 'top_destinations')}",
            f"  sources:       {g(row, 'top_sources')}",
        ]
        for k, lab in (("top_routers", "routers"), ("oui_mix", "OUI"), ("direction_mix", "direction")):
            if str(row.get(k, "")) not in ("", "nan"):
                lines.append(f"  {lab + ':':14s} {row[k]}")
        lines.append(f"  temporal:      {g(row, 'temporal_pattern_mix')}")
        lines.append("")
    with open(os.path.join(out_dir, "region_profiles.txt"), "w") as fh:
        fh.write("\n".join(lines) + "\n")


def _plot_region_heatmap(out_dir, s):
    """Region x property heatmap: colour = value normalised per column, number = raw value."""
    cols = [c for c in ["n_flows", "share_of_divergence", "p_value", "silhouette",
                        "port_concentration", "ip_concentration", "sync_jaccard",
                        "packets_vs_global", "span_vs_global", "frac_filter_marked",
                        "frac_iot", "frac_ech"] if c in s.columns and s[c].notna().any()]
    if not cols or len(s) == 0:
        return
    plt = _mpl()
    M = s.sort_values("region_rank")
    data = M[cols].to_numpy(dtype=float)
    lo, hi = np.nanmin(data, 0), np.nanmax(data, 0)
    dn = (data - lo) / np.where(hi > lo, hi - lo, 1.0)
    fig, ax = plt.subplots(figsize=(1.15 * len(cols) + 3.5, 0.42 * len(M) + 2))
    im = ax.imshow(np.nan_to_num(dn, nan=0.0), aspect="auto", cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols, rotation=40, ha="right", fontsize=8)
    svc = M["service_mix"].astype(str).str.split(":").str[0]
    ax.set_yticks(range(len(M)))
    ax.set_yticklabels([f"#{int(a)} R{int(b)} {c}" for a, b, c in zip(M.region_rank, M.region_id, svc)],
                       fontsize=8)
    for i in range(len(M)):
        for j in range(len(cols)):
            v = data[i, j]
            if v == v:
                ax.text(j, i, f"{v:.2g}", ha="center", va="center", fontsize=6.5,
                        color="white" if dn[i, j] < 0.6 else "black")
    ax.set_title("significant regions x properties (colour = per-column normalised, text = value)")
    fig.colorbar(im, ax=ax, fraction=0.025, label="relative (per column)")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "region_heatmap.png"), dpi=140, bbox_inches="tight")
    plt.close(fig)


def _plot_Di_by_service(out_dir, D, region, order, flows_df, rank_of):
    """D_i of the flows of each significant region, grouped by the region's dominant
    service, against all other flows of A."""
    plt = _mpl()
    svc_of = flows_df.groupby("region_id")["service"].agg(lambda x: Counter(x).most_common(1)[0][0])
    fig, ax = plt.subplots(figsize=(11, 5))
    rest = ~np.isin(region, order)
    ax.hist(D[rest], bins=60, color="#cccccc", alpha=0.7, density=True, label=f"other flows (n={int(rest.sum())})")
    groups = {}
    for r in order:
        groups.setdefault(svc_of.get(r, "?"), []).extend(D[region == r].tolist())
    cmap = plt.get_cmap("tab10")
    for gi, (svc, vals) in enumerate(sorted(groups.items(), key=lambda kv: -len(kv[1]))):
        ax.hist(vals, bins=30, color=cmap(gi % 10), alpha=0.6, density=True, label=f"{svc} (n={len(vals)})")
    ax.set_xlabel("D_i (local KL contribution)"); ax.set_ylabel("density")
    ax.set_title("D_i by dominant service of the significant region (right = more specific to the target set)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "Di_by_service.png"), dpi=140, bbox_inches="tight")
    plt.close(fig)


def _plot_region_similarity(out_dir, flows_df, order, rank_of):
    """Jaccard overlap of the endpoint sets (sources + destinations) between significant
    regions. High overlap = probably the same behaviour split into several regions."""
    if len(order) < 2:
        return
    plt = _mpl()
    f = flows_df.copy()
    f["dest"] = [f"{a}:{b}" if a else f"{p}/{b}" for a, b, p in zip(f["dst_ip"], f["dst_port"], f["protocol"])]
    sets = {r: set(f.loc[f.region_id == r, "src_ip"].astype(str)) | set(f.loc[f.region_id == r, "dest"])
            for r in order}
    n = len(order)
    J = np.array([[len(sets[a] & sets[b]) / max(len(sets[a] | sets[b]), 1) for b in order] for a in order])
    svc = f.groupby("region_id")["service"].agg(lambda x: Counter(x).most_common(1)[0][0])
    labs = [f"#{rank_of.get(r)} R{r} {svc.get(r, '')}" for r in order]
    fig, ax = plt.subplots(figsize=(0.55 * n + 4, 0.5 * n + 3))
    im = ax.imshow(J, cmap="magma", vmin=0, vmax=1)
    ax.set_xticks(range(n)); ax.set_xticklabels(labs, rotation=60, ha="right", fontsize=7)
    ax.set_yticks(range(n)); ax.set_yticklabels(labs, fontsize=7)
    for i in range(n):
        for j in range(n):
            if i != j and J[i, j] >= 0.05:
                ax.text(j, i, f"{J[i, j]:.2f}", ha="center", va="center", fontsize=6,
                        color="white" if J[i, j] < 0.6 else "black")
    ax.set_title("overlap of endpoints between regions (Jaccard)\nhigh = same sources/destinations, "
                 "likely one behaviour split in several regions")
    fig.colorbar(im, ax=ax, fraction=0.04)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "region_similarity.png"), dpi=140, bbox_inches="tight")
    plt.close(fig)

def _region_summary(r, rank_of, info, flows_df, A, idx, prof_cache, step, contrast_rows):
    sub = flows_df[flows_df.region_id == r]
    ports = Counter(p for p in sub["dst_port"] if p == p and p is not None)
    ips = Counter(list(sub["src_ip"]) + [x for x in sub["dst_ip"] if x])
    sync, peak_frac, peak_time = _synchrony([A[i] for i in idx],
                                            [prof_cache[i]["_active_idx"] for i in idx], step)
    con = [c for c in contrast_rows if c["region_id"] == r]
    top = sorted(con, key=lambda c: -abs(c["cohens_d"]))[:3]
    return {
        "region_rank": rank_of.get(r), "region_id": r, "n_flows": len(idx),
        "p_value": float(info.loc[r, "p_value"]),
        "share_of_divergence": float(info.loc[r, "share_of_divergence"]),
        "silhouette": float(info.loc[r, "silhouette"]) if "silhouette" in info.columns else np.nan,
        "service_mix": _mix(sub["service"]), "protocol_mix": _mix(sub["protocol"]),
        "top_destinations": _mix([f"{a}:{b}" if a else f"{p}/{b}" for a, b, p in
                                  zip(sub["dst_ip"], sub["dst_port"], sub["protocol"])]),
        "top_routers": _mix(sub["router"]) if (sub["router"] != "").any() else "",
        "top_sources": _mix(sub["src_ip"]),
        "port_concentration": _concentration(ports) if ports else np.nan,
        "ip_concentration": _concentration(ips),
        "temporal_pattern_mix": _mix(sub["temporal_pattern"]),
        "sync_jaccard": sync,
        "peak_coactivity_frac": peak_frac, "peak_coactivity_time": peak_time,
        "first_seen": sub["start_time"].min(), "last_seen": sub["end_time"].max(),
        "median_n_packets": float(sub["n_packets"].median()),
        "frac_filter_marked": float(sub["filter_marked"].mean()),
        "oui_mix": _mix(sub["oui"]) if (sub["oui"].astype(str) != "").any() else "",
        "frac_iot": _frac(sub["is_iot"]), "frac_ech": _frac(sub["has_ech"]),
        "direction_mix": _mix(sub["conn_direction"]) if (sub["conn_direction"].astype(str) != "").any() else "",
        "most_different_features": "; ".join(f"{c['feature']} (d={c['cohens_d']:+.2f})" for c in top),
    }


def _synchrony(flows, active_idx, step, max_pairs_members=150):
    """Do the region's flows act at the SAME absolute moments? sync_jaccard = mean
    pairwise Jaccard of absolute active-bin sets; peak_coactivity = largest fraction of
    members active in one bin (and when)."""
    if not flows:
        return np.nan, np.nan, None
    g0 = min(f["t0"] for f in flows)
    sets = []
    for f, ai in zip(flows, active_idx):
        k0 = int(round((f["t0"] - g0) / step))
        sets.append(set((ai + k0).tolist()))
    cnt = Counter(b for s in sets for b in s)
    if cnt:
        b, c = cnt.most_common(1)[0]
        peak_frac, peak_time = c / len(sets), g0 + b * step
    else:
        peak_frac, peak_time = np.nan, None
    sel = sets if len(sets) <= max_pairs_members else \
        [sets[i] for i in np.random.default_rng(0).choice(len(sets), max_pairs_members, replace=False)]
    js = []
    for i in range(len(sel)):
        for j in range(i + 1, len(sel)):
            u = len(sel[i] | sel[j])
            if u:
                js.append(len(sel[i] & sel[j]) / u)
    return (float(np.mean(js)) if js else np.nan), peak_frac, peak_time


def _reg_label(args, channels):
    reg = getattr(args, "reg", "target")
    if isinstance(reg, int):
        return channels[reg] if 0 <= reg < len(channels) else f"ch{reg}"
    return channels[0] if reg == "target" and channels else str(reg)


def _sigalpha_sweep(out_dir, XA, region, reg_df, channels, args):
    """sig-alpha sweep for THIS run's representation (e.g. --reg concat): one panel per
    threshold, significant regions coloured. sig-alpha only selects which of the run's
    regions count as significant, so nothing is recomputed. A region keeps the same
    colour in every panel (colour = its rank), so you can follow it across thresholds."""
    raw = getattr(args, "sweep_sig_alphas", "0.01,0.025,0.05,0.1,0.15,0.2")
    alphas = sorted(float(x) for x in (raw if isinstance(raw, (list, tuple))
                                       else str(raw).replace(" ", "").split(",")) if x != "")
    if not alphas:
        return
    plt = _mpl()
    xy = XA[:, :2]
    label = _reg_label(args, channels)
    has = len(reg_df) > 0
    colour = {}
    if has:
        cmap = plt.get_cmap("tab20")
        for k, rid in enumerate(reg_df.sort_values("rank")["region_id"]):
            colour[int(rid)] = cmap(k % 20)
    fig, axes = plt.subplots(1, len(alphas), figsize=(3.4 * len(alphas), 3.6), squeeze=False)
    rows = []
    for j, sa in enumerate(alphas):
        ax = axes[0][j]
        ids = sorted(int(x) for x in reg_df.loc[reg_df["p_value"] <= sa, "region_id"]) if has else []
        in_sig = np.isin(region, ids) if ids else np.zeros(len(region), bool)
        ax.scatter(xy[~in_sig, 0], xy[~in_sig, 1], s=3, c="#e6e6e6", zorder=1, rasterized=True)
        for rid in ids:
            m = region == rid
            ax.scatter(xy[m, 0], xy[m, 1], s=9, color=colour.get(rid, "k"), zorder=3)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"sig-alpha = {sa}\n{len(ids)} reg / {int(in_sig.sum())} flows", fontsize=9)
        rows.append({"sig_alpha": sa, "n_significant_regions": len(ids),
                     "n_anomalous_flows": int(in_sig.sum())})
    perms = getattr(args, "perms", 20)
    fig.suptitle(f"sig-alpha sweep -- reg = {label}   (with {perms} permutations the smallest "
                 f"attainable p is {1 / (perms + 1):.3f})", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    fig.savefig(os.path.join(out_dir, "sweep_sigalpha.png"), dpi=140, bbox_inches="tight")
    plt.close(fig)
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, "sweep_sigalpha_counts.csv"), index=False)
    _log(f"sweep_sigalpha.png: " + ", ".join(f"{r['sig_alpha']}->{r['n_significant_regions']}" for r in rows))


def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _plot_region_map(out_dir, XA, region, order, rank_of, svc_by_region, flows_df, A):
    plt = _mpl()
    xy = XA[:, :2]
    fig, ax = plt.subplots(1, 2, figsize=(17, 7.2))
    insig = ~np.isin(region, order)
    for a in ax:
        a.scatter(xy[insig, 0], xy[insig, 1], s=3, c="#e4e4e4", zorder=1, rasterized=True)
        a.set_xlabel("PC1"); a.set_ylabel("PC2")
    cmap = plt.get_cmap("tab20")
    for gi, r in enumerate(order):
        m = region == r
        ax[0].scatter(xy[m, 0], xy[m, 1], s=14, color=cmap(gi % 20), zorder=3,
                      label=f"#{rank_of.get(r)} R{r} {svc_by_region.get(r, '?')} (n={int(m.sum())})")
        cx, cy = np.median(xy[m, 0]), np.median(xy[m, 1])
        ax[0].annotate(f"#{rank_of.get(r)} {svc_by_region.get(r, '')}", (cx, cy), fontsize=7,
                       weight="bold", ha="center",
                       bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.8), zorder=4)
    ax[0].set_title(f"{len(order)} significant regions (label = rank, service)")
    ax[0].legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.0, 1.0), framealpha=0.9)
    pats = flows_df["temporal_pattern"].str.replace(r" \(.*\)", "", regex=True)
    cid_to_i = {f["cid"]: i for i, f in enumerate(A)}
    pal = plt.get_cmap("Set1")
    for gi, p in enumerate(sorted(pats.unique())):
        ii = [cid_to_i[c] for c in flows_df.loc[pats == p, "conn_id"]]
        ax[1].scatter(xy[ii, 0], xy[ii, 1], s=14, color=pal(gi % 9), label=f"{p} (n={len(ii)})", zorder=3)
    ax[1].set_title("anomalous flows by temporal pattern")
    ax[1].legend(fontsize=8)
    fig.suptitle("Significant regions in the Chronos-2 REG embedding (PCA 2-D)", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "regions_map.png"), dpi=140, bbox_inches="tight")
    plt.close(fig)


def _time_strip(flows, step, pc_idx, max_cols=4000):
    g0 = min(f["t0"] for f in flows)
    g1 = max(f["t0"] + (len(f["frame"]) - 1) * step for f in flows)
    nb = int(round((g1 - g0) / step)) + 1
    pool = max(1, int(math.ceil(nb / max_cols)))
    M = np.zeros((len(flows), nb // pool + 1), dtype=np.float32)
    for i, f in enumerate(flows):
        k0 = int(round((f["t0"] - g0) / step))
        cols = (k0 + np.arange(len(f["frame"]))) // pool
        np.maximum.at(M[i], cols, np.log1p(f["frame"][:, pc_idx]))
    return M, g0, g1


def _region_pdf(rdir, r, rank_of, A, B, idx, D, prof_cache, channels, step, pc_idx,
                summary_df, contrast_df, near_rows, cap):
    plt = _mpl()
    import matplotlib.dates as mdates
    from matplotlib.backends.backend_pdf import PdfPages
    rank = rank_of.get(r)
    idx = idx[np.argsort([A[i]["t0"] for i in idx])]              # chronological
    flows = [A[i] for i in idx]
    srow = summary_df[summary_df.region_id == r].iloc[0]
    path = os.path.join(rdir, f"region_{rank:02d}_R{r}.pdf")
    with PdfPages(path) as pdf:
        # ---- page 1: overview ----
        fig = plt.figure(figsize=(16, 11))
        gs = fig.add_gridspec(3, 1, height_ratios=[0.5, 1.7, 1.2])
        ax0 = fig.add_subplot(gs[0]); ax0.axis("off")
        lines = [f"REGION #{rank} (id {r})   flows={int(srow.n_flows)}   p={srow.p_value:.3f}   "
                 f"share of divergence={srow.share_of_divergence:.1%}   silhouette={srow.silhouette:.2f}",
                 f"verdict: {srow.verdict}   packets x{srow.packets_vs_global:.1f} and duration "
                 f"x{srow.span_vs_global:.1f} vs all flows",
                 f"service: {_short(srow.service_mix)}     protocol: {_short(srow.protocol_mix)}",
                 f"destinations: {_short(srow.top_destinations)}",
                 f"sources: {_short(srow.top_sources)}",
                 f"temporal patterns: {_short(srow.temporal_pattern_mix)}",
                 f"synchrony (mean Jaccard of active bins) = {srow.sync_jaccard:.2f};  "
                 f"peak co-activity {srow.peak_coactivity_frac:.0%} of flows at {srow.peak_coactivity_time}",
                 f"seen {srow.first_seen} -> {srow.last_seen};  filter-marked {srow.frac_filter_marked:.0%}"
                 + (f";  IoT {srow.frac_iot:.0%};  ECH {srow.frac_ech:.0%}" if srow.frac_iot == srow.frac_iot else "")
                 + (f";  OUI: {_short(srow.oui_mix, 3)}" if srow.oui_mix else ""),
                 f"most different from reference: {srow.most_different_features}"]
        ax0.text(0.0, 1.0, "\n".join(lines), va="top", family="monospace", fontsize=9)
        M, g0, g1 = _time_strip(flows, step, pc_idx)
        ax1 = fig.add_subplot(gs[1])
        ax1.imshow(M, aspect="auto", cmap="magma", interpolation="nearest",
                   extent=[mdates.date2num(g0), mdates.date2num(g1), len(flows), 0])
        ax1.xaxis_date(); ax1.set_ylabel("flow (by start time)")
        ax1.set_title("time-strip: every flow of the region on the ABSOLUTE time axis "
                      "(colour = log(1+packets)); vertical stripes = synchronised activity")
        ax2 = fig.add_subplot(gs[2])
        for f in flows:
            t = [f["t0"] + k * step for k in range(len(f["frame"]))]
            ax2.plot(t, f["frame"][:, pc_idx], lw=0.5, alpha=0.35, color="#1f77b4")
        ax2.set_ylabel(channels[pc_idx]); ax2.set_title("overlay of all flows (absolute time)")
        fig.autofmt_xdate(); fig.tight_layout()
        pdf.savefig(fig); plt.close(fig)

        # ---- page 2: contrast vs reference + nearest normal flows ----
        fig, ax = plt.subplots(1, 2, figsize=(16, 6.5))
        con = contrast_df[contrast_df.region_id == r].sort_values("cohens_d")
        ax[0].barh(con["feature"], con["cohens_d"],
                   color=["#d62728" if v > 0 else "#1f77b4" for v in con["cohens_d"]])
        ax[0].axvline(0, color="k", lw=0.8)
        ax[0].set_title("region vs reference: Cohen's d per feature\n(|d|>0.8 large difference)")
        ax[0].tick_params(axis="y", labelsize=7)
        for f in flows:
            v = f["frame"][:, pc_idx]
            ax[1].plot(np.arange(len(v)), v, lw=0.6, alpha=0.25, color="#d62728")
        for nr in [n for n in near_rows if n["region_id"] == r]:
            v = B[nr["_b_index"]]["frame"][:, pc_idx]
            ax[1].plot(np.arange(len(v)), v, lw=1.0, alpha=0.8, color="#2ca02c")
        ax[1].set_xlabel("bins since flow start"); ax[1].set_ylabel(channels[pc_idx])
        ax[1].set_title("red = region flows, green = 10 nearest NORMAL (reference) flows")
        fig.tight_layout(); pdf.savefig(fig); plt.close(fig)

        # ---- pages 3..: every member flow (8 per page, all channels) ----
        sel = idx if not cap else idx[np.argsort(-D[idx])][:cap]
        sel = sel[np.argsort([A[i]["t0"] for i in sel])]
        per = 8
        for s in range(0, len(sel), per):
            chunk = sel[s:s + per]
            fig, axes = plt.subplots(len(chunk), len(channels),
                                     figsize=(3.4 * len(channels), 1.6 * len(chunk) + 0.8),
                                     squeeze=False)
            for ri, i in enumerate(chunk):
                f = A[i]; pr = prof_cache[i]
                t = [f["t0"] + k * step for k in range(len(f["frame"]))]
                for c, ch in enumerate(channels):
                    a = axes[ri][c]
                    a.plot(t, f["frame"][:, c], lw=0.7, color="#1f77b4")
                    a.tick_params(labelsize=5)
                    loc = mdates.AutoDateLocator(minticks=2, maxticks=3)
                    a.xaxis.set_major_locator(loc)
                    a.xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc))
                    a.xaxis.get_offset_text().set_fontsize(5)
                    if ri == 0:
                        a.set_title(ch, fontsize=8)
                axes[ri][0].set_ylabel(f"{f['cid'][:34]}\nD={D[i]:.1e}  {pr['temporal_pattern']}",
                                       fontsize=5.5)
            fig.suptitle(f"REGION #{rank} (id {r}) -- flows {s + 1}-{s + len(chunk)} of {len(sel)}"
                         + (f" (top {cap} by D_i)" if cap and len(idx) > cap else ""), fontsize=10)
            fig.tight_layout(rect=[0, 0, 1, 0.97]); pdf.savefig(fig); plt.close(fig)


def _review_sample(out_dir, A, flows_df, region, sig_ids, channels, step, pc_idx, n, rng):
    """Blinded set for manual labelling: n flagged flows (spread over regions) + n
    random non-flagged flows, shuffled. review_key.csv holds the answers -- keep it
    away from the reviewer. Precision estimate = fraction of flagged ones the reviewer
    calls anomalous; the normal half measures the reviewer's own base rate."""
    if n <= 0 or len(flows_df) == 0:
        return
    per_region = max(1, n // max(1, flows_df.region_id.nunique()))
    flagged = pd.concat([g.sample(min(len(g), per_region), random_state=0)
                         for _, g in flows_df.groupby("region_id")]).head(n)
    cid_to_i = {f["cid"]: i for i, f in enumerate(A)}
    normal_pool = np.flatnonzero(~np.isin(region, list(sig_ids)))
    norm = rng.choice(normal_pool, min(n, len(normal_pool)), replace=False)
    items = [(cid_to_i[c], "flagged", int(rid)) for c, rid in zip(flagged.conn_id, flagged.region_id)]
    items += [(int(i), "normal", -1) for i in norm]
    perm = rng.permutation(len(items))
    rows, key = [], []
    for b, k in enumerate(perm, 1):
        i, lab, rid = items[k]
        f = A[i]
        s_ip, s_p, d_ip, d_p = flow_endpoints(f)
        T = len(f["frame"])
        rows.append({"blind_id": b, "conn_id": f["cid"], "protocol": f.get("proto", ""),
                     "src_ip": s_ip, "src_port": s_p, "dst_ip": d_ip, "dst_port": d_p,
                     "start_time": f["t0"], "end_time": f["t0"] + (T - 1) * step,
                     "n_packets": f["n_pkts"],
                     "verdict (normal | unusual-benign | suspicious | cannot-tell)": "",
                     "reason": "", "missing_information": ""})
        key.append({"blind_id": b, "truth": lab, "region_id": rid})
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, "review_sample.csv"), index=False)
    pd.DataFrame(key).to_csv(os.path.join(out_dir, "review_key.csv"), index=False)
    plt = _mpl()
    from matplotlib.backends.backend_pdf import PdfPages
    with PdfPages(os.path.join(out_dir, "review_sample.pdf")) as pdf:
        for s in range(0, len(rows), 8):
            chunk = rows[s:s + 8]
            fig, axes = plt.subplots(len(chunk), 1, figsize=(12, 1.5 * len(chunk) + 0.6), squeeze=False)
            for ri, row in enumerate(chunk):
                f = A[cid_to_i[row["conn_id"]]]
                axes[ri][0].plot(f["frame"][:, pc_idx], lw=0.7)
                axes[ri][0].set_ylabel(f"#{row['blind_id']}", fontsize=8)
                axes[ri][0].tick_params(labelsize=6)
            fig.suptitle("blinded review sample (packet count per bin; labels in review_key.csv)",
                         fontsize=9)
            fig.tight_layout(rect=[0, 0, 1, 0.96]); pdf.savefig(fig); plt.close(fig)
    _log(f"review_sample.csv/.pdf: {len(rows)} blinded flows (key in review_key.csv)")


def _write_config(out_dir, args, path, channels, A, B, region, sig_ids):
    cfg = {k: (v if isinstance(v, (int, float, str, bool, type(None), list)) else str(v))
           for k, v in vars(args).items()}
    cfg.update({"features_file": str(path), "channels": channels, "n_A": len(A), "n_B": len(B),
                "n_regions": int(len(set(region[region >= 0].tolist()))),
                "n_significant_regions": len(sig_ids)})
    with open(os.path.join(out_dir, "run_config.json"), "w") as fh:
        json.dump(cfg, fh, indent=2, default=str)


def _write_readme(out_dir, channels):
    txt = f"""OUTPUTS OF THIS RUN
regions_map.png        embedding map; left = significant regions (label #rank), right = anomalous
                       flows coloured by temporal pattern
clustering.png         KL significant regions (left) vs filter-marked flows (right)
sweep_sigalpha.png     this run's regions at each threshold of --sig-alphas (one panel per
                       threshold; a region keeps its colour); counts in sweep_sigalpha_counts.csv
anomalous_flows.csv    one row per flow in a significant region (p <= sig-alpha):
  region_rank/region_id/region_p_value/region_share_of_divergence/region_service
  conn_id, protocol    unit id ("ip:port<->ip:port"; host runs: the IP; router logs:
                       "router|device|proto/port") and L4 protocol
  router, device       router folder and device (MAC as logged) -- router-log data only
  oui, is_iot          device vendor prefix (first 3 MAC bytes) and IoT flag (new router logs)
  conn_direction       local / out / in (new router logs)
  has_ech              1 if the connection had a client hello with a mitm_force SNI (ECH)
  src_ip/src_port, dst_ip/dst_port   dst = endpoint with the LOWER port (service side);
                       router logs: src = device, dst_port from the log, no IPs
  service              name of the destination port (HTTPS/QUIC, DNS, ...; hi:<port> if unknown)
  start_time/end_time/duration_h/span_bins/active_bins/active_frac    timing of the flow
  temporal_pattern     continuous | single-burst | periodic (~N min) | sporadic | intermittent
  n_bursts/max_burst_bins/median_gap_bins/gap_cv   burst structure (gap_cv ~0 = regular)
  period_min/period_strength   dominant period of activity (autocorrelation peak, 0..1)
  peak_time/peak_packets/dominant_hour
  <channel>_mean_active/<channel>_max   for channels {channels}
  D_i/D_rank_in_region  local KL contribution (higher = more specific to the target set)
  typicality           distance to region centre / region median (>>1 = atypical member)
  point_silhouette     how well the flow belongs to its region (-1..1)
  filter_marked/filter_score   flagged by the independent filter (method5/lof) used for B
  nearest_normal_cid/nearest_normal_dist   closest REFERENCE flow in the embedding
region_summary.csv     per region: verdict, packets_vs_global, span_vs_global,
                       service/protocol/endpoint mix, temporal-pattern mix,
                       oui_mix, frac_iot, frac_ech, direction_mix (new router logs),
                       sync_jaccard (0..1, flows active at the same moments), peak co-activity,
                       port/ip concentration, the 3 features most different from reference
region_composition.csv EVERY service / protocol / destination / source / router / OUI / direction /
                       IoT / ECH / temporal-pattern value per region, with n_flows and pct_of_region
region_profiles.txt    readable profile of every significant region with ALL values, a verdict
                       (COHERENT = >=60% on one service or one source/destination, else "mixed /
                       possibly noise") and packets/duration compared with all flows of A
region_heatmap.png     significant regions x properties (colour per column, value in the cell)
Di_by_service.png      D_i of each region's flows by dominant service, vs all other flows
region_similarity.png  endpoint overlap (Jaccard) between regions: high = one behaviour split
                       into several regions
region_contrast.csv    per region x feature: medians, Cohen's d, KS statistic/p vs reference
nearest_reference.csv  10 closest NORMAL flows to each region centre (side-by-side review)
regions/region_XX_R<id>.pdf   p1 overview + time-strip + overlay, p2 contrast + nearest normal,
                       p3.. ALL member flows (8 per page, every channel, absolute time)
cases/case_XX_R<id>.pdf  dossier per significant region for a networking specialist: case sheet
                       (identity, scale, why flagged in network terms, service guide, missing data),
                       device history over the whole period, peer comparison, raw packet excerpt
                       (with --raw-logs-dir); excerpts also in cases/case_packets.csv
case_review_form.csv   one row per case: verdict (normal | unusual-benign | suspicious |
                       cannot-tell), reason, missing information, reviewer, date
review_sample.csv/.pdf blinded flagged+normal flows for manual labelling (same four verdicts; do
                       this BEFORE reading the case files, which are not blind); answers in
review_key.csv         (do not show the key to the reviewer)
embeddings_proj.npz    XA/XB projected embeddings, cids, D_i, region labels
run_config.json        all arguments and counts
"""
    with open(os.path.join(out_dir, "README_outputs.txt"), "w") as fh:
        fh.write(txt)