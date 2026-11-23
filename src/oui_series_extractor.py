#!/usr/bin/env python3
"""
oui_series_extractor.py  (src/)

Builds the input for the "which device type behaves differently from router to router"
experiment (option 2: all series together, normalised per OUI).

Saves the WHOLE series (one row per router x OUI x direction), NOT cut into windows, so
you can inspect the series lengths first and choose the window size afterwards with
series_to_windows.py.

A SERIES is all traffic of ONE DIRECTION of a (router, OUI) pair, binned in time:
    out series = packets SENT by that OUI on that router
    in  series = packets RECEIVED by that OUI on that router
So each (router, OUI) gives TWO unidirectional series. Channels per bin:
    packet_count   packets in that direction in the bin
    bytes          bytes in that direction
    n_conns        connections of that OUI ACTIVE in the bin (first..last packet bin,
                   including silent intermediate bins), on that router

Only REAL vendor OUIs are kept: randomized (locally-administered) MACs are not stable
device-type identifiers, so the same prefix on two routers is different devices.

Output parquet columns (one row per router x OUI x direction series):
    series_id, router, oui, direction, start, end, n_bins, n_active_bins,
    packet_count (list[n_bins]), bytes (list[n_bins]), n_conns (list[n_bins])
n_bins is the full length of the series (first to last active bin); the channel lists have
that length. Cut into windows later with series_to_windows.py.

USAGE
  python src/oui_series_extractor.py --input-dir /mnt/new-router-data \
      --bin-seconds 300 --out data/parquet/oui_series.parquet
"""
import argparse
import glob
import json
import os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

DEFAULT_BIN_S = 300


def _log(m):
    print(f"[oui-windows] {m}", flush=True)


def is_real_oui(oui):
    """True for a real vendor OUI (universally administered). False for randomized /
    locally-administered MACs (second-least-significant bit of the first octet = 1)."""
    s = str(oui).strip().lower().replace(":", "").replace("-", "").replace(".", "")
    if len(s) < 2:
        return False
    try:
        return not (int(s[:2], 16) & 0b10)
    except ValueError:
        return False


def _log_files(rd):
    mp = os.path.join(rd, "manifest.json")
    if os.path.exists(mp):
        try:
            files = json.load(open(mp)).get("files", [])
            if files:
                return [os.path.join(rd, f) for f in files]
        except Exception:
            pass
    return sorted(glob.glob(os.path.join(rd, "flows_*.log")))


def process_router(router_dir, router_name, out_path, bin_s):
    """Stream one router, aggregate per (oui, direction, bin), cut into windows, write a
    parquet part. Memory is O(active oui*bins), never O(packets)."""
    # per (oui, dir, bin): [packets, bytes]
    acc = defaultdict(lambda: [0, 0.0])
    # a connection is ACTIVE from its first packet's bin to its last packet's bin,
    # INCLUDING intermediate bins with no traffic. We record each connection's bin span
    # while streaming, then count active connections per (oui, bin) at the end.
    conn_span = {}                                     # local_id -> [oui, first_bin, last_bin]
    spans = []                                         # (oui, first_bin, last_bin) finalised
    active = {}                                        # local_id -> [oui, clock_ms] or None
    n_pkts = orphan = bad = 0
    binms = bin_s * 1000

    def _finalise(lid):
        s = conn_span.pop(lid, None)
        if s is not None and s[1] is not None:
            spans.append((s[0], s[1], s[2]))

    for fp in _log_files(router_dir):
        if not os.path.exists(fp):
            continue
        with open(fp, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                p = line.rstrip("\n").split(",")
                try:
                    if p[0] == "new_conn":
                        _finalise(p[1])                    # a reused id ends the previous conn
                        oui = p[3]
                        if not is_real_oui(oui):
                            active[p[1]] = None
                            conn_span[p[1]] = [oui, None, None]
                            continue
                        start = int(p[2])
                        active[p[1]] = [oui, start]
                        conn_span[p[1]] = [oui, None, None]
                    elif p[0] == "pkt":
                        a = active.get(p[1])
                        if a is None:
                            orphan += 1
                            continue
                        oui, clock = a
                        clock += int(p[2])                 # rel ms since previous packet
                        a[1] = clock
                        size = int(p[3])
                        d = "out" if p[4].strip() == "out" else "in"
                        b = (clock // binms) * binms
                        e = acc[(oui, d, b)]
                        e[0] += 1; e[1] += size
                        n_pkts += 1
                        sp = conn_span.get(p[1])           # extend the active bin span
                        if sp is not None:
                            if sp[1] is None or b < sp[1]:
                                sp[1] = b
                            if sp[2] is None or b > sp[2]:
                                sp[2] = b
                except (IndexError, ValueError):
                    bad += 1
    for lid in list(conn_span):                            # finalise the still-open ones
        _finalise(lid)

    if not acc:
        return 0, 0, orphan, bad

    # active connections per bin, grouped BY OUI (so each series reads only its own bins,
    # instead of scanning all conn_bins per series). A pathological span (clock bug) is
    # capped so one connection cannot generate billions of bins.
    MAX_SPAN_BINS = 100_000                             # ~ a year of 5-min bins; safety cap
    conn_bins_by_oui = defaultdict(lambda: defaultdict(int))   # oui -> {bin: n_active}
    for oui, fb, lb in spans:
        nb_span = (lb - fb) // binms + 1
        if nb_span > MAX_SPAN_BINS:
            lb = fb + (MAX_SPAN_BINS - 1) * binms
        cb = conn_bins_by_oui[oui]
        for b in range(fb, lb + binms, binms):
            cb[b] += 1

    # assemble per (oui, direction) dense series, then cut into windows
    series = defaultdict(dict)                          # (oui, dir) -> {bin_ms: [pk, by]}
    for (oui, d, b), (pk, by) in acc.items():
        series[(oui, d)][b] = [pk, by]
    rows = []
    step = binms
    for (oui, d), bins in series.items():
        bset = sorted(bins)
        b0, b1 = bset[0], bset[-1]
        nb = (b1 - b0) // step + 1
        pk = np.zeros(nb, np.float32); by = np.zeros(nb, np.float32); nc = np.zeros(nb, np.float32)
        for b, (p_, y_) in bins.items():
            i = (b - b0) // step
            pk[i] = p_; by[i] = y_
        for b, c in conn_bins_by_oui.get(oui, {}).items():
            if b0 <= b <= b1:
                nc[(b - b0) // step] += c
        rows.append({
            "series_id": f"{router_name}|{oui}|{d}",
            "router": router_name, "oui": oui, "direction": d,
            "start": pd.Timestamp(b0, unit="ms"), "end": pd.Timestamp(b1, unit="ms"),
            "n_bins": int(nb), "n_active_bins": int((pk > 0).sum()),
            "packet_count": pk.tolist(), "bytes": by.tolist(), "n_conns": nc.tolist(),
        })
    if rows:
        pq.write_table(pa.Table.from_pandas(pd.DataFrame(rows), preserve_index=False), out_path)
    return len(rows), n_pkts, orphan, bad




def _work(t):
    rd, name, out_path, bin_s = t
    try:
        return out_path, process_router(rd, name, out_path, bin_s)
    except Exception as e:
        return None, ("ERR", str(e))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--out", default="/root/router_anomaly/data/parquet/oui_series.parquet")
    ap.add_argument("--bin-seconds", type=int, default=DEFAULT_BIN_S)
    ap.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    ap.add_argument("--only-routers", default=None)
    args = ap.parse_args()

    rdirs = sorted(d for d in glob.glob(os.path.join(args.input_dir, "*")) if os.path.isdir(d))
    if args.only_routers:
        keep = set(args.only_routers.split(","))
        rdirs = [d for d in rdirs if os.path.basename(d) in keep]
    if not rdirs:
        _log(f"no router folders in {args.input_dir}"); return
    parts_dir = args.out + ".parts"
    os.makedirs(parts_dir, exist_ok=True)
    jobs = [(d, os.path.basename(d.rstrip("/"))[:12], os.path.join(parts_dir, f"{i:06d}.parquet"),
             args.bin_seconds) for i, d in enumerate(rdirs)]
    _log(f"{len(jobs)} routers, bin={args.bin_seconds}s, workers={args.workers}")

    parts, tot = [], np.zeros(4, np.int64)
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(_work, j): j[1] for j in jobs}
        for n, fu in enumerate(as_completed(futs), 1):
            out_path, res = fu.result()
            if out_path is None:
                _log(f"ERROR {futs[fu]}: {res[1]}")
            else:
                nrows, npk, orph, bad = res
                tot += np.array([nrows, npk, orph, bad])
                if nrows:
                    parts.append(out_path)
            if n % 50 == 0 or n == len(jobs):
                _log(f"{n}/{len(jobs)} routers; series so far {tot[0]:,}")

    if not parts:
        _log("no series produced"); return
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    writer = None
    for i, p in enumerate(sorted(parts), 1):
        t = pq.read_table(p)
        if writer is None:
            writer = pq.ParquetWriter(args.out, t.schema)
        writer.write_table(t)
        os.remove(p)
    if writer:
        writer.close()
    try:
        os.rmdir(parts_dir)
    except OSError:
        pass
    _log(f"series={tot[0]:,} packets={tot[1]:,} orphan={tot[2]:,} bad={tot[3]:,}")
    _log(f"wrote {args.out}")
    # quick summary (read only 3 columns, streamed -> memory-safe)
    t = pq.read_table(args.out, columns=["router", "oui", "direction"])
    router = t["router"].to_pylist(); oui = t["oui"].to_pylist()
    pairs = len(set(zip(router, oui)))
    _log(f"{t.num_rows:,} series over {pairs:,} (router,oui) pairs, "
         f"{len(set(oui)):,} real OUIs, {len(set(router)):,} routers")


if __name__ == "__main__":
    main()