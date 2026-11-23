#!/usr/bin/env python3
"""
series_to_windows.py -- cut the whole series from oui_series_extractor.py into fixed
disjoint windows of --window bins. Run this AFTER choosing a window size from
series_lengths.py.

Each series (one row: router x OUI x direction, with channel lists of length n_bins) is
split into ceil(n_bins / W) windows; the last window is zero-padded to W. Every window is
kept by default; use --min-active to drop near-empty ones.

Output parquet columns (one row per window):
    window_id, series_id, router, oui, direction, start, end, n_active_bins,
    packet_count (list[W]), bytes (list[W]), n_conns (list[W])

USAGE
  python src/series_to_windows.py data/parquet/oui_series.parquet --window 48 \
      -o data/parquet/oui_windows.parquet
"""
import argparse
import os

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("series_parquet")
    ap.add_argument("--window", type=int, required=True, help="window length in bins")
    ap.add_argument("--bin-seconds", type=int, default=300)
    ap.add_argument("--min-active", type=int, default=0,
                    help="drop windows with fewer than this many active bins (default 0 = keep all)")
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--batch-rows", type=int, default=20000, help="series per write batch")
    args = ap.parse_args()
    W = args.window
    step = pd.Timedelta(seconds=args.bin_seconds)

    pf = pq.ParquetFile(args.series_parquet)
    writer = None
    n_series = n_win = 0
    buf = []

    def flush():
        nonlocal writer
        if not buf:
            return
        tab = pa.Table.from_pandas(pd.DataFrame(buf), preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(args.out, tab.schema)
        writer.write_table(tab)
        buf.clear()

    for batch in pf.iter_batches(batch_size=args.batch_rows,
                                 columns=["series_id", "router", "oui", "direction", "start",
                                          "packet_count", "bytes", "n_conns"]):
        b = batch.to_pydict()
        for i in range(len(b["series_id"])):
            n_series += 1
            pk = np.asarray(b["packet_count"][i], np.float32)
            by = np.asarray(b["bytes"][i], np.float32)
            nc = np.asarray(b["n_conns"][i], np.float32)
            nb = len(pk)
            t0 = pd.Timestamp(b["start"][i])
            for w0 in range(0, nb, W):
                wpk = pk[w0:w0 + W]; wby = by[w0:w0 + W]; wnc = nc[w0:w0 + W]
                if len(wpk) < W:
                    pad = W - len(wpk)
                    wpk = np.r_[wpk, np.zeros(pad, np.float32)]
                    wby = np.r_[wby, np.zeros(pad, np.float32)]
                    wnc = np.r_[wnc, np.zeros(pad, np.float32)]
                n_act = int((wpk > 0).sum())
                if args.min_active > 0 and n_act < args.min_active:
                    continue
                ws = t0 + w0 * step
                buf.append({
                    "window_id": f"{b['series_id'][i]}|{int(ws.value // 1_000_000)}",
                    "series_id": b["series_id"][i], "router": b["router"][i],
                    "oui": b["oui"][i], "direction": b["direction"][i],
                    "start": ws, "end": ws + (W - 1) * step, "n_active_bins": n_act,
                    "packet_count": wpk.tolist(), "bytes": wby.tolist(), "n_conns": wnc.tolist(),
                })
                n_win += 1
        if len(buf) >= args.batch_rows:
            flush()
    flush()
    if writer:
        writer.close()
    print(f"[windows] {n_series:,} series -> {n_win:,} windows of {W} bins, wrote {args.out}")


if __name__ == "__main__":
    main()