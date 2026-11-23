#!/usr/bin/env python3
"""
pcap_to_parquet.py  (lives in  Anomaly_detection/src/)

Fast, fault-tolerant, resumable extractor: pcap tree -> one Parquet per session,
one row per TCP/UDP packet keyed by connection.

Direction (initiator detection): for each connection the "initiator" (client)
is decided from protocol evidence, in order of confidence:
    1. a pure SYN (SYN set, ACK not)        -> the sender is the client
    2. a SYN+ACK                            -> the RECEIVER is the client
    3. a TLS ClientHello                    -> the sender is the client
    4. fallback: whoever sent the first packet of the connection
Because the deciding packet is not always the first one seen (reordering, or a
capture that starts mid-handshake), each file is read in TWO passes: pass 1
parses and buffers the file's packets and fixes each new connection's initiator
from the strongest evidence in that file; pass 2 walks the buffered packets in
capture order and writes them with the final direction and a continuous
inter-arrival gap. direction='fwd' means initiator->responder, 'bwd' the reply.
A connection's initiator is locked once its first file is written, so labels
stay consistent across the hourly splits (later files never relabel it).

Speed: dpkt dissection, sessions in parallel (one worker each). Pass 1 holds one
pcap file's packets in memory; with huge hourly files, lower --workers.

Durability + resume: each input pcap -> its own Parquet PART (atomic rename), so
a finished file is safe the instant it's done; a kill loses at most the current
file's buffer. Reruns skip existing parts and complete sessions. Parts are then
merged (atomically) into data/parquet/<hash>.parquet (kept with --keep-parts).

Robustness: truncated final record stops that file at its last valid packet;
an unreadable file is skipped and left un-parted so a rerun retries it; a failed
session is reported but does not abort the run.

Columns
    capture_id, conn_id, src_ip/dst_ip, src_port/dst_port (initiator-oriented),
    protocol, direction, is_forward, ts, datetime, bytes, payload_len, iat,
    tcp_window, tcp_flags

Usage:
    python src/pcap_to_parquet.py
    python src/pcap_to_parquet.py --only e00ab1501d16
===============================================================================
USAGE  --  STEP 1 of the pipeline (pcap -> per-packet Parquet)
===============================================================================
Input : data/unfiltered_pcap/<session_hash>/traffic_<epoch_ms>.pcap
Output: data/parquet/<session_hash>.parquet   (one row per packet)

Basic:
    python src/pcap_to_parquet.py                       # all sessions
    python src/pcap_to_parquet.py --only 2eadb48f       # one session (name or prefix)
    python src/pcap_to_parquet.py --only 2eadb48f --overwrite   # re-extract

Options:
    --only <hash>        one session folder (name or prefix)
    --overwrite          re-extract even if the .parquet already exists
    --include-icmp       also extract ICMP (ports=0); OFF by default (TCP/UDP only)
    --workers N          parallel worker processes (default: auto)
    --combined <path>    also write one merged Parquet of all sessions
    --keep-parts         keep intermediate per-file parts (debug)
    --root / --pcap-dir / --out-dir   override the default paths

Notes:
  * bytes / payload_len come from the DECLARED header lengths (ip.len, udp.ulen,
    tcp data-offset), NOT len(buf) -- so truncated captures still report the real
    on-wire size. Values are clamped to [headers, 65535] to drop offload artefacts.
  * timestamps are corrected PER FILE (epoch from the filename + relative time),
    which fixes sessions whose files mix relative and absolute time.

Next step:  aggregate_to_forecast_csv.py   (Parquet -> per-bin features)
===============================================================================
"""

import argparse
import glob
import os
import re
import socket
import struct
import tempfile
import shutil
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

import dpkt
import dpkt.dpkt
import dpkt.pcap
import dpkt.pcapng
import dpkt.ethernet
import dpkt.sll
import dpkt.ip
import dpkt.ip6
import dpkt.tcp
import dpkt.udp
import dpkt.icmp
import dpkt.icmp6
import dpkt.loopback
try:
    import dpkt.sll2
    _HAS_SLL2 = True
except Exception:
    _HAS_SLL2 = False

_TRUNC_ERRORS = (dpkt.dpkt.NeedData, struct.error, EOFError)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_ROOT = os.path.dirname(SCRIPT_DIR)
_CLOCK_CHECK_EVERY = 250_000

# Initiator-evidence confidence levels (higher wins).
_EV_NONE, _EV_CH, _EV_SYNACK, _EV_SYN = 0, 1, 2, 3

SCHEMA = pa.schema([
    ("capture_id",  pa.string()),
    ("conn_id",     pa.string()),
    ("src_ip",      pa.string()),
    ("dst_ip",      pa.string()),
    ("src_port",    pa.int32()),
    ("dst_port",    pa.int32()),
    ("protocol",    pa.string()),
    ("direction",   pa.string()),
    ("is_forward",  pa.bool_()),
    ("ts",          pa.float64()),
    ("datetime",    pa.timestamp("ns")),
    ("bytes",       pa.int32()),
    ("payload_len", pa.int32()),
    ("iat",         pa.float64()),
    ("tcp_window",  pa.float64()),
    ("tcp_flags",   pa.string()),
])
COLUMNS = [f.name for f in SCHEMA]

_TCP_FLAGS = [(0x01, "F"), (0x02, "S"), (0x04, "R"), (0x08, "P"),
              (0x10, "A"), (0x20, "U"), (0x40, "E"), (0x80, "C")]


def _flags_str(f):
    return "".join(ch for bit, ch in _TCP_FLAGS if f & bit)


# ---- link-layer handling -----------------------------------------------------

def _open_reader(fh):
    magic = fh.read(4)
    fh.seek(0)
    if magic == b"\x0a\x0d\x0d\x0a":
        return dpkt.pcapng.Reader(fh)
    return dpkt.pcap.Reader(fh)


def _l2_len(datalink):
    """Bytes of link-layer header stripped before L3, per pcap datalink type."""
    return {1: 14, 113: 16, 276: 20, 0: 4, 12: 0, 14: 0, 101: 0}.get(datalink, 14)


def _make_l3_extractor(datalink):
    def from_raw(buf):
        return dpkt.ip6.IP6(buf) if (buf and (buf[0] >> 4) == 6) else dpkt.ip.IP(buf)

    if datalink == 1:
        return lambda buf: dpkt.ethernet.Ethernet(buf).data
    if datalink == 113:
        return lambda buf: dpkt.sll.SLL(buf).data
    if datalink == 276 and _HAS_SLL2:
        return lambda buf: dpkt.sll2.SLL2(buf).data
    if datalink in (12, 14, 101):
        return from_raw
    if datalink == 0:
        return lambda buf: dpkt.loopback.Loopback(buf).data
    return lambda buf: dpkt.ethernet.Ethernet(buf).data


def _dissect(l3, ts, l2_len, include_icmp=False):
    # Real ON-WIRE lengths come from the IP/L4 HEADERS (ip.len, tcp.off, udp.ulen),
    # NOT from len(buf): captures may be truncated (only headers saved) while the
    # header still declares the true size. Using len(buf) would report every large
    # packet as ~42 bytes with 0 payload. We reconstruct the real frame size and
    # payload length from the declared header fields instead.
    if isinstance(l3, dpkt.ip.IP):
        v6 = False
        ip_hl = int(l3.hl) * 4
        real_l3 = int(l3.len)                          # IPv4 total length (hdr+payload)
        if real_l3 <= 0:                               # TSO/LRO offload can report 0
            real_l3 = ip_hl + len(bytes(l3.data))
    elif isinstance(l3, dpkt.ip6.IP6):
        v6 = True
        ip_hl = 40
        real_l3 = 40 + int(l3.plen)
        if int(l3.plen) <= 0:
            real_l3 = 40 + len(bytes(l3.data))
    else:
        return None

    l4 = l3.data
    if isinstance(l4, dpkt.tcp.TCP):
        proto = "TCP"
        fl = int(l4.flags)
        syn, ack = bool(fl & 0x02), bool(fl & 0x10)
        window = int(l4.win)
        flags = _flags_str(fl)
        data = l4.data
        is_ch = (len(data) >= 6 and data[0] == 0x16 and data[1] == 0x03 and data[5] == 0x01)
        sport, dport = int(l4.sport), int(l4.dport)
        l4_hl = int(l4.off) * 4                         # TCP data offset (words) -> bytes
    elif isinstance(l4, dpkt.udp.UDP):
        proto, window, flags = "UDP", None, None
        syn = ack = is_ch = False
        sport, dport = int(l4.sport), int(l4.dport)
        l4_hl = 8
    elif include_icmp and isinstance(l4, (dpkt.icmp.ICMP, dpkt.icmp6.ICMP6)):
        proto, window = "ICMP", None
        syn = ack = is_ch = False
        try:
            flags = f"icmp_type{int(l4.type)}"
        except Exception:
            flags = "icmp"
        sport, dport = 0, 0
        l4_hl = 8
    else:
        return None

    # Sanity bounds. IP total length is a 16-bit field (<= 65535); values above
    # that come from TSO/LRO/GRO offload or corrupt headers where the reassembled
    # size or a misread field is reported. A single captured packet cannot exceed
    # ~64 KB on the wire, so clamp; also guard the tiny/zero end (offload with
    # ip.len=0 and no captured payload). This keeps byte features meaningful.
    _MIN_FRAME = ip_hl + l4_hl                          # at least the headers
    _MAX_FRAME = 65535
    if real_l3 < ip_hl + l4_hl:
        real_l3 = ip_hl + l4_hl
    elif real_l3 > _MAX_FRAME:
        real_l3 = _MAX_FRAME

    real_bytes = l2_len + real_l3                       # full frame size on the wire
    payload = real_l3 - ip_hl - l4_hl                   # declared L4 payload length
    if payload < 0:
        payload = 0

    fam = socket.AF_INET6 if v6 else socket.AF_INET
    return {
        "src_ip": socket.inet_ntop(fam, l3.src),
        "dst_ip": socket.inet_ntop(fam, l3.dst),
        "src_port": sport,
        "dst_port": dport,
        "protocol": proto,
        "bytes": int(real_bytes),
        "payload_len": int(payload),
        "tcp_window": window,
        "tcp_flags": flags,
        "ts": float(ts),
        "syn": syn, "ack": ack, "is_ch": is_ch,
    }


def _evidence(f):
    """(confidence, initiator_endpoint) for who opened this connection."""
    if f["protocol"] != "TCP":
        return _EV_NONE, None
    if f["syn"] and not f["ack"]:
        return _EV_SYN, (f["src_ip"], f["src_port"])       # pure SYN: sender = client
    if f["syn"] and f["ack"]:
        return _EV_SYNACK, (f["dst_ip"], f["dst_port"])    # SYN+ACK: receiver = client
    if f["is_ch"]:
        return _EV_CH, (f["src_ip"], f["src_port"])        # ClientHello: sender = client
    return _EV_NONE, None


def _canonical(src_ip, src_port, dst_ip, dst_port, proto):
    a, b = (src_ip, src_port), (dst_ip, dst_port)
    lo, hi = (a, b) if a <= b else (b, a)
    return f"{lo[0]}:{lo[1]}<->{hi[0]}:{hi[1]}/{proto}"


def _flush(buffer, writer):
    df = pd.DataFrame(buffer, columns=COLUMNS)
    df["datetime"] = pd.to_datetime(df["ts"], unit="s")
    writer.write_table(pa.Table.from_pandas(df, schema=SCHEMA, preserve_index=False))


def _emit(tag, files_done, files_total, seen, t0):
    el = time.perf_counter() - t0
    rate = seen / el if el > 0 else 0.0
    print(f"[{tag}] {files_done}/{files_total} files | {seen:,} pkts | "
          f"{el:6.1f}s | {rate:,.0f} pkt/s", flush=True)


def _iter_file_packets(path, tag, counters):
    """Yield (ts, buf, extract_l3) for every readable packet; tolerant of
    truncation. Increments counters['bad_files'] if the file is unreadable."""
    try:
        fh = open(path, "rb")
    except OSError as e:
        counters["bad_files"] += 1
        print(f"[{tag}] WARN cannot open {os.path.basename(path)}: {e}", flush=True)
        return
    try:
        try:
            reader = _open_reader(fh)
            dl = reader.datalink()
            extract_l3 = _make_l3_extractor(dl)
            l2_len = _l2_len(dl)
        except Exception as e:
            counters["bad_files"] += 1
            print(f"[{tag}] WARN unreadable {os.path.basename(path)}: {e}", flush=True)
            return
        it = iter(reader)
        while True:
            try:
                ts, buf = next(it)
            except StopIteration:
                break
            except _TRUNC_ERRORS:
                counters["truncated_files"] += 1
                print(f"[{tag}] WARN truncated {os.path.basename(path)} "
                      f"(stopped at last complete packet)", flush=True)
                break
            except Exception as e:
                counters["truncated_files"] += 1
                print(f"[{tag}] WARN read error in {os.path.basename(path)}: "
                      f"{type(e).__name__}: {e}", flush=True)
                break
            yield ts, buf, extract_l3, l2_len
    finally:
        fh.close()


def _file_time_offset(pcap_path):
    """Per-FILE time correction. Each capture file 'traffic_<epoch_ms>.pcap'
    encodes its real wall-clock start time in the filename. Some files store
    RELATIVE (since-boot) timestamps, others already store absolute epoch time,
    and a single session can mix both (e.g. the capture device rebooted or its
    clock got NTP-synced mid-session), each relative file having its OWN boot
    base. So the offset MUST be computed per file, not per session.

    Returns the offset to ADD to this file's packet times so they become absolute
    epoch seconds: (filename_epoch - first_packet_ts) if the file is relative, or
    0.0 if it already looks absolute (first packet within a day of the filename
    time) or the filename has no epoch.
    """
    m = re.search(r"(\d{10,})", os.path.basename(pcap_path))
    if not m:
        return 0.0
    raw = int(m.group(1))
    fname_s = raw / 1000.0 if raw > 1e11 else float(raw)     # 13-digit names are ms
    try:
        with open(pcap_path, "rb") as fh:
            first = next(iter(_open_reader(fh)), None)
    except Exception:
        return 0.0
    if first is None:
        return 0.0
    diff = fname_s - float(first[0])
    return diff if abs(diff) >= 86400 else 0.0               # within a day -> already absolute


def _write_part(part_path, pcap_path, capture_id, conn_state, batch_size, counters, tag,
                time_offset=0.0, include_icmp=False):
    """Two-pass over one pcap: (1) buffer + finalize initiators for new
    connections, (2) assign direction/IAT in capture order and write the part
    atomically. Returns True if a part was finalized, False if the file was
    unreadable (left for a later retry)."""
    tmp_path = part_path + ".tmp"
    bad_before = counters["bad_files"]

    # ---- PASS 1: parse once, buffer, decide initiators ----
    records = []
    new_this_file = set()
    for ts, buf, extract_l3, l2_len in _iter_file_packets(pcap_path, tag, counters):
        counters["seen"] += 1
        try:
            f = _dissect(extract_l3(buf), ts, l2_len, include_icmp)
        except Exception:
            f = None
        if f is None:
            counters["skipped"] += 1
            continue

        key = _canonical(f["src_ip"], f["src_port"], f["dst_ip"], f["dst_port"], f["protocol"])
        st = conn_state.get(key)
        if st is None:
            st = {"ep_a": (f["src_ip"], f["src_port"]),      # first-seen sender
                  "ep_b": (f["dst_ip"], f["dst_port"]),      # first-seen receiver
                  "init_ep": (f["src_ip"], f["src_port"]),   # provisional initiator
                  "ev": _EV_NONE, "last": None}
            conn_state[key] = st
            new_this_file.add(key)

        # Only connections first seen in THIS file may still be relabeled;
        # ones carried from earlier (already-written) files are locked.
        if key in new_this_file:
            lvl, init_ep = _evidence(f)
            if lvl > st["ev"]:
                st["ev"] = lvl
                st["init_ep"] = init_ep

        records.append((key, f["src_ip"], f["src_port"], f["dst_ip"], f["dst_port"],
                        f["protocol"], f["bytes"], f["payload_len"],
                        f["tcp_window"], f["tcp_flags"], f["ts"] + time_offset))

    if counters["bad_files"] > bad_before:
        return False   # unreadable: no part, retry on a later run

    # ---- PASS 2: assign final direction + IAT, write ----
    writer = pq.ParquetWriter(tmp_path, SCHEMA, compression="snappy")
    buffer = []
    try:
        for (key, sip, sport, dip, dport, proto, nbytes, plen, win, flags, ts) in records:
            st = conn_state[key]
            init_ep = st["init_ep"]
            resp_ep = st["ep_b"] if init_ep == st["ep_a"] else st["ep_a"]
            is_fwd = (sip, sport) == init_ep
            if st["last"] is None:
                iat = 0.0
            else:
                iat = ts - st["last"]
                if iat < 0:
                    iat = 0.0
            st["last"] = ts

            buffer.append({
                "capture_id": capture_id, "conn_id": key,
                "src_ip": init_ep[0], "dst_ip": resp_ep[0],
                "src_port": init_ep[1], "dst_port": resp_ep[1],
                "protocol": proto,
                "direction": "fwd" if is_fwd else "bwd",
                "is_forward": bool(is_fwd),
                "ts": ts, "datetime": None,
                "bytes": nbytes, "payload_len": plen,
                "iat": iat,
                "tcp_window": (np.nan if win is None else float(win)),
                "tcp_flags": flags,
            })
            if len(buffer) >= batch_size:
                _flush(buffer, writer)
                counters["written"] += len(buffer)
                buffer = []
        if buffer:
            _flush(buffer, writer)
            counters["written"] += len(buffer)
    finally:
        writer.close()

    os.replace(tmp_path, part_path)   # atomic finalize
    return True


def _merge_parquets(files, out_path):
    writer = pq.ParquetWriter(out_path, SCHEMA, compression="snappy")
    try:
        for f in files:
            if not os.path.exists(f):
                continue
            pf = pq.ParquetFile(f)
            for i in range(pf.num_row_groups):
                writer.write_table(pf.read_row_group(i))
    finally:
        writer.close()


def _process_session(capture_id, pcap_paths, out_dir, batch_size,
                     progress_every, progress_secs, overwrite, keep_parts,
                     include_icmp=False):
    counters = {"seen": 0, "written": 0, "skipped": 0, "connections": 0,
                "bad_files": 0, "truncated_files": 0, "resumed_files": 0}
    tag = capture_id[:8]
    total = len(pcap_paths)
    parts_dir = os.path.join(out_dir, capture_id)
    final_path = os.path.join(out_dir, capture_id + ".parquet")
    os.makedirs(parts_dir, exist_ok=True)

    conn_state = {}
    part_paths = []
    t0 = time.perf_counter()
    last_print = t0
    print(f"[{tag}] start: {total} pcap files", flush=True)

    n_corrected = 0
    for fi, path in enumerate(pcap_paths, 1):
        stem = os.path.splitext(os.path.basename(path))[0]
        part_path = os.path.join(parts_dir, stem + ".parquet")

        if (not overwrite) and os.path.exists(part_path):
            counters["resumed_files"] += 1
            part_paths.append(part_path)
        else:
            time_offset = _file_time_offset(path)          # PER-FILE offset
            if time_offset:
                n_corrected += 1
            ok = _write_part(part_path, path, capture_id, conn_state,
                             batch_size, counters, tag, time_offset, include_icmp)
            if ok:
                part_paths.append(part_path)

        if counters["seen"] % _CLOCK_CHECK_EVERY == 0:
            now = time.perf_counter()
            if now - last_print >= progress_secs:
                _emit(tag, fi, total, counters["seen"], t0)
                last_print = now
        if (fi % progress_every == 0) or (time.perf_counter() - last_print >= progress_secs):
            _emit(tag, fi, total, counters["seen"], t0)
            last_print = time.perf_counter()

    existing = [p for p in part_paths if os.path.exists(p)]
    tmp_final = final_path + ".tmp"
    _merge_parquets(existing, tmp_final)
    os.replace(tmp_final, final_path)
    if not keep_parts:
        shutil.rmtree(parts_dir, ignore_errors=True)

    counters["connections"] = len(conn_state)
    el = time.perf_counter() - t0
    extra = ""
    if counters["resumed_files"]:
        extra += f" | resumed {counters['resumed_files']}"
    if counters["truncated_files"] or counters["bad_files"]:
        extra += f" | truncated {counters['truncated_files']}, bad {counters['bad_files']}"
    if n_corrected:
        extra += f" | time-corrected {n_corrected}/{total} files (relative->absolute)"
    print(f"[{tag}] finished -> {os.path.basename(final_path)} | "
          f"{counters['written']:,} rows this run | {counters['connections']:,} conns | "
          f"{el:6.1f}s | {(counters['seen']/el if el > 0 else 0):,.0f} pkt/s{extra}",
          flush=True)
    return counters, final_path


def _session_worker(capture_id, pcap_paths, out_dir, batch_size,
                    progress_every, progress_secs, overwrite, keep_parts,
                    include_icmp=False):
    return _process_session(capture_id, pcap_paths, out_dir, batch_size,
                            progress_every, progress_secs, overwrite, keep_parts,
                            include_icmp)


def _ts_key(path):
    m = re.search(r"(\d+)", os.path.basename(path))
    return (int(m.group(1)) if m else 0, os.path.basename(path))


def discover_sessions(pcap_dir):
    if not os.path.isdir(pcap_dir):
        raise FileNotFoundError(f"pcap dir not found: {pcap_dir}")
    sessions = {}
    for entry in sorted(os.listdir(pcap_dir)):
        sub = os.path.join(pcap_dir, entry)
        if not os.path.isdir(sub):
            continue
        pcaps = glob.glob(os.path.join(sub, "*.pcap")) + glob.glob(os.path.join(sub, "*.pcapng"))
        pcaps = sorted(set(pcaps), key=_ts_key)
        if pcaps:
            sessions[entry] = pcaps
    return sessions


def _sum_counters(dst, src):
    for k in dst:
        dst[k] += src.get(k, 0)


def main():
    ap = argparse.ArgumentParser(description="Fast, fault-tolerant, resumable pcap->parquet extractor with SYN/SYN-ACK/ClientHello initiator detection.")
    ap.add_argument("--root", default=DEFAULT_ROOT)
    ap.add_argument("--pcap-dir", default=None,
                    help="Override input dir (default: <root>/data/unfiltered_pcap)")
    ap.add_argument("--out-dir", default=None,
                    help="Override output dir (default: <root>/data/parquet)")
    ap.add_argument("--only", default=None, help="One session hash folder (name or prefix)")
    ap.add_argument("--combined", default=None,
                    help="Merge ALL sessions into this single parquet (still parallel per session)")
    ap.add_argument("--workers", type=int, default=None,
                    help="Parallel worker processes (default: min(cpu_count, #sessions))")
    ap.add_argument("--batch-size", type=int, default=200_000,
                    help="Rows buffered before each Parquet flush (default 200k)")
    ap.add_argument("--overwrite", action="store_true",
                    help="Ignore existing parts/outputs and reprocess everything")
    ap.add_argument("--keep-parts", action="store_true",
                    help="Keep per-file parquet parts under <out>/<hash>/ after merging")
    ap.add_argument("--progress-every", type=int, default=50,
                    help="Print progress every N pcap files per session (default 50)")
    ap.add_argument("--progress-secs", type=float, default=5.0,
                    help="Also print progress at least every N seconds (default 5)")
    ap.add_argument("--include-icmp", action="store_true",
                    help="Also extract ICMP packets (as host-to-host connections with "
                         "ports 0). Default off. Turn on for captures with ICMP-based attacks.")
    args = ap.parse_args()

    pcap_dir = args.pcap_dir or os.path.join(args.root, "data", "unfiltered_pcap")
    out_dir = args.out_dir or os.path.join(args.root, "data", "parquet")

    sessions = discover_sessions(pcap_dir)
    if args.only:
        sessions = {k: v for k, v in sessions.items()
                    if k == args.only or k.startswith(args.only)}
    if not sessions:
        print(f"No sessions with pcaps found under {pcap_dir}")
        return

    if args.combined:
        work_dir = tempfile.mkdtemp(prefix="p2p_")
        os.makedirs(os.path.dirname(os.path.abspath(args.combined)), exist_ok=True)
    else:
        work_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)

    pending = {}
    skipped_sessions = 0
    for cid, paths in sessions.items():
        final = os.path.join(work_dir, cid + ".parquet")
        if (not args.overwrite) and (not args.combined) and os.path.exists(final):
            skipped_sessions += 1
        else:
            pending[cid] = paths

    print(f"Found {len(sessions)} session(s) under {pcap_dir}; "
          f"{len(pending)} to process, {skipped_sessions} already complete.")
    for k, v in sessions.items():
        mark = "" if k in pending else "  [done]"
        print(f"  {k}  ({len(v)} pcap files){mark}")

    totals = {"seen": 0, "written": 0, "skipped": 0, "connections": 0,
              "bad_files": 0, "truncated_files": 0, "resumed_files": 0}
    produced = []
    failed = []

    if pending:
        n_workers = args.workers or min(os.cpu_count() or 1, len(pending))
        n_workers = max(1, min(n_workers, len(pending)))
        print(f"Using {n_workers} worker(s).")

        jobs = [(cid, paths, work_dir, args.batch_size, args.progress_every,
                 args.progress_secs, args.overwrite, args.keep_parts, args.include_icmp)
                for cid, paths in pending.items()]

        if n_workers == 1:
            for job in jobs:
                cid = job[0]
                try:
                    counters, final_path = _session_worker(*job)
                    _sum_counters(totals, counters)
                    produced.append(final_path)
                except Exception as e:
                    failed.append(cid)
                    print(f"[{cid[:8]}] FAILED: {type(e).__name__}: {e} "
                          f"(finished files are safe on disk; rerun to resume)", flush=True)
        else:
            with ProcessPoolExecutor(max_workers=n_workers) as ex:
                futs = {ex.submit(_session_worker, *job): job[0] for job in jobs}
                for fut in as_completed(futs):
                    cid = futs[fut]
                    try:
                        counters, final_path = fut.result()
                        _sum_counters(totals, counters)
                        produced.append(final_path)
                    except Exception as e:
                        failed.append(cid)
                        print(f"[{cid[:8]}] FAILED: {type(e).__name__}: {e} "
                              f"(finished files are safe on disk; rerun to resume)", flush=True)

    if args.combined:
        finals = [os.path.join(work_dir, cid + ".parquet") for cid in sessions]
        finals = [f for f in finals if os.path.exists(f)]
        print("Merging sessions ...", flush=True)
        tmp_comb = args.combined + ".tmp"
        _merge_parquets(sorted(finals), tmp_comb)
        os.replace(tmp_comb, args.combined)
        shutil.rmtree(work_dir, ignore_errors=True)
        print(f"Wrote combined parquet: {args.combined}")
    else:
        print(f"Outputs in {out_dir} ({len(produced)} written this run, "
              f"{skipped_sessions} already complete).")

    if failed:
        print(f"WARNING: {len(failed)} session(s) failed: "
              f"{', '.join(c[:8] for c in failed)} -- rerun to resume them.")

    print(f"Packets seen: {totals['seen']:,} | written: {totals['written']:,} | "
          f"skipped (non IP/TCP/UDP): {totals['skipped']:,} | "
          f"resumed files: {totals['resumed_files']:,} | "
          f"truncated files: {totals['truncated_files']:,} | "
          f"bad files: {totals['bad_files']:,} | "
          f"connections (summed over sessions): {totals['connections']:,}")


if __name__ == "__main__":
    main()