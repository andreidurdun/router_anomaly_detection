#!/usr/bin/env python3
"""
refactor_sliding_results.py  —  reorganize EXISTING sliding results into per-representation folders.

The sliding script used to write every series to
    results/sliding/<connection>/<target>/...
so runs of different representations (per-flow 1min / 5min, router x oui) landed in the same
place. This moves each series into a representation subfolder (the new "tag" layout):
    results/sliding/<tag>/<connection>/<target>/...
so the dashboard's "Representation" selector can separate them. NO experiment is re-run.

Each series is grouped by its (context C, horizon H) — that pair IS the representation here
(flow 1min / flow 5min / router x oui were each run with a different C,H). The folder name:
    1) --map "C48_H6=flow_5min,C128_H12=flow_1min,C288_H12=oui_router"  -> your chosen name;
    2) else meta["tag"], or "<mode>_<datafile>_C<C>_H<H>" from a newer sliding_meta.json;
    3) else plain "C<C>_H<H>".
NOTE: the mode is NO LONGER guessed from the connection name. Old flow conn_ids
(router_index_timestamp) have no "_to_", so the old heuristic mislabelled every flow series as
"oui-router". Use --map to name representations correctly in one pass.

Collision note: if two representations shared a connection NAME (the 1min and 5min per-flow
parquets reuse the same conn_ids), only the LAST run's sliding_scores.csv / sliding_series.csv /
meta survived on disk; the PNGs of the others, which carry C/H in their name, survived too. Those
orphan PNGs are moved to their own tag folder and reported; their per-series scores are gone and
cannot be recovered without re-running those representations (now collision-safe via the tag layout).

    python refactor_sliding_results.py --dir results/sliding --dry-run      # preview everything
    python refactor_sliding_results.py --dir results/sliding \
        --map "C48_H6=flow_5min,C128_H12=flow_1min,C288_H12=oui_router"     # move all, named

    # recover ONLY oui x router (intact) and delete the collided flow folders to re-run them:
    python refactor_sliding_results.py --dir results/sliding \
        --only "C288_H12" --map "C288_H12=oui_router" --drop-unselected --dry-run   # check first
    python refactor_sliding_results.py --dir results/sliding \
        --only "C288_H12" --map "C288_H12=oui_router" --drop-unselected
"""
import argparse
import glob
import json
import os
import re
import shutil

import pandas as pd

PNG_RE = re.compile(r"sliding_C(\d+)_H(\d+)_")


def read_meta(folder):
    f = os.path.join(folder, "sliding_meta.json")
    if os.path.exists(f):
        try:
            return json.load(open(f))
        except Exception:
            return None
    return None


def png_ch(folder):
    """Set of (C, H) found in the PNG filenames of a series folder."""
    out = set()
    for p in glob.glob(os.path.join(folder, "sliding_C*_H*_*.png")):
        m = PNG_RE.search(os.path.basename(p))
        if m:
            out.add((int(m.group(1)), int(m.group(2))))
    return out


def build_tag(meta, C, H, tagmap):
    """Representation tag for a series. Priority:
       1) an explicit --map entry for this C/H (e.g. 'C48_H6=flow_5min');
       2) the tag/mode/data recorded in sliding_meta.json (newer runs);
       3) plain 'C<C>_H<H>' (old runs have no mode -> do NOT guess it)."""
    key = f"C{C}_H{H}"
    if tagmap and key in tagmap:
        return tagmap[key]
    if meta and meta.get("tag"):
        return meta["tag"]
    parts = []
    if meta and meta.get("mode"):
        parts.append(str(meta["mode"]))
    if meta and meta.get("data"):
        parts.append(os.path.splitext(str(meta["data"]))[0])
    parts.append(key)
    return "_".join(parts)


def primary_ch(meta, folder):
    """(C, H) of the surviving scores/series in this folder."""
    if meta and meta.get("context") is not None and meta.get("horizon") is not None:
        return int(meta["context"]), int(meta["horizon"])
    chs = png_ch(folder)
    return sorted(chs)[0] if chs else (None, None)


def move_or_copy(src, dst, copy):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if copy:
        shutil.copy2(src, dst)
    else:
        shutil.move(src, dst)


def regenerate_summary(tag_dir):
    rows = []
    for sc in glob.glob(os.path.join(tag_dir, "*", "*", "sliding_scores.csv")):
        tgt_dir = os.path.dirname(sc)
        conn = os.path.basename(os.path.dirname(tgt_dir))
        target = os.path.basename(tgt_dir)
        try:
            df = pd.read_csv(sc)
        except Exception:
            continue
        meta = read_meta(tgt_dir) or {}
        row = dict(connection=conn, target=target,
                   n_bins=meta.get("n", ""), n_windows=len(meta.get("starts", [])) or "",
                   n_eval=len(df))
        for c in df.columns:
            if c.startswith("is_anom_m"):
                row["anom_" + c.replace("is_anom_", "")] = int(df[c].astype(bool).sum())
            if c == "is_anom_consensus":
                row["anom_consensus"] = int(df[c].astype(bool).sum())
        rows.append(row)
    if rows:
        pd.DataFrame(rows).to_csv(os.path.join(tag_dir, "summary.csv"), index=False)
    return len(rows)


def main():
    ap = argparse.ArgumentParser(description="Reorganize flat sliding results into per-representation tag folders")
    ap.add_argument("--dir", required=True, help="results/sliding directory")
    ap.add_argument("--dry-run", action="store_true", help="only print the plan")
    ap.add_argument("--copy", action="store_true", help="copy instead of move")
    ap.add_argument("--map", default="",
                    help="comma list of C<C>_H<H>=name pairs to name representations, e.g. "
                         "'C48_H6=flow_5min,C128_H12=flow_1min,C288_H12=oui_router'")
    ap.add_argument("--only", default="",
                    help="comma list of C<C>_H<H> to process; all other (C,H) are left untouched "
                         "(e.g. 'C288_H12' to recover only oui x router and re-run the rest later)")
    ap.add_argument("--drop-unselected", action="store_true",
                    help="with --only: DELETE the flat folders of the (C,H) you did NOT select "
                         "(their partial/collided data; use when you will re-run those representations)")
    args = ap.parse_args()

    tagmap = {}
    for kv in str(args.map).split(","):
        kv = kv.strip()
        if "=" in kv:
            k, v = kv.split("=", 1)
            tagmap[k.strip()] = v.strip()

    only = {s.strip() for s in str(args.only).split(",") if s.strip()}
    if args.drop_unselected and not only:
        raise SystemExit("--drop-unselected requires --only (it deletes the (C,H) you did not select).")

    root = args.dir
    if not os.path.isdir(root):
        raise SystemExit(f"not a directory: {root}")

    # flat series = <root>/<connection>/<target>/sliding_scores.csv  (depth 2)
    flat = sorted(glob.glob(os.path.join(root, "*", "*", "sliding_scores.csv")))
    if not flat:
        print("No flat-layout series found (nothing like <root>/<conn>/<target>/sliding_scores.csv). "
              "Results may already be in the tag layout — nothing to do.")
        return

    print(f"Found {len(flat)} flat series under {root}\n")
    tags_touched = set()
    tag_counts = {}
    skipped = []            # (tgt_dir, conn, target, key) left untouched because of --only
    n_moved = n_orphan = n_collide = 0

    for sc in flat:
        tgt_dir = os.path.dirname(sc)
        target = os.path.basename(tgt_dir)
        conn = os.path.basename(os.path.dirname(tgt_dir))
        meta = read_meta(tgt_dir)
        C, H = primary_ch(meta, tgt_dir)
        if C is None:
            print(f"  SKIP {conn}/{target}: no meta and no C/H in PNG names")
            continue
        key = f"C{C}_H{H}"
        if only and key not in only:
            skipped.append((tgt_dir, conn, target, key))
            continue
        tag = build_tag(meta, C, H, tagmap)
        tags_touched.add(tag)
        tag_counts[tag] = tag_counts.get(tag, 0) + 1
        dst_dir = os.path.join(root, tag, conn, target)

        # detect stray PNGs of OTHER (C,H) that collided into this folder
        other_ch = sorted(ch for ch in png_ch(tgt_dir) if ch != (C, H))
        if other_ch:
            n_collide += 1

        print(f"  {conn}/{target}  ->  {tag}/{conn}/{target}"
              + (f"   (+{len(other_ch)} orphan PNG set(s) {other_ch})" if other_ch else ""))

        if args.dry_run:
            continue

        # 1) move the primary (C,H) files: everything EXCEPT the stray-other-CH PNGs
        os.makedirs(dst_dir, exist_ok=True)
        for item in os.listdir(tgt_dir):
            src = os.path.join(tgt_dir, item)
            m = PNG_RE.search(item)
            if m and (int(m.group(1)), int(m.group(2))) in other_ch:
                continue  # leave for step 2
            move_or_copy(src, os.path.join(dst_dir, item), args.copy)
        n_moved += 1

        # 2) relocate orphan PNGs (png-only, no recoverable scores) to their own tag folders
        for (c2, h2) in other_ch:
            if only and f"C{c2}_H{h2}" not in only:
                continue  # unselected representation: leave its orphan PNGs to be dropped/re-run
            otag = build_tag(None, c2, h2, tagmap)
            odst = os.path.join(root, otag, conn, target)
            os.makedirs(odst, exist_ok=True)
            for p in glob.glob(os.path.join(tgt_dir, f"sliding_C{c2}_H{h2}_*.png")):
                move_or_copy(p, os.path.join(odst, os.path.basename(p)), args.copy)
                n_orphan += 1
            tags_touched.add(otag)

        # 3) remove now-empty source dirs (only when moving)
        if not args.copy:
            for d in (tgt_dir, os.path.dirname(tgt_dir)):
                if os.path.isdir(d) and not os.listdir(d):
                    os.rmdir(d)

    # report / drop the (C,H) left untouched by --only
    if skipped:
        skip_keys = {}
        for (_, _, _, key) in skipped:
            skip_keys[key] = skip_keys.get(key, 0) + 1
        print(f"\n{len(skipped)} series left untouched by --only "
              f"(representations {dict(sorted(skip_keys.items()))}).")
        if args.drop_unselected:
            conns_to_clean = set()
            for (tgt_dir, conn, target, key) in skipped:
                conns_to_clean.add(os.path.dirname(tgt_dir))
                if args.dry_run:
                    print(f"  [dry-run] would DELETE {conn}/{target}  ({key})")
                else:
                    shutil.rmtree(tgt_dir, ignore_errors=True)
            if not args.dry_run:
                for cdir in conns_to_clean:               # drop now-empty connection dirs
                    if os.path.isdir(cdir) and not os.listdir(cdir):
                        os.rmdir(cdir)
                print(f"  deleted {len(skipped)} unselected flat series "
                      f"(you will re-run these representations).")
        else:
            print("  (kept in place; add --drop-unselected to delete them before re-running.)")

    # tidy old top-level summary_*.csv into an _old_summaries/ folder
    old_sum = glob.glob(os.path.join(root, "summary_*.csv"))
    if old_sum and not args.dry_run:
        odir = os.path.join(root, "_old_summaries")
        os.makedirs(odir, exist_ok=True)
        for s in old_sum:
            move_or_copy(s, os.path.join(odir, os.path.basename(s)), args.copy)

    if not args.dry_run:
        print()
        for tag in sorted(tags_touched):
            n = regenerate_summary(os.path.join(root, tag))
            print(f"  regenerated summary.csv for '{tag}'  ({n} series)")

    n_series = n_moved if not args.dry_run else sum(tag_counts.values())
    print(f"\n{'[dry-run] would move' if args.dry_run else 'Moved'} {n_series} series into "
          f"{len(tags_touched)} representation(s):")
    for tag in sorted(tags_touched):
        print(f"    {tag:30s}  {tag_counts.get(tag, 0)} series"
              + ("  (recoverable scores)" if tag_counts.get(tag, 0) else "  (orphan PNGs only — scores lost)"))
    if n_collide:
        print(f"\nWARNING: {n_collide} connection folder(s) held PNGs from more than one (C,H), i.e. the "
              f"same connection name was reused across representations (the 1-min and 5-min per-flow "
              f"parquets share conn_ids). For each such connection only the LAST run's "
              f"sliding_scores.csv/series/meta survived; {n_orphan} orphan PNG(s) were relocated to the "
              f"other representation's tag folder, but those overwritten scores are GONE. To get complete "
              f"results for the affected representation(s), re-run them with the new tag layout "
              f"(forecast_sliding_anomaly.py writes results/sliding/<tag>/... so runs no longer collide).")


if __name__ == "__main__":
    main()