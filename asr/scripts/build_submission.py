#!/usr/bin/env python3
"""Build a NADI2026 subtask-1 submission folder from a decoded hypothesis directory.

Eight UTF-8 files, one per dialect (Algeria.txt ... Yemen.txt), one hypothesis per line, in the
original released parquet row order for that dialect. Supply a manifest with a zero-based
"row" field recorded per country while reading the released files. The scorer pairs by line
number, so preserving this order is essential. See ../README.md for the manifest contract.

Usage:
  build_submission.py --hyp runs/test/<dir> --manifest manifests/nadi_robust_test.jsonl \
                      --out work/asr/submission [--expect 500]
"""
import argparse
import json
import os
import sys

DIALECTS = ["Algeria", "Egypt", "Jordan", "Mauritania", "Morocco", "Palestine", "UAE", "Yemen"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hyp", required=True, help="dir containing hyp.shard_*")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--expect", type=int, default=0, help="expected lines per dialect (0 = infer)")
    a = ap.parse_args()

    order = {}   # utt_id -> (dialect, row)
    per_dia_n = {}
    for line in open(a.manifest, encoding="utf-8"):
        r = json.loads(line)
        order[r["utt_id"]] = (r["dataset"], r["row"])
        per_dia_n[r["dataset"]] = per_dia_n.get(r["dataset"], 0) + 1

    hyps = {}
    for fn in sorted(os.listdir(a.hyp)):
        if not fn.startswith("hyp.shard_"):
            continue
        for line in open(os.path.join(a.hyp, fn), encoding="utf-8"):
            line = line.strip()
            if line:
                j = json.loads(line)
                hyps[j["utt_id"]] = j["hyp"]

    os.makedirs(a.out, exist_ok=True)
    problems = []
    total = 0
    for dia in DIALECTS:
        want = [(row, uid) for uid, (d, row) in order.items() if d == dia]
        want.sort()
        lines = []
        for row, uid in want:
            h = hyps.get(uid)
            if h is None:
                problems.append("MISSING hyp for %s" % uid)
                h = ""
            # a stray newline would shift every following line against the reference
            h = " ".join(h.replace("\n", " ").replace("\r", " ").split())
            if not h:
                problems.append("EMPTY hyp for %s" % uid)
            lines.append(h)
        n = a.expect or per_dia_n[dia]
        if len(lines) != n:
            problems.append("%s: %d lines, expected %d" % (dia, len(lines), n))
        with open(os.path.join(a.out, "%s.txt" % dia), "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        total += len(lines)
        print("  %-12s %d lines" % (dia, len(lines)))

    print("TOTAL %d lines -> %s" % (total, a.out))
    empt = [p for p in problems if p.startswith("EMPTY")]
    miss = [p for p in problems if not p.startswith("EMPTY")]
    if empt:
        print("NOTE: %d empty hypotheses (model produced no text; kept as blank lines)" % len(empt))
    if miss:
        print("PROBLEMS:")
        for p in miss[:20]:
            print("   ", p)
        sys.exit(1)
    print("PROBLEMS: none" if not empt else "PROBLEMS: none besides the empty lines noted above")


if __name__ == "__main__":
    main()
