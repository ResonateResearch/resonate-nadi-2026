#!/usr/bin/env python3
"""Pivot-anchored ROVER over already-decoded hypothesis sets; CPU only.

Takes N runs/eval/<dir> that each contain hyp.shard_* in the nadi_val_eval.py schema, aligns
every system to the FIRST system's hypothesis (the pivot -- must be the strongest system),
and takes a weighted per-slot vote. Deletions vote for the empty word; insertions relative to
the pivot are dropped (the pivot anchors length, which is what keeps a strong pivot from being
dragged around by weaker voters).

Ties go to the pivot. Writes hyp.shard_0 and a local-approximation report.
See ../README.md for the exact submitted nine-system order and weights.
Public-export changes: documentation and local scorer import/output naming.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from local_score import normalize, report  # noqa: E402


def load_by_utt(d):
    out = {}
    for fn in sorted(os.listdir(d)):
        if fn.startswith("hyp.shard_"):
            for line in open(os.path.join(d, fn), encoding="utf-8"):
                line = line.strip()
                if line:
                    j = json.loads(line)
                    out[j["utt_id"]] = j
    return out


def align_to_pivot(piv, hyp):
    """Levenshtein-align hyp onto pivot. Returns list len(piv): word aligned to piv[i],
    "" for a deletion, None if unaligned. Insertions are dropped."""
    la, lb = len(piv), len(hyp)
    D = [[0] * (lb + 1) for _ in range(la + 1)]
    for i in range(la + 1):
        D[i][0] = i
    for j in range(lb + 1):
        D[0][j] = j
    for i in range(1, la + 1):
        for j in range(1, lb + 1):
            D[i][j] = min(D[i - 1][j] + 1, D[i][j - 1] + 1,
                          D[i - 1][j - 1] + (0 if piv[i - 1] == hyp[j - 1] else 1))
    out = [None] * la
    i, j = la, lb
    while i > 0 or j > 0:
        if i > 0 and j > 0 and D[i][j] == D[i - 1][j - 1] + (0 if piv[i - 1] == hyp[j - 1] else 1):
            out[i - 1] = hyp[j - 1]
            i -= 1
            j -= 1
        elif i > 0 and D[i][j] == D[i - 1][j] + 1:
            out[i - 1] = ""
            i -= 1
        else:
            j -= 1
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--systems", nargs="+", required=True, help="eval dirs; FIRST is the pivot")
    ap.add_argument("--weights", nargs="*", type=float, default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    ws = a.weights or [1.0] * len(a.systems)
    assert len(ws) == len(a.systems)

    S = [load_by_utt(d) for d in a.systems]
    utts = sorted(set.intersection(*[set(s) for s in S]))
    print("[rover] %d systems, %d common utts" % (len(S), len(utts)), flush=True)

    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "hyp.shard_0"), "w", encoding="utf-8") as f:
        for u in utts:
            piv = normalize(S[0][u]["hyp"]).split()
            if piv:
                votes = [dict() for _ in piv]
                for s, w in zip(S, ws):
                    for i, wd in enumerate(align_to_pivot(piv, normalize(s[u]["hyp"]).split())):
                        if wd is None:
                            continue
                        votes[i][wd] = votes[i].get(wd, 0.0) + w
                merged = []
                for i, v in enumerate(votes):
                    if not v:
                        merged.append(piv[i])
                        continue
                    # tie -> pivot word
                    bw = max(v.items(), key=lambda x: (x[1], x[0] == piv[i]))[0]
                    if bw:
                        merged.append(bw)
                hyp = " ".join(merged)
            else:
                hyp = ""
            f.write(json.dumps({"utt_id": u, "dataset": S[0][u]["dataset"],
                                "ref": S[0][u]["ref"], "hyp": hyp}, ensure_ascii=False) + "\n")
    print("[rover] wrote %s/hyp.shard_0" % a.out, flush=True)
    r = report(a.out)
    with open(os.path.join(a.out, "wer_report_local.json"), "w") as f:
        json.dump(r, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
