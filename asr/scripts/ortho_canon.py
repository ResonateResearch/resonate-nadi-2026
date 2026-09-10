#!/usr/bin/env python3
"""Per-country orthographic canonicalization from competition training text only.

Use explicit country labels in the training manifest. Evaluation country
identity comes from the released country-organized task structure.
Public-export changes: require a training manifest, remove storage-depth
assumptions, and name the local-approximation scorer/output accurately.
"""
import argparse, collections, json, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from local_score import normalize, report  # noqa: E402

FOLD = str.maketrans({"ى": "ي", "ة": "ه", "إ": "ا", "أ": "ا", "آ": "ا", "ﻻ": "لا"})


def fold(s):
    return s.translate(FOLD)


def build_lexicon(train_manifest, minfrac, mincount):
    bydia = collections.defaultdict(lambda: collections.defaultdict(collections.Counter))
    for line in open(train_manifest, encoding="utf-8"):
        j = json.loads(line)
        dia = j.get("dialect") or j.get("dataset")
        if dia not in {"Algeria", "Egypt", "Jordan", "Mauritania", "Morocco", "Palestine", "UAE", "Yemen"}:
            raise ValueError("Training rows require an explicit valid dialect/country field")
        for w in normalize(j["text"].split("<asr_text>")[-1]).split():
            bydia[dia][fold(w)][w] += 1
    C = {}
    for d, m in bydia.items():
        c = {}
        for k, v in m.items():
            tot = sum(v.values())
            top, n = v.most_common(1)[0]
            if tot >= mincount and n / tot >= minfrac:
                c[k] = top
        C[d] = c
    return C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--train", required=True, help="training JSONL with explicit dialect fields")
    ap.add_argument("--minfrac", type=float, default=0.6)
    ap.add_argument("--mincount", type=int, default=30)
    a = ap.parse_args()

    C = build_lexicon(a.train, a.minfrac, a.mincount)
    print("[ortho] lexicon: %s" % {d: len(v) for d, v in sorted(C.items())}, flush=True)

    os.makedirs(a.out, exist_ok=True)
    n = ch = 0
    with open(os.path.join(a.out, "hyp.shard_0"), "w", encoding="utf-8") as f:
        for fn in sorted(os.listdir(a.inp)):
            if not fn.startswith("hyp.shard_"):
                continue
            for line in open(os.path.join(a.inp, fn), encoding="utf-8"):
                line = line.strip()
                if not line:
                    continue
                j = json.loads(line)
                lex = C.get(j["dataset"], {})
                new = " ".join(lex.get(fold(w), w) for w in normalize(j["hyp"]).split())
                n += 1
                ch += (new != normalize(j["hyp"]))
                j["hyp"] = new
                f.write(json.dumps(j, ensure_ascii=False) + "\n")
    print("[ortho] %d utts, %d changed (%.1f%%) -> %s" % (n, ch, 100 * ch / max(n, 1), a.out))
    r = report(a.out)
    with open(os.path.join(a.out, "wer_report_local.json"), "w") as f:
        json.dump(r, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
