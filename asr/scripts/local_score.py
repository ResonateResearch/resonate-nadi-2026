#!/usr/bin/env python3
"""Local approximation for NADI 2026 ASR evaluation; NOT the official scorer.

The implemented normalization was calibrated against E11/checkpoint-6000's
confirmed organizer result (49.7195% macro WER), comparing eight per-variety
WER and eight CER values. It removes diacritics, replaces punctuation with
spaces, and collapses whitespace without letter folding. CER retains spaces.
Leaderboard feedback also guided subsequent experimental directions.

The training-harness scorer in train/common.py uses different normalization.
Only organizer-returned scores are official. Normalization/scoring operations
are unchanged in this public export; the former filename was official_wer.py.
"""
import json
import os
import re

# Arabic letters live at U+0621..U+064A and must never appear in the diacritic class.
_DIAC_CPS = (
    list(range(0x0610, 0x061B))   # Arabic signs / honorifics
    + list(range(0x064B, 0x0660))  # tanween, harakat, shadda, sukun
    + [0x0670]                     # superscript alef
    + list(range(0x06D6, 0x06EE))  # Quranic annotation marks
)
assert all(not (0x0621 <= c <= 0x064A) for c in _DIAC_CPS), "diacritic class overlaps Arabic letters"

DIAC = re.compile("[" + "".join(map(chr, _DIAC_CPS)) + "]")
TATWEEL = re.compile(chr(0x0640))          # kashida; inside the letter block by design
PUNCT_AR = "،؛؟٪٫٬٭۔"
PUNCT_EN = r"""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""
PUNCT = re.compile("[" + re.escape(PUNCT_AR + PUNCT_EN) + "]")


def normalize(s):
    """Local normalization: diacritics out, punctuation -> space, whitespace collapsed."""
    return " ".join(PUNCT.sub(" ", DIAC.sub("", s)).split())


# Guard: real dialectal Arabic must survive intact apart from the trailing question mark.
assert normalize("أنت مين كمان؟") == "أنت مين كمان", "normalizer is destroying Arabic text"
assert normalize("مَرْحَبًا،") == "مرحبا", "diacritic stripping broken"


def score_pairs(pairs):
    """pairs: [(ref, hyp)] -> (corpus_wer, corpus_cer, n_scored). Rows with empty refs dropped."""
    import jiwer
    refs = [normalize(r) for r, _ in pairs]
    hyps = [normalize(h) for _, h in pairs]
    keep = [(r, h) for r, h in zip(refs, hyps) if r.strip()]
    if not keep:
        return 0.0, 0.0, 0
    r = [x[0] for x in keep]
    h = [x[1] for x in keep]
    return float(jiwer.wer(r, h)), float(jiwer.cer(r, h)), len(keep)


def load_hyps(eval_dir):
    """Read runs/eval/<name>/hyp.shard_* -> {dialect: [(ref, hyp)]}."""
    per = {}
    for fn in sorted(os.listdir(eval_dir)):
        if not fn.startswith("hyp.shard_"):
            continue
        with open(os.path.join(eval_dir, fn), encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                per.setdefault(d.get("dataset", ""), []).append((d["ref"], d["hyp"]))
    return per


def report(eval_dir, verbose=True):
    """Local report for one eval dir. Returns dict with macro/pooled/per-variety rates."""
    per = load_hyps(eval_dir)
    out = {"per_dialect": {}, "n": sum(len(v) for v in per.values())}
    for dia in sorted(per):
        w, c, n = score_pairs(per[dia])
        out["per_dialect"][dia] = {"wer": w, "cer": c, "n": n}
    dias = sorted(out["per_dialect"])
    if dias:
        out["country_av_wer"] = sum(out["per_dialect"][d]["wer"] for d in dias) / len(dias)
        out["country_av_cer"] = sum(out["per_dialect"][d]["cer"] for d in dias) / len(dias)
        allp = [p for d in dias for p in per[d]]
        out["micro_wer"], out["micro_cer"], _ = score_pairs(allp)
    if verbose:
        print("=== %s (n=%d) ===" % (os.path.basename(eval_dir.rstrip("/")), out["n"]))
        for d in dias:
            v = out["per_dialect"][d]
            print("  %-12s wer %.6f  cer %.6f  (n=%d)" % (d, v["wer"], v["cer"], v["n"]))
        if dias:
            print("  country_av_wer %.6f   country_av_cer %.6f" % (out["country_av_wer"], out["country_av_cer"]))
            print("  micro_wer      %.6f   micro_cer      %.6f" % (out["micro_wer"], out["micro_cer"]))
    return out


if __name__ == "__main__":
    import sys
    for d in sys.argv[1:]:
        report(d)
        print()
