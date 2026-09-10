#!/usr/bin/env python3
"""Assemble manifests/E15.jsonl = E11_dial.jsonl + the E15 short clips.

E11_dial.jsonl already carries {"audio","text","dialect"} for the 185,798 E11 rows; the new
short-clip rows come from manifests/E15_shortclips.jsonl (built by scripts/build_shortclips.py)
and are reduced to the same three keys so every row of E15.jsonl is format-identical.
"""
import json
import os

RF = os.path.abspath(os.environ.get("NADI_ASR_WORKSPACE", "work/asr"))
E11 = os.path.join(RF, "manifests/E11_dial.jsonl")
SHORT = os.path.join(RF, "manifests/E15_shortclips.jsonl")
OUT = os.path.join(RF, "manifests/E15.jsonl")
PREFIX = "language Arabic<asr_text>"
DIALECTS = {"Algeria", "Egypt", "Jordan", "Mauritania", "Morocco", "Palestine", "UAE", "Yemen"}


def dialect_from_path(p):
    """Same rule as asr/train/common.py: scan parts after /audio/."""
    parts = p.split("/audio/")[-1].split("/")
    for c in parts:
        if c in DIALECTS:
            return c
    return ""


def main():
    n_e11 = n_new = 0
    with open(OUT, "w", encoding="utf-8") as o:
        for line in open(E11, encoding="utf-8"):
            r = json.loads(line)
            assert "original_val" not in r["audio"], f"LEAKAGE in E11 row: {r['audio']}"
            assert r["text"].startswith(PREFIX), f"bad prefix: {r['text'][:40]}"
            d = r.get("dialect") or dialect_from_path(r["audio"])
            assert d in DIALECTS, f"no dialect for {r['audio']}"
            o.write(json.dumps({"audio": r["audio"], "text": r["text"], "dialect": d},
                               ensure_ascii=False) + "\n")
            n_e11 += 1
        for line in open(SHORT, encoding="utf-8"):
            r = json.loads(line)
            assert "original_val" not in r["audio"] and "original_val" not in r["src"], \
                f"LEAKAGE in short row: {r}"
            assert r["text"].startswith(PREFIX), f"bad prefix: {r['text'][:40]}"
            d = r.get("dialect") or dialect_from_path(r["audio"])
            assert d in DIALECTS, f"no dialect for {r['audio']}"
            o.write(json.dumps({"audio": r["audio"], "text": r["text"], "dialect": d},
                               ensure_ascii=False) + "\n")
            n_new += 1
    print(f"[E15] {n_e11} E11 rows + {n_new} short-clip rows = {n_e11 + n_new} -> {OUT}")


if __name__ == "__main__":
    main()
