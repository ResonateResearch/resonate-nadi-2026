#!/usr/bin/env python3
"""Build the E16 manifests.

E16 = E14's manifest (E11_dial.jsonl, dialect-conditioned) + BOUNDARY-FILTERED short clips.

Why filter rather than take all 21,381: the clip TEXT is clean (verbatim gold word spans), but
the clip AUDIO boundaries are not. Measured on the source FLACs, 68.2% of E15 clips have
speech-level energy in the 30 ms immediately outside a cut edge, versus 45.0% for the real
val sub-1s clips they were built to imitate (both-edges-hot 21.8% vs 8.6%). A cut that lands
mid-speech puts audible words under the clip that are absent from the reference -> the model is
supervised to ignore audio, which is an insertion/length-prior shift. That is exactly E15's
observed failure signature (insertions rose in EVERY >1s bucket while deletions fell).

Two filters, both computable offline from the manifest + the CTC alignment, no audio reads:
  gap    >= 0.06  the manifest 'gap' field is min(lgap,rgap) over NON-edge boundaries
                  (9.0 sentinel = true utterance edge, which is a genuinely clean cut).
                  So this keeps clips whose every artificial cut has >=60 ms of silence.
  edgew  >= 0.04  the first and last word of the span must each have >=40 ms of aligned
                  extent. build_shortclips.py's own docstring notes CTC spans run ~40 ms
                  narrow per side, so a word with <=40 ms of support (mostly the 1-frame
                  conjunction 'و', mean 0.021 s) is truncated by the builder's own error model.

Modes:
  --mode filtered  (default) 185,798 + 15,417 = 201,215   <- E16-A / E16-D
  --mode identity            185,798 + 21,381 = 207,179   <- E16-C, must equal manifests/E15.jsonl
  --mode none                185,798                      <- E16-B (control), == E11_dial.jsonl
"""
import argparse, json, os, sys, collections
from pathlib import Path

RF = os.path.abspath(os.environ.get("NADI_ASR_WORKSPACE", "work/asr"))
MAN = os.path.join(RF, "manifests")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "train"))
from common import dialect_of, dialect_from_path  # trainer's own resolver  # noqa: E402

DIALECTS = ("Algeria", "Egypt", "Jordan", "Mauritania", "Morocco", "Palestine", "UAE", "Yemen")
BASE_EXPECT = {"Algeria": 23225, "Egypt": 23224, "Jordan": 23224, "Mauritania": 23225,
               "Morocco": 23225, "Palestine": 23225, "UAE": 23225, "Yemen": 23225}
SHORT_EXPECT = {"Algeria": 4281, "Egypt": 2866, "Jordan": 681, "Mauritania": 1337,
                "Morocco": 3141, "Palestine": 3838, "UAE": 1337, "Yemen": 3900}
FILT_EXPECT = {"Algeria": 2736, "Egypt": 2198, "Jordan": 524, "Mauritania": 1093,
               "Morocco": 2640, "Palestine": 2509, "UAE": 1053, "Yemen": 2664}

GAP_MIN, EDGEW_MIN = 0.06, 0.04


def row3(r):
    """Project to exactly {audio,text,dialect}; resolve+verify the dialect."""
    d = r.get("dialect") or ""
    if d not in DIALECTS:
        d = dialect_from_path(r["audio"])
    assert d in DIALECTS, "unresolved dialect: %r" % r["audio"]
    assert "original_val" not in r["audio"], "VAL LEAKAGE: %s" % r["audio"]
    assert "original_val" not in (r.get("src") or ""), "VAL LEAKAGE via src: %s" % r.get("src")
    assert r["text"].startswith("language Arabic<asr_text>"), "bad text prefix: %r" % r["text"][:40]
    return {"audio": r["audio"], "text": r["text"], "dialect": d}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["filtered", "identity", "none"], default="filtered")
    ap.add_argument("--out", default=None)
    ap.add_argument("--check-audio", action="store_true")
    a = ap.parse_args()
    out = a.out or os.path.join(MAN, {"filtered": "E16.jsonl", "identity": "E16_all.jsonl",
                                      "none": "E16_noclip.jsonl"}[a.mode])

    base = [json.loads(l) for l in open(os.path.join(MAN, "E11_dial.jsonl"), encoding="utf-8")]
    assert len(base) == 185798, len(base)
    assert collections.Counter(r["dialect"] for r in base) == BASE_EXPECT

    short = [json.loads(l) for l in open(os.path.join(MAN, "E15_shortclips.jsonl"), encoding="utf-8")]
    assert len(short) == 21381, len(short)
    assert collections.Counter(r["dialect"] for r in short) == SHORT_EXPECT

    if a.mode == "none":
        keep = []
    elif a.mode == "identity":
        keep = short
    else:
        AL = {}
        for line in open(os.path.join(MAN, "nadi_original.align.jsonl"), encoding="utf-8"):
            j = json.loads(line)
            if "words" in j:
                AL[j["audio"]] = [w for w in j["words"] if w.get("aligned")]
        keep = []
        for r in short:
            ws = AL[r["src"]]
            i, j = r["span"]
            assert 0 <= i <= j < len(ws), "span out of range: %s" % r["audio"]
            edgew = min(ws[i]["end"] - ws[i]["start"], ws[j]["end"] - ws[j]["start"])
            if r["gap"] >= GAP_MIN - 1e-9 and edgew >= EDGEW_MIN - 1e-9:
                keep.append(r)
        assert collections.Counter(r["dialect"] for r in keep) == FILT_EXPECT, \
            collections.Counter(r["dialect"] for r in keep)
        assert len(keep) == 15417, len(keep)

    rows = [row3(r) for r in base] + [row3(r) for r in keep]
    exp = {"filtered": 201215, "identity": 207179, "none": 185798}[a.mode]
    assert len(rows) == exp, "row count %d != %d" % (len(rows), exp)

    # schema must be EXACTLY 3 keys on every row (guards the HF CastError on mixed schemas)
    ks = collections.Counter(tuple(sorted(r)) for r in rows)
    assert list(ks) == [("audio", "dialect", "text")], ks
    # what the trainer recomputes must equal what we wrote
    assert all(dialect_of(r) == r["dialect"] for r in rows), "dialect_of mismatch"
    # NOTE: repeated audio paths are EXPECTED in the base (E11 carries augmented variants that
    # reuse a path across rows); only the short-clip block must be duplicate-free.
    sa = [r["audio"] for r in rows[len(base):]]
    assert len(set(sa)) == len(sa), "duplicate short-clip audio paths"
    if a.check_audio:
        miss = [r["audio"] for r in rows if not os.path.exists(r["audio"])]
        assert not miss, "%d missing audio files, e.g. %s" % (len(miss), miss[:3])

    with open(out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    cnt = collections.Counter(r["dialect"] for r in rows)
    print("[build_e16b] mode=%s rows=%d -> %s" % (a.mode, len(rows), out))
    print("[build_e16b] per-dialect: %s" % dict(sorted(cnt.items())))
    print("[build_e16b] row max/min = %.4f" % (max(cnt.values()) / min(cnt.values())))
    print("[build_e16b] short clips kept: %d (%.1f%% of 21381)" % (len(keep), 100 * len(keep) / 21381))


if __name__ == "__main__":
    main()
