#!/usr/bin/env python3
"""E15 short-clip re-segmentation: cut sub-1s clips out of the REAL NADI train audio.

WHY
---
The NADI2026 train audio has a HARD FLOOR at 1.001 s (0.0% of the 12,800 real train clips
are <=1 s) while the val set we are scored on is 10.5% <=1 s and reaches down to 0.43 s.
Short clips are also the worst WER bucket (68.4% @ <=1 s vs 39.8% @ 5 s+).  We may not add
new data, so we re-segment the audio we already have, using the CTC forced alignment in
manifests/nadi_original.align.jsonl (produced by scripts/align_ctc.py).

ALGORITHM
---------
For every aligned train utterance we enumerate every contiguous span of words whose padded
duration lands in [0.43, 1.00] s (the observed val <=1 s range).  Boundaries are placed at
the MIDPOINT of the inter-word gap and then padded OUTWARD by PAD, clamped so the cut can
never cross into the neighbouring word's aligned extent:

    left  = max(prev_end,  prev_end + gap/2 - PAD)      (utterance start: word0.start - EDGE)
    right = min(next_start, this_end  + gap/2 + PAD)    (utterance end:   wordN.end   + EDGE)

The verifier for align_ctc.py measured that emitted word spans are the VOICED extent and run
~40 ms narrow on each side, hence the outward padding is mandatory; and that 15.0% of the
131,586 inter-word gaps are exactly 20 ms, which is a STRUCTURAL artifact of the CTC
delimiter state occupying >=1 frame and NOT a real pause.  We therefore rank candidates by
their weakest boundary gap and fill each quota cell from tier A (both gaps >= 40 ms) before
tier B (> 20 ms) and only fall back to tier C (structural 20 ms) if a cell cannot be filled
otherwise.  53% of gaps are <= 60 ms so a hard ">= 40 ms" rule would discard ~2/3 of the
boundaries -- we prefer, we do not require.

Utterances are gated on the CTC score at the WITHIN-DIALECT 10th percentile.  The score is a
mean per-frame log-prob and is NOT comparable across dialects (the XLSR acoustic model is
MSA-biased, so Morocco/Mauritania score lower for model reasons, not alignment reasons), so a
global threshold would silently delete those dialects.

TARGETING
---------
Quotas are chosen so that E11.jsonl + these rows reproduces the val duration profile:
  * per-dialect allocation proportional to that dialect's OWN val <=1 s share
    (Algeria 17.4 ... Jordan 3.3), not uniform;
  * within a dialect, the (word-count x duration-bin) joint distribution of the val <=1 s
    clips is reproduced (val mean 3.14 words, word hist {1:11%,2:16%,3:38%,4:27%,5:6%});
  * the global total is sized so the combined set lands at val's 10.5% <=1 s.
Deficit in any cell is redistributed to the nearest duration bin, then the nearest word
count, inside the same dialect, so the totals hold even where the audio cannot supply a cell.

AUDIO
-----
Source is ONLY audio/original/<Dialect>/*.flac (the real TRAIN audio).  audio/original_val/
is the evaluation set; touching it would be leakage and is asserted against in three places.
Each source file is read and low-pass filtered ONCE (order-10 Butterworth @ 2.5 kHz, the
identical FINAL_LP filter used by scripts/build_e11.py and scripts/augment.py) and the spans
are sliced out of the filtered signal -- filtering per-slice would put an IIR startup
transient at the head of every clip.  Output: FLAC, 16 kHz, mono, under audio/E15_short/.

Measured: the real NADI audio is ALREADY band-limited -- (2.6-4 kHz)/(0.3-2.5 kHz) energy is
-67.1 dB for audio/original and -67.5 dB for the val clips -- so this filter is close to a
no-op here (it takes the cuts to -80.8 dB).  It is applied anyway for consistency with the
rest of the E11 tree; both figures sit far below any log-mel dynamic-range floor.

Usage:
  PY scripts/build_shortclips.py --workers 48
  PY scripts/build_shortclips.py --dry-run          # quotas + fill report, no audio written
"""
import argparse
import collections
import json
import os
import sys

import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfilt

RF = os.path.abspath(os.environ.get("NADI_ASR_WORKSPACE", "work/asr"))
ALIGN = os.path.join(RF, "manifests/nadi_original.align.jsonl")
OUT_AUDIO = os.path.join(RF, "audio/E15_short")
OUT_MANIFEST = os.path.join(RF, "manifests/E15_shortclips.jsonl")

SR = 16000
FADE_N = int(0.006 * SR)   # 6 ms raised-cosine fade at each cut edge (anti-discontinuity)
SEED = 20260720
PREFIX = "language Arabic<asr_text>"

# --- identical to build_e11.py / augment.py FINAL_LP -------------------------------------
_SOS = butter(10, 2500, btype="low", fs=SR, output="sos")

DIALECTS = ["Algeria", "Egypt", "Jordan", "Mauritania", "Morocco", "Palestine", "UAE", "Yemen"]

# --- cut geometry ------------------------------------------------------------------------
PAD = 0.035          # outward pad from the gap midpoint (aligned spans run ~40 ms narrow)
EDGE = 0.050         # outward pad at a true utterance edge
GAP_A = 0.040        # tier A: a real pause
GAP_STRUCT = 0.021   # <= this is the structural CTC-delimiter gap, not a pause
# NB gaps are quantised to the 20 ms CTC frame stride, so in practice only two tiers exist:
# exactly 20 ms (structural) and >= 40 ms (a real pause).  Tier B therefore only ever catches
# 40 ms gaps that came out as 0.0399999... in float; qc_shortclips.py rounds to 4 dp and
# correctly reports them as tier A, which is why its A/C split is 94.2% / 5.8% with B empty.
DUR_LO, DUR_HI = 0.43, 1.0001   # val <=1 s clips span 0.43 .. 1.00 s
MAX_WORDS = 6
SCORE_PCTL = 10.0    # drop the worst decile of each dialect (within-dialect, never global)

DUR_BINS = [0.43, 0.50, 0.60, 0.70, 0.80, 0.90, 1.0001]
N_BINS = len(DUR_BINS) - 1

# val <=1 s joint histogram: [word_count][duration_bin] counts, n=1103 (measured, see docstring)
VAL_JOINT = {
    1: [7, 23, 33, 17, 18, 10],
    2: [4, 16, 28, 40, 45, 46],
    3: [3, 21, 56, 83, 119, 147],
    4: [1, 6, 23, 56, 82, 132],
    5: [0, 0, 3, 4, 14, 42],
    6: [0, 0, 1, 2, 8, 13],
}
# per-dialect <=1 s share in val (%)
VAL_SHORT_SHARE = {"Algeria": 17.4, "Yemen": 16.2, "Palestine": 15.9, "Morocco": 13.6,
                   "Egypt": 12.5, "Mauritania": 6.2, "UAE": 6.2, "Jordan": 3.3}
# E11.jsonl rows per dialect, and how many of them are already <= 1 s (measured)
E11_ROWS_PER_DIALECT = 23225
E11_SHORT_PER_DIALECT = {"Algeria": 46, "Egypt": 70, "Jordan": 23, "Mauritania": 23,
                         "Morocco": 93, "Palestine": 46, "UAE": 23, "Yemen": 70}
E11_TOTAL_ROWS = 185798
E11_TOTAL_SHORT = 372
TARGET_SHORT_SHARE = 0.105       # val overall <=1 s share
MAX_SPANS_PER_UTT = 6
MAX_WORD_JACCARD = 0.65          # reject a span >65% word-identical to one already kept from that utt


# ==========================================================================================
# candidate enumeration
# ==========================================================================================
def usable_words(rec):
    """Aligned, non-degenerate words only.  Punctuation-only tokens are emitted with
    aligned=false by align_ctc.py (1,393 of them) and 2 words have zero duration."""
    return [w for w in rec["words"]
            if w.get("aligned", True) and (w["end"] - w["start"]) > 1e-6 and w["w"].strip()]


def enumerate_spans(rec):
    """All contiguous word spans of rec whose padded duration is in [DUR_LO, DUR_HI].

    Returns list of dicts: i, j (word indices into usable_words), t0, t1, dur, nw, gap
    where gap is the weaker of the two boundary gaps (utterance edges score as 9.0 = best).
    """
    dur_file = rec["duration"]
    ws = usable_words(rec)
    n = len(ws)
    out = []
    for i in range(n):
        for j in range(i, min(n, i + MAX_WORDS)):
            if i == 0:
                t0 = max(0.0, ws[0]["start"] - EDGE)
                lgap = 9.0
            else:
                g = ws[i]["start"] - ws[i - 1]["end"]
                lgap = g
                # midpoint of the gap, padded outward, never crossing the previous word
                t0 = max(ws[i - 1]["end"], ws[i - 1]["end"] + g / 2.0 - PAD)
            if j == n - 1:
                t1 = min(dur_file, ws[-1]["end"] + EDGE)
                rgap = 9.0
            else:
                g = ws[j + 1]["start"] - ws[j]["end"]
                rgap = g
                t1 = min(ws[j + 1]["start"], ws[j]["end"] + g / 2.0 + PAD)
            d = t1 - t0
            if d > DUR_HI:
                break            # widening j only grows d
            if d < DUR_LO:
                continue
            out.append({"i": i, "j": j, "t0": round(t0, 4), "t1": round(t1, 4),
                        "dur": d, "nw": j - i + 1, "gap": min(lgap, rgap),
                        "text": " ".join(w["w"] for w in ws[i:j + 1])})
    return out


def dur_bin(d):
    b = int(np.digitize(d, DUR_BINS)) - 1
    return min(max(b, 0), N_BINS - 1)


def tier(gap):
    if gap >= GAP_A:
        return 0
    if gap > GAP_STRUCT:
        return 1
    return 2


# ==========================================================================================
# quota computation
# ==========================================================================================
def compute_quotas():
    """Per-dialect totals, then the (word-count, duration-bin) split inside each dialect."""
    # exact per-dialect need to hit that dialect's own val <=1 s share
    need = {}
    for d in DIALECTS:
        p = VAL_SHORT_SHARE[d] / 100.0
        need[d] = (p * E11_ROWS_PER_DIALECT - E11_SHORT_PER_DIALECT[d]) / (1.0 - p)
    # global size so that (E11_short + N) / (E11_rows + N) == TARGET_SHORT_SHARE
    p = TARGET_SHORT_SHARE
    n_total = (p * E11_TOTAL_ROWS - E11_TOTAL_SHORT) / (1.0 - p)
    alpha = n_total / sum(need.values())
    per_dialect = {d: need[d] * alpha for d in DIALECTS}

    jt = sum(sum(v) for v in VAL_JOINT.values())
    quotas = {}
    for d in DIALECTS:
        cells = {}
        for k, row in VAL_JOINT.items():
            for b, c in enumerate(row):
                q = c / jt * per_dialect[d]
                if q > 0:
                    cells[(k, b)] = q
        # integerise deterministically (largest remainder)
        target = int(round(per_dialect[d]))
        base = {kk: int(np.floor(v)) for kk, v in cells.items()}
        rem = sorted(cells, key=lambda kk: (-(cells[kk] - base[kk]), kk))
        short = target - sum(base.values())
        for kk in rem[:max(0, short)]:
            base[kk] += 1
        quotas[d] = {kk: v for kk, v in base.items() if v > 0}
    return quotas, per_dialect, alpha


# ==========================================================================================
# selection
# ==========================================================================================
def select_for_dialect(d, recs, quota, log):
    """Greedy quota fill: tier A first, then B, then C; diversity-capped per utterance."""
    pool = collections.defaultdict(list)      # (nw, bin) -> list of candidate dicts
    for r in recs:
        for c in enumerate_spans(r):
            c["src"] = r["audio"]
            c["utt"] = os.path.splitext(os.path.basename(r["audio"]))[0]
            c["score"] = r["score"]
            pool[(c["nw"], dur_bin(c["dur"]))].append(c)
    for kk in pool:
        # deterministic: best boundary gap first, then highest alignment score, then id
        pool[kk].sort(key=lambda c: (-min(c["gap"], 0.5), -c["score"], c["utt"], c["i"], c["j"]))

    used = collections.defaultdict(list)      # utt -> list of (i, j) already taken
    chosen = []
    n_tier = collections.Counter()

    def try_take(c):
        prev = used[c["utt"]]
        if len(prev) >= MAX_SPANS_PER_UTT:
            return False
        s = set(range(c["i"], c["j"] + 1))
        for (pi, pj) in prev:
            p = set(range(pi, pj + 1))
            inter = len(s & p)
            if inter and inter / len(s | p) > MAX_WORD_JACCARD:
                return False
        prev.append((c["i"], c["j"]))
        chosen.append(c)
        n_tier[tier(c["gap"])] += 1
        return True

    taken_per_cell = collections.Counter()
    # Scarcity-first cell order.  Filling cheap cells first lets 2-3 word spans consume the
    # utterances that the (much rarer) 4-6 word spans need, because a 4-word span overlapping
    # an already-taken 3-word span fails the Jaccard test.  Train speech runs 2.82 w/s vs
    # val's 3.87, so long-word-count sub-1s spans are the binding constraint -- fill them first.
    cell_order = sorted(quota, key=lambda kk: (len(pool[kk]) / max(quota[kk], 1), kk))
    # pass 1: exact cells, tier by tier
    for t in (0, 1, 2):
        for kk in cell_order:
            q = quota[kk]
            want = q - taken_per_cell[kk]
            if want <= 0:
                continue
            for c in pool[kk]:
                if want <= 0:
                    break
                if c.get("_taken") or tier(c["gap"]) != t:
                    continue
                if try_take(c):
                    c["_taken"] = True
                    taken_per_cell[kk] += 1
                    want -= 1

    # pass 2: redistribute the deficit -> nearest duration bin, then nearest word count
    deficit = sum(max(0, q - taken_per_cell[kk]) for kk, q in quota.items())
    if deficit:
        order = []
        for kk, q in quota.items():
            need = q - taken_per_cell[kk]
            if need > 0:
                k, b = kk
                subs = [(k, bb) for bb in sorted(range(N_BINS), key=lambda x: (abs(x - b), -x))]
                subs += [(kk2, bb) for kk2 in sorted(VAL_JOINT, key=lambda x: (abs(x - k), x))
                         for bb in sorted(range(N_BINS), key=lambda x: (abs(x - b), -x))]
                order.append((kk, need, subs))
        for kk, need, subs in sorted(order):
            for sub in subs:
                if need <= 0:
                    break
                for c in pool.get(sub, []):
                    if need <= 0:
                        break
                    if c.get("_taken"):
                        continue
                    if try_take(c):
                        c["_taken"] = True
                        need -= 1
    got = len(chosen)
    want = sum(quota.values())
    log.append(f"[{d:<11}] quota={want} selected={got} ({100.0*got/max(want,1):.1f}%) "
               f"tierA={n_tier[0]} tierB={n_tier[1]} tierC={n_tier[2]} "
               f"utts_used={len(used)} pool={sum(len(v) for v in pool.values())}")
    return chosen


# ==========================================================================================
# audio cutting (one worker per SOURCE FILE -> filter once, slice many)
# ==========================================================================================
def cut_file(job):
    src, dialect, spans = job
    assert "/audio/original/" in src, f"source not from audio/original: {src}"
    assert "original_val" not in src, f"LEAKAGE: val audio as cut source: {src}"
    out = []
    try:
        w, sr = sf.read(src, dtype="float32", always_2d=False)
        if w.ndim > 1:
            w = w.mean(axis=1)
        assert sr == SR, f"sr={sr} for {src}"
        y = sosfilt(_SOS, w).astype(np.float32)      # filter the WHOLE file: no IIR transient
        n = len(y)
        for sp in spans:
            a = max(0, int(round(sp["t0"] * SR)))
            b = min(n, int(round(sp["t1"] * SR)))
            if b - a < int(0.30 * SR):
                out.append({"error": f"{src}: degenerate slice {a}:{b}"})
                continue
            seg = y[a:b].copy()
            # Raised-cosine fade at both cut points. Slicing mid-signal leaves a step
            # discontinuity worth ~8 dB of broadband splatter vs real val clips (measured),
            # which survives the low-pass because the LP is applied BEFORE slicing. A short
            # fade removes it, keeping the clips spectrally matched to val like the rest of E11.
            _fn = min(FADE_N, (b - a) // 4)
            if _fn > 0:
                ramp = 0.5 * (1.0 - np.cos(np.linspace(0.0, np.pi, _fn, dtype=np.float32)))
                seg[:_fn] *= ramp
                seg[-_fn:] *= ramp[::-1]
            m = float(np.max(np.abs(seg)))
            if m > 1.0:
                seg = seg / m * 0.99
            dst = os.path.join(OUT_AUDIO, dialect, sp["name"] + ".flac")
            sf.write(dst, seg, SR, format="FLAC")
            out.append({"audio": dst, "text": PREFIX + sp["text"], "dialect": dialect,
                        "duration": round((b - a) / SR, 4), "src": src,
                        "span": [sp["i"], sp["j"]], "t0": sp["t0"], "t1": sp["t1"],
                        "gap": round(min(sp["gap"], 9.0), 4), "nw": sp["nw"]})
    except Exception as e:                                     # noqa: BLE001
        out.append({"error": f"{src}: {e}"})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=48)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    np.random.seed(SEED)

    recs = []
    n_err = 0
    for line in open(ALIGN, encoding="utf-8"):
        r = json.loads(line)
        if "error" in r:
            n_err += 1
            continue
        # HARD leakage guard: the cut source must be the real TRAIN tree, never the val tree.
        assert "/audio/original/" in r["audio"], f"unexpected source tree: {r['audio']}"
        assert "original_val" not in r["audio"], f"LEAKAGE: {r['audio']}"
        recs.append(r)
    print(f"[align] {len(recs)} aligned utterances ({n_err} alignment errors skipped)", flush=True)

    # within-dialect score gate (never a global threshold: score is not cross-dialect comparable)
    by_d = collections.defaultdict(list)
    for r in recs:
        by_d[r["dialect"]].append(r)
    kept = {}
    for d in DIALECTS:
        sc = np.array([r["score"] for r in by_d[d]])
        thr = float(np.percentile(sc, SCORE_PCTL))
        kept[d] = [r for r in by_d[d] if r["score"] >= thr]
        print(f"[gate] {d:<11} p{SCORE_PCTL:g}={thr:.3f}  kept {len(kept[d])}/{len(by_d[d])}",
              flush=True)

    quotas, per_dialect, alpha = compute_quotas()
    print(f"[quota] alpha={alpha:.4f} total={sum(sum(q.values()) for q in quotas.values())} "
          + " ".join(f"{d}={sum(quotas[d].values())}" for d in DIALECTS), flush=True)

    log = []
    chosen = []
    for d in DIALECTS:
        chosen += [(d, c) for c in select_for_dialect(d, kept[d], quotas[d], log)]
    for line in log:
        print(line, flush=True)
    print(f"[select] {len(chosen)} spans total", flush=True)

    if a.dry_run:
        ds = np.array([c["dur"] for _, c in chosen])
        nw = np.array([c["nw"] for _, c in chosen])
        print(f"[dry] dur mean={ds.mean():.3f} median={np.median(ds):.3f} "
              f"words mean={nw.mean():.2f} wps={np.mean(nw/ds):.2f}")
        print("[dry] word hist", {k: round(100*v/len(nw), 1)
                                  for k, v in sorted(collections.Counter(nw.tolist()).items())})
        return

    # group by source file so each file is read+filtered exactly once
    jobs = collections.defaultdict(list)
    for d, c in chosen:
        c["name"] = f"{c['utt']}__w{c['i']:02d}-{c['j']:02d}"
        jobs[(c["src"], d)].append(c)
    for d in DIALECTS:
        os.makedirs(os.path.join(OUT_AUDIO, d), exist_ok=True)
    job_list = sorted(((s, d, sorted(v, key=lambda c: (c["i"], c["j"])))
                       for (s, d), v in jobs.items()), key=lambda x: x[0])
    print(f"[cut] {len(job_list)} source files -> {len(chosen)} clips, workers={a.workers}",
          flush=True)

    from multiprocessing import Pool
    n_ok, n_bad = 0, 0
    with open(OUT_MANIFEST, "w", encoding="utf-8") as mf, Pool(a.workers) as pool:
        for idx, res in enumerate(pool.imap_unordered(cut_file, job_list, chunksize=16)):
            for row in res:
                if "error" in row:
                    n_bad += 1
                    if n_bad <= 10:
                        print("[err]", row["error"], flush=True)
                    continue
                mf.write(json.dumps(row, ensure_ascii=False) + "\n")
                n_ok += 1
            if (idx + 1) % 2000 == 0:
                print(f"  {idx+1}/{len(job_list)} files, {n_ok} clips", flush=True)
    print(f"[done] wrote {n_ok} clips ({n_bad} errors) -> {OUT_MANIFEST}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
