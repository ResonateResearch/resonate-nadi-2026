#!/usr/bin/env python3
"""Build the recovered E11 base and augmentation-source manifests.

Recipe per dialect (all 8, balanced — tail-tilt was tested in E10 and regressed):
  original      x3 (4,800)                 — already at val band, no LP
  clean-neural  x3 (4,800)                 — already at val band, no LP
  VC (kNN-VC)   7,500/dialect, LOW-PASSED order-10 butter @2.5kHz -> new flac tree
                (VC has spurious vocoder energy at 3-4kHz, -41dB vs val's -55dB)
  aug_src       6,125/dialect real rows for augment.py (dialect-matched additive /
                channel=LP-only / speed; FINAL_LP 2.5kHz applied to every output).
                The source suggested SNR U(14,30); historical launch overrides were
                not recovered. Configure SNR explicitly; see PREPARATION.md.

Emits E11_base.jsonl (orig+clean+VC-LP, absolute paths) + manifests/aug_src_e11/{D}.jsonl.
Total after aug: 38,400+38,400+60,000+49,000 = 185,800 rows (~E7's public-free step count).
"""
import argparse
import json
import os
import random
import sys
from glob import glob
from multiprocessing import Pool

import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfilt

RF = os.path.abspath(os.environ.get("NADI_ASR_WORKSPACE", "work/asr"))
LP_OUT = os.path.join(RF, "audio/E11_vc_lp")
DIALECTS = ["Algeria", "Egypt", "Jordan", "Mauritania", "Morocco", "Palestine", "UAE", "Yemen"]
VC_PER_DIALECT = 7500
AUG_PER_DIALECT = 6125
SEED = 20260717
SR = 16000
_SOS = butter(10, 2500, btype="low", fs=SR, output="sos")


def stem(p):
    return os.path.splitext(os.path.basename(p))[0]


def lp_one(args):
    src, dst = args
    try:
        w, sr = sf.read(src, dtype="float32", always_2d=False)
        if w.ndim > 1:
            w = w.mean(axis=1)
        assert sr == SR, f"sr={sr}"
        y = sosfilt(_SOS, w).astype(np.float32)
        m = np.max(np.abs(y))
        if m > 1.0:
            y = y / m * 0.99
        sf.write(dst, y, SR, format="FLAC")
        return None
    except Exception as e:
        return f"{src}: {e}"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--vc-root", required=True,
                    help="generated ReDimNet-bank VC root containing <Dialect>/wavs/*.wav")
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    vc_root = os.path.abspath(a.vc_root)
    rng = random.Random(SEED)
    txt = {}
    for l in open(os.path.join(RF, "manifests/nadi_original.jsonl"), encoding="utf-8"):
        r = json.loads(l)
        txt[stem(r["audio"])] = r["text"]
    print(f"[map] {len(txt)} transcripts", flush=True)

    base = open(os.path.join(RF, "manifests/E11_base.jsonl"), "w", encoding="utf-8")
    os.makedirs(os.path.join(RF, "manifests/aug_src_e11"), exist_ok=True)
    lp_jobs = []
    vc_rows = []
    n_base = 0

    for d in DIALECTS:
        ids = [stem(p) for p in sorted(glob(os.path.join(RF, "audio/original", d, "*.flac")))]
        # real x3 + clean x3
        for i in ids:
            t = txt.get(i)
            if t is None:
                continue
            op = os.path.join(RF, "audio/original", d, f"{i}.flac")
            cp = os.path.join(RF, "audio/clean-neural", d, f"{i}.flac")
            for _ in range(3):
                base.write(json.dumps({"audio": op, "text": t}, ensure_ascii=False) + "\n")
                n_base += 1
            if os.path.exists(cp):
                for _ in range(3):
                    base.write(json.dumps({"audio": cp, "text": t}, ensure_ascii=False) + "\n")
                    n_base += 1
        # VC: balanced subsample -> LP tree
        pool = sorted(glob(os.path.join(vc_root, d, "wavs", "*.wav")))
        if not pool:
            raise ValueError(f"no VC files for {d} under {vc_root}")
        sel = rng.sample(pool, min(VC_PER_DIALECT, len(pool)))
        os.makedirs(os.path.join(LP_OUT, d), exist_ok=True)
        kept = 0
        for p in sel:
            src_id = os.path.basename(p).split("__")[0]
            t = txt.get(src_id)
            if t is None:
                continue
            dst = os.path.join(LP_OUT, d, stem(p) + ".flac")
            lp_jobs.append((p, dst))
            vc_rows.append({"audio": dst, "text": t})
            kept += 1
        # aug source (sample real with replacement)
        real = [(os.path.join(RF, "audio/original", d, f"{i}.flac"), txt[i]) for i in ids if i in txt]
        with open(os.path.join(RF, "manifests/aug_src_e11", f"{d}.jsonl"), "w", encoding="utf-8") as af:
            for _ in range(AUG_PER_DIALECT):
                ap, t = rng.choice(real)
                af.write(json.dumps({"audio": ap, "text": t}, ensure_ascii=False) + "\n")
        print(f"[{d:<11}] base rows ok, vc selected {kept}", flush=True)

    print(f"[lp] filtering {len(lp_jobs)} VC files (order-10 @2.5kHz)...", flush=True)
    n_err = 0
    with Pool(a.workers) as p:
        for j, err in enumerate(p.imap_unordered(lp_one, lp_jobs, chunksize=64)):
            if err:
                n_err += 1
                if n_err <= 10:
                    print("[err]", err, flush=True)
            if (j + 1) % 10000 == 0:
                print(f"  {j+1}/{len(lp_jobs)}", flush=True)
    for r in vc_rows:
        if os.path.exists(r["audio"]):
            base.write(json.dumps(r, ensure_ascii=False) + "\n")
            n_base += 1
    base.close()
    print(f"[done] E11_base.jsonl = {n_base} rows, lp_errors={n_err}", flush=True)


if __name__ == "__main__":
    main()
