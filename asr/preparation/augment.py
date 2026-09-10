#!/usr/bin/env python3
"""Offline waveform augmentation for the NADI robust-ASR experiments.

Applies ONE randomly-chosen aug per utterance (research-style single-view), drawn from the
6 WAVEFORM augs below. SpecAugment is feature-domain and applied live in the collator
(--specaug), NOT here.

  additive : mix a NADI-extracted noise segment (noiselab noise_pool) @ SNR U(0,20) dB
  musan    : mix a MUSAN noise/music clip @ SNR U(0,20) dB
  reverb   : convolve with a real room impulse response (OpenSLR RIRS_NOISES/real_rirs)
  rir      : convolve with a simulated RIR (OpenSLR RIRS_NOISES/simulated_rirs)
  channel  : telephony sim - band-limit 300-3400 Hz, 16k->8k->16k, mu-law codec round-trip
  speed    : speed-perturb by 0.9 or 1.1 (resample; changes duration, transcript unchanged)

Driver reads a source manifest (jsonl {"audio","text"}), augments each row, writes a FLAC
under --out-audio and a new manifest at --out-manifest. Deterministic per-row via SEED+index.
Parallel across processes with --workers.

Usage:
  PY augment.py --src manifests/E2_clean.jsonl --out-audio audio/aug_clean13 \
     --out-manifest manifests/E3_clean_aug.jsonl --workers 24
"""
import argparse
import glob
import io
import json
import os
import random
import sys

import numpy as np
import soundfile as sf
import soxr
from scipy.signal import butter, sosfilt, fftconvolve

SR = 16000
WORKSPACE = os.path.abspath(os.environ.get("NADI_ASR_WORKSPACE", "work/asr"))
NOISE_POOL = os.environ.get("NADI_NOISE_POOL", os.path.join(WORKSPACE, "noise_pool"))
MUSAN_DIR = os.environ.get("MUSAN_DIR", os.path.join(WORKSPACE, "corpora/musan"))
RIR_ROOT = os.environ.get("RIR_ROOT", os.path.join(WORKSPACE, "corpora/RIRS_NOISES"))
AUGS = ["additive", "musan", "reverb", "rir", "channel", "speed"]

# lazily-built, per-process file lists (shared read-only corpora)
_POOLS = {}
NOISE_DIALECT = os.environ.get("NOISE_DIALECT", "")  # if set, additive noise restricted to this dialect
# E11 domain-matching knobs (defaults preserve original behavior):
SNR_LO = float(os.environ.get("AUG_SNR_LO", "0"))    # additive-noise SNR draw lower bound (dB)
SNR_HI = float(os.environ.get("AUG_SNR_HI", "20"))   # additive-noise SNR draw upper bound (dB)
CHANNEL_LP_HZ = float(os.environ.get("CHANNEL_LP_HZ", "0"))  # >0: channel aug = pure low-pass at this Hz
FINAL_LP_HZ = float(os.environ.get("FINAL_LP_HZ", "0"))      # >0: low-pass EVERY output (post-aug) at this Hz


def _list(kind):
    if kind in _POOLS:
        return _POOLS[kind]
    if kind == "noise":
        sub = NOISE_DIALECT if NOISE_DIALECT else "*"
        files = glob.glob(os.path.join(NOISE_POOL, "segments", sub, "*.flac"))
    elif kind == "musan":
        files = glob.glob(os.path.join(MUSAN_DIR, "noise", "**", "*.wav"), recursive=True) \
            + glob.glob(os.path.join(MUSAN_DIR, "music", "**", "*.wav"), recursive=True)
    elif kind == "real_rir":
        files = glob.glob(os.path.join(RIR_ROOT, "real_rirs_isotropic_noises", "*.wav")) \
            + glob.glob(os.path.join(RIR_ROOT, "**", "real_rir*", "**", "*.wav"), recursive=True)
    elif kind == "sim_rir":
        files = glob.glob(os.path.join(RIR_ROOT, "simulated_rirs", "**", "*.wav"), recursive=True)
    else:
        files = []
    _POOLS[kind] = files
    return files


def _read(path, target_sr=SR):
    w, sr = sf.read(path, dtype="float32", always_2d=False)
    if w.ndim > 1:
        w = w.mean(axis=1)
    if sr != target_sr:
        w = soxr.resample(w, sr, target_sr)
    return w.astype(np.float32)


def _rms(x):
    return float(np.sqrt(np.mean(x ** 2) + 1e-12))


def _fit(noise, n):
    """tile/crop noise to length n."""
    if len(noise) == 0:
        return np.zeros(n, np.float32)
    if len(noise) < n:
        reps = int(np.ceil(n / len(noise)))
        noise = np.tile(noise, reps)
    start = 0 if len(noise) == n else random.randint(0, len(noise) - n)
    return noise[start:start + n]


def _mix_noise(w, rng, files):
    if not files:
        return w
    noise = _read(rng.choice(files))
    noise = _fit(noise, len(w))
    snr = rng.uniform(SNR_LO, SNR_HI)
    scale = _rms(w) / (_rms(noise) + 1e-9) / (10 ** (snr / 20))
    out = w + noise * scale
    return out.astype(np.float32)


def aug_additive(w, rng):
    return _mix_noise(w, rng, _list("noise"))


def aug_musan(w, rng):
    return _mix_noise(w, rng, _list("musan"))


def _convolve_rir(w, rir):
    peak = int(np.argmax(np.abs(rir)))
    y = fftconvolve(w, rir)[peak:peak + len(w)]
    m = np.max(np.abs(y)) + 1e-9
    return (y / m * (np.max(np.abs(w)) + 1e-9)).astype(np.float32)


def aug_reverb(w, rng):
    files = _list("real_rir")
    if not files:
        return w
    return _convolve_rir(w, _read(rng.choice(files)))


def aug_rir(w, rng):
    files = _list("sim_rir")
    if not files:
        return w
    return _convolve_rir(w, _read(rng.choice(files)))


_SOS = butter(6, [300, 3400], btype="band", fs=SR, output="sos")
# order-10 low-pass for band-limit domain matching (order 6 leaks: -12dB@3k; order 10: -54dB@4k)
_LP_SOS = {}


def _lp(w, hz):
    if hz not in _LP_SOS:
        _LP_SOS[hz] = butter(10, hz, btype="low", fs=SR, output="sos")
    return sosfilt(_LP_SOS[hz], w).astype(np.float32)


def _mulaw(x, mu=255):
    x = np.clip(x, -1, 1)
    y = np.sign(x) * np.log1p(mu * np.abs(x)) / np.log1p(mu)
    q = np.round((y + 1) / 2 * mu)
    yq = q / mu * 2 - 1
    return (np.sign(yq) * (1 / mu) * ((1 + mu) ** np.abs(yq) - 1)).astype(np.float32)


def aug_channel(w, rng):
    if CHANNEL_LP_HZ > 0:   # E11 mode: pure low-pass (val has real sub-300Hz energy; mu-law adds noise floor)
        return _lp(w, CHANNEL_LP_HZ)
    y = sosfilt(_SOS, w).astype(np.float32)
    y = soxr.resample(soxr.resample(y, SR, 8000), 8000, SR)
    y = _mulaw(y / (np.max(np.abs(y)) + 1e-9)) * (np.max(np.abs(w)) + 1e-9)
    return y.astype(np.float32)


def aug_speed(w, rng):
    factor = rng.choice([0.9, 1.1])
    return soxr.resample(w, int(SR * factor), SR).astype(np.float32)


AUG_FN = {"additive": aug_additive, "musan": aug_musan, "reverb": aug_reverb,
          "rir": aug_rir, "channel": aug_channel, "speed": aug_speed}


def process_row(args):
    idx, row, out_audio, seed, aug_only = args
    rng = random.Random(seed + idx)
    random.seed(seed + idx)  # for _fit's global random
    kind = rng.choice(aug_only)
    try:
        w = _read(row["audio"])
        w = AUG_FN[kind](w, rng)
        if FINAL_LP_HZ > 0:          # domain-match: guarantee output band <= val band (after speed etc)
            w = _lp(w, FINAL_LP_HZ)
        m = np.max(np.abs(w))
        if m > 1.0:
            w = w / m * 0.99
        dest = os.path.join(out_audio, kind, f"{idx}.flac")
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        sf.write(dest, w, SR, format="FLAC")
        return {"audio": dest, "text": row["text"], "aug": kind}
    except Exception as e:
        return {"error": f"{row.get('audio')}: {e}"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="source manifest jsonl {audio,text}")
    ap.add_argument("--out-audio", required=True)
    ap.add_argument("--out-manifest", required=True)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--augs", nargs="+", choices=AUGS, default=AUGS,
                    help="subset of aug kinds; retained NADI recipe uses additive channel speed")
    a = ap.parse_args()

    # A missing external pool must not silently turn a requested augmentation into a no-op.
    pools = {"additive": "noise", "musan": "musan", "reverb": "real_rir", "rir": "sim_rir"}
    for kind in a.augs:
        if kind in pools and not _list(pools[kind]):
            ap.error(f"empty pool for {kind}; configure NADI_NOISE_POOL, MUSAN_DIR, or RIR_ROOT")
    a.out_audio = os.path.abspath(a.out_audio)
    os.makedirs(os.path.dirname(os.path.abspath(a.out_manifest)), exist_ok=True)

    rows = [json.loads(l) for l in open(a.src, encoding="utf-8") if l.strip()]
    os.makedirs(a.out_audio, exist_ok=True)
    tasks = [(i, r, a.out_audio, a.seed, a.augs) for i, r in enumerate(rows)]

    from multiprocessing import Pool
    n_err, counts = 0, {}
    with open(a.out_manifest, "w", encoding="utf-8") as mf, Pool(a.workers) as pool:
        for j, res in enumerate(pool.imap_unordered(process_row, tasks, chunksize=64)):
            if "error" in res:
                n_err += 1
                if n_err <= 10:
                    print("[err]", res["error"], flush=True)
                continue
            counts[res["aug"]] = counts.get(res["aug"], 0) + 1
            mf.write(json.dumps({"audio": res["audio"], "text": res["text"]},
                                ensure_ascii=False) + "\n")
            if (j + 1) % 5000 == 0:
                print(f"  {j+1}/{len(tasks)}  err={n_err}  {counts}", flush=True)
    print(f"[done] wrote {sum(counts.values())} rows, err={n_err}, dist={counts}")


if __name__ == "__main__":
    main()
