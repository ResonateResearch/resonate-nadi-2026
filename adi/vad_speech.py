"""Generate local training exclusion lists with the NADI project's Silero VAD.

Adapted from the NADI project vad_speech.py (2026). Release changes supply
portable cache/output roots, explicit split and worker controls, and require
the installed silero-vad package instead of a torch.hub download fallback.
Speech detection, severity rules, decoding, and reports retain the source logic.
"""

import argparse
import os
import sys
import subprocess

# dataset -> (HF id, cache root). reuse the already-downloaded caches.
DATASETS = {
    "adi17": ("ArabicSpeech/ADI17", "ADI17"),
    "adi20": ("ArabicSpeech/ADI20", "ADI20"),
}

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("targets", nargs="*", help="adi17 and/or adi20 (default: both)")
parser.add_argument("--data_root", required=True, help="Root containing ADI17/ and ADI20/ caches")
parser.add_argument("--num_workers", type=int, default=64, help="CPU map workers (default: 64)")
parser.add_argument("--splits", nargs="+", default=["train"], help="Source splits to process (default: train)")
args = parser.parse_args()
if args.num_workers < 1:
    parser.error("--num_workers must be positive")
_targets = [a.lower() for a in args.targets] or list(DATASETS)
_bad = [t for t in _targets if t not in DATASETS]
if _bad:
    parser.error(f"unknown dataset(s): {_bad}; choose from {list(DATASETS)}")

# >1 dataset: run each in its own process. the HF cache root differs per dataset
# and is resolved when `datasets` imports, so it must be set before that import.
if len(_targets) > 1:
    for t in _targets:
        subprocess.run([
            sys.executable, os.path.abspath(__file__), t,
            "--data_root", args.data_root, "--num_workers", str(args.num_workers),
            "--splits", *args.splits,
        ], check=True)
    sys.exit(0)

DATASET = _targets[0]
HF_ID, CACHE_NAME = DATASETS[DATASET]
CACHE_ROOT = os.path.abspath(os.path.join(args.data_root, CACHE_NAME))
os.environ["HF_HUB_CACHE"] = os.path.join(CACHE_ROOT, "hub")
os.environ["HF_HOME"]      = CACHE_ROOT
os.environ["HF_DATASETS_CACHE"] = os.path.join(CACHE_ROOT, "datasets")
# dataset already fully cached -> force offline so map workers never race on a download
os.environ["HF_HUB_OFFLINE"]      = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import defaultdict, Counter

import io
import csv
import torch
import soundfile as sf
from tqdm import tqdm
from datasets import load_dataset, Audio

# ---- config ----
SR            = 16000     # silero-vad supports 16k (and 8k); we resample on decode
NUM_WORKERS   = args.num_workers  # silero runs on CPU per worker
VAD_THRESHOLD = 0.5       # speech-prob cutoff; raise (0.6-0.7) to reject music false-pos
MIN_SPEECH_MS = 250       # drop speech segments shorter than this
MIN_SIL_MS    = 100       # merge segments separated by silence shorter than this
BAD_RATIO     = 0.50      # flag clip if speech fraction < this (majority non-speech)
EMPTY_SEC     = 0.50      # OR flag if absolute speech seconds < this (music/silence/dead)
OUT_DIR_DS = CACHE_ROOT   # prepared lists sit beside each corpus's datasets/ cache

# NOTE: CUDA + fork (num_proc>1) is unsafe, and silero-vad is fast on CPU, so the
# map runs the model on CPU. Each worker lazily builds its own model (module-global).
_VAD = None  # (model, get_speech_timestamps) per worker process


def get_vad():
    """Lazy CPU model from the installed silero-vad package and its bundled weights."""
    global _VAD
    if _VAD is not None:
        return _VAD
    torch.set_num_threads(1)   # avoid oversubscription with NUM_WORKERS
    from silero_vad import load_silero_vad, get_speech_timestamps
    model = load_silero_vad()
    model.eval()
    _VAD = (model, get_speech_timestamps)
    return _VAD


def speech_seconds(wav_np):
    """Sum of speech-segment durations (s) for a mono 16k float array."""
    model, get_speech_timestamps = get_vad()
    wav = torch.from_numpy(np.ascontiguousarray(wav_np, dtype=np.float32))
    ts = get_speech_timestamps(
        wav, model,
        sampling_rate=SR,
        threshold=VAD_THRESHOLD,
        min_speech_duration_ms=MIN_SPEECH_MS,
        min_silence_duration_ms=MIN_SIL_MS,
    )
    samp = sum(t["end"] - t["start"] for t in ts)
    return samp / SR


def clip_id(ex):
    """Stable id: ADI17 has an 'id' column; ADI20 has none -> use audio filename."""
    if ex.get("id"):
        return ex["id"]
    p = (ex.get("audio") or {}).get("path")
    if p:
        return os.path.splitext(os.path.basename(p))[0]
    return ""   # caller falls back to split#index


def load_wav(a):
    """Decode audio dict (decode=False -> {bytes, path}) to a mono float32 16k array."""
    b = a.get("bytes")
    arr, sr = sf.read(io.BytesIO(b) if b is not None else a["path"],
                      dtype="float32", always_2d=False)
    if arr.ndim > 1:                   # stereo -> mono
        arr = arr.mean(axis=1)
    if sr != SR:                       # both ADI datasets are 16k; resample just in case
        import torchaudio
        arr = torchaudio.functional.resample(
            torch.from_numpy(arr), sr, SR).numpy()
    return np.ascontiguousarray(arr, dtype=np.float32)


def add_vad(ex):
    """Per-clip id + total dur + speech dur. -1 + err string on failure."""
    sid = ""
    try:
        sid = clip_id(ex)
        arr = load_wav(ex["audio"])
        total = len(arr) / SR
        return {"id": sid, "dur": float(total),
                "speech": float(speech_seconds(arr)), "err": ""}
    except Exception as e:
        return {"id": sid, "dur": -1.0, "speech": -1.0, "err": repr(e)[:200]}


def flag_reason(dur, sp, ratio):
    """None if clip is fine, else severity label."""
    if sp < EMPTY_SEC or ratio < 0.10:
        return "empty"     # near-zero speech: music-only / silence / dead audio
    if ratio < 0.30:
        return "severe"    # mostly non-speech
    if ratio < BAD_RATIO:
        return "low"       # majority non-speech
    return None


def dial_stats(tot, sp):
    tot = np.asarray(tot, dtype=np.float64)
    sp  = np.asarray(sp,  dtype=np.float64)
    if tot.size == 0:
        return {}
    ratio = np.divide(sp, tot, out=np.zeros_like(sp), where=tot > 0)
    return {
        "n":          int(tot.size),
        "total_h":    float(tot.sum() / 3600),
        "speech_h":   float(sp.sum() / 3600),
        "speech_pct": float(100 * sp.sum() / tot.sum()) if tot.sum() > 0 else 0.0,
        "clip_mean":  float(100 * ratio.mean()),   # mean per-clip speech %
        "clip_med":   float(100 * np.median(ratio)),
    }


def fmt(s):
    return "  ".join(f"{k}={v:.2f}" if isinstance(v, float) else f"{k}={v}"
                     for k, v in s.items())


def report_split(split, tot_by_dialect, sp_by_dialect, problematic):
    """Print stats + write 2 plots + npy + problematic csv for ONE split -> OUT_DIR_DS."""
    tag = f"{DATASET.upper()} [{split}]"
    dialects = sorted(tot_by_dialect, key=lambda k: -np.sum(tot_by_dialect[k]))
    if not dialects:
        print(f"{tag}: no clips -> skip plots")
        return
    all_tot = np.array([v for k in dialects for v in tot_by_dialect[k]])
    all_sp  = np.array([v for k in dialects for v in sp_by_dialect[k]])

    print(f"\n=== {tag} speech (VAD) stats per dialect ===")
    for dial in dialects:
        print(f"{dial:>6}: {fmt(dial_stats(tot_by_dialect[dial], sp_by_dialect[dial]))}")
    print(f"{'ALL':>6}: {fmt(dial_stats(all_tot, all_sp))}")

    # per-clip speech ratio (fraction 0..1) per dialect, for the plots
    ratio_by_dialect = {
        d: np.divide(np.asarray(sp_by_dialect[d]), np.asarray(tot_by_dialect[d]),
                     out=np.zeros(len(tot_by_dialect[d])),
                     where=np.asarray(tot_by_dialect[d]) > 0)
        for d in dialects
    }

    # ---- fig1: histogram grid per dialect of per-clip speech ratio ----
    ncol = 4
    nrow = (len(dialects) + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 2.6 * nrow), squeeze=False)
    bins = np.linspace(0, 1, 40)
    for ax, dial in zip(axes.ravel(), dialects):
        r = ratio_by_dialect[dial]
        ax.hist(r, bins=bins, color="steelblue")
        tt = np.sum(tot_by_dialect[dial])
        pct = 100 * np.sum(sp_by_dialect[dial]) / tt if tt > 0 else 0.0
        ax.set_title(f"{dial}  (n={r.size}, {pct:.0f}% speech)", fontsize=9)
        ax.set_yscale("log")
        ax.set_xlabel("speech fraction")
    for ax in axes.ravel()[len(dialects):]:
        ax.axis("off")
    fig.suptitle(f"{tag} per-clip speech-fraction distribution per dialect "
                 f"(silero-vad, thr={VAD_THRESHOLD})", y=1.005)
    fig.tight_layout()
    p1 = os.path.join(OUT_DIR_DS, f"{split}_speech_frac_per_dialect.png")
    fig.savefig(p1, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"saved -> {p1}")

    # ---- fig2: speech vs non-speech hours bar + per-clip ratio boxplot ----
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(15, 5))
    sp_h  = np.array([np.sum(sp_by_dialect[d])  / 3600 for d in dialects])
    tot_h = np.array([np.sum(tot_by_dialect[d]) / 3600 for d in dialects])
    nonsp_h = tot_h - sp_h
    a0.bar(dialects, sp_h,   color="seagreen",           label="speech")
    a0.bar(dialects, nonsp_h, bottom=sp_h, color="lightgray", label="non-speech")
    for i, d in enumerate(dialects):
        a0.text(i, tot_h[i], f"{100*sp_h[i]/tot_h[i]:.0f}%",
                ha="center", va="bottom", fontsize=8)
    a0.set_ylabel("hours"); a0.set_title(f"{tag} speech vs non-speech hours per dialect")
    a0.tick_params(axis="x", rotation=45); a0.legend()

    a1.boxplot([ratio_by_dialect[d] for d in dialects], labels=dialects, showfliers=False)
    a1.set_ylabel("per-clip speech fraction"); a1.set_title("speech-fraction spread")
    a1.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    p2 = os.path.join(OUT_DIR_DS, f"{split}_speech_compare_dialect.png")
    fig.savefig(p2, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"saved -> {p2}")

    # ---- dump raw arrays ----
    pnpy = os.path.join(OUT_DIR_DS, f"{split}_vad_by_dialect.npy")
    np.save(pnpy,
            {"total": {k: np.asarray(v) for k, v in tot_by_dialect.items()},
             "speech": {k: np.asarray(v) for k, v in sp_by_dialect.items()}},
            allow_pickle=True)
    print(f"saved -> {pnpy}")

    # ---- dump problematic sample ids ----
    # reason: empty (<0.5s or <10% speech) | severe (<30%) | low (<50%) | decode_fail
    problematic.sort(key=lambda r: r[5])   # lowest speech ratio first
    pcsv = os.path.join(OUT_DIR_DS, f"{split}_problematic.csv")
    with open(pcsv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "split", "dialect", "dur_s", "speech_s", "speech_ratio", "reason"])
        for sid, sp_, dial, tt, ss, ratio, reason in problematic:
            w.writerow([sid, sp_, dial, f"{tt:.3f}", f"{ss:.3f}", f"{ratio:.4f}", reason])

    n_total = sum(len(v) for v in tot_by_dialect.values())
    by_reason = defaultdict(int)
    for r in problematic:
        by_reason[r[6]] += 1
    print(f"saved -> {pcsv}")
    print(f"{tag} flagged {len(problematic)}/{n_total} clips "
          f"({100*len(problematic)/max(n_total,1):.2f}%): "
          + "  ".join(f"{k}={v}" for k, v in sorted(by_reason.items())))


print(f"\n########## {DATASET.upper()} ({HF_ID}) ##########")
os.makedirs(OUT_DIR_DS, exist_ok=True)
print(f"output dir -> {OUT_DIR_DS}")
ds = load_dataset(HF_ID, cache_dir=os.environ["HF_DATASETS_CACHE"])
missing_splits = [split for split in args.splits if split not in ds]
if missing_splits:
    sys.exit(f"Missing source splits {missing_splits}; available: {list(ds)}")

# Validate silero-vad loads NOW (fail loud), then reset so each map worker loads its
# own model -> avoids fork-sharing a torch model across 64 procs (can break inference).
print("loading silero-vad ...")
get_vad()
_VAD = None

for split in args.splits:
    # decode=False -> {bytes, path}; we decode via soundfile (avoids torchcodec AudioDecoder)
    d = ds[split].cast_column("audio", Audio(decode=False))
    # add_vad emits 'id' itself (ADI20 has no id column); keep 'dialect' from source
    drop = [c for c in d.column_names if c != "dialect"]
    d = d.map(add_vad, num_proc=NUM_WORKERS, remove_columns=drop,
              desc=f"vad[{split}]")
    durs   = np.asarray(d["dur"],    dtype=np.float64)
    speech = np.asarray(d["speech"], dtype=np.float64)
    dials  = d["dialect"]
    ids    = d["id"]

    # per-split accumulators (kept separate per split -> separate plots)
    tot_by_dialect = defaultdict(list)   # per-clip total seconds
    sp_by_dialect  = defaultdict(list)   # per-clip speech seconds
    problematic    = []                  # (id, split, dialect, dur, speech, ratio, reason)
    for i, (sid, dial, tt, ss) in enumerate(tqdm(zip(ids, dials, durs, speech),
                                  total=len(durs), desc=f"agg[{split}]")):
        sid = sid or f"{split}#{i}"     # fallback if no id / no audio path
        if tt < 0 or ss < 0:
            problematic.append((sid, split, dial, tt, ss, -1.0, "decode_fail"))
            continue
        tot_by_dialect[dial].append(tt)
        sp_by_dialect[dial].append(ss)
        ratio = ss / tt if tt > 0 else 0.0
        reason = flag_reason(tt, ss, ratio)
        if reason:
            problematic.append((sid, split, dial, tt, ss, ratio, reason))

    n_fail = int((durs < 0).sum())
    if n_fail:
        errs = Counter(e for e in d["err"] if e)
        top = "  |  ".join(f"{c}x {msg}" for msg, c in errs.most_common(3))
        print(f"[{split}] WARN {n_fail}/{len(durs)} clips failed -> {top}")
    if len(durs) and n_fail == len(durs):
        sys.exit(f"[{split}] ALL clips failed -> fix the error above (see [{split}] WARN).")
    print(f"[{split}] {len(durs)} clips")

    report_split(split, tot_by_dialect, sp_by_dialect, problematic)
