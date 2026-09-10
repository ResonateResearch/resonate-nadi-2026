#!/usr/bin/python3
"""Build a fixed-length segment manifest for ADI-20 training.

Why: training currently draws ONE random `sentence_len` crop per clip per
epoch, so a dialect's share of updates is its CLIP count, not its hours. Clip
lengths differ ~5x across dialects (BAH 26.2s mean vs JOR 4.8s), which makes the
per-epoch class prior lopsided (BAH 1.5% of samples vs JOR 8.2%) even though the
subset is hour-balanced. It also means only ~64% of the audio is reachable in a
given epoch -- an 26s clip contributes 5s and the rest is unseen.

Cutting every kept clip into `seg_len` segments fixes both: segment count becomes
proportional to hours, so the hour balance the subset already enforces carries
straight through to the sample counts (~5.3x imbalance -> ~1.08x), and every
second of audio is reachable each epoch.

Durations come from metadata:
    ADI-17  -> parsed from the id (`<ytid>_<startCS>-<endCS>`)
    ADI-20  -> `adi20_duration_cache` (measured wav headers)
    VC      -> `duration_s` in each metadata.jsonl
If the ADI-20 duration cache is missing, the shared selection code measures
local audio headers and writes that cache first. The full HyperPyYAML recipe
also constructs its model and augmentation objects, using local model files;
this script does not run a model forward pass. It writes the manifest and a
configuration sidecar.

The clip set is `_build_keep_ids()` plus the VC selection, i.e. EXACTLY the
clips `train_ca_mhfa_adi.py` would train on under the same yaml -- the manifest
is that same set expanded, never a different sample.

Output CSV (one row = one training example):
    ID        utterance id (join key into the arrow corpora; VC uses its own id)
    dialect   label
    source    adi17 | adi20 | vc
    path      wav path for `vc`, empty for the arrow-backed corpora
    start_s   segment start inside the clip, seconds
    dur_s     segment length, seconds (< seg_len only for a kept tail)
A `<out>.meta.json` sidecar records the config the manifest was built from --
check it before reusing a manifest, the contents depend on subset_target_h /
subset_seed / subset_min_dur / drop_reasons.

Knobs (script-only, NOT yaml keys -- stripped from argv before hyperpyyaml):
    --out         manifest path (default: next to adi20_duration_cache)
    --seg_len     segment length in seconds (default 5.0, = sentence_len)
    --hop         segment hop in seconds (default = seg_len, non-overlapping)
    --min_seg     drop a tail shorter than this (default 2.0)
    --balance     truncate every dialect to the smallest dialect's count
    --max_per_dialect  hard cap on segments per dialect (0 = no cap)

Run:
> python make_segment_manifest.py hparams/train_wavlm_base_plus.yaml \
>     --data_root "$ADI_DATA_ROOT" --ssl_hub "$WAVLM_BASE_PLUS" \
>     --out "$ADI_DATA_ROOT/adi20_segments_5s.csv" --seg_len 5 --hop 5 --min_seg 2

Public-release modifications (2026): portable examples and offline execution
without implicit Hub credentials; segmentation and selection logic retained.

Author
    * nadi project, 2026
"""

import csv
import json
import logging
import os
import random
import sys
from collections import defaultdict

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"

from hyperpyyaml import load_hyperpyyaml

import speechbrain as sb
from train_ca_mhfa_adi import (
    _adi20_durations,
    _build_keep_ids,
    _dur_from_id,
    _adi17_source,
    _load_adi17_id_dialect,
    _load_adi20_id_dialect,
)

logger = logging.getLogger(__name__)

FIELDS = ["ID", "dialect", "source", "path", "start_s", "dur_s"]


def _pop_arg(argv, flag, default):
    """Pull `--flag value` out of argv before sb.parse_arguments sees it.

    Keeps hyperpyyaml from rejecting a script-only knob as an unknown override.
    """
    if flag in argv:
        i = argv.index(flag)
        val = argv[i + 1]
        del argv[i:i + 2]
        return val
    return default


def segments(dur, seg_len, hop, min_seg):
    """(start, length) windows covering a clip of `dur` seconds.

    Clip shorter than seg_len -> one whole-clip segment (kept if it clears
    min_seg; the loader pads it and `lens` masks the padding, same as today).
    Otherwise stride by `hop`, and keep the ragged tail only when it is at least
    min_seg long -- a 0.3s remainder is noise, not a training example.
    """
    if dur < min_seg:
        return []
    if dur <= seg_len:
        return [(0.0, round(dur, 3))]

    out = []
    start = 0.0
    while start + seg_len <= dur + 1e-6:
        out.append((round(start, 3), seg_len))
        start += hop
    tail = dur - start
    if tail >= min_seg:
        out.append((round(start, 3), round(tail, 3)))
    return out


def vc_selection(hparams):
    """(id, dialect, path, duration) for the VC clips the recipe would keep.

    Mirrors `load_vc_split`: each dialect takes only the hours real audio could
    not supply (`subset_target_h` - the real hours `_build_keep_ids` published),
    same seed and same shuffle, so the selection matches the training run.
    """
    folder = hparams.get("vc_data_folder")
    if not folder:
        return []
    root = os.path.join(folder, str(hparams.get("vc_variant", "clean")))
    if not os.path.isdir(root):
        raise FileNotFoundError(f"vc_variant dir not found: {root}")

    min_dur = float(hparams.get("vc_min_dur", hparams.get("subset_min_dur", 3.0)))
    real_h = hparams.get("subset_real_hours") or {}
    target_s = float(hparams.get("subset_target_h", 53.0)) * 3600
    seed = hparams.get("subset_seed", hparams.get("seed", 1986))

    picked, total_s = [], defaultdict(float)
    for dia in sorted(os.listdir(root)):
        meta = os.path.join(root, dia, "metadata.jsonl")
        if not os.path.isfile(meta):
            continue
        rows = []
        with open(meta, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                if r["duration_s"] >= min_dur:
                    rows.append(r)
        budget_s = (
            max(0.0, target_s - real_h.get(dia, 0.0)) if real_h else float("inf")
        )
        random.Random(seed).shuffle(rows)
        for r in rows:
            if total_s[r["dialect"]] >= budget_s:
                continue
            # Rebuilt from `root`: the recorded wav_path points at the machine
            # that generated the clip (same reason load_vc_split rebuilds it).
            picked.append((
                r["id"],
                r["dialect"],
                os.path.join(root, dia, "wavs", r["id"] + ".wav"),
                float(r["duration_s"]),
            ))
            total_s[r["dialect"]] += r["duration_s"]
    return picked


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    argv = sys.argv[1:]
    out_path = _pop_arg(argv, "--out", None)
    seg_len = float(_pop_arg(argv, "--seg_len", 5.0))
    hop = float(_pop_arg(argv, "--hop", 0) or seg_len)
    min_seg = float(_pop_arg(argv, "--min_seg", 2.0))
    max_per_dialect = int(_pop_arg(argv, "--max_per_dialect", 0))
    balance = "--balance" in argv
    if balance:
        argv.remove("--balance")

    hparams_file, run_opts, overrides = sb.parse_arguments(argv)
    with open(hparams_file, encoding="utf-8") as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    if seg_len <= 0 or hop <= 0 or min_seg <= 0:
        raise ValueError("seg_len / hop / min_seg must all be > 0")
    if min_seg > seg_len:
        raise ValueError(f"min_seg={min_seg} > seg_len={seg_len}")
    if out_path is None:
        cache = hparams.get("adi20_duration_cache") or "."
        out_path = os.path.join(
            os.path.dirname(cache), f"adi20_segments_{seg_len:g}s.csv"
        )

    # ---- the exact clip set the recipe would train on under this yaml.
    if not hparams.get("use_subset", False):
        raise ValueError(
            "use_subset is False -- this manifest is defined by the subset. "
            "Enable it, or segment the full split with a separate tool."
        )
    keep = _build_keep_ids(hparams)

    # ---- id -> (dialect, duration) for both arrow corpora. Only ids in `keep`
    # matter; everything else is dropped as we go so nothing large is retained.
    dur20 = _adi20_durations(hparams)
    clips = []          # (ID, dialect, source, path, duration)

    src17 = _adi17_source(hparams)
    if src17 is not None:
        ids17, dia17 = _load_adi17_id_dialect(src17)
        for uid, dia in zip(ids17, dia17):
            if uid in keep:
                d = _dur_from_id(uid)
                if d > 0:
                    clips.append((uid, dia, "adi17", "", d))
    seen17 = {c[0] for c in clips}

    ids20, dia20 = _load_adi20_id_dialect(hparams)
    n_nodur = 0
    for uid, dia in zip(ids20, dia20):
        if uid in keep and uid not in seen17:
            d = dur20.get(uid, 0.0)
            if d > 0:
                clips.append((uid, dia, "adi20", "", d))
            else:
                n_nodur += 1
    if n_nodur:
        logger.warning(
            f"[seg] {n_nodur} kept ADI-20 ids have no measured duration "
            f"-> skipped (rebuild adi20_duration_cache to include them)"
        )

    n_real = len(clips)
    clips += [(i, d, "vc", p, s) for i, d, p, s in vc_selection(hparams)]
    logger.info(
        f"[seg] clips: real={n_real} vc={len(clips) - n_real} total={len(clips)}"
    )

    # ---- expand every clip into segments.
    rows_by_dialect = defaultdict(list)
    for uid, dia, source, path, dur in clips:
        for start, length in segments(dur, seg_len, hop, min_seg):
            rows_by_dialect[dia].append(
                {
                    "ID": uid,
                    "dialect": dia,
                    "source": source,
                    "path": path,
                    "start_s": start,
                    "dur_s": length,
                }
            )

    # ---- optional count capping. Segmentation alone lands the dialects within
    # a few percent of each other (counts track hours, which the subset already
    # balances), so this is a trim rather than a fix -- shuffled on subset_seed
    # so the kept subset is reproducible and not biased to whole clips.
    cap = max_per_dialect or 0
    if balance:
        smallest = min(len(v) for v in rows_by_dialect.values())
        cap = smallest if cap == 0 else min(cap, smallest)
    if cap:
        rng = random.Random(hparams.get("subset_seed", hparams.get("seed", 1986)))
        for dia, rows in rows_by_dialect.items():
            if len(rows) > cap:
                rng.shuffle(rows)
                rows_by_dialect[dia] = rows[:cap]

    # ---- write. Sorted by dialect then ID then start so the file is stable
    # across runs and diffable; the loader shuffles, so order is not a prior.
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    total = 0
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for dia in sorted(rows_by_dialect):
            for r in sorted(rows_by_dialect[dia], key=lambda x: (x["ID"], x["start_s"])):
                w.writerow(r)
                total += 1

    meta = {
        "seg_len": seg_len,
        "hop": hop,
        "min_seg": min_seg,
        "balance": balance,
        "max_per_dialect": max_per_dialect,
        "n_segments": total,
        "n_clips": len(clips),
        # The manifest is only valid for these subset settings -- changing any
        # of them changes the clip set, so rebuild rather than reuse.
        "subset": {
            k: hparams.get(k)
            for k in (
                "subset_mode", "subset_target_h", "subset_min_dur",
                "subset_seed", "subset_restrict_to_manifest_videos",
                "subset_csv", "drop_problematic", "drop_reasons",
                "vc_data_folder", "vc_variant", "vc_min_dur",
            )
        },
        "per_dialect": {
            d: len(rows_by_dialect[d]) for d in sorted(rows_by_dialect)
        },
    }
    with open(out_path + ".meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    # ---- report: the point of the manifest is the count spread, so print it.
    counts = meta["per_dialect"]
    hi, lo = max(counts.values()), min(counts.values())
    print(f"\nmanifest -> {out_path}")
    print(f"  clips {len(clips)} -> segments {total}  "
          f"({seg_len:g}s, hop {hop:g}s, tail >= {min_seg:g}s)")
    print(f"{'dial':6s}{'segments':>10s}{'share':>8s}{'hours':>9s}")
    for d in sorted(counts, key=lambda x: -counts[x]):
        h = sum(r["dur_s"] for r in rows_by_dialect[d]) / 3600
        print(f"{d:6s}{counts[d]:>10d}{100 * counts[d] / total:>7.2f}%{h:>9.1f}")
    print(f"  max/min ratio {hi / lo:.2f}x   (uniform share = {100 / len(counts):.2f}%)")
