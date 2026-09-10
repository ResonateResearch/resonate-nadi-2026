#!/usr/bin/env python3
"""Release adapter: create train-only VC metadata/config from exported NADI audio.

The original VC pipeline consumed a separate metadata tree. This adapter
reconstructs that interface from nadi_original.jsonl without bundling any data.
"""
import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import soundfile as sf

DIALECTS = ["Algeria", "Egypt", "Jordan", "Mauritania", "Morocco", "Palestine", "UAE", "Yemen"]
PREFIX = "language Arabic<asr_text>"
EXPERIMENT = "clean_neural__redimnet__thr0.45_coh0.7"


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    # JSON is valid YAML, allowing this adapter to avoid a YAML-writing dependency.
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--original-manifest", required=True, type=Path)
    ap.add_argument("--clean-audio-root", required=True, type=Path,
                    help="directory containing <Dialect>/<source_id>.flac")
    ap.add_argument("--workspace", type=Path,
                    default=Path(os.environ.get("NADI_VC_WORKSPACE", "work/vc")))
    a = ap.parse_args()
    workspace = a.workspace.resolve()
    clean = a.clean_audio_root.resolve()
    metadata = workspace / "metadata"
    groups = defaultdict(list)
    with a.original_manifest.open(encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            row = json.loads(line)
            path = Path(row["audio"])
            if "/audio/original/" not in path.as_posix() or "original_val" in path.as_posix():
                raise ValueError(f"source must come from the exported NADI train tree: {path}")
            country = path.parent.name
            if country not in DIALECTS or not row["text"].startswith(PREFIX):
                raise ValueError("invalid NADI dialect or training prefix")
            if not (clean / country / f"{path.stem}.flac").is_file():
                raise FileNotFoundError(f"missing clean-neural counterpart for {path}")
            info = sf.info(path)
            groups[country].append({"id": path.stem, "duration_s": info.duration,
                                    "transcription": row["text"][len(PREFIX):]})
    for country in DIALECTS:
        if not groups[country]:
            raise ValueError(f"no original train rows for {country}")
        dump(metadata / country / "train" / "metadata.json", groups[country])
    dump(workspace / "data/clean_neural/dataset.yaml", {
        "name": "clean_neural", "audio_root": str(clean), "metadata_root": str(metadata),
        "layout": "country_flat", "ext": "flac", "countries": DIALECTS, "splits": ["train"],
    })
    dump(workspace / "exp" / EXPERIMENT / "config.yaml", {
        "dataset": "clean_neural", "embed_model": "redimnet",
        "cluster": {"linkage": "complete", "threshold": 0.45, "cohesion": 0.70, "min_dur_s": 30},
        "knnvc": {"topk": 4, "prematched": True},
    })
    print(f"[vc config] {sum(map(len, groups.values()))} train clips -> {workspace}")


if __name__ == "__main__":
    main()
