#!/usr/bin/env python3
"""Export only NADI train/validation and optionally NADI-derived parquet audio.

Extracted from the original preparation script's NADI stages. External Arabic
anchor, code-switch retention, merged dev, and mixed smoke stages are excluded.
See PREPARATION.md for the separately supplied denoising/augmentation inputs.
"""
import argparse
import io
import json
import os
from pathlib import Path

import pyarrow.parquet as pq
import soundfile as sf

DIALECTS = ["Algeria", "Egypt", "Jordan", "Mauritania", "Morocco", "Palestine", "UAE", "Yemen"]
VERSIONS = ["aug-additive", "aug-fullchain", "aug-channel", "clean-neural"]
PREFIX = "language Arabic<asr_text>"


def write_audio(dest, blob):
    """Preserve the original export's bytes-first, resumable audio handling."""
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    if blob[:4] in (b"fLaC", b"RIFF"):
        tmp = dest.with_suffix(dest.suffix + ".tmp")
        tmp.write_bytes(blob)
        tmp.rename(dest)
    else:
        wav, sr = sf.read(io.BytesIO(blob))
        sf.write(dest, wav, sr, format="FLAC")
    return dest


def parquet_rows(root, dialect, split):
    shards = sorted((root / dialect).glob(f"{split}-*.parquet"))
    if not shards:
        raise FileNotFoundError(f"no {split} shards for {root}/{dialect}")
    for shard in shards:
        table = pq.read_table(shard, columns=["id", "audio", "transcription"])
        for row in table.to_pylist():
            text = (row["transcription"] or "").strip()
            if text:
                yield row, text


def write_row(handle, row):
    handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def export_train(root, name, audio, manifests):
    count = 0
    with (manifests / f"nadi_{name.replace('-', '_')}.jsonl").open("w", encoding="utf-8") as out:
        for dialect in DIALECTS:
            for row, text in parquet_rows(root, dialect, "train"):
                path = write_audio(audio / name / dialect / f"{row['id']}.flac", row["audio"]["bytes"])
                write_row(out, {"audio": str(path), "text": PREFIX + text})
                count += 1
    print(f"[{name}] train rows: {count}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--nadi-root", type=Path, required=True,
                    help="source root containing <Dialect>/{train,validation}-*.parquet")
    ap.add_argument("--workspace", type=Path,
                    default=Path(os.environ.get("NADI_ASR_WORKSPACE", "work/asr")))
    ap.add_argument("--derived-root", type=Path,
                    help="optional external root containing <version>/<Dialect>/train-*.parquet")
    ap.add_argument("--versions", nargs="+", choices=VERSIONS, default=["clean-neural"])
    ap.add_argument("--dev-per-dialect", type=int, default=38,
                    help="size of deterministic NADI-only dev sample per dialect")
    a = ap.parse_args()
    workspace = a.workspace.resolve()
    audio, manifests = workspace / "audio", workspace / "manifests"
    manifests.mkdir(parents=True, exist_ok=True)
    export_train(a.nadi_root, "original", audio, manifests)
    if a.derived_root:
        for version in a.versions:
            export_train(a.derived_root / version, version, audio, manifests)

    # The sample is for convenience; it is not the historical duration-matched 512-item probe.
    with (manifests / "nadi_val_eval.jsonl").open("w", encoding="utf-8") as full, \
            (manifests / "nadi_dev.jsonl").open("w", encoding="utf-8") as dev:
        for dialect in DIALECTS:
            for index, (row, text) in enumerate(parquet_rows(a.nadi_root, dialect, "validation")):
                path = write_audio(audio / "original_val" / dialect / f"{row['id']}.flac", row["audio"]["bytes"])
                result = {"utt_id": f"nadi-{dialect}-{row['id']}", "audio": str(path),
                          "refs": [text], "domain": "nadi", "dataset": dialect}
                write_row(full, result)
                if index < a.dev_per_dialect:
                    write_row(dev, result)


if __name__ == "__main__":
    main()
