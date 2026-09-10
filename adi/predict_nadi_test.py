#!/usr/bin/python3
"""Write NADI-2026 Subtask-2 prediction files for a local blind dataset.

Inputs: a Hugging Face Dataset saved with Dataset.save_to_disk, containing
``audio`` and ``idx``, and an explicit SpeechBrain ``--checkpoint`` directory.
``--dataset_split train`` selects a split if the local input is a DatasetDict;
the shared task packages its blind data under that split name.

Each row is scored once, using a central crop of at most ``eval_max_len``
seconds (30 in the released configuration). Submission logits are
``aam_scale * cosine`` (scale 32 in that configuration), reordered to the
official dialect order. Row order is preserved. This computes no official
accuracy or C_avg because blind reference labels are unavailable.

See README.md for the complete command. No checkpoint is selected implicitly.
README.md explains how to supply a local checkpoint and its matching label encoder.

Public-release modifications (2026): local dataset input, mandatory checkpoint,
offline model loading without implicit Hub credentials, and explicit CPU support.

Author
    * nadi project, 2026
"""

import contextlib
import csv as _csv
import json
import os
import pathlib
import sys
import zipfile
from collections import Counter

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"

import torch
from datasets import Audio, Dataset, DatasetDict, load_from_disk
from hyperpyyaml import load_hyperpyyaml

import speechbrain as sb
from speechbrain.utils.checkpoints import Checkpointer
from train_ca_mhfa_adi import _read_chunk

# NADI-2026 Subtask-2 submission column order (from the shared-task notebook).
# logits.tsv columns and predictions.tsv indices MUST follow this exact order,
# which differs from our training label-encoder order -> we permute below.
NADI_SUBMISSION_ORDER = [
    "MSA", "BAH", "TUN", "ALG", "EGY", "IRA", "JOR", "KSA", "KUW", "LEB",
    "LIB", "MAU", "MOR", "OMA", "PAL", "QAT", "SUD", "SYR", "UAE", "YEM",
]


def _pop_arg(argv, flag, default):
    """Pull `--flag value` out of argv before sb.parse_arguments sees it.

    Keeps hyperpyyaml from rejecting a script-only knob as an unknown override.
    """
    if flag in argv:
        i = argv.index(flag)
        if i + 1 >= len(argv) or argv[i + 1].startswith("--"):
            raise ValueError(f"{flag} requires a value")
        val = argv[i + 1]
        del argv[i:i + 2]
        return val
    return default


if __name__ == "__main__":
    argv = sys.argv[1:]
    ckpt_path = _pop_arg(argv, "--checkpoint", None)
    dataset_path = _pop_arg(argv, "--dataset", None)
    dataset_split = _pop_arg(argv, "--dataset_split", "train")
    if ckpt_path is None or dataset_path is None:
        raise ValueError("Both --checkpoint and --dataset are required; see README.md")
    ckpt_dir = pathlib.Path(ckpt_path)
    if not Checkpointer._is_checkpoint_dir(ckpt_dir):
        raise FileNotFoundError(
            f"{ckpt_dir} is not a SpeechBrain checkpoint directory "
            "(expected a CKPT+* directory containing CKPT.yaml)"
        )
    if not pathlib.Path(dataset_path).is_dir():
        raise FileNotFoundError(f"Local dataset directory not found: {dataset_path}")

    hparams_file, run_opts, overrides = sb.parse_arguments(argv)
    with open(hparams_file, encoding="utf-8") as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    device = run_opts.get("device", "cpu")
    sr = hparams["sample_rate"]
    cap = hparams.get("eval_max_len", 0.0)
    cap_n = int(cap * sr) if cap and cap > 0 else None  # center-crop cap or None

    # ---- label encoder (same 20-class ADI-20 encoder built during training).
    enc = sb.dataio.encoder.CategoricalEncoder()
    if not enc.load_if_possible(hparams["label_encoder_file"]):
        raise FileNotFoundError(
            f"label encoder not found at {hparams['label_encoder_file']} "
            "-- run training first."
        )
    missing = [d for d in NADI_SUBMISSION_ORDER if d not in enc.lab2ind]
    if missing:
        raise ValueError(f"encoder is missing submission dialects: {missing}")

    # ---- modules + the exact checkpoint selected by the caller.
    modules = hparams["modules"]
    checkpointer = hparams["checkpointer"]
    # SpeechBrain's own directory -> Checkpoint(meta, paramfiles) constructor.
    found = Checkpointer._construct_checkpoint_objects([ckpt_dir])[0]
    checkpointer.load_checkpoint(found)
    which = f"ckpt_{ckpt_dir.name.replace('CKPT+', '')}"
    print(f"[predict] ckpt={which}  path={found.path}  "
          f"ErrorRate={found.meta.get('ErrorRate', float('nan')):.4f}")
    ssl_model = modules["ssl_model"].to(device).eval()
    embedding_model = modules["embedding_model"].to(device).eval()
    classifier = modules["classifier"].to(device).eval()

    # ---- user-prepared local blind data, kept in its original row order.
    ds = load_from_disk(dataset_path)
    if isinstance(ds, DatasetDict):
        ds = ds[dataset_split]
    if not isinstance(ds, Dataset):
        raise TypeError("--dataset must contain a saved Dataset or DatasetDict")
    missing_columns = {"audio", "idx"} - set(ds.column_names)
    if missing_columns:
        raise ValueError(f"Dataset is missing columns: {sorted(missing_columns)}")
    if not len(ds):
        raise ValueError("The prediction dataset is empty")
    ds = ds.cast_column("audio", Audio(decode=False))  # keep {bytes,path}
    print(
        f"[predict] NADI-2026 test n={len(ds)} "
        f"cap={'none' if cap_n is None else f'{cap}s'} device={device}"
    )

    eval_prec = run_opts.get("eval_precision", "bf16")  # CLI --eval_precision
    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(eval_prec)
    use_amp = amp_dtype is not None and device.startswith("cuda")

    # Submission logits = scale * cosine (the softmax logit at inference, margin 0).
    scale = float(hparams["aam_scale"])
    # column j of the submission -> index of that dialect in our encoder's logits.
    perm = [enc.lab2ind[d] for d in NADI_SUBMISSION_ORDER]

    ids, durs, sub_logits = [], [], []
    lens = torch.ones(1, device=device)  # single full utterance = all valid
    log_every = max(1, len(ds) // 20)

    # Row order is the submission order -- sequential loop, never shuffled.
    for i in range(len(ds)):
        row = ds[i]
        sig = _read_chunk(row["audio"], cap_n, sr, mode="center")
        durs.append(sig.shape[-1] / sr)
        wav = sig.unsqueeze(0).to(device)

        ctx = (
            torch.autocast(device_type="cuda", dtype=amp_dtype)
            if use_amp
            else contextlib.nullcontext()
        )
        with torch.no_grad(), ctx:
            feats = ssl_model(wav).permute(1, 0, 2, 3)  # [B, L, T, F]
            emb = embedding_model(feats, lens)          # [B, emb_dim]
            out = classifier(emb)                       # [B, 1, n_class] cosine

        logit = (scale * out.squeeze()).float().cpu()   # [n_class] inference logit
        sub_logits.append(logit[perm].tolist())         # -> NADI column order
        ids.append(str(row["idx"]))

        if (i + 1) % log_every == 0:
            print(f"[predict] {i + 1}/{len(ds)}", flush=True)

    sub_preds = [int(max(range(len(r)), key=r.__getitem__)) for r in sub_logits]

    # ---- NADI-2026 Subtask-2 submission files -------------------------------
    # Official format (from the shared-task notebook): two TAB-separated files
    #   logits.tsv       -> 20 logits per sample, NADI column order, no header
    #   predictions.tsv  -> predicted class index (into NADI order), one per line
    # zipped as submission.zip. Neither file carries ids, so ROW ORDER is the
    # only alignment to the reference -- keep it identical to the dataset order.
    sub_dir = os.path.join(
        hparams["output_folder"], f"submission_nadi_test_{which}"
    )
    os.makedirs(sub_dir, exist_ok=True)
    logits_tsv = os.path.join(sub_dir, "logits.tsv")
    preds_tsv = os.path.join(sub_dir, "predictions.tsv")
    preds_csv = os.path.join(sub_dir, "predictions.csv")  # convenience idx,dialect
    zip_path = os.path.join(sub_dir, "submission.zip")

    with open(logits_tsv, "w", encoding="utf-8", newline="") as fl:
        for r in sub_logits:
            fl.write("\t".join(f"{v:.6f}" for v in r) + "\n")
    with open(preds_tsv, "w", encoding="utf-8", newline="") as fp:
        for p in sub_preds:
            fp.write(f"{p}\n")
    with open(preds_csv, "w", encoding="utf-8", newline="") as fc:
        w = _csv.writer(fc)
        w.writerow(["idx", "dialect"])
        for uid, p in zip(ids, sub_preds):
            w.writerow([uid, NADI_SUBMISSION_ORDER[p]])

    # zip logits.tsv + predictions.tsv at archive root (exact submission names).
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(logits_tsv, arcname="logits.tsv")
        zf.write(preds_tsv, arcname="predictions.tsv")

    # ---- sanity summary: no labels to score against, so eyeball the shape of
    # the predictions instead. A collapsed distribution (one dialect taking most
    # of the dataset) or a duration profile far from the dev set means something is
    # wrong upstream, and it is the only check available on a blind set.
    counts = Counter(NADI_SUBMISSION_ORDER[p] for p in sub_preds)
    durs_sorted = sorted(durs)
    summary = {
        "n": len(ids),
        "checkpoint": which,
        "checkpoint_path": str(found.path),
        "checkpoint_error_rate": found.meta.get("ErrorRate"),
        "eval_max_len": cap,
        "duration_s": {
            "min": round(durs_sorted[0], 2),
            "median": round(durs_sorted[len(durs_sorted) // 2], 2),
            "max": round(durs_sorted[-1], 2),
            "mean": round(sum(durs) / len(durs), 2),
        },
        "predicted_counts": {d: counts.get(d, 0) for d in NADI_SUBMISSION_ORDER},
    }
    with open(
        os.path.join(sub_dir, "summary.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\nsubmission -> {zip_path}")
    print(f"  logits.tsv       {len(sub_logits)} rows x {len(NADI_SUBMISSION_ORDER)} cols")
    print(f"  predictions.tsv  {len(sub_preds)} rows")
    print(f"  predictions.csv  {len(sub_preds)} rows (idx,dialect)")
    d = summary["duration_s"]
    print(f"  durations        min={d['min']}s median={d['median']}s max={d['max']}s")
    print("  predicted class distribution:")
    for dial, c in counts.most_common():
        print(f"    {dial}  {c:4d}  ({100 * c / len(sub_preds):.1f}%)")
