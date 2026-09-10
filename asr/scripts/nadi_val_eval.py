#!/usr/bin/env python3
"""Decode NADI ASR manifests and optionally report training-harness WER.

Rows contain {utt_id, audio, refs:[text], dataset:<country>}. The submitted
configuration uses --num-beams 10 --no-repeat-ngram 6, with --dialect-cond set
to match the checkpoint's training run; the CLI default is greedy decoding.
The --merge mode scores existing hyp.shard_* files using common.normalize_text.
Use local_score.py for the paper's local validation approximation.

Each process uses one visible CUDA device. Optional --num-shards N --shard-id i
partitions the manifest across separate decoder processes. See ../README.md for
complete commands, exact voter order, and empty-reference blind-test inputs.
"""
import argparse
import json
import os
import sys

import torch

# same non-negotiable as inference: cuDNN fused SDPA crashes some cross-attn shapes
torch.backends.cuda.enable_cudnn_sdp(False)

# Public export: resolve the bundled trainer, not a private installation.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "train"))
from data_collator import load_audio          # noqa: E402
from callbacks import generate_transcripts     # noqa: E402
from common import dialect_of, normalize_text   # noqa: E402

RF = os.environ.get("NADI_ASR_WORKSPACE", "work/asr")
VAL = os.path.join(RF, "manifests", "nadi_val_eval.jsonl")
BASE_ID = "CohereLabs/cohere-transcribe-arabic-07-2026"


def read_rows(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def corpus_wer(pairs):
    """pairs: list of (ref, hyp) already produced; normalize then aggregate edits/words."""
    import jiwer
    refs = [normalize_text(r) for r, _ in pairs]
    hyps = [normalize_text(h) for _, h in pairs]
    keep = [(r, h) for r, h in zip(refs, hyps) if r.strip()]
    if not keep:
        return 0.0, 0
    r = [x[0] for x in keep]
    h = [x[1] for x in keep]
    return float(jiwer.wer(r, h)), len(keep)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", help="ckpt dir or base model id (base=%s)" % BASE_ID)
    ap.add_argument("--model-id", default=BASE_ID, help="processor source (not fine-tuned)")
    ap.add_argument("--val", default=VAL)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--shard-id", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-new-tokens", type=int, default=448)
    ap.add_argument("--no-repeat-ngram", type=int, default=6)
    ap.add_argument("--num-beams", type=int, default=1, help=">1 = beam search (do_sample stays off)")
    ap.add_argument("--length-penalty", type=float, default=None, help="beam length norm exponent")
    ap.add_argument("--min-new-tokens", type=int, default=None)
    ap.add_argument("--language", default="ar")
    ap.add_argument("--dialect-cond", "--dialect_cond", dest="dialect_cond", type=int, default=0,
                   help="1 = dialect conditioning at decode: group utts by their 'dataset' "
                        "dialect and inject the country name into the <|startofcontext|> slot "
                        "(must match how the checkpoint was trained). 0 = off (default).")
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    if a.merge:
        pairs, per_dialect = [], {}
        for fn in sorted(os.listdir(a.out_dir)):
            if not fn.startswith("hyp.shard_"):
                continue
            for line in open(os.path.join(a.out_dir, fn), encoding="utf-8"):
                d = json.loads(line)
                pairs.append((d["ref"], d["hyp"]))
                per_dialect.setdefault(d["dataset"], []).append((d["ref"], d["hyp"]))
        overall, n = corpus_wer(pairs)
        report = {"checkpoint": None, "overall_wer": overall, "n_scored": n, "per_dialect": {}}
        for dia in sorted(per_dialect):
            w, dn = corpus_wer(per_dialect[dia])
            report["per_dialect"][dia] = {"wer": w, "n": dn}
        with open(os.path.join(a.out_dir, "wer_report.json"), "w") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print("=== NADI val WER (nrn6) ===")
        print("OVERALL %.4f  (n=%d)" % (overall, n))
        for dia in sorted(report["per_dialect"]):
            v = report["per_dialect"][dia]
            print("  %-12s %.4f  (n=%d)" % (dia, v["wer"], v["n"]))
        return

    ckpt = a.checkpoint if a.checkpoint != "base" else BASE_ID
    from transformers import AutoProcessor, CohereAsrForConditionalGeneration
    print(f"[load] {ckpt}", flush=True)
    model = CohereAsrForConditionalGeneration.from_pretrained(ckpt, dtype=torch.bfloat16).to("cuda:0")
    model.eval()
    processor = AutoProcessor.from_pretrained(a.model_id)

    rows = read_rows(a.val)
    mine = [r for i, r in enumerate(rows) if i % a.num_shards == a.shard_id]
    missing = [r["audio"] for r in mine if not os.path.exists(r["audio"])]
    if missing:
        raise FileNotFoundError(f"{len(missing)} manifest audio files are missing; refusing a partial evaluation")
    if a.limit:
        mine = mine[: a.limit]
    gk = {}
    if a.no_repeat_ngram > 0:
        gk["no_repeat_ngram_size"] = a.no_repeat_ngram
    if a.num_beams > 1:
        gk["num_beams"] = a.num_beams
    if a.length_penalty is not None:
        gk["length_penalty"] = a.length_penalty
    if a.min_new_tokens is not None:
        gk["min_new_tokens"] = a.min_new_tokens
    gk = gk or None
    fn = os.path.join(a.out_dir, f"hyp.shard_{a.shard_id}")
    print(f"[infer] shard {a.shard_id+1}/{a.num_shards}: {len(mine)} utts", flush=True)
    # dialect conditioning: one prompt per generate call, so batches must be
    # dialect-homogeneous -> decode dialect by dialect (row order within a group, and the
    # single-group order when off, are unchanged; scoring is order-independent anyway).
    groups = [("", mine)]
    if a.dialect_cond:
        by = {}
        for r in mine:
            by.setdefault(dialect_of(r), []).append(r)
        groups = sorted(by.items())
        print("[dialect] groups: %s" % {d or "<none>": len(g) for d, g in groups}, flush=True)
    done = 0
    with open(fn, "w", encoding="utf-8") as out:
        for dia, grp in groups:
            for i in range(0, len(grp), a.batch_size):
                chunk = grp[i:i + a.batch_size]
                wavs = [load_audio(r["audio"]) for r in chunk]
                hyps = generate_transcripts(model, processor, wavs, a.language,
                                            a.max_new_tokens, batch_size=a.batch_size,
                                            gen_kwargs=gk, dialect=(dia or None))
                for r, h in zip(chunk, hyps):
                    out.write(json.dumps({"utt_id": r["utt_id"], "dataset": r.get("dataset", ""),
                                          "ref": r["refs"][0], "hyp": h}, ensure_ascii=False) + "\n")
                out.flush()
                done += len(chunk)
                if (i // a.batch_size) % 20 == 0:
                    print(f"  {done}/{len(mine)}", flush=True)
    print("[done shard]", flush=True)


if __name__ == "__main__":
    main()
