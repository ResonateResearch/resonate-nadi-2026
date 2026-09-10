#!/usr/bin/env python
# coding=utf-8
"""CTC forced alignment for the NADI2026 train clips.

Produces word-level timestamps by forced-aligning each clip against its KNOWN
reference transcript (not free recognition) with a character-level Arabic CTC
model (jonatasgrosman/wav2vec2-large-xlsr-53-arabic).

The Viterbi forced-alignment over the blank-interleaved label sequence is
implemented here from scratch in numpy (log space); torchaudio is not used.

Output: one json per utterance
    {"audio", "dialect", "text", "duration", "n_frames", "frame_stride",
     "time_offset", "words": [{"w", "start", "end", "aligned"}], "score"}
or on failure
    {"audio", "error"}

Usage (one visible CUDA device):
    CUDA_VISIBLE_DEVICES=0 python asr/preparation/align_ctc.py \
        --manifest manifests/nadi_original.jsonl \
        --out manifests/nadi_original.align.jsonl
"""
import argparse
import json
import os
import re
import sys
import time
import unicodedata
from pathlib import Path

# Device visibility, cache location, and offline mode are controlled by the caller.

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
import torch  # noqa: E402
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "train"))
from common import clean_target  # noqa: E402

MODEL_ID = "jonatasgrosman/wav2vec2-large-xlsr-53-arabic"
SR = 16000
NEG = -1e30

# Characters that are absent from the aligner vocab but are trivially
# substitutable for ALIGNMENT PURPOSES ONLY. The emitted `text` / `w` fields
# always carry the ORIGINAL untouched characters.
CHAR_MAP = {
    "ڨ": "ق",  # ڨ -> ق
    "ڤ": "ف",  # ڤ -> ف
    "ڢ": "ف",  # ڢ -> ف
    "پ": "ب",  # پ -> ب
    "ٱ": "ا",  # ٱ -> ا  (alef wasla)
}
# Arabic diacritics + tatweel: present in the vocab but essentially never
# emitted by the acoustic model (only ~800 occurrences in 713k chars of our
# transcripts). Dropping them from the alignment target avoids forcing the
# path through labels the model will not fire on.
DIACRITICS_RE = re.compile("[\\u064B-\\u065F\\u0670\\u0640]")

# Systematic latency correction, in seconds, added to every emitted timestamp.
# Two effects push the raw frame index earlier than the true acoustic event:
#   (a) frame t has receptive field samples [t*320, t*320+400), i.e. its centre
#       sits 400/2/16000 = 12.5 ms after t*20 ms;
#   (b) CTC is peaky and fires a character slightly before its acoustic centre.
# Calibrated empirically on a 200-clip held-in sample (25/dialect) by sweeping
# the offset that MINIMISES the energy at predicted inter-word boundaries:
# the curve is smooth and unimodal with a minimum at +0.030 s (see
# qc_alignment.py, which re-measures this on held-out data).
TIME_OFFSET = 0.030


# ----------------------------------------------------------------------------
# CTC forced alignment (Viterbi over blank-interleaved targets, log space)
# ----------------------------------------------------------------------------
def ctc_forced_align(logprob: np.ndarray, tokens: np.ndarray, blank: int = 0):
    """Viterbi forced alignment.

    logprob : (T, V) float32, log-softmax over the CTC vocabulary.
    tokens  : (L,) int64 target label ids (no blanks, no repeats collapsed).

    Returns (path, path_score) where path[t] is the index into the
    blank-interleaved state sequence  b y1 b y2 b ... yL b  (length S = 2L+1)
    and path_score is the mean per-frame log-prob along the best path.
    """
    T, _ = logprob.shape
    L = int(tokens.shape[0])
    if L == 0:
        raise ValueError("empty alignment target")
    S = 2 * L + 1
    # a CTC path may skip blank states, so the minimum feasible number of
    # frames is L plus one extra frame for every pair of adjacent equal labels
    # (which MUST be separated by a blank).
    min_T = L + int(np.sum(tokens[1:] == tokens[:-1])) if L > 1 else 1
    if T < min_T:
        raise ValueError(
            f"audio too short for transcript: T={T} frames < {min_T} required "
            f"({L} labels)")

    labels = np.full(S, blank, dtype=np.int64)
    labels[1::2] = tokens

    # (T, S) emission log-probs gathered onto the state sequence
    em = logprob[:, labels]

    # skip transition s-2 -> s is legal only into a non-blank state whose label
    # differs from the previous non-blank label
    skip = np.zeros(S, dtype=bool)
    if S > 2:
        skip[2:] = (labels[2:] != blank) & (labels[2:] != labels[:-2])

    alpha = np.full(S, NEG, dtype=np.float64)
    alpha[0] = em[0, 0]
    if S > 1:
        alpha[1] = em[0, 1]

    bp = np.zeros((T, S), dtype=np.int8)  # 0: stay, 1: from s-1, 2: from s-2
    ar = np.arange(S)
    for t in range(1, T):
        a1 = np.empty(S, dtype=np.float64)
        a1[0] = NEG
        a1[1:] = alpha[:-1]
        a2 = np.full(S, NEG, dtype=np.float64)
        if S > 2:
            a2[2:] = np.where(skip[2:], alpha[:-2], NEG)
        stack = np.stack((alpha, a1, a2))          # (3, S)
        idx = np.argmax(stack, axis=0)
        alpha = stack[idx, ar] + em[t]
        bp[t] = idx

    # a valid path ends either on the final blank or on the final label
    if S > 1 and alpha[S - 2] > alpha[S - 1]:
        s = S - 2
    else:
        s = S - 1

    path = np.zeros(T, dtype=np.int64)
    for t in range(T - 1, -1, -1):
        path[t] = s
        if t > 0:
            s = s - int(bp[t, s])
    if path[0] not in (0, 1):
        raise ValueError(f"backtrace did not reach a start state (got {path[0]})")

    score = float(em[np.arange(T), path].mean())
    return path, score


def char_spans_from_path(path: np.ndarray, n_labels: int):
    """Frame span [first, last+1) for each non-blank label index."""
    spans = np.zeros((n_labels, 2), dtype=np.int64)
    state = path  # odd states 2k+1 correspond to label k
    for k in range(n_labels):
        fr = np.nonzero(state == (2 * k + 1))[0]
        if fr.size == 0:
            raise ValueError(f"label {k} never occupied on the Viterbi path")
        spans[k, 0] = fr[0]
        spans[k, 1] = fr[-1] + 1
    return spans


# ----------------------------------------------------------------------------
# text -> alignment target
# ----------------------------------------------------------------------------
def build_target(text, vocab, delim="|"):
    """Split `text` on whitespace, map to alignable label ids.

    Returns (tokens, tok_char_slices) where
      tokens          : list[int] label ids (words joined by the delimiter id)
      tok_char_slices : list of (orig_token, start_idx_or_None, end_idx_or_None)
                        indices into `tokens` covering the token's own chars.
    """
    raw_tokens = text.split()
    tokens = []
    slices = []
    for tok in raw_tokens:
        chars = []
        for ch in tok:
            ch = DIACRITICS_RE.sub("", ch)
            if not ch:
                continue
            ch = CHAR_MAP.get(ch, ch)
            if unicodedata.category(ch).startswith("Z"):
                continue
            if ch in vocab and ch != delim:
                chars.append(vocab[ch])
        if not chars:
            slices.append((tok, None, None))
            continue
        if tokens:  # word delimiter between consecutive alignable words
            tokens.append(vocab[delim])
        a = len(tokens)
        tokens.extend(chars)
        slices.append((tok, a, len(tokens)))
    return tokens, slices


# ----------------------------------------------------------------------------
def feat_lengths(n_samples, kernels, strides):
    L = n_samples
    for k, s in zip(kernels, strides):
        L = (L - k) // s + 1
    return int(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--model", default=MODEL_ID)
    ap.add_argument("--time-offset", type=float, default=TIME_OFFSET,
                    help="seconds added to every word start/end to compensate "
                         "the CTC emission latency (see TIME_OFFSET); "
                         "pass 0 for raw frame times")
    ap.add_argument("--fp16", action="store_true", default=True)
    ap.add_argument("--fp32", dest="fp16", action="store_false")
    ap.add_argument("--resume", action="store_true",
                    help="skip audios already present in --out and append")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available; select a CUDA device using CUDA_VISIBLE_DEVICES")
    dev = torch.device("cuda:0")  # first device visible to this process
    print(f"[env] CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')} "
          f"-> {torch.cuda.get_device_name(0)}", flush=True)

    proc = Wav2Vec2Processor.from_pretrained(args.model)
    model = Wav2Vec2ForCTC.from_pretrained(
        args.model, dtype=torch.float16 if args.fp16 else torch.float32)
    model.to(dev).eval()
    cfg = model.config
    total_stride = int(np.prod(cfg.conv_stride))
    frame_stride = total_stride / SR
    vocab = proc.tokenizer.get_vocab()
    blank = proc.tokenizer.pad_token_id
    print(f"[model] conv_stride={list(cfg.conv_stride)} total={total_stride} "
          f"-> {frame_stride*1000:.1f} ms/frame ; blank id={blank} ; "
          f"vocab={len(vocab)} ; time_offset={args.time_offset:+.3f}s",
          flush=True)

    rows = []
    with open(args.manifest, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if args.limit:
        rows = rows[: args.limit]

    done = set()
    mode = "w"
    if args.resume and os.path.exists(args.out):
        with open(args.out, encoding="utf-8") as f:
            for line in f:
                try:
                    done.add(json.loads(line)["audio"])
                except Exception:
                    pass
        mode = "a"
        print(f"[resume] {len(done)} utterances already done", flush=True)
    rows = [r for r in rows if r["audio"] not in done]
    print(f"[data] {len(rows)} utterances to align", flush=True)

    # pre-read durations so we can bucket by length (big padding win)
    metas = []
    for r in rows:
        try:
            info = sf.info(r["audio"])
            metas.append((info.frames, r))
        except Exception as e:
            metas.append((-1, r))
            print(f"[warn] cannot stat {r['audio']}: {e}", flush=True)
    metas.sort(key=lambda x: x[0])

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    fout = open(args.out, mode, encoding="utf-8")
    n_ok = n_err = 0
    t0 = time.time()

    for bstart in range(0, len(metas), args.batch_size):
        batch = metas[bstart: bstart + args.batch_size]
        waves, keep = [], []
        for nframes, r in batch:
            try:
                wav, sr = sf.read(r["audio"], dtype="float32", always_2d=False)
                if wav.ndim > 1:
                    wav = wav.mean(axis=1)
                if sr != SR:
                    raise ValueError(f"unexpected sample rate {sr}")
                if wav.size < 1000:
                    raise ValueError(f"audio too short: {wav.size} samples")
                waves.append(wav)
                keep.append(r)
            except Exception as e:
                fout.write(json.dumps({"audio": r["audio"],
                                       "error": f"load: {type(e).__name__}: {e}"},
                                      ensure_ascii=False) + "\n")
                n_err += 1
        if not waves:
            continue

        try:
            inputs = proc(waves, sampling_rate=SR, return_tensors="pt",
                          padding=True, return_attention_mask=True)
            iv = inputs.input_values.to(dev, dtype=model.dtype)
            am = inputs.attention_mask.to(dev)
            with torch.inference_mode():
                logits = model(iv, attention_mask=am).logits.float()
                logprobs = torch.log_softmax(logits, dim=-1).cpu().numpy()
        except Exception as e:
            for r in keep:
                fout.write(json.dumps({"audio": r["audio"],
                                       "error": f"forward: {type(e).__name__}: {e}"},
                                      ensure_ascii=False) + "\n")
                n_err += 1
            fout.flush()
            torch.cuda.empty_cache()
            continue

        for i, r in enumerate(keep):
            try:
                n_s = waves[i].shape[0]
                T = min(feat_lengths(n_s, cfg.conv_kernel, cfg.conv_stride),
                        logprobs.shape[1])
                lp = np.ascontiguousarray(logprobs[i, :T].astype(np.float64))
                duration = n_s / SR

                text = clean_target(r["text"])
                tokens, slices = build_target(text, vocab)
                if not tokens:
                    raise ValueError("no alignable characters in transcript")
                path, score = ctc_forced_align(lp, np.asarray(tokens, dtype=np.int64),
                                               blank=blank)
                cspans = char_spans_from_path(path, len(tokens))

                words = []
                prev_end = 0.0
                for tok, a, b in slices:
                    if a is None:
                        words.append({"w": tok, "start": round(prev_end, 4),
                                      "end": round(prev_end, 4), "aligned": False})
                        continue
                    f0 = int(cspans[a:b, 0].min())
                    f1 = int(cspans[a:b, 1].max())
                    st = min(max(f0 * frame_stride + args.time_offset, 0.0), duration)
                    en = min(max(f1 * frame_stride + args.time_offset, st), duration)
                    words.append({"w": tok, "start": round(st, 4),
                                  "end": round(en, 4), "aligned": True})
                    prev_end = en

                dialect = os.path.basename(os.path.dirname(r["audio"]))
                fout.write(json.dumps({
                    "audio": r["audio"], "dialect": dialect, "text": text,
                    "duration": round(duration, 4), "n_frames": int(T),
                    "frame_stride": frame_stride, "time_offset": args.time_offset,
                    "words": words,
                    "score": round(score, 5),
                }, ensure_ascii=False) + "\n")
                n_ok += 1
            except Exception as e:
                fout.write(json.dumps({"audio": r["audio"],
                                       "error": f"align: {type(e).__name__}: {e}"},
                                      ensure_ascii=False) + "\n")
                n_err += 1

        fout.flush()
        if (bstart // args.batch_size) % 25 == 0:
            el = time.time() - t0
            n = n_ok + n_err
            print(f"[prog] {n}/{len(metas)}  ok={n_ok} err={n_err}  "
                  f"{el:.0f}s  {n/max(el,1e-9):.1f} utt/s", flush=True)

    fout.close()
    el = time.time() - t0
    print(f"[done] ok={n_ok} err={n_err} in {el:.0f}s -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
