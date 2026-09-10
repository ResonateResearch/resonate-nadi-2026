# coding=utf-8
"""Shared ASR training-harness utilities.

This retains the original training-harness normalization. It is different
from scripts/local_score.py and is not an organizer scorer. Source-port
provenance and license status are described in the root THIRD_PARTY.md.
"""
import re
from typing import Dict, List, Tuple

import jiwer

# --- self-contained training-harness normalization ---------------------------
_TAG_RE = re.compile(r"<\w+>")
_PUNC_RE = re.compile(r"[,.،;؛@#؟?!&$ـ_\[\]\(\)]+\ *")


def normalize_text(text: str, is_asmo: bool = False, remove_punc: bool = True) -> str:
    """Remove markup, fold selected letters, optionally remove punctuation, and
    collapse spaces using the original training-harness rules.

    The Arabic branch folds ى/ة/إ/أ/آ to ي/ه/ا/ا/ا, lowercases text, and replaces
    سين/جيم with س/ج. The optional ASMO branch uses the character substitutions
    below instead. All operations are implemented here; no external normalizer
    file is required. This differs from the local approximation in local_score.py.
    """
    text = _TAG_RE.sub("", text).replace("< ⁇ n ⁇ >", "")
    text = text.replace("<", " <").replace(">", "> ")
    text = " ".join(
        i for i in text.replace("▁", " ").split(" ")
        if not (i.startswith("<") or i.endswith(">"))
    )
    if is_asmo:
        text = (text.replace("i", "j").replace("I", "g").replace("E", "G")
                    .replace("B", "G").replace("C", "G"))
    else:
        text = (text.replace("ى", "ي").replace("ة", "ه").replace("إ", "ا")
                    .replace("أ", "ا").replace("آ", "ا").lower())
        text = text.replace("سين", "س").replace("جيم", "ج")
    if remove_punc:
        text = _PUNC_RE.sub(" ", text)
    return " ".join(text.split())


def compute_wer(ref: str, hyp: str, **kw) -> float:
    """Single-reference WER on normalized text (proxy used inline during training)."""
    r = normalize_text(ref, **kw)
    h = normalize_text(hyp, **kw)
    if not r.strip():
        return 0.0 if not h.strip() else 1.0
    return float(jiwer.wer(r, h))


def compute_mr_wer(refs: List[str], hyp: str, **kw) -> float:
    """Multi-reference WER: min WER over references (matches MR-WER spirit)."""
    vals = [compute_wer(r, hyp, **kw) for r in refs if r is not None]
    return min(vals) if vals else 1.0


def mr_wer_counts(refs: List[str], hyp: str, **kw) -> Tuple[int, int]:
    """Sibling of compute_mr_wer for CORPUS-level aggregation (does NOT change it).

    Returns (n_edits, n_ref_words) for the SAME reference compute_mr_wer would pick:
    the one with the lowest utterance WER (ties -> first). Summing these over a group and
    dividing gives corpus WER = sum(edits) / sum(ref_words), which is what the NADI
    country_av_wer is built from. Blank/None references contribute (0, 0) and are dropped,
    matching the training-harness aggregation in scripts/nadi_val_eval.py:corpus_wer.
    """
    hn = normalize_text(hyp, **kw)
    best = None                       # (rate, edits, n_ref_words)
    for r in refs or []:
        if r is None:
            continue
        rn = normalize_text(r, **kw)
        if not rn.strip():
            continue
        out = jiwer.process_words(rn, hn)
        edits = out.substitutions + out.deletions + out.insertions
        nref = out.substitutions + out.deletions + out.hits
        rate = edits / nref if nref else 0.0
        if best is None or rate < best[0]:
            best = (rate, edits, nref)
    return (0, 0) if best is None else (best[1], best[2])


# --- Kaldi-style IO (wav.scp / text_ar are "<id> <value>" per line) -----------
def read_kaldi(path: str) -> Dict[str, str]:
    d: Dict[str, str] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            parts = line.split(" ", 1)
            d[parts[0]] = parts[1].strip() if len(parts) > 1 else ""
    return d


def clean_target(text: str) -> str:
    """Extract the transcript from 'language Arabic<asr_text>TRANSCRIPT'."""
    return text.split("<asr_text>", 1)[1] if "<asr_text>" in text else text


# --- dialect conditioning (opt-in, --dialect_cond) ----------------------------
NADI_DIALECTS = ("Algeria", "Egypt", "Jordan", "Mauritania",
                 "Morocco", "Palestine", "UAE", "Yemen")
_DIALECT_BY_LOWER = {d.lower(): d for d in NADI_DIALECTS}


def dialect_from_path(path: str) -> str:
    """Canonical NADI dialect parsed out of an audio path, or "" if there is none.

    Depth-agnostic: scans every component after the last '/audio/' segment, so it handles
        <bucket>/<Dialect>/<id>.flac              (original, clean-neural, E11_vc_lp)
        <bucket>/<Dialect>/<augtype>/<n>.flac     (E11_aug)
    and any future layout, and returns "" for non-NADI corpora (k2hub, CS, ...).
    """
    parts = str(path).split("/")
    if "audio" in parts:
        parts = parts[len(parts) - parts[::-1].index("audio"):]
    for seg in parts:
        d = _DIALECT_BY_LOWER.get(seg.lower())
        if d:
            return d
    return ""


def dialect_of(ex) -> str:
    """Canonical dialect of a manifest row: explicit 'dialect'/'dataset' field first
    (validated against the 8 NADI countries -- so dev-set markers like 'ar'/'cs' fall
    through), robust audio-path parse second, "" if unknown (-> no cue)."""
    for k in ("dialect", "dataset"):
        v = ex.get(k)
        if v:
            d = _DIALECT_BY_LOWER.get(str(v).strip().lower())
            if d:
                return d
    return dialect_from_path(ex.get("audio", "") or "")
