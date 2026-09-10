# coding=utf-8
"""Data collator for Cohere ASR fine-tuning, including the July Arabic checkpoint.

Cohere-transcribe is a classic ENCODER-DECODER seq2seq ASR model (Parakeet/FastConformer
encoder -> small cross-attention transformer decoder). Audio conditions the decoder through
cross-attention (NOT masked_scatter of audio embeddings, as in Voxtral).

The decoder prompt is the fixed task/language prefix from
`processor.get_decoder_prompt_ids(language, punctuation)`:
    ▁ <|startofcontext|> <|startoftranscript|> <|emo:undefined|> <|ar|> <|ar|>
      <|pnc|> <|noitn|> <|notimestamp|> <|nodiarize|>            (10 tokens for language='ar')

The model's loss_function shifts internally (causal-LM style), so we build a SINGLE decoder
stream and mask the whole prompt — identical in structure to the Voxtral collator:

    decoder_input_ids = prompt + target + [eos]           # eos id = 3 (<|endoftext|>)
    labels            = [-100]*len(prompt) + target + [eos]   # same length; loss shifts

The encoder is fed `input_features` + `attention_mask` from the feature extractor. Audio is
capped at 30 s so the feature extractor never energy-splits it (its fast path is <= 30 s),
guaranteeing exactly one feature row per example (asserted).
"""
from dataclasses import dataclass
from typing import Any, Dict, List

import librosa
import numpy as np
import soundfile as sf
import torch

MAX_AUDIO_SECONDS = 30.0        # feature-extractor fast path is <= 30 s -> single chunk, no split
MAX_TOTAL_TOKENS = 1000         # prompt + target + eos must stay < decoder max_seq_len (1024)
SOC_TOKEN = "<|startofcontext|>"   # contextual-biasing slot of the decoder prompt


def build_decoder_prompt(processor, language: str = "ar", dialect=None) -> List[int]:
    """Decoder prompt ids, optionally with a DIALECT CUE injected into the
    <|startofcontext|> contextual-biasing region:

        base  : ▁ <|startofcontext|> <|startoftranscript|> <|emo:undefined|> <|ar|> ...
        cued  : ▁ <|startofcontext|> ▁Al ger ia <|startoftranscript|> <|emo:undefined|> ...

    dialect None/"" -> the stock 10-token prompt, byte-identical to
    processor.get_decoder_prompt_ids(language, punctuation=True).

    The cue is the plain country NAME (pretrained sub-words), not a spare token: its
    embedding already carries semantics and it still works when embed_tokens is frozen
    (stage1 / stage2 / warmup).  Consequence: the cue is 1-4 tokens depending on the
    country (Egypt=1, Yemen=2, Algeria/Jordan/Morocco/Palestine/UAE=3, Mauritania=4),
    so the prompt length is NOT constant -- every caller must take
    plen = len(<the list this function returned>) and never assume 10.

    The cue lives in the PROMPT only; the collator masks it with -100 so the model is
    never trained to emit it.
    """
    prompt = list(processor.get_decoder_prompt_ids(language=language, punctuation=True))
    if not dialect:
        return prompt
    tok = processor.tokenizer
    cue = tok(str(dialect).strip(), add_special_tokens=False)["input_ids"]
    cue = cue[0] if (cue and isinstance(cue[0], list)) else cue
    assert cue, f"empty dialect cue for {dialect!r}"
    soc = tok.convert_tokens_to_ids(SOC_TOKEN)
    i = prompt.index(soc) + 1 if soc in prompt else 1
    return prompt[:i] + list(cue) + prompt[i:]


def load_audio(path: str, sr: int = 16000) -> np.ndarray:
    """Fast decode + resample. soundfile native read (float32) + soxr resample is ~26x faster
    than librosa.load's float64/audioread path (81 ms -> 3 ms/audio on the 8 kHz wavs), which
    is what starves the GPUs. Falls back to librosa for formats soundfile can't decode."""
    try:
        wav, nsr = sf.read(path, dtype="float32", always_2d=False)
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
    except Exception:
        wav, nsr = librosa.load(path, sr=None, mono=True)
        wav = wav.astype(np.float32)
    if nsr != sr:
        wav = librosa.resample(wav, orig_sr=nsr, target_sr=sr, res_type="soxr_hq")
    if wav.shape[0] > int(MAX_AUDIO_SECONDS * sr):
        wav = wav[: int(MAX_AUDIO_SECONDS * sr)]
    return np.ascontiguousarray(wav, dtype=np.float32)


def _spec_augment(feats: torch.Tensor, attn: torch.Tensor, n_freq: int, freq_param: int,
                  n_time: int, time_param: int, live_bins: int = 0,
                  time_ratio: float = 0.0) -> torch.Tensor:
    """In-place SpecAugment on log-mel features [B, T, F] (time-major; F=128 mel bins).
    Masks set to per-utt mean. Time masks respect each utt's valid length from attn.
    live_bins > 0 restricts frequency masks to the first live_bins mel bins — for band-limited
    audio whose upper bins carry no energy, masks landing there regularize nothing.
    time_ratio > 0 additionally caps each time mask at time_ratio * valid frames. The absolute
    time_mask_param (100 frames = 1.00 s) is longer than a sub-1s utterance, so without a ratio
    cap a short clip is masked ~79% on average and blanked ENTIRELY ~32% of the time while its
    full transcript is still supervised — i.e. trained to emit words from a constant fill.
    Default 0.0 keeps the legacy absolute-only behaviour bit-identical."""
    B, T, F = feats.shape
    f_top = live_bins if 0 < live_bins < F else F
    for b in range(B):
        fill = feats[b].mean()
        for _ in range(n_freq):                          # frequency masks on the F (last) axis
            f = int(torch.randint(0, freq_param + 1, (1,)).item())
            if 0 < f < f_top:
                f0 = int(torch.randint(0, f_top - f + 1, (1,)).item())
                feats[b, :, f0:f0 + f] = fill
        valid = int(attn[b].sum().item()) if attn is not None else T
        valid = max(1, min(valid, T))
        t_cap = time_param
        if time_ratio > 0:
            t_cap = max(1, min(time_param, int(time_ratio * valid)))
        for _ in range(n_time):                          # time masks on the T (middle) axis
            t = min(int(torch.randint(0, t_cap + 1, (1,)).item()), valid)
            if t > 0:
                t0 = int(torch.randint(0, valid - t + 1, (1,)).item())
                feats[b, t0:t0 + t, :] = fill
    return feats


@dataclass
class DataCollatorForCohereASR:
    processor: Any
    language: str = "ar"
    sampling_rate: int = 16000
    specaug: bool = False           # always-on SpecAugment during training (feature-domain)
    n_freq_masks: int = 2
    freq_mask_param: int = 27
    n_time_masks: int = 2
    time_mask_param: int = 100
    time_mask_ratio: float = 0.0    # >0: also cap each time mask at ratio*valid frames (short-clip safety)
    specaug_live_bins: int = 0      # >0: freq masks restricted to first N mel bins (band-limited audio)
    dialect_cond: bool = False      # True: per-sample dialect cue in the <|startofcontext|> slot

    def __post_init__(self):
        self.tok = self.processor.tokenizer
        self.fe = self.processor.feature_extractor
        self.prompt = list(self.processor.get_decoder_prompt_ids(language=self.language, punctuation=True))
        self.eos = self.tok.eos_token_id
        pid = self.tok.pad_token_id
        self.pad_id = pid if pid is not None else self.eos
        self.tgt_cap = MAX_TOTAL_TOKENS - len(self.prompt) - 1
        self._prompt_cache = {"": self.prompt}     # dialect -> prompt ids (cue tokenized once)

    def _prompt_for(self, dialect) -> List[int]:
        """Per-sample decoder prompt. Feature off, or row without a dialect -> the shared
        base prompt (identical object), so mixed manifests (NADI + anchor Arabic/CS) work."""
        if not self.dialect_cond or not dialect:
            return self.prompt
        key = str(dialect)
        p = self._prompt_cache.get(key)
        if p is None:
            p = build_decoder_prompt(self.processor, self.language, key)
            self._prompt_cache[key] = p
        return p

    def _encode_target(self, text: str, cap: int = None) -> List[int]:
        ids = self.tok(text, add_special_tokens=False)["input_ids"]
        ids = ids[0] if (ids and isinstance(ids[0], list)) else ids
        return ids[: self.tgt_cap if cap is None else cap]

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        audios = [load_audio(f["audio"], self.sampling_rate) for f in features]
        targets = [f["target"] for f in features]

        feats = self.fe(audios, sampling_rate=self.sampling_rate,
                        return_tensors="pt", return_attention_mask=True)
        input_features = feats["input_features"]
        enc_attn = feats["attention_mask"]
        # guard: audio <= 30 s must yield exactly one feature row per example (no energy-split)
        assert input_features.shape[0] == len(features), \
            f"feature rows {input_features.shape[0]} != batch {len(features)} (long audio got chunked)"

        if self.specaug:
            input_features = _spec_augment(
                input_features, enc_attn, self.n_freq_masks, self.freq_mask_param,
                self.n_time_masks, self.time_mask_param, self.specaug_live_bins,
                self.time_mask_ratio)

        # PER-SAMPLE prompt: with --dialect_cond the cue length differs per country, so the
        # label mask must use THIS row's prompt length (a shared len(p) would leak cue tokens
        # into the loss / mask away real target tokens). Feature off -> every p is self.prompt.
        prompts = [self._prompt_for(f.get("dialect")) for f in features]
        tgt_ids = [self._encode_target(t, MAX_TOTAL_TOKENS - len(p) - 1)
                   for t, p in zip(targets, prompts)]
        fulls = [p + t + [self.eos] for p, t in zip(prompts, tgt_ids)]
        maxlen = max(len(f) for f in fulls)

        dec_in, labels, dec_attn = [], [], []
        for p, t, f in zip(prompts, tgt_ids, fulls):
            pad = maxlen - len(f)
            dec_in.append(f + [self.pad_id] * pad)
            labels.append([-100] * len(p) + t + [self.eos] + [-100] * pad)
            dec_attn.append([1] * len(f) + [0] * pad)

        return {
            "input_features": input_features,
            "attention_mask": enc_attn,
            "decoder_input_ids": torch.tensor(dec_in, dtype=torch.long),
            "decoder_attention_mask": torch.tensor(dec_attn, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }
