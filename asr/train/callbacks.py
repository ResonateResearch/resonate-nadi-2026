# coding=utf-8
"""Training callbacks for Cohere ASR; see THIRD_PARTY.md for port provenance.

- MakeCheckpointInferableCallback : every checkpoint gets the full processor (feature
  extractor + tokenizer) + a post-save assertion.
- DevWERCallback                  : inline multi-ref dev-WER on a small domain-rep subset
  (overall / per-domain) every eval; exposes .last_dev_wer for retention.
- TopKCheckpointCallback          : keep only the K lowest-dev-WER checkpoints on disk
  (+ always the latest, for resume); prune the rest.
- WandbSampleInferenceCallback    : base vs current predictions on fixed anchors + random
  utts, logged to a wandb.Table.

Cohere-transcribe is encoder-decoder: generation runs the encoder on `input_features`, then
decodes from the fixed `decoder_input_ids` prompt. Special tokens (`<|ar|>`, `<|pnc|>`, ...)
are stripped by `skip_special_tokens=True` (no `lang:xx` hack needed, unlike Voxtral).
All inference/IO runs on the main process only.
"""
import json
import os
import random
import shutil

import torch
from transformers import TrainerCallback

from common import compute_mr_wer, compute_wer, dialect_of, mr_wer_counts
from data_collator import build_decoder_prompt, load_audio

_CKPT = "checkpoint-{}"


def _is_main(args) -> bool:
    return getattr(args, "process_index", 0) == 0


def _unwrap(trainer):
    return trainer.accelerator.unwrap_model(trainer.model)


@torch.no_grad()
def generate_transcripts(model, processor, wavs, language="ar",
                         max_new_tokens=448, batch_size=16, gen_kwargs=None, dialect=None):
    """Greedy batched decode of waveforms via the encoder-decoder + language prompt.
    gen_kwargs: extra model.generate() args (e.g. no_repeat_ngram_size). Waveforms must be
    <= 30 s (feature extractor stays single-chunk); callers use load_audio which enforces it.

    dialect: optional country name injected into the <|startofcontext|> slot (dialect
    conditioning). ONE dialect per call -- the prompt is built once and shared by the whole
    call, so callers must group utterances by dialect (NADI test data is already delivered
    per country). CRITICAL: plen is taken from the SAME list that becomes decoder_input_ids,
    because the cue is 1-4 tokens long; slicing gen[plen:] with a stale plen=10 silently
    corrupts every hypothesis (drops or keeps leading real tokens)."""
    device = next(model.parameters()).device
    mdtype = next(model.parameters()).dtype
    was_training = model.training
    prev_cache = getattr(model.config, "use_cache", False)
    gc_was_on = bool(getattr(model, "is_gradient_checkpointing", False))
    model.eval()
    if gc_was_on:
        model.gradient_checkpointing_disable()      # let KV-cache work -> fast greedy decode
    model.config.use_cache = True
    tok = processor.tokenizer
    fe = processor.feature_extractor
    prompt = build_decoder_prompt(processor, language=language, dialect=dialect)
    plen = len(prompt)      # MUST come from `prompt` itself: it is 10 + len(cue), not 10
    outs = []
    try:
        for i in range(0, len(wavs), batch_size):
            batch = wavs[i:i + batch_size]
            feats = fe(batch, sampling_rate=16000, return_tensors="pt", return_attention_mask=True)
            assert feats["input_features"].shape[0] == len(batch), "long audio got chunked in generate"
            di = torch.tensor([prompt] * len(batch), dtype=torch.long, device=device)
            gen = model.generate(
                input_features=feats["input_features"].to(device).to(mdtype),
                attention_mask=feats["attention_mask"].to(device),
                decoder_input_ids=di,
                max_new_tokens=max_new_tokens, do_sample=False, **(gen_kwargs or {}))
            for row in gen:
                outs.append(tok.decode(row[plen:], skip_special_tokens=True).strip())
    finally:
        model.config.use_cache = prev_cache
        if gc_was_on:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        if was_training:
            model.train()
    return outs


def _generate_rows(model, processor, rows, wavs, language="ar", max_new_tokens=448,
                   gen_kwargs=None, dialect_cond=False):
    """generate_transcripts over parallel (rows, wavs). With dialect_cond, rows are grouped
    by their canonical dialect (common.dialect_of: 'dialect'/'dataset' field, else path;
    non-NADI rows -> "" -> no cue) so each generate call is dialect-homogeneous, then the
    hypotheses are scattered back into the original row order.
    dialect_cond False -> a single pass, identical to calling generate_transcripts directly."""
    if not dialect_cond:
        return generate_transcripts(model, processor, wavs, language, max_new_tokens,
                                    gen_kwargs=gen_kwargs)
    groups = {}
    for i, r in enumerate(rows):
        groups.setdefault(dialect_of(r), []).append(i)
    outs = [""] * len(rows)
    for dia, idx in groups.items():
        hyps = generate_transcripts(model, processor, [wavs[i] for i in idx], language,
                                    max_new_tokens, gen_kwargs=gen_kwargs, dialect=dia or None)
        for i, h in zip(idx, hyps):
            outs[i] = h
    return outs


def _load_dev(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _dialect_key(row):
    """Grouping key for the dialect-macro corpus WER.

    Prefers a VALIDATED NADI dialect (dialect_of checks the value against the 8 countries
    and falls back to an audio-path parse), so dev-set markers like 'ar'/'cs' cannot
    masquerade as dialects and silently dilute the competition metric. Returns "" for a
    row that is not one of the 8 NADI dialects; the caller skips those rows, and if NO row
    resolves the caller degenerates to a single micro bucket rather than reporting a bogus
    macro over non-dialect groups.
    """
    d = dialect_of(row)
    if d:
        return d
    for k in ("dataset", "domain"):
        v = row.get(k)
        if isinstance(v, str) and v.strip():
            return ""      # a real but non-NADI marker (e.g. 'ar'/'cs') -> excluded
    return ""


# ----------------------------------------------------------------------------- inferable
class MakeCheckpointInferableCallback(TrainerCallback):
    def __init__(self, processor):
        self.processor = processor

    def on_save(self, args, state, control, **kwargs):
        if not _is_main(args):
            return control
        ckpt = os.path.join(args.output_dir, _CKPT.format(state.global_step))
        if os.path.isdir(ckpt):
            self.processor.save_pretrained(ckpt)
            from transformers import AutoProcessor
            p = AutoProcessor.from_pretrained(ckpt)
            assert getattr(p, "feature_extractor", None) is not None, \
                f"checkpoint {ckpt} lost its audio feature_extractor!"
        return control


# ----------------------------------------------------------------------------- dev-WER
class DevWERCallback(TrainerCallback):
    """Fires on a step schedule (NOT the HF eval loop) so retention has a fresh dev-WER at
    every save step."""
    def __init__(self, processor, dev_path, language="ar", every=1000, max_new_tokens=448, nrn6=0,
                 dialect_cond=False):
        self.processor = processor
        self.rows = _load_dev(dev_path)
        self.language = language
        self.dialect_cond = dialect_cond
        self.every = every
        self.max_new_tokens = max_new_tokens
        self.gen_kwargs = {"no_repeat_ngram_size": nrn6} if nrn6 and nrn6 > 0 else None
        self.last_dev_wer = None            # overall (micro-avg of per-utt WER)
        self.last_dev_wer_balanced = None   # macro over domains (upweights the smaller CS set)
        # NADI country_av_wer mirror: macro over `dataset` (dialect) of per-dialect CORPUS WER
        self.last_dev_wer_macro_dialect = None
        self.last_dev_wer_per_dialect = {}  # {dialect: corpus WER}
        self.trainer = None

    def attach(self, trainer):
        self.trainer = trainer

    def _run(self, state):
        model = _unwrap(self.trainer)
        wavs = [load_audio(r["audio"]) for r in self.rows]
        hyps = _generate_rows(model, self.processor, self.rows, wavs, self.language,
                              self.max_new_tokens, gen_kwargs=self.gen_kwargs,
                              dialect_cond=self.dialect_cond)
        per_domain, all_w = {}, []
        per_dialect = {}          # dialect -> [sum_edits, sum_ref_words] (CORPUS WER accumulator)
        for r, h in zip(self.rows, hyps):
            w = compute_mr_wer(r["refs"], h)
            all_w.append(w)
            per_domain.setdefault(r.get("domain", "all"), []).append(w)
            e, nw = mr_wer_counts(r.get("refs", []), h)
            if nw > 0:
                key = _dialect_key(r)
                acc = per_dialect.setdefault(key or "_nondialect", [0, 0])
                acc[0] += e
                acc[1] += nw
        overall = sum(all_w) / max(len(all_w), 1)
        dom_means = {d: sum(ws) / len(ws) for d, ws in per_domain.items()}
        self.last_dev_wer = overall
        self.last_dev_wer_balanced = sum(dom_means.values()) / max(len(dom_means), 1)
        # competition metric mirror: per-dialect corpus WER, then UNWEIGHTED mean over dialects
        # Only the 8 real NADI dialects count toward the competition-mirror macro. Rows that
        # are not a NADI dialect ('ar'/'cs' markers, other experiments' dev sets) land in
        # "_nondialect" and are excluded; if NOTHING resolved, fall back to that single
        # bucket so the metric is a plain micro corpus WER instead of silently bogus.
        dia_wer = {d: v[0] / v[1] for d, v in per_dialect.items()
                   if v[1] > 0 and d != "_nondialect"}
        if not dia_wer:
            dia_wer = {d: v[0] / v[1] for d, v in per_dialect.items() if v[1] > 0}
        self.last_dev_wer_per_dialect = dia_wer
        self.last_dev_wer_macro_dialect = (
            sum(dia_wer.values()) / len(dia_wer) if dia_wer else None)
        logs = {"dev_wer": overall, "dev_wer_balanced": self.last_dev_wer_balanced}
        if self.last_dev_wer_macro_dialect is not None:
            logs["dev_wer_macro_dialect"] = self.last_dev_wer_macro_dialect
        for dom, m in dom_means.items():
            logs[f"dev_wer_{dom}"] = m
        for dia in sorted(dia_wer):
            logs[f"dev_cwer_{dia}"] = dia_wer[dia]   # per-dialect CORPUS WER (weak-dialect watch)
        try:
            import wandb
            if wandb.run is not None:
                wandb.log(logs, step=state.global_step)
        except Exception:
            pass
        print(f"[dev-WER @ {state.global_step}] " +
              " ".join(f"{k}={v:.4f}" for k, v in logs.items()))

    def on_step_end(self, args, state, control, **kwargs):
        if not _is_main(args) or self.trainer is None:
            return control
        if state.global_step > 0 and state.global_step % self.every == 0:
            self._run(state)
        return control


# ----------------------------------------------------------------------------- top-K retention
class TopKCheckpointCallback(TrainerCallback):
    def __init__(self, k, metric_fn):
        self.k = k
        self.metric_fn = metric_fn
        self.scores = {}
        self._loaded = False

    def _scores_path(self, args):
        return os.path.join(args.output_dir, "topk_scores.json")

    def _restore(self, args):
        self._loaded = True
        try:
            with open(self._scores_path(args), encoding="utf-8") as f:
                saved = {int(s): float(w) for s, w in json.load(f).items()}
        except (OSError, ValueError):
            return
        self.scores.update({s: w for s, w in saved.items()
                            if os.path.isdir(os.path.join(args.output_dir, _CKPT.format(s)))})
        if self.scores:
            print(f"[topk] restored {len(self.scores)} checkpoint scores")

    def on_save(self, args, state, control, **kwargs):
        if not _is_main(args):
            return control
        if not self._loaded:
            self._restore(args)
        step = state.global_step
        score = self.metric_fn()
        self.scores[step] = score if score is not None else float("inf")

        ranked = sorted(self.scores.items(), key=lambda kv: kv[1])
        keep = {s for s, _ in ranked[: self.k]}
        keep.add(step)
        for s in list(self.scores.keys()):
            if s in keep:
                continue
            d = os.path.join(args.output_dir, _CKPT.format(s))
            if os.path.isdir(d):
                shutil.rmtree(d, ignore_errors=True)
                print(f"[topk] pruned {d} (dev_wer={self.scores[s]:.4f})")
            self.scores.pop(s, None)
        with open(self._scores_path(args), "w", encoding="utf-8") as f:
            json.dump(self.scores, f, indent=1)
        return control


# ----------------------------------------------------------------------------- live W&B samples
class WandbSampleInferenceCallback(TrainerCallback):
    def __init__(self, processor, dev_path, language="ar", every=250,
                 n_anchor=8, n_random=4, max_new_tokens=448, dialect_cond=False):
        self.processor = processor
        self.rows = _load_dev(dev_path)
        self.language = language
        self.dialect_cond = dialect_cond
        self.every = every
        self.n_anchor = min(n_anchor, len(self.rows))
        self.n_random = n_random
        self.max_new_tokens = max_new_tokens
        self.anchors = self.rows[: self.n_anchor]
        self.pool = self.rows[self.n_anchor:]
        self.base_cache = {}
        self._cache_path = None
        self.trainer = None

    def attach(self, trainer):
        self.trainer = trainer

    def _uid(self, r, i):
        return r.get("utt_id") or r.get("audio") or f"{r.get('dataset','d')}-{i}"

    def _base_preds(self, model, rows):
        todo = [(i, r) for i, r in enumerate(rows) if self._uid(r, i) not in self.base_cache]
        if todo:
            wavs = [load_audio(r["audio"]) for _, r in todo]
            preds = _generate_rows(model, self.processor, [r for _, r in todo], wavs,
                                   self.language, self.max_new_tokens,
                                   dialect_cond=self.dialect_cond)
            for (i, r), p in zip(todo, preds):
                self.base_cache[self._uid(r, i)] = p

    def _save_cache(self):
        if self._cache_path:
            with open(self._cache_path, "w", encoding="utf-8") as f:
                json.dump(self.base_cache, f, ensure_ascii=False, indent=1)

    def on_train_begin(self, args, state, control, **kwargs):
        if not _is_main(args) or self.trainer is None:
            return control
        self._cache_path = os.path.join(args.output_dir, "sample_base_cache.json")
        if os.path.exists(self._cache_path):
            with open(self._cache_path, encoding="utf-8") as f:
                self.base_cache.update(json.load(f))
            print(f"[samples] loaded {len(self.base_cache)} base predictions from cache")
        if state.global_step == 0:
            self._base_preds(_unwrap(self.trainer), self.rows)
            self._save_cache()
        elif not self.base_cache:
            print("[samples] resumed without base cache: base_pred column will be blank")
        return control

    def on_step_end(self, args, state, control, **kwargs):
        if not _is_main(args) or self.trainer is None:
            return control
        step = state.global_step
        if step == 0 or step % self.every != 0:
            return control
        try:
            import wandb
            if wandb.run is None:
                return control
        except Exception:
            return control

        model = _unwrap(self.trainer)
        rnd = random.sample(self.pool, min(self.n_random, len(self.pool))) if self.pool else []
        rows = self.anchors + rnd
        wavs = [load_audio(r["audio"]) for r in rows]
        cur = _generate_rows(model, self.processor, rows, wavs, self.language,
                             self.max_new_tokens, dialect_cond=self.dialect_cond)
        cols = ["step", "utt_id", "dataset", "audio", "reference",
                "base_pred", "current_pred", "base_wer", "current_wer"]
        table = wandb.Table(columns=cols)
        for i, (r, h, wav) in enumerate(zip(rows, cur, wavs)):
            uid = self._uid(r, i)
            ref = r["refs"][0] if r.get("refs") else ""
            bp = self.base_cache.get(uid, "")
            table.add_data(
                step, uid, r.get("dataset", ""),
                wandb.Audio(wav, sample_rate=16000),
                ref, bp, h,
                round(compute_wer(ref, bp), 4) if ref else None,
                round(compute_mr_wer(r.get("refs", [ref]), h), 4) if ref else None,
            )
        wandb.log({"live_samples": table}, step=step)
        return control
