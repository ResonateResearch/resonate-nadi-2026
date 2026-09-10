# coding=utf-8
"""Cohere ASR fine-tuning, used in full_ft mode for the NADI Arabic system.

Public-export changes: default to the Arabic July model, explicitly record
seed 42, and disable automatic remote experiment reporting. See ../README.md
for the NADI settings; the general trainer retains other optional modes.

Cohere-transcribe is a classic ENCODER-DECODER seq2seq ASR model:
    model.encoder                  Parakeet/FastConformer encoder (48L, d=1280)   ENCODER  (92% of params)
    model.decoder.proj             Linear 1280->1024 (encoder_hidden -> dec space) PROJECTOR / adaptor
    model.decoder.layers[i]        self_attn + encoder_attn(cross) + mlp(fc1/fc2)  DECODER  (8L, d=1024, 6.5%)
    proj_out                       LM head, tied to decoder.embed_tokens           HEAD
Audio conditions the decoder via cross-attention. The loss shifts internally, so
the collator masks the whole prompt and appends eos (see data_collator.py).

Stages (superset; pick with --stage):
    full_ft  : train everything (encoder+proj+decoder+head).            [recommended for a small dedicated ASR model]
    warmup   : train the proj adaptor only (quick front-end alignment).
    stage1   : train encoder + proj, freeze decoder + head.            [Voxtral-style front-end stage]
    stage2   : freeze all, attach LoRA to the decoder layers only.     [Voxtral-style decoder stage]
"""
import argparse
import os

import torch
import torch.nn.functional as F

# cuDNN's fused SDPA (MHA-graph) kernel crashes on some cross-attention seq-length combos
# ("mha_graph.execute(...).is_good() == false"); it killed a Phase-A run at 91%. Disable just
# that backend -> SDPA falls back to flash / mem-efficient (same speed, robust).
torch.backends.cuda.enable_cudnn_sdp(False)

from datasets import DatasetDict, load_dataset
from transformers import AutoProcessor, CohereAsrForConditionalGeneration, Trainer, TrainingArguments

from common import clean_target, dialect_of
from data_collator import DataCollatorForCohereASR

# --- module-name prefixes (verified against the real checkpoint) ---
ENCODER_KEYS = ("model.encoder.",)
PROJECTOR_KEYS = ("model.decoder.proj.",)
DECODER_BODY = ("model.decoder.layers.", "model.decoder.embed_tokens.",
                "model.decoder.pos_emb.", "model.decoder.embedding_layernorm.",
                "model.decoder.norm.")
HEAD_KEYS = ("proj_out.",)
LORA_REGEX = r".*model\.decoder\.layers\.\d+\..*\.(q_proj|k_proj|v_proj|o_proj|fc1|fc2)$"
MODEL_ID = "CohereLabs/cohere-transcribe-arabic-07-2026"


def _startswith_any(n, keys):
    return any(n.startswith(k) for k in keys)


def set_trainable(model, stage: str):
    """stage in {full_ft, warmup, stage1}. Returns (n_trainable, n_total)."""
    if stage == "full_ft":
        train_keys = ENCODER_KEYS + PROJECTOR_KEYS + DECODER_BODY + HEAD_KEYS
        guard = ()
    elif stage == "warmup":
        train_keys = PROJECTOR_KEYS
        guard = ENCODER_KEYS + DECODER_BODY + HEAD_KEYS
    else:  # stage1: front-end
        train_keys = ENCODER_KEYS + PROJECTOR_KEYS
        guard = DECODER_BODY + HEAD_KEYS
    n_train = n_total = 0
    bad = []
    for n, p in model.named_parameters():
        is_train = _startswith_any(n, train_keys)
        p.requires_grad_(is_train)
        n_total += p.numel()
        if is_train:
            n_train += p.numel()
            if guard and _startswith_any(n, guard):
                bad.append(n)
    assert not bad, f"GUARD: frozen-zone params marked trainable: {bad[:5]}"
    assert 0 < n_train <= n_total, "degenerate trainable set"
    return n_train, n_total


def _is_head(n):
    # proj_out (LM head) is TIED to model.decoder.embed_tokens -> same tensor. Treat both as "head".
    return _startswith_any(n, HEAD_KEYS) or n.startswith("model.decoder.embed_tokens.")


class CastFloatInputsTrainer(Trainer):
    """Cast float inputs (mel input_features) to the model dtype (bf16); per-group LRs
    (encoder / body / tied-head); and a correctness-critical custom loss.

    Label alignment: this encoder-decoder model's loss SHIFTS internally
    (logits[i] predicts labels[i+1]). HF Trainer's built-in
    label_smoothing_factor would apply the NON-shifting encoder-decoder branch and misalign by
    one token, so we compute the shifted (optionally label-smoothed) loss ourselves. With
    label_smoothing=0 this reproduces the model's own verified loss exactly.
    """
    encoder_lr = None
    head_lr = None
    label_smoothing = 0.0

    def _prepare_inputs(self, inputs):
        inputs = super()._prepare_inputs(inputs)
        mdtype = getattr(self.model, "dtype", None)
        if mdtype is not None:
            for k, v in list(inputs.items()):
                if torch.is_tensor(v) and v.is_floating_point():
                    inputs[k] = v.to(dtype=mdtype)
        return inputs

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if getattr(self, "mwer", 0):
            return self._mwer_loss(model, inputs)
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        shift_logits = logits[..., :-1, :].contiguous().float()
        shift_labels = labels[..., 1:].contiguous()
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1),
            ignore_index=-100, label_smoothing=self.label_smoothing)
        return (loss, outputs) if return_outputs else loss

    @staticmethod
    def _word_edit(a, b):
        if a == b:
            return 0
        la, lb = len(a), len(b)
        if not la or not lb:
            return max(la, lb)
        prev = list(range(lb + 1))
        for i in range(1, la + 1):
            cur = [i] + [0] * lb
            ai = a[i - 1]
            for j in range(1, lb + 1):
                cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ai != b[j - 1]))
            prev = cur
        return prev[lb]

    def _mwer_loss(self, model, inputs):
        """Minimum-WER sequence training. Generate N-best (no-grad, BN frozen), score each hyp
        with gradient, minimize expected normalized word-edit risk over the renormalized
        hypothesis distribution, plus a small CE anchor to the reference."""
        raw = model.module if hasattr(model, "module") else model
        tok = self.mwer_tok
        nt = self._normtext
        eos = tok.eos_token_id
        pad = tok.pad_token_id if tok.pad_token_id is not None else eos
        prompt = self.mwer_prompt
        plen = len(prompt)
        inf = inputs["input_features"]
        am = inputs["attention_mask"]
        labels = inputs["labels"]
        B = inf.size(0)
        N = self.mwer_nbest
        device = inf.device

        # freeze BatchNorm (encoder) for the whole step: gen (eval-stats) and score must match,
        # and a short MWER stage must not repopulate running stats.
        for m in raw.modules():
            if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
                m.eval()

        # reference text per utt (strip the -100 prompt/pad, decode target+eos)
        refs = []
        for b in range(B):
            ids = labels[b][labels[b] != -100].tolist()
            refs.append(nt(tok.decode(ids, skip_special_tokens=True)).split())

        # 1) N-best generation, no grad
        prev_cache = getattr(raw.config, "use_cache", False)
        raw.config.use_cache = True
        di = torch.tensor([prompt] * B, dtype=torch.long, device=device)
        with torch.no_grad():
            gen = raw.generate(input_features=inf, attention_mask=am, decoder_input_ids=di,
                               max_new_tokens=self.mwer_max_new, do_sample=False,
                               num_beams=N, num_return_sequences=N, no_repeat_ngram_size=6)
        raw.config.use_cache = False
        gen = gen.view(B, N, -1)

        # 2) build scoring batch (prompt + hyp + eos), compute risk
        hyp_full, risks = [], []
        for b in range(B):
            for i in range(N):
                h = gen[b, i, plen:].tolist()
                if eos in h:
                    h = h[:h.index(eos)]
                h = [x for x in h if x != pad]
                risks.append(self._word_edit(refs[b], nt(tok.decode(h, skip_special_tokens=True)).split())
                             / max(len(refs[b]), 1))
                hyp_full.append(prompt + h + [eos])
        Lmax = max(len(f) for f in hyp_full)
        dec_in = torch.full((B * N, Lmax), pad, dtype=torch.long, device=device)
        dec_lab = torch.full((B * N, Lmax), -100, dtype=torch.long, device=device)
        dec_am = torch.zeros((B * N, Lmax), dtype=torch.long, device=device)
        for k, f in enumerate(hyp_full):
            dec_in[k, :len(f)] = torch.tensor(f, device=device)
            dec_am[k, :len(f)] = 1
            dec_lab[k, plen:len(f)] = torch.tensor(f[plen:], device=device)  # score hyp+eos, not prompt

        # 3) scored forward (grad), encoder features repeated per hyp
        inf_rep = inf.repeat_interleave(N, 0)
        am_rep = am.repeat_interleave(N, 0)
        out = raw(input_features=inf_rep, attention_mask=am_rep,
                  decoder_input_ids=dec_in, decoder_attention_mask=dec_am)
        # token log-probs via fused cross_entropy (no [B*N,L,V] float materialization -> OOM-safe)
        V = out.logits.size(-1)
        tgt = dec_lab[:, 1:]
        m = (tgt != -100).float()
        tok_lp = -F.cross_entropy(out.logits[:, :-1].reshape(-1, V),
                                  tgt.reshape(-1).clamp(min=0), reduction="none").view(tgt.shape)
        seq_lp = (tok_lp * m).sum(-1).view(B, N)            # sum log-prob per hyp
        W = torch.tensor(risks, device=device, dtype=torch.float).view(B, N)

        # 4) MWER: expected risk over renormalized hyp distribution, mean-risk baseline
        p = F.softmax(seq_lp, dim=1)
        Wbar = (p.detach() * W).sum(1, keepdim=True)
        mwer = (p * (W - Wbar)).sum(1).mean()

        # CE anchor to the reference (teacher forced)
        ce_out = raw(input_features=inf, attention_mask=am,
                     decoder_input_ids=inputs["decoder_input_ids"],
                     decoder_attention_mask=inputs["decoder_attention_mask"])
        ce = F.cross_entropy(ce_out.logits[..., :-1, :].reshape(-1, V),
                             labels[..., 1:].reshape(-1),
                             ignore_index=-100, label_smoothing=self.label_smoothing)
        raw.config.use_cache = prev_cache
        return mwer + self.mwer_ce * ce

    def create_optimizer(self):
        # single-LR path (e.g. Phase-A warmup) -> HF default (keeps its decay/no-decay split)
        if self.optimizer is not None or not self.encoder_lr:
            return super().create_optimizer()
        decay = set(self.get_decay_parameter_names(self.model))
        head_lr = self.head_lr if self.head_lr else self.args.learning_rate
        buckets = {}   # (lr, wd) -> [params]
        seen = set()
        for n, p in self.model.named_parameters():
            if not p.requires_grad or id(p) in seen:
                continue
            seen.add(id(p))
            if _startswith_any(n, ENCODER_KEYS):
                lr = self.encoder_lr
            elif _is_head(n):
                lr = head_lr
            else:
                lr = self.args.learning_rate
            wd = self.args.weight_decay if n in decay else 0.0
            buckets.setdefault((lr, wd), []).append(p)
        opt_cls, opt_kw = Trainer.get_optimizer_cls_and_kwargs(self.args)
        opt_kw.pop("lr", None); opt_kw.pop("weight_decay", None)
        groups = [{"params": ps, "lr": lr, "weight_decay": wd} for (lr, wd), ps in buckets.items()]
        n_by = {f"lr={lr:g},wd={wd:g}": sum(p.numel() for p in ps) for (lr, wd), ps in buckets.items()}
        print(f"[optim] param groups: {n_by}")
        self.optimizer = opt_cls(groups, **opt_kw)
        return self.optimizer


def parse_args():
    p = argparse.ArgumentParser("cohere-transcribe ASR SFT")
    p.add_argument("--model_id", default=MODEL_ID)
    p.add_argument("--train_files", nargs="+", required=True)
    p.add_argument("--eval_files", nargs="*", default=[])
    p.add_argument("--output_dir", required=True)
    p.add_argument("--stage", choices=["full_ft", "warmup", "stage1", "stage2"], default="full_ft")
    p.add_argument("--init_from", default="", help="load weights only (e.g. a stage1 ckpt)")
    p.add_argument("--language", default="ar")
    p.add_argument("--dialect_cond", type=int, default=0,
                   help="1 = dialect conditioning: inject the row's country name into the "
                        "<|startofcontext|> prompt slot (prompt-only, masked with -100 in the "
                        "labels). Dialect = manifest 'dialect' field, else parsed from the audio "
                        "path; rows with no dialect keep the stock prompt. 0 = off (default, "
                        "byte-identical to the pre-feature behaviour).")

    p.add_argument("--sr", type=int, default=16000)
    p.add_argument("--batch_size", type=int, default=24)
    p.add_argument("--eval_batch_size", type=int, default=24)
    p.add_argument("--grad_acc", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-4, help="base LR (body: proj + decoder layers)")
    p.add_argument("--encoder_lr", type=float, default=0.0, help=">0 enables per-group LRs (encoder/body/head)")
    p.add_argument("--head_lr", type=float, default=0.0, help="LR for the tied embed/proj_out head (~0.25x lr); 0 = use base lr")
    p.add_argument("--label_smoothing", type=float, default=0.0)
    p.add_argument("--adam_beta2", type=float, default=0.999)
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--max_steps", type=int, default=-1)
    p.add_argument("--lr_scheduler_type", default="cosine")
    p.add_argument("--warmup_ratio", type=float, default=0.03)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--log_steps", type=int, default=25)
    p.add_argument("--gradient_checkpointing", type=int, default=1)
    p.add_argument("--sync_bn", type=int, default=1, help="convert encoder BatchNorm -> SyncBatchNorm under DDP")

    p.add_argument("--num_workers", type=int, default=10)
    p.add_argument("--save_steps", type=int, default=1000)
    p.add_argument("--eval_steps", type=int, default=1000)

    # monitoring / retention
    p.add_argument("--dev_subset", default="", help="jsonl with {audio, refs:[...], domain, dataset}")
    p.add_argument("--select_metric", choices=["balanced", "macro_dialect", "micro"],
                   default="balanced",
                   help="metric TopKCheckpointCallback ranks/prunes checkpoints on. "
                        "'balanced' (default, legacy) = macro over `domain` of mean-utterance WER; "
                        "'macro_dialect' = macro over `dataset` of per-dialect CORPUS WER "
                        "(mirrors the NADI country_av_wer); 'micro' = overall mean-utterance WER")
    p.add_argument("--topk", type=int, default=10)
    p.add_argument("--sample_every", type=int, default=250)
    p.add_argument("--sample_anchor", type=int, default=8)
    p.add_argument("--sample_random", type=int, default=4)
    p.add_argument("--gen_max_new_tokens", type=int, default=448)

    # stage2 LoRA
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)

    p.add_argument("--resume", type=int, default=0)
    p.add_argument("--specaug", type=int, default=0, help="1 = always-on SpecAugment (feature masks) during training")
    p.add_argument("--specaug_live_bins", type=int, default=0,
                   help=">0: restrict freq masks to first N mel bins (band-limited audio); freq width scales 27->18")
    p.add_argument("--specaug_time_ratio", type=float, default=0.0,
                   help="cap each SpecAugment time mask at ratio*valid_frames (0 = legacy absolute-only cap). "
                        "Needed when the manifest contains sub-1s clips: the absolute cap is 100 frames = 1.00 s, "
                        "so short clips get blanked entirely while their transcript is still supervised")
    p.add_argument("--mwer", type=int, default=0, help="1 = MWER (min-WER) sequence-level loss (stage-2)")
    p.add_argument("--mwer_nbest", type=int, default=4)
    p.add_argument("--mwer_ce", type=float, default=0.05, help="CE-anchor weight in MWER loss")
    p.add_argument("--mwer_max_new", type=int, default=160)
    p.add_argument("--probe_nrn6", type=int, default=6, help="no_repeat_ngram_size for the dev-probe decode (0=off); default 6 matches offline scoring")
    return p.parse_args()


def main():
    a = parse_args()
    # MWER builds ONE rectangular decoder_input_ids from a single cached prompt
    # (trainer.mwer_prompt) and slices gen[:, :, plen:] with one plen; dialect cues have
    # per-country lengths (1-4 tokens), so the two are incompatible until MWER grows a
    # per-sample/left-padded prompt. Fail fast instead of silently mis-slicing N-best.
    assert not (a.mwer and a.dialect_cond), \
        "--mwer and --dialect_cond are mutually exclusive (MWER assumes one fixed-length prompt)"
    rank0 = int(os.environ.get("RANK", "0")) == 0
    world = int(os.environ.get("WORLD_SIZE", "1"))

    use_bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8
    src = a.init_from or a.model_id
    model = CohereAsrForConditionalGeneration.from_pretrained(
        src, dtype=torch.bfloat16 if use_bf16 else torch.float16, device_map=None)
    model.config.use_cache = False
    processor = AutoProcessor.from_pretrained(a.model_id)

    # --- trainable / frozen ---
    if a.stage == "stage2":
        for p in model.parameters():
            p.requires_grad_(False)
        from peft import LoraConfig, get_peft_model
        lconf = LoraConfig(
            task_type="SEQ_2_SEQ_LM", r=a.lora_r, lora_alpha=a.lora_alpha, lora_dropout=a.lora_dropout,
            target_modules=LORA_REGEX,
        )
        model = get_peft_model(model, lconf)
        if rank0:
            model.print_trainable_parameters()
            leaked = [n for n, p in model.named_parameters()
                      if p.requires_grad and ("model.encoder" in n or "decoder.proj" in n or n.startswith("proj_out"))]
            assert not leaked, f"LoRA leaked onto the front-end/head: {leaked[:5]}"
            assert any("decoder.layers" in n and "lora_" in n
                       for n, p in model.named_parameters() if p.requires_grad), \
                "no LoRA adapters on decoder layers"
    else:
        n_tr, n_tot = set_trainable(model, a.stage)
        if rank0:
            print(f"[freeze] stage={a.stage} trainable={n_tr:,}/{n_tot:,} = {n_tr/n_tot:.4%}")
        # BatchNorm in the encoder: sync stats across GPUs when the encoder trains (avoids
        # degenerate per-replica stats on long/noisy utterances under length-bucketing).
        if a.sync_bn and world > 1 and a.stage in ("full_ft", "stage1"):
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            if rank0:
                print("[bn] converted encoder BatchNorm -> SyncBatchNorm (DDP)")

    if a.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

    # --- data ---
    data_files = {"train": list(a.train_files)}
    if a.eval_files:
        data_files["validation"] = list(a.eval_files)
    raw = load_dataset("json", data_files=data_files)

    def _prep(ex):
        out = {"audio": ex["audio"], "target": clean_target(ex["text"])}
        if a.dialect_cond:
            out["dialect"] = dialect_of(ex)      # "" when unknown -> stock prompt for that row
        return out

    keep = ("audio", "dialect") if a.dialect_cond else ("audio",)
    ds = DatasetDict()
    for split in raw:
        ds[split] = raw[split].map(
            _prep, remove_columns=[c for c in raw[split].column_names if c not in keep])
        if rank0:
            print(f"[data] split={split} rows={len(ds[split])}")
            if a.dialect_cond:
                from collections import Counter
                cnt = Counter(ds[split]["dialect"])
                cov = 1.0 - cnt.get("", 0) / max(len(ds[split]), 1)
                print(f"[dialect] split={split} coverage={cov:.2%} {dict(cnt)}")

    collator = DataCollatorForCohereASR(processor=processor, language=a.language,
                                        sampling_rate=a.sr, specaug=bool(a.specaug),
                                        specaug_live_bins=a.specaug_live_bins,
                                        freq_mask_param=18 if a.specaug_live_bins > 0 else 27,
                                        time_mask_ratio=a.specaug_time_ratio,
                                        dialect_cond=bool(a.dialect_cond))

    # --- callbacks ---
    from callbacks import (MakeCheckpointInferableCallback, DevWERCallback,
                           TopKCheckpointCallback, WandbSampleInferenceCallback)
    callbacks = [MakeCheckpointInferableCallback(processor)]
    dev_cb = None
    if a.dev_subset:
        dev_cb = DevWERCallback(processor, a.dev_subset, language=a.language,
                                every=a.eval_steps, max_new_tokens=a.gen_max_new_tokens,
                                nrn6=a.probe_nrn6, dialect_cond=bool(a.dialect_cond))
        callbacks.append(dev_cb)
        # retention (and best-dev selection); default stays the CS-weighted macro dev-WER
        def _select_metric(cb=dev_cb, which=a.select_metric):
            if which == "macro_dialect":
                v = cb.last_dev_wer_macro_dialect
                # dev set produced no scorable dialect groups -> fall back to the legacy metric
                return v if v is not None else cb.last_dev_wer_balanced
            if which == "micro":
                return cb.last_dev_wer
            return cb.last_dev_wer_balanced
        if rank0:
            print(f"[topk] checkpoint selection metric = {a.select_metric}")
        callbacks.append(TopKCheckpointCallback(k=a.topk, metric_fn=_select_metric))
    if a.sample_every > 0 and a.dev_subset:
        callbacks.append(WandbSampleInferenceCallback(
            processor, a.dev_subset, language=a.language,
            every=a.sample_every, n_anchor=a.sample_anchor, n_random=a.sample_random,
            max_new_tokens=a.gen_max_new_tokens, dialect_cond=bool(a.dialect_cond)))

    targs = TrainingArguments(
        output_dir=a.output_dir,
        per_device_train_batch_size=a.batch_size,
        per_device_eval_batch_size=a.eval_batch_size,
        gradient_accumulation_steps=a.grad_acc,
        learning_rate=a.lr,
        num_train_epochs=a.epochs,
        max_steps=a.max_steps,
        lr_scheduler_type=a.lr_scheduler_type,
        warmup_ratio=a.warmup_ratio,
        weight_decay=a.weight_decay,
        adam_beta2=a.adam_beta2,
        max_grad_norm=a.max_grad_norm,
        logging_steps=a.log_steps,
        gradient_checkpointing=bool(a.gradient_checkpointing),
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataloader_num_workers=a.num_workers,
        dataloader_pin_memory=True,
        dataloader_persistent_workers=a.num_workers > 0,
        dataloader_prefetch_factor=2 if a.num_workers > 0 else None,
        save_strategy="steps",
        save_steps=a.save_steps,
        save_total_limit=None,          # retention handled by TopKCheckpointCallback
        eval_strategy="steps" if a.eval_files else "no",
        eval_steps=a.eval_steps,
        bf16=use_bf16,
        fp16=not use_bf16,
        ddp_find_unused_parameters=False,
        remove_unused_columns=False,
        seed=42,
        report_to=[],  # public export: no automatic remote logging or credential requirement
        run_name=a.output_dir.rstrip("/").split("/")[-1],
    )

    trainer = CastFloatInputsTrainer(
        model=model, args=targs,
        train_dataset=ds["train"], eval_dataset=ds.get("validation"),
        data_collator=collator, processing_class=processor,
        callbacks=callbacks,
    )
    if a.encoder_lr > 0:
        trainer.encoder_lr = a.encoder_lr
    if a.head_lr > 0:
        trainer.head_lr = a.head_lr
    trainer.label_smoothing = a.label_smoothing
    trainer.mwer = a.mwer
    if a.mwer:
        from common import normalize_text as _nt
        trainer.mwer_nbest = a.mwer_nbest
        trainer.mwer_ce = a.mwer_ce
        trainer.mwer_max_new = a.mwer_max_new
        trainer.mwer_prompt = list(processor.get_decoder_prompt_ids(language=a.language, punctuation=True))
        trainer.mwer_tok = processor.tokenizer
        trainer._normtext = _nt
    for cb in callbacks:
        if hasattr(cb, "attach"):
            cb.attach(trainer)

    trainer.train(resume_from_checkpoint=bool(a.resume))


if __name__ == "__main__":
    main()
