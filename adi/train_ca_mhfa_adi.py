#!/usr/bin/python3
"""Train WavLM Base+ with CA-MHFA for 20-class Arabic dialect identification.

The public configuration matches the documented submission_wavlm run. It initializes
WavLM from Base+ and trains the pooling and classifier from scratch; the optional
VoxCeleb checkpoint transfer supported by this trainer is disabled in that record.
Training combines locally prepared ADI-17/ADI-20 Arrow caches with same-dialect
voice-converted audio and uses a five-second segment manifest with start jitter.

Run on one H100; see README.md for the required local inputs and preparation:
> python train_ca_mhfa_adi.py hparams/train_wavlm_base_plus.yaml \
>     --data_root "$ADI_DATA_ROOT" --ssl_hub "$WAVLM_BASE_PLUS" \
>     --device cuda:0 --precision bf16 --eval_precision bf16

Public-release modifications (2026): portable local inputs, corrected stale
configuration comments, and offline execution without implicit Hub credentials.

Author
    * nadi project, 2026 (adapted from VoxCeleb/SpeakerRec/train_ca_mhfa.py)

Inherited SpeechBrain recipe attribution
    * Mirco Ravanelli, Hwidong Na, Nauman Dawalatabad, 2020
      (VoxCeleb/SpeakerRec/train_speaker_embeddings.py lineage)
"""

import csv
import glob
import io
import json
import math
import os
import random
import re
import sys
from collections import defaultdict

# The Hub reads these flags during import. Public recipe imports therefore
# establish offline local-data/model operation before importing SpeechBrain.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"

import numpy as np
import soundfile as sf
import torch
from hyperpyyaml import load_hyperpyyaml

import speechbrain as sb
from speechbrain.augment.preparation import prepare_csv
from speechbrain.utils.checkpoints import Checkpointer
from speechbrain.utils.data_utils import get_all_files
from speechbrain.utils.distributed import if_main_process, run_on_main
from speechbrain.utils.logger import get_logger

logger = get_logger(__name__)


def prepare_aug_csv(folder, csv_file, ext="wav", max_length=None):
    """Build a MUSAN/RIR augmentation CSV from an already-downloaded folder."""
    if os.path.isfile(csv_file):
        return
    filelist = get_all_files(folder, match_and=["." + ext])
    if not filelist:
        raise ValueError(f"No .{ext} files found under {folder}")
    prepare_csv(filelist, csv_file, max_length)


def _read_chunk(audio, n_samples, target_sr, mode, start_s=None, dur_s=None):
    """Read one waveform chunk from an (undecoded) HF Audio value.

    `audio` is {"bytes": ..., "path": ...} (Audio(decode=False)). We seek inside
    the file with soundfile so long clips are never fully decoded.

    mode="random"  -> random crop of n_samples (train)
    mode="center"  -> centered crop of n_samples (eval)
    Clips shorter than n_samples are returned whole (PaddedBatch pads them).

    start_s/dur_s (segment manifest) override `mode`: read exactly that window.
    The start is clamped into the clip, so per-epoch jitter may push it past
    either edge harmlessly, and offsets are computed at the FILE's sample rate
    (resampling, if any, happens after the read).
    """
    b = audio.get("bytes")
    src = io.BytesIO(b) if b is not None else audio["path"]
    with sf.SoundFile(src) as f:
        total = len(f)
        sr = f.samplerate
        if start_s is not None:
            n = min(int(dur_s * sr), total) if dur_s else total
            s0 = max(0, min(int(start_s * sr), total - n))
            f.seek(s0)
            wav = f.read(frames=n, dtype="float32")
        elif n_samples is not None and total > n_samples:
            if mode == "random":
                start = random.randint(0, total - n_samples)
            else:  # center
                start = (total - n_samples) // 2
            f.seek(start)
            wav = f.read(frames=n_samples, dtype="float32")
        else:
            wav = f.read(dtype="float32")

    wav = np.asarray(wav, dtype=np.float32)
    if wav.ndim > 1:  # stereo -> mono
        wav = wav.mean(axis=1)
    sig = torch.from_numpy(wav)
    if sr != target_sr:  # ADI-17 is 16 kHz; resample only if it ever isn't
        import torchaudio

        sig = torchaudio.functional.resample(sig, sr, target_sr)
    return sig


def classification_metrics(y_true, y_pred, n_classes):
    """Single-label multiclass metrics from raw label lists (dependency-free).

    Returns accuracy, error_rate, macro_f1, macro_precision, macro_recall.
    (Micro-F1 is omitted: for single-label classification it equals accuracy.)
    Macro stats average over the classes present in y_true OR y_pred.
    """
    n = len(y_true)
    if n == 0:
        return {
            "accuracy": 0.0, "error_rate": 1.0,
            "macro_f1": 0.0, "macro_precision": 0.0, "macro_recall": 0.0,
        }
    tp = [0] * n_classes
    fp = [0] * n_classes
    fn = [0] * n_classes
    correct = 0
    present = set()
    for t, p in zip(y_true, y_pred):
        present.add(t)
        present.add(p)
        if t == p:
            tp[t] += 1
            correct += 1
        else:
            fp[p] += 1
            fn[t] += 1
    acc = correct / n
    precs, recs, f1s = [], [], []
    for c in sorted(present):
        pd, rd = tp[c] + fp[c], tp[c] + fn[c]
        pr = tp[c] / pd if pd else 0.0
        rc = tp[c] / rd if rd else 0.0
        precs.append(pr)
        recs.append(rc)
        f1s.append(2 * pr * rc / (pr + rc) if (pr + rc) else 0.0)
    mean = lambda xs: sum(xs) / len(xs) if xs else 0.0
    return {
        "accuracy": acc,
        "error_rate": 1.0 - acc,
        "macro_f1": mean(f1s),
        "macro_precision": mean(precs),
        "macro_recall": mean(recs),
    }


class DialectBrain(sb.core.Brain):
    """WavLM + CA-MHFA pooling -> AAM-softmax dialect classifier."""

    def compute_forward(self, batch, stage):
        """SSL front-end -> CA-MHFA pooling -> dialect classifier."""
        batch = batch.to(self.device)
        wavs, lens = batch.sig

        # Waveform augmentation (noise / reverb / speed / freq+chunk drop) during
        # training. Forced to fp32 with autocast OFF: compute_forward runs inside
        # the bf16 training context, and SpeedPerturb's torchaudio Resample keeps
        # an fp32 sinc kernel while migrating only device, not dtype -- a bf16
        # waveform reaching it dies in conv1d. Augmentation is signal processing
        # (SNR ratios, sinc resampling), so fp32 is the right precision anyway;
        # the SSL front-end below re-enters autocast and casts as needed.
        if (
            stage == sb.Stage.TRAIN
            and self.hparams.apply_augment
            and hasattr(self.hparams, "wav_augment")
        ):
            dev_type = "cuda" if str(self.device).startswith("cuda") else "cpu"
            with torch.autocast(device_type=dev_type, enabled=False):
                wavs, lens = self.hparams.wav_augment(wavs.float(), lens)

        # SSL layer-wise features. output_all_hiddens=True -> [L, B, T, F].
        feats = self.modules.ssl_model(wavs)
        feats = feats.permute(1, 0, 2, 3)  # -> [B, num_layers, T, F]

        # CA-MHFA pooling. `lens` (relative) masks padded frames in attention.
        embeddings = self.modules.embedding_model(feats, lens)  # [B, emb_dim]

        # Dialect classifier (AAM head weights live in the classifier).
        outputs = self.modules.classifier(embeddings)  # [B, 1, n_class]

        return outputs, lens

    def compute_objectives(self, predictions, batch, stage):
        """AAM-softmax loss over dialect labels."""
        predictions, lens = predictions
        uttid = batch.id
        dialect, _ = batch.dialect_encoded

        # Replicate labels for the augmentation copies (concat_original etc.).
        if (
            stage == sb.Stage.TRAIN
            and self.hparams.apply_augment
            and hasattr(self.hparams, "wav_augment")
        ):
            dialect = self.hparams.wav_augment.replicate_labels(dialect)

        loss = self.hparams.compute_cost(predictions, dialect, lens)

        if stage != sb.Stage.TRAIN:
            # Accumulate class predictions vs targets for full metric suite.
            preds = predictions.argmax(dim=-1).reshape(-1)  # [B]
            targs = dialect.reshape(-1)                     # [B]
            self.y_pred.extend(preds.tolist())
            self.y_true.extend(targs.tolist())

        return loss

    def on_fit_start(self):
        """Set up gradual SSL unfreezing (top-down) before the DDP wrap.

        Same recipe as the VoxCeleb CA-MHFA run: the WavLM upstream starts fully
        masked (only CA-MHFA pooling + the dialect head train), then transformer
        blocks are unfrozen from the TOP down, `unfreeze_rate` block(s) per epoch
        from `unfreeze_start_epoch`. Paired with a warmup->cosine LR (see _lr_at).

        Freezing is done by ZEROING each masked block's gradient, NOT by toggling
        requires_grad: DDP snapshots requires_grad when it wraps the modules and
        only reduces params trainable at that instant, so flipping it later would
        leave newly unfrozen params un-all-reduced (silent per-rank drift). Every
        SSL param stays requires_grad=True and in the reducer; a masked block
        still runs forward, produces a grad, and the hook zeros it -> no update,
        no unused-param crash. MUST run before super().on_fit_start() (the wrap).
        """
        hf_model = getattr(self.modules.ssl_model, "model", None)

        # masked_spec_embed is only used on the (disabled) SpecAugment path -> it
        # never gets a grad, so DDP's reducer would wait forever. Freeze it out.
        mse = getattr(hf_model, "masked_spec_embed", None)
        if mse is not None and mse.requires_grad:
            mse.requires_grad_(False)

        # Disable LayerDrop (config.layerdrop=0.05): a randomly skipped encoder
        # layer gets no grad -> DDP unused-param crash, and it corrupts CA-MHFA's
        # per-layer softmax weighting (a dropped layer copies the previous state).
        cfg = getattr(hf_model, "config", None)
        if cfg is not None and getattr(cfg, "layerdrop", 0.0):
            cfg.layerdrop = 0.0

        # Map each trainable SSL param -> transformer-block index (built pre-wrap:
        # clean names, and id() stays valid since the wrap doesn't replace tensors).
        self.n_ssl_layers = int(
            getattr(self.hparams, "n_ssl_transformer_layers", 12)
        )
        self._ssl_param_layer = {
            id(p): self._layer_index(name)
            for name, p in self.modules.ssl_model.named_parameters()
            if p.requires_grad
        }
        self.n_unfrozen = 0  # top-N transformer blocks currently trainable

        # One grad-mask hook per trainable SSL param; reads self.n_unfrozen live,
        # so advancing the schedule needs no re-registration.
        for p in self.modules.ssl_model.parameters():
            if p.requires_grad and not getattr(p, "_ca_mhfa_masked", False):
                p.register_hook(self._make_mask_hook(p))
                p._ca_mhfa_masked = True

        # DDP wrap + default single-group init_optimizers + checkpoint resume.
        super().on_fit_start()

    def _layer_index(self, name):
        """Transformer-block index for an SSL param name (schedule bucket)."""
        m = re.search(r"encoder\.layers\.(\d+)\.", name)
        if m:
            return int(m.group(1))
        if "feature_extractor" in name:
            return -1  # CNN feature extractor: frozen the whole run
        # Shared pre-encoder params -- feature_projection, pos_conv_embed and
        # encoder.layer_norm (do_stable_layer_norm=False, so it runs BEFORE the
        # blocks). They have no block index of their own, so which bucket they
        # land in is a choice:
        #   "top"    -- bucket n-1, unfrozen on the FIRST ramp step, as configured
        #               in the released seed-420 run record.
        #   "bottom" -- bucket 0, unfrozen LAST (and never, under `max_unfrozen`).
        #               They sit at the bottom of the stack -- they produce the
        #               input to block 0, so training them shifts what every
        #               frozen block sees. Matches ULMFiT (most general layers
        #               fine-tuned last) and BERT-style LLRD (embedding group
        #               gets the lowest LR).
        where = str(getattr(self.hparams, "pre_encoder_bucket", "top")).lower()
        return 0 if where == "bottom" else self.n_ssl_layers - 1

    def _ssl_param_trainable(self, p):
        """Is `p`'s block currently unfrozen? Single source of truth for the
        schedule -- used by the grad mask and by `optimizers_step`."""
        layer = self._ssl_param_layer.get(id(p), -1)
        if layer < 0:  # CNN / untracked -> always frozen
            return False
        # Top-down: a block is live once it's within the top n_unfrozen.
        return layer >= self.n_ssl_layers - self.n_unfrozen

    def _make_mask_hook(self, p):
        """Backward hook: pass the grad if p's block is unfrozen, else zero it."""

        def hook(grad):
            return grad if self._ssl_param_trainable(p) else grad * 0.0

        return hook

    def optimizers_step(self):
        """Drop frozen blocks' grads entirely, then step as usual.

        The mask hook zeroes a frozen block's grad, but zero is not None, and
        AdamW applies its decoupled weight decay to every param whose `.grad` is
        not None -- so a masked block would still shrink by (1 - lr * wd) on
        every step despite never receiving a learning signal. Setting
        `.grad = None` here makes the optimizer skip those params outright.

        This cannot be done from the backward hook: returning None from
        `Tensor.register_hook` means "leave the gradient unchanged", which would
        un-freeze the param. Doing it here is also DDP-safe -- the reducer has
        already finished its allreduce by the time backward returns. Gradient
        clipping is unaffected: a None grad is skipped by `clip_grad_norm_`, and
        these grads were zero, so they contributed nothing to the norm anyway.
        """
        for p in self.modules.ssl_model.parameters():
            if p.grad is not None and not self._ssl_param_trainable(p):
                p.grad = None
        super().optimizers_step()

    def _unfrozen_at(self, epoch):
        """How many top transformer blocks are trainable at `epoch` (1-based)."""
        start = int(getattr(self.hparams, "unfreeze_start_epoch", 2))
        rate = int(getattr(self.hparams, "unfreeze_rate", 1))
        # `max_unfrozen` caps the ramp: the bottom blocks stay frozen all run.
        cap = getattr(self.hparams, "max_unfrozen", None)
        cap = self.n_ssl_layers if cap is None else min(int(cap), self.n_ssl_layers)
        if epoch < start:
            return 0
        return min(cap, (epoch - start + 1) * rate)

    def _lr_at(self, epoch):
        """Per-epoch LR: warmup->cosine, repeated `lr_cycles` times (SGDR warm restart).

        The run is split into `lr_cycles` equal periods; each period restarts the LR
        at `lr` via a fresh `warmup_epochs` linear ramp, then cosine-decays to
        `final_lr`. lr_cycles=1 is the original single warmup+cosine over the whole run.
        """
        peak = float(self.hparams.lr)
        final = float(getattr(self.hparams, "final_lr", 0.0))
        warm = max(1, int(getattr(self.hparams, "warmup_epochs", 2)))
        total = int(self.hparams.number_of_epochs)
        cycles = max(1, int(getattr(self.hparams, "lr_cycles", 1)))
        period = max(1, total // cycles)          # epochs per restart cycle
        e = (epoch - 1) % period + 1              # 1..period within the current cycle
        if e <= warm:
            return peak * e / warm
        prog = min(1.0, max(0.0, (e - warm) / max(1, period - warm)))
        return final + (peak - final) * 0.5 * (1.0 + math.cos(math.pi * prog))

    def _margin_at(self, epoch):
        """Per-epoch AAM margin: linear ramp `margin_start` -> `aam_margin`.

        A full margin from step 0 fights an untrained head: every class is near
        every other, so pushing cos(theta+m) hard on top of that mostly produces
        large early losses and unstable updates. Ramping it in lets the head
        first learn a rough separation, then tightens it -- the same reason the
        LR warms up. margin_warmup_epochs<=0 (default) keeps the constant margin.
        """
        target = float(self.hparams.aam_margin)
        warm = int(getattr(self.hparams, "margin_warmup_epochs", 0))
        if warm <= 0:
            return target
        start = float(getattr(self.hparams, "margin_start", 0.0))
        # epoch `warm` is the first at full margin; before that, linear in between.
        prog = min(1.0, max(0.0, (epoch - 1) / warm))
        return start + (target - start) * prog

    def _set_margin(self, margin):
        """Push a new margin into the AAM loss.

        AdditiveAngularMargin precomputes cos_m/sin_m/th/mm in __init__, so
        assigning `.margin` alone silently changes nothing -- all four have to be
        recomputed or the loss keeps using the original margin.
        """
        loss_fn = self.hparams.compute_cost.loss_fn
        loss_fn.margin = margin
        loss_fn.cos_m = math.cos(margin)
        loss_fn.sin_m = math.sin(margin)
        loss_fn.th = math.cos(math.pi - margin)
        loss_fn.mm = math.sin(math.pi - margin) * margin

    def _wandb_run(self):
        """The live wandb Run on the main process, or None (W&B off / other rank)."""
        wl = getattr(self, "wandb_logger", None)
        return getattr(wl, "run", None) if wl is not None else None

    def _wandb_log(self, payload):
        """Log to W&B against the global `optimizer_step` x-axis.

        Everything (per-batch train + per-epoch valid/test) is logged WITHOUT an
        explicit wandb step and carries `optimizer_step`; build_wandb_logger sets
        that as the step metric for all series, so per-batch and per-epoch points
        share one monotonic x-axis and never collide.
        """
        run = self._wandb_run()
        if run is None:
            return
        run.log({**payload, "optimizer_step": self.optimizer_step})

    def on_fit_batch_end(self, batch, outputs, loss, should_step):
        """Per-step train logging so W&B charts populate live (not once/epoch)."""
        super().on_fit_batch_end(batch, outputs, loss, should_step)
        if not should_step or self._wandb_run() is None:
            return
        every = int(getattr(self.hparams, "wandb_log_every", 25))
        if self.optimizer_step % every == 0:
            self._wandb_log({
                "train/loss": float(loss),
                "lr": self.optimizer.param_groups[0]["lr"],
                "ssl_unfrozen": self.n_unfrozen,
                "epoch": self.hparams.epoch_counter.current,
            })

    def _log_wandb_confusion(self, split):
        """Log a confusion-matrix heatmap for the accumulated preds to W&B.

        No-op unless W&B is on (main process) and class names are attached.
        """
        run = self._wandb_run()
        names = getattr(self, "dialect_names", None)
        if run is None or not names or not self.y_true:
            return
        import wandb

        cm = wandb.plot.confusion_matrix(
            y_true=self.y_true,
            preds=self.y_pred,
            class_names=names,
            title=f"{split} confusion",
        )
        self._wandb_log({f"{split}/confusion_matrix": cm})

    def on_stage_start(self, stage, epoch=None):
        if stage == sb.Stage.TRAIN and epoch is not None:
            # Advance the unfreeze schedule + set this epoch's LR (both derived
            # purely from `epoch`, so resume is exact -- no scheduler state).
            self.n_unfrozen = self._unfrozen_at(epoch)
            lr = self._lr_at(epoch)
            for group in self.optimizer.param_groups:
                group["lr"] = lr
            margin = self._margin_at(epoch)
            self._set_margin(margin)
            logger.info(
                "epoch %d: SSL blocks unfrozen=%d/%d (top-down)  lr=%.3e  margin=%.3f",
                epoch, self.n_unfrozen, self.n_ssl_layers, lr, margin,
            )
        if stage != sb.Stage.TRAIN:
            self.y_true = []
            self.y_pred = []

    def on_stage_end(self, stage, stage_loss, epoch=None):
        stage_stats = {"loss": stage_loss}
        if stage == sb.Stage.TRAIN:
            self.train_stats = stage_stats
        else:
            metrics = classification_metrics(
                self.y_true, self.y_pred, self.hparams.out_n_neurons
            )
            # Keep the capitalized ErrorRate key for the checkpointer's min_key;
            # attach accuracy / macro-F1 / macro-P / macro-R too.
            stage_stats["ErrorRate"] = metrics.pop("error_rate")
            stage_stats.update(metrics)

        if stage == sb.Stage.VALID:
            # LR + unfreeze are set per-epoch in on_stage_start (warmup->cosine +
            # gradual unfreeze); just report the current values here.
            lr = self.optimizer.param_groups[0]["lr"]
            margin = self.hparams.compute_cost.loss_fn.margin
            meta = {
                "epoch": epoch,
                "lr": lr,
                "ssl_unfrozen": self.n_unfrozen,
                "margin": margin,
            }

            self.hparams.train_logger.log_stats(
                stats_meta=meta,
                train_stats=self.train_stats,
                valid_stats=stage_stats,
            )
            # Per-epoch valid to W&B (main process only; no-op if off). Direct
            # run.log against optimizer_step -- consistent with the per-batch
            # train points, so both share one x-axis.
            self._wandb_log({
                "epoch": epoch,
                "lr": lr,
                "ssl_unfrozen": self.n_unfrozen,
                "margin": margin,
                "train/epoch_loss": self.train_stats["loss"],
                **{f"valid/{k}": v for k, v in stage_stats.items()},
            })
            self._log_wandb_confusion("valid")
            # Save every epoch: keep the <ckpts_to_keep> most recent checkpoints
            # (default = number_of_epochs, i.e. all of them) plus the best-by-
            # ErrorRate. Lower ckpts_to_keep to cap disk.
            keep = int(getattr(self.hparams, "ckpts_to_keep",
                               self.hparams.number_of_epochs))
            self.checkpointer.save_and_keep_only(
                meta={"ErrorRate": stage_stats["ErrorRate"]},
                min_keys=["ErrorRate"],
                num_to_keep=keep,
                keep_recent=True,
            )
        elif stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                stats_meta={"epoch": epoch}, test_stats=stage_stats
            )
            # Final test metrics to W&B.
            self._wandb_log({f"test/{k}": v for k, v in stage_stats.items()})
            self._log_wandb_confusion("test")


def _stem(path):
    """Utterance id = audio filename without directory or extension.

    Matches vad_speech.py::clip_id (basename then splitext) so the ids here are
    the exact keys the problematic-clip CSVs were written with.
    """
    return os.path.splitext(os.path.basename(path))[0]


def _arrow_names(hf_name):
    """('ArabicSpeech/ADI17') -> (cache subdir 'ArabicSpeech___adi17', file 'adi17')."""
    org, repo = hf_name.split("/")
    low = repo.lower()
    return f"{org}___{low}", low


def _arrow_shards(cache_dir, hf_name, split):
    """Locate the prepared arrow shards for one corpus split.

    ADI-17 / ADI-20 were already built into an on-disk arrow cache by the
    original `load_dataset` download, at
        <cache_dir>/datasets/<Org>___<repo>/default/0.0.0/<hash>/<repo>-<split>-*.arrow
    We open those shards directly with `Dataset.from_file` (memory-mapped, zero
    copy) instead of going through `load_dataset`, which resolves against the
    single-valued (import-frozen) HF_HUB_CACHE and cannot serve two corpora in
    one process without re-downloading. The `cache-*.arrow` files in the same
    dir are map/filter intermediates and are deliberately NOT matched.
    """
    dir_name, prefix = _arrow_names(hf_name)
    base = os.path.join(cache_dir, "datasets", dir_name, "default", "0.0.0")
    shards = []
    for h in glob.glob(os.path.join(base, "*")):
        # Sharded splits -> "<prefix>-<split>-00000-of-000NN.arrow";
        # single-shard splits -> "<prefix>-<split>.arrow" (no -of- suffix).
        shards += glob.glob(os.path.join(h, f"{prefix}-{split}-*-of-*.arrow"))
        shards += glob.glob(os.path.join(h, f"{prefix}-{split}.arrow"))
    return sorted(shards)


def _hf_ids(dataset, audio_key, id_key):
    """Utterance ids aligned to rows: the `id` column if present, else the
    stem of audio.path. ADI-17 carries an `id` column (audio.path is None);
    ADI-20 has no id column and the id lives in audio.path."""
    if id_key and id_key in dataset.column_names:
        return list(dataset[id_key])
    import pyarrow.compute as pc

    paths = pc.struct_field(dataset.data.column(audio_key), "path").to_pylist()
    return [_stem(p) for p in paths]


def _load_problematic(hparams):
    """Set of low-speech clip ids to drop (VAD-flagged), filtered by reason."""
    if not hparams.get("drop_problematic", False):
        return set()
    reasons = hparams.get("drop_reasons")
    reasons = set(reasons) if reasons else None
    prob = set()
    for cf in hparams["drop_problematic_csvs"]:
        with open(cf, newline="") as f:
            for row in csv.DictReader(f):
                if reasons is None or row.get("reason") in reasons:
                    prob.add(row["id"])
    return prob


_ID_RANGE = re.compile(r"_(\d+)-(\d+)$")


def _dur_from_id(uid):
    """ADI-17 clip duration (s) parsed from its id `<ytid>_<startCS>-<endCS>`
    (frame numbers are centiseconds; verified to <0.1s vs decoded length)."""
    m = _ID_RANGE.search(uid)
    return (int(m.group(2)) - int(m.group(1))) / 100.0 if m else 0.0


def _adi17_source(hparams):
    """The hf_sources entry that is the ADI-17 corpus (has the `id` column)."""
    for src in hparams["hf_sources"]:
        if "ADI17" in src["name"].upper().replace("-", ""):
            return src
    return None


def _load_adi17_id_dialect(src):
    """(ids, dialects) for ADI-17 train, read straight from the arrow (no audio)."""
    from datasets import Dataset, concatenate_datasets

    shards = _arrow_shards(src["cache_dir"], src["name"], src["splits"]["train"])
    d = concatenate_datasets([Dataset.from_file(s) for s in shards])
    return list(d["id"]), list(d["dialect"])


def _manifest_durations(hparams):
    """id -> duration (s) from the 53h manifest CSV (ID,dialect,duration)."""
    col = hparams.get("subset_id_col", "ID")
    with open(hparams["subset_csv"], newline="") as f:
        return {r[col]: float(r["duration"]) for r in csv.DictReader(f)}


def _load_adi20_id_dialect(hparams):
    """(ids, dialects) for the ADI-20-repo train split; ids are audio.path stems."""
    from datasets import Audio, Dataset, concatenate_datasets

    src = _adi20_source(hparams)
    if src is None:
        return [], []
    shards = _arrow_shards(src["cache_dir"], src["name"], src["splits"]["train"])
    d = concatenate_datasets([Dataset.from_file(s) for s in shards])
    # decode=False keeps `audio` as {"bytes","path"}; without it this version of
    # `datasets` tries to build a torchcodec AudioDecoder just to read a column.
    d = d.cast_column(hparams["audio_key"], Audio(decode=False))
    return _hf_ids(d, hparams["audio_key"], src.get("id_key")), list(d["dialect"])


def _adi20_source(hparams):
    """The hf_sources entry that is the ADI-20 repo (the non-ADI-17 corpus)."""
    for src in hparams["hf_sources"]:
        if "ADI17" not in src["name"].upper().replace("-", ""):
            return src
    return None


def _adi20_durations(hparams):
    """id -> duration (s) for the ADI-20 repo's train split.

    The manifest CSV covers the ids it lists. Everything else needs measuring:
    unlike ADI-17, an ADI-20 id carries no usable timing. TUN ids are bare
    counters (`27_120_1`), MSA ids are UUIDs, and BAH's `_split_N` spans
    contradict the audio (an id claiming 28.359-33.6 decodes to 21.19s). So the
    remainder is read from the wav headers -- `Audio(decode=False)` + soundfile
    opens the container and reports its length without decoding a sample -- and
    cached to `adi20_duration_cache`. Delete that file to force a rebuild.

    Returns {} when no cache path is configured, which leaves the ADI-20-only
    dialects on their exact manifest ids (the pre-top-up behaviour).
    """
    cache = hparams.get("adi20_duration_cache")
    if not cache:
        return {}

    dur = _manifest_durations(hparams)
    if os.path.isfile(cache):
        with open(cache, newline="") as f:
            for r in csv.DictReader(f):
                d = float(r["duration"])
                if d > 0:  # -1 marks a clip whose header would not parse
                    dur[r["ID"]] = d
        logger.info(f"[subset] adi20 durations: {len(dur)} ids ({cache})")
        return dur

    logger.warning(
        f"[subset] {cache} missing -- measuring ADI-20 wav headers once "
        "(~20 min, then cached). Delete the file to redo it."
    )
    import io as _io
    from concurrent.futures import ThreadPoolExecutor

    import pyarrow.compute as pc
    from datasets import Audio, Dataset, concatenate_datasets

    src = _adi20_source(hparams)
    shards = _arrow_shards(src["cache_dir"], src["name"], src["splits"]["train"])
    d = concatenate_datasets([Dataset.from_file(s) for s in shards])
    d = d.cast_column(hparams["audio_key"], Audio(decode=False))
    ids = _hf_ids(d, hparams["audio_key"], src.get("id_key"))
    col = d.data.column(hparams["audio_key"])
    byts, paths = pc.struct_field(col, "bytes"), pc.struct_field(col, "path")
    todo = [(i, u) for i, u in enumerate(ids) if u not in dur]

    def measure(t):
        i, u = t
        b = byts[i].as_py()
        try:
            with sf.SoundFile(_io.BytesIO(b) if b is not None else paths[i].as_py()) as f:
                return u, round(len(f) / f.samplerate, 3)
        except Exception:
            return u, -1.0

    with ThreadPoolExecutor(max_workers=12) as ex:
        rows = list(ex.map(measure, todo, chunksize=64))
    os.makedirs(os.path.dirname(cache) or ".", exist_ok=True)
    with open(cache, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ID", "duration"])
        w.writerows(rows)
    dur.update({u: v for u, v in rows if v > 0})
    logger.info(f"[subset] measured {len(rows)} ADI-20 clips -> {cache}")
    return dur


def _build_keep_ids(hparams):
    """Set of utterance ids to train on.

    mode='exact'       : the 53h manifest ids minus low-speech clips. Only the
                         ~353k clips whose exact id is in the HF release survive
                         (the manifest's `_split_N` re-segmented clips are absent).
    mode='video_topup' : sample <subset_target_h> per dialect from ADI-17's own
                         train segments (shuffled on a fixed seed, low-speech +
                         <min_dur dropped) -- real HF audio, replacing the
                         manifest's `_split` clips that the release does not
                         ship. Set `subset_restrict_to_manifest_videos` to draw
                         only from the videos the manifest itself selected (the
                         original 53h behaviour). The ADI-20-only dialects
                         (BAH/TUN/MSA) start from their exact manifest ids and
                         top up from the rest of the ADI-20 train split, which
                         needs `adi20_duration_cache`.
    """
    id_col = hparams.get("subset_id_col", "ID")
    with open(hparams["subset_csv"], newline="") as f:
        manifest = {row[id_col] for row in csv.DictReader(f)}
    prob = _load_problematic(hparams)
    keep = manifest - prob
    logger.info(
        f"[subset] manifest ids={len(manifest)}  dropped low-speech="
        f"{len(manifest) - len(keep)}  keep(exact)={len(keep)}"
    )

    mode = hparams.get("subset_mode", "exact")
    if mode == "exact":
        return keep
    if mode != "video_topup":
        raise ValueError(f"unknown subset_mode={mode!r} (exact|video_topup)")

    src17 = _adi17_source(hparams)
    if src17 is None:
        logger.warning("[subset] video_topup: no ADI-17 source -> falling back to exact")
        return keep

    target_s = hparams.get("subset_target_h", 53.0) * 3600
    min_dur = hparams.get("subset_min_dur", 3.0)
    seed = hparams.get("subset_seed", hparams.get("seed", 1986))

    # The manifest's clips imply a set of source videos. Restricting the topup to
    # those videos keeps the 53h subset's channel/speaker mix, but it also caps
    # each dialect a few hours below what ADI-17 actually holds, and past ~53h it
    # only digs deeper into videos already represented. Off by default so
    # <subset_target_h> can draw on the whole train split.
    restrict = bool(hparams.get("subset_restrict_to_manifest_videos", False))
    keep_videos = {k[:11] for k in keep}          # youtube id = 11 chars
    ids17, dia17 = _load_adi17_id_dialect(src17)
    a17_ids = set(ids17)

    # ADI-17 segments that are not low-speech and are long enough -- grouped by
    # dialect for stratified sampling.
    cand = defaultdict(list)
    for uid, dia in zip(ids17, dia17):
        if restrict and uid[:11] not in keep_videos:
            continue
        if uid not in prob and _dur_from_id(uid) >= min_dur:
            cand[dia].append(uid)

    rng = random.Random(seed)
    final = set()
    real_h = defaultdict(float)   # hours of REAL audio kept, per dialect
    for dia, seg_ids in cand.items():
        rng.shuffle(seg_ids)
        acc = 0.0
        for uid in seg_ids:
            if acc >= target_s:
                break
            final.add(uid)
            acc += _dur_from_id(uid)
        real_h[dia] = acc
        logger.info(
            f"[subset] video_topup {dia}: {acc / 3600:.1f}h from ADI-17"
            f" (pool {sum(_dur_from_id(u) for u in seg_ids) / 3600:.1f}h)"
        )

    # ADI-20-only dialects (BAH/TUN/MSA): keep manifest ids not sourced from
    # ADI-17. (Truly-missing `_split` ids added here are inert -- load_combined_split
    # simply finds no matching row and drops them.)
    final |= {k for k in keep if k not in a17_ids}

    # ...and top those dialects up to the same target from the REST of the ADI-20
    # train split. Without this they are stuck at their ~53h manifest slice while
    # every ADI-17 dialect is sampled to <subset_target_h>: BAH/TUN would sit at
    # roughly half the prior of the other 18 classes. The corpus has the material
    # (~220h more BAH, ~87h more TUN); what it lacks is a duration in the id, so
    # this needs the measured table from `_adi20_durations`.
    dur20 = _adi20_durations(hparams)
    if dur20:
        ids20, dia20 = _load_adi20_id_dialect(hparams)
        have = defaultdict(float)
        for uid, dia in zip(ids20, dia20):
            if uid in final:
                have[dia] += dur20.get(uid, 0.0)
        pool = defaultdict(list)
        for uid, dia in zip(ids20, dia20):
            # Only dialects ADI-17 cannot serve; the rest are already at target.
            if dia in cand or uid in final or uid in prob:
                continue
            if dur20.get(uid, 0.0) >= min_dur:
                pool[dia].append(uid)
        for dia in have:
            real_h[dia] += have[dia]
        for dia, uids in pool.items():
            rng.shuffle(uids)
            acc = have[dia]
            n0 = len(final)
            for uid in uids:
                if acc >= target_s:
                    break
                final.add(uid)
                acc += dur20[uid]
            real_h[dia] = real_h[dia] - have[dia] + acc
            logger.info(
                f"[subset] adi20_topup {dia}: {have[dia] / 3600:.1f}h manifest "
                f"+ {len(final) - n0} clips -> {acc / 3600:.1f}h"
            )
    elif hparams.get("vc_data_folder"):
        logger.warning(
            "[subset] no adi20_duration_cache -> ADI-20-only dialects unmeasured; "
            "the VC cap will treat their real hours as 0"
        )

    # Published for load_vc_split, which fills each dialect's remaining hours.
    hparams["subset_real_hours"] = {d: h for d, h in real_h.items()}
    logger.info(f"[subset] video_topup total keep ids={len(final)}")
    return final


def load_vc_split(hparams, features, column_names):
    """Voice-conversion generated train audio as an HF dataset, or None.

    Layout: ``<vc_data_folder>/<vc_variant>/<DIALECT>/{metadata.jsonl,wavs/}``.
    Each jsonl line carries id / dialect / duration_s / source_id / target_voice;
    `clean` and `remix` are the same generated clips, dry and re-mixed with
    background noise, so `vc_variant` picks a rendering, never a different set.

    Sampled per dialect up to `subset_target_h` MINUS the real hours
    `_build_keep_ids` already kept, so real+VC lands on the target rather than
    overshooting it for dialects ADI-17 can nearly serve on its own. Dialects
    with no folder here get nothing -- real audio is all they have.

    Returned with `features`/`column_names` copied from the real train set so it
    concatenates directly. Rows are {"bytes": None, "path": ...}: the wavs are
    read from disk by `_read_chunk`, never embedded in the arrow table.
    """
    import pyarrow as pa
    from datasets import Dataset
    from datasets.info import DatasetInfo

    folder = hparams.get("vc_data_folder")
    if not folder:
        return None
    variant = str(hparams.get("vc_variant", "clean"))
    root = os.path.join(folder, variant)
    if not os.path.isdir(root):
        raise FileNotFoundError(
            f"vc_variant={variant!r} not found under vc_data_folder={folder!r} "
            f"(expected {root})"
        )
    min_dur = float(hparams.get("vc_min_dur", hparams.get("subset_min_dur", 3.0)))

    # Each dialect only takes the hours real audio could not supply, so the two
    # sources sum to <subset_target_h> instead of overshooting it. Absent real
    # hours (subset off, or an unmeasured dialect) means "take everything".
    real_h = hparams.get("subset_real_hours") or {}
    target_s = float(hparams.get("subset_target_h", 53.0)) * 3600
    seed = hparams.get("subset_seed", hparams.get("seed", 1986))

    paths, dialects, total_s = [], [], defaultdict(float)
    for dia in sorted(os.listdir(root)):
        meta = os.path.join(root, dia, "metadata.jsonl")
        if not os.path.isfile(meta):
            continue
        rows, n_short = [], 0
        with open(meta, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                if r["duration_s"] < min_dur:
                    n_short += 1
                    continue
                rows.append(r)

        budget_s = max(0.0, target_s - real_h.get(dia, 0.0)) if real_h else float("inf")
        random.Random(seed).shuffle(rows)
        n_cap = 0
        for r in rows:
            if total_s[r["dialect"]] >= budget_s:
                n_cap += 1
                continue
            # Rebuilt from `root`, not taken from r["wav_path"]: the recorded
            # path points at the clean/ tree of the machine that generated it.
            paths.append(os.path.join(root, dia, "wavs", r["id"] + ".wav"))
            dialects.append(r["dialect"])
            total_s[r["dialect"]] += r["duration_s"]
        logger.info(
            f"[vc] {dia}: real {real_h.get(dia, 0.0) / 3600:.1f}h + vc "
            f"{total_s[dia] / 3600:.1f}h = {(real_h.get(dia, 0.0) + total_s[dia]) / 3600:.1f}h"
            + (f" ({n_short} < {min_dur}s dropped)" if n_short else "")
            + (f" ({n_cap} over budget)" if n_cap else "")
        )

    if not paths:
        raise FileNotFoundError(f"no metadata.jsonl found under {root}")
    logger.info(
        f"[vc] variant={variant} dialects={len(total_s)} clips={len(paths)} "
        f"total={sum(total_s.values()) / 3600:.1f}h"
    )

    cols = {
        hparams["audio_key"]: pa.StructArray.from_arrays(
            [
                pa.array([None] * len(paths), type=pa.binary()),
                pa.array(paths, type=pa.string()),
            ],
            names=["bytes", "path"],
        ),
        hparams["label_key"]: pa.array(dialects, type=pa.string()),
    }
    table = pa.table({c: cols[c] for c in column_names})
    # Dataset(table, info=...) rather than from_dict: Audio.encode_example
    # imports torchcodec unconditionally, and the struct is already built.
    return Dataset(table, info=DatasetInfo(features=features))


def load_combined_split(hparams, which, keep_ids=None, return_ids=False):
    """Concatenate one logical split ('train'|'valid'|'test') across all
    hf_sources (ADI-17 + ADI-20), harmonized to [audio, dialect] with
    Audio(decode=False). If keep_ids is given (train only), each corpus is first
    filtered to rows whose utterance id is in the set.

    return_ids=True also returns the utterance ids aligned to the kept rows.
    They cannot be recovered afterwards -- harmonizing drops ADI-17's `id`
    column and its audio.path is None -- so the segment manifest, which joins on
    id, has to collect them here.
    """
    from datasets import Audio, Dataset, concatenate_datasets

    audio_key = hparams["audio_key"]
    label_key = hparams["label_key"]
    parts, part_ids = [], []
    for src in hparams["hf_sources"]:
        split = src["splits"][which]
        shards = _arrow_shards(src["cache_dir"], src["name"], split)
        if not shards:
            raise FileNotFoundError(
                f"No prepared arrow shards for {src['name']} split='{split}' "
                f"under {src['cache_dir']} -- was the dataset downloaded there?"
            )
        chunks = [Dataset.from_file(s) for s in shards]
        d = concatenate_datasets(chunks) if len(chunks) > 1 else chunks[0]

        n_full = len(d)
        if keep_ids is not None or return_ids:
            ids = _hf_ids(d, audio_key, src.get("id_key"))
            if keep_ids is not None:
                idx = [i for i, x in enumerate(ids) if x in keep_ids]
                d = d.select(idx)
                ids = [ids[i] for i in idx]
            part_ids.append(ids)

        drop = [c for c in d.column_names if c not in (audio_key, label_key)]
        if drop:
            d = d.remove_columns(drop)
        d = d.cast_column(audio_key, Audio(decode=False))
        parts.append(d)
        logger.info(
            f"[data] {src['name']} split='{split}' rows={len(d)}"
            + (f"/{n_full} (subset)" if keep_ids is not None else "")
        )
    out = concatenate_datasets(parts) if len(parts) > 1 else parts[0]
    if return_ids:
        return out, [uid for ids in part_ids for uid in ids]
    return out


def load_segment_table(hparams, train_ids):
    """Arrow table of manifest segments, joined to `train_ids` by utterance id.

    One row per training example: (row, start_s, dur_s, dialect), where `row`
    indexes the concatenated train dataset. The audio is NOT copied -- the
    pipeline random-accesses the arrow by `row` -- so 1.5M segments cost ~20MB
    rather than re-materializing every clip once per segment.

    Segments whose id is absent from the train set are skipped: a manifest built
    under different subset settings addresses clips this run did not keep, and
    silently training on the intersection would hide that. The count is logged,
    and a manifest that matches almost nothing raises.
    """
    import pyarrow as pa
    from datasets import Dataset

    path = hparams["segment_manifest"]
    id2row = {}
    for i, uid in enumerate(train_ids):
        id2row[uid] = i  # ids are unique across corpora (VC adds a __vN suffix)

    rows, starts, durs, dialects = [], [], [], []
    n_miss, n_total = 0, 0
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            n_total += 1
            i = id2row.get(r["ID"])
            if i is None:
                n_miss += 1
                continue
            rows.append(i)
            starts.append(float(r["start_s"]))
            durs.append(float(r["dur_s"]))
            dialects.append(r["dialect"])
    del id2row

    if not rows:
        raise ValueError(
            f"segment_manifest {path} matched 0 of {n_total} rows against the "
            f"{len(train_ids)} kept clips -- it was built for different subset "
            f"settings (see the .meta.json sidecar). Rebuild it."
        )
    if n_miss:
        logger.warning(
            f"[seg] {n_miss}/{n_total} manifest segments reference clips this "
            f"run did not keep -> skipped (stale manifest?)"
        )
    # The opposite direction is the silent one: a kept clip with no segment is
    # simply never trained on, with nothing in the manifest to notice it. A
    # handful is normal (clips shorter than the builder's min_seg); a large
    # share means the manifest was built for a different subset -- most often a
    # different `seed`, since subset_seed drives the video_topup sampling.
    n_uncovered = len(train_ids) - len(set(rows))
    if n_uncovered > 0.01 * len(train_ids):
        logger.warning(
            f"[seg] {n_uncovered}/{len(train_ids)} kept clips have NO segment "
            f"-> they will never be trained on. Check subset_seed/target_h "
            f"against {path}.meta.json and rebuild if they differ."
        )
    logger.info(
        f"[seg] {len(rows)} segments over {len(set(rows))} clips "
        f"({sum(durs) / 3600:.1f}h) from {path}"
    )
    table = pa.table(
        {
            "row": pa.array(rows, type=pa.int32()),
            "start_s": pa.array(starts, type=pa.float32()),
            "dur_s": pa.array(durs, type=pa.float32()),
            hparams["label_key"]: pa.array(dialects, type=pa.string()),
        }
    )
    return Dataset(table)


def dataio_prep(hparams):
    """Build the SB datasets from the combined ADI-17 + ADI-20 arrow caches.

    Train = the configured real-audio selection plus voice-conversion top-up;
    valid/test = the full concatenated dev/test splits of both corpora.
    """
    audio_key = hparams["audio_key"]
    label_key = hparams["label_key"]

    seg_manifest = hparams.get("segment_manifest")

    keep = _build_keep_ids(hparams) if hparams.get("use_subset", False) else None
    # The manifest joins on utterance id, and harmonizing drops ADI-17's `id`
    # column, so the ids have to come back with the dataset.
    train_hf, train_ids = load_combined_split(
        hparams, "train", keep_ids=keep, return_ids=True
    )
    valid_hf = load_combined_split(hparams, "valid")
    test_hf = load_combined_split(hparams, "test")

    # Synthetic voice-conversion audio, TRAIN ONLY -- valid/test stay real.
    vc_hf = load_vc_split(hparams, train_hf.features, train_hf.column_names)
    if vc_hf is not None:
        from datasets import concatenate_datasets

        n_real = len(train_hf)
        if seg_manifest:
            # VC rows carry their id in audio.path (no `id` column), and they
            # are appended after the real rows, so row index == position here.
            train_ids += _hf_ids(vc_hf, audio_key, None)
        train_hf = concatenate_datasets([train_hf, vc_hf])
        logger.info(
            f"[data] train real={n_real} + vc={len(vc_hf)} -> {len(train_hf)}"
        )

    logger.info(
        f"[data] combined train={len(train_hf)} "
        f"valid={len(valid_hf)} test={len(test_hf)}"
    )

    # ---- label encoder over the dialect strings (columnar read, no audio).
    label_encoder = sb.dataio.encoder.CategoricalEncoder()
    label_encoder.load_or_create(
        path=hparams["label_encoder_file"],
        from_iterables=[train_hf[label_key]],
    )
    n_dialects = len(label_encoder)
    if n_dialects != hparams["out_n_neurons"]:
        raise ValueError(
            f"out_n_neurons={hparams['out_n_neurons']} but the encoder found "
            f"{n_dialects} dialects across the combined train split -- fix the yaml."
        )

    target_sr = hparams["sample_rate"]
    train_n = int(hparams["sentence_len"] * target_sr)
    eval_n = int(hparams["eval_len"] * target_sr)

    # Variable-length train crop (LMFT): per clip, with prob crop_var_prob draw
    # the crop length uniformly from crop_len_range; otherwise use the fixed
    # crop_anchor_len. Clips shorter than the drawn length are returned whole
    # (padded + masked). Off in the main run (fixed sentence_len crops).
    var_crop = hparams.get("variable_crop", False)
    _r = hparams.get("crop_len_range", [2.0, 10.0])
    crop_lo, crop_hi = int(_r[0] * target_sr), int(_r[1] * target_sr)
    anchor_n = int(hparams.get("crop_anchor_len", 5.0) * target_sr)
    var_prob = float(hparams.get("crop_var_prob", 0.5))
    if var_crop:
        logger.info(
            "[crop] variable: anchor=%.1fs (p=%.2f) else uniform(%.1f, %.1f)s",
            anchor_n / target_sr, 1.0 - var_prob,
            crop_lo / target_sr, crop_hi / target_sr,
        )

    def wrap(d):
        # Columns already normalized to [audio, dialect] (no `id` to rename).
        return sb.dataio.dataset.DynamicItemDataset.from_arrow_dataset(d)

    valid_data = wrap(valid_hf)
    test_data = wrap(test_hf)

    # ---- audio pipelines: random crop for train, centered crop for eval.
    @sb.utils.data_pipeline.takes(audio_key)
    @sb.utils.data_pipeline.provides("sig")
    def train_audio_pipeline(audio):
        if var_crop:
            n = (
                random.randint(crop_lo, crop_hi)
                if random.random() < var_prob
                else anchor_n
            )
        else:
            n = train_n
        return _read_chunk(audio, n, target_sr, mode="random")

    @sb.utils.data_pipeline.takes(audio_key)
    @sb.utils.data_pipeline.provides("sig")
    def eval_audio_pipeline(audio):
        return _read_chunk(audio, eval_n, target_sr, mode="center")

    # Segment mode: one example per manifest window instead of one per clip, so
    # a dialect's share of updates tracks its HOURS rather than its clip count
    # (clip lengths differ ~5x across dialects) and every second of audio is
    # reachable each epoch instead of the one crop a clip yields. `row` indexes
    # train_hf, which stays the single owner of the audio.
    seg_jitter = float(hparams.get("segment_jitter_s", 0.0))

    @sb.utils.data_pipeline.takes("row", "start_s", "dur_s")
    @sb.utils.data_pipeline.provides("sig")
    def segment_audio_pipeline(row, start_s, dur_s):
        start = float(start_s)
        if seg_jitter > 0:
            # Restores the crop augmentation fixed windows would otherwise cost:
            # without it a clip yields the SAME few windows every epoch. Reads
            # clamp the start into the clip, so overshoot at either edge is safe.
            start += random.uniform(-seg_jitter, seg_jitter)
        return _read_chunk(
            train_hf[int(row)][audio_key],
            None,
            target_sr,
            mode="random",
            start_s=start,
            dur_s=float(dur_s),
        )

    if seg_manifest:
        train_data = wrap(load_segment_table(hparams, train_ids))
        train_data.add_dynamic_item(segment_audio_pipeline)
        logger.info(
            f"[seg] segment mode: {len(train_data)} examples/epoch "
            f"(jitter +-{seg_jitter:g}s)"
        )
    else:
        train_data = wrap(train_hf)
        train_data.add_dynamic_item(train_audio_pipeline)
    del train_ids
    sb.dataio.dataset.add_dynamic_item(
        [valid_data, test_data], eval_audio_pipeline
    )

    # ---- label pipeline: dialect string -> encoded index.
    @sb.utils.data_pipeline.takes(label_key)
    @sb.utils.data_pipeline.provides(label_key, "dialect_encoded")
    def label_pipeline(dialect):
        yield dialect
        yield label_encoder.encode_sequence_torch([dialect])

    datasets = [train_data, valid_data, test_data]
    sb.dataio.dataset.add_dynamic_item(datasets, label_pipeline)
    sb.dataio.dataset.set_output_keys(datasets, ["id", "sig", "dialect_encoded"])

    return train_data, valid_data, test_data, label_encoder


def transfer_from_voxceleb(folder, modules, include_classifier=False):
    """Recover ssl_model + embedding_model from a CA-MHFA checkpoint (partial).

    From the VoxCeleb speaker run (include_classifier=False) only the backbone +
    pooling transfer; the 20-class dialect head starts fresh. For the LMFT stage
    (include_classifier=True) `folder` is the MAIN ADI run's save dir, so the
    trained classifier is carried over too. Picks the best (min ErrorRate) ckpt,
    falling back to the most recent one if no ErrorRate meta exists.
    """
    recoverables = {
        "ssl_model": modules["ssl_model"],
        "embedding_model": modules["embedding_model"],
    }
    if include_classifier:
        recoverables["classifier"] = modules["classifier"]
    ckpter = Checkpointer(folder, recoverables=recoverables)
    found = ckpter.recover_if_possible(min_key="ErrorRate")
    if found is None:
        found = ckpter.recover_if_possible()
    if found is None:
        raise FileNotFoundError(
            f"No checkpoint found under pretrained_ca_mhfa_folder={folder}"
        )


def build_wandb_logger(hparams):
    """Build a SpeechBrain WandBLogger, or None if use_wandb is False.

    WandBLogger only calls wandb.init on the main process and its log_stats is
    main-process-only, so this is DDP-safe to build on every rank.

    Run name: if `wandb_run_name` is set, it is used as both the display name and
    the run id (id + resume="allow" -> re-running continues the SAME W&B run, in
    step with the checkpointer resume). If it is null/empty, name + id are omitted
    so W&B auto-generates a fresh random run name each launch.
    """
    if not hparams.get("use_wandb", False):
        return None

    import wandb

    from speechbrain.utils.train_logger import WandBLogger

    tracked = [
        "seed", "ssl_hub", "out_n_neurons", "number_of_epochs", "batch_size",
        "lr", "final_lr", "warmup_epochs", "weight_decay", "sentence_len",
        "eval_len", "freeze_ssl", "n_ssl_transformer_layers",
        "unfreeze_start_epoch", "unfreeze_rate", "subset_mode", "apply_augment",
    ]
    config = {k: hparams[k] for k in tracked if k in hparams}

    init_kwargs = dict(
        initializer=wandb.init,
        project=hparams["wandb_project"],
        entity=hparams.get("wandb_entity"),
        resume="allow",
        mode=hparams.get("wandb_mode", "online"),
        dir=hparams["output_folder"],
        config=config,
    )
    run_name = hparams.get("wandb_run_name")
    if run_name:  # explicit name -> stable id so a re-run resumes the same run
        init_kwargs["name"] = run_name
        init_kwargs["id"] = run_name
    # else: leave name/id unset -> W&B auto-generates a random run name + id.

    wl = WandBLogger(**init_kwargs)
    # Plot every series against the global optimizer_step so per-batch train
    # points and per-epoch valid points share one monotonic x-axis (no explicit
    # wandb step is ever passed -> no step collisions).
    if getattr(wl, "run", None) is not None:
        wl.run.define_metric("optimizer_step")
        wl.run.define_metric("*", step_metric="optimizer_step", step_sync=True)
    return wl


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True

    # Cap the cuFFT plan cache (default 4096 plans/device). AddReverb convolves
    # via FFT at size = padded batch length; with `variable_crop` that length is
    # a new arbitrary integer nearly every step, so two plans (rfft + irfft) get
    # cached per step and their workspaces grow the reserved VRAM without bound
    # until plan creation fails with CUFFT_INTERNAL_ERROR. A few plans is plenty
    # -- creation is cheap next to a training step, and the math is unchanged.
    for _dev in range(torch.cuda.device_count()):
        torch.backends.cuda.cufft_plan_cache[_dev].max_size = 16

    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    sb.utils.distributed.ddp_init_group(run_opts)

    with open(hparams_file, encoding="utf-8") as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # The two corpora are read straight from their prepared on-disk arrow caches
    # (Dataset.from_file, see load_combined_split) -- no hub lookup, no download.
    # Force offline so `datasets` never phones home for either repo's metadata.
    if hparams.get("hf_offline", True):
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

    # Build MUSAN (noise) + RIR (reverb) augmentation CSVs from local folders.
    if hparams["apply_augment"] and hparams.get("prepare_augment", True):
        run_on_main(
            prepare_aug_csv,
            kwargs={
                "folder": hparams["musan_folder"],
                "csv_file": hparams["noise_annotation"],
                "ext": hparams["augment_ext"],
                "max_length": hparams["noise_max_length"],
            },
        )
        run_on_main(
            prepare_aug_csv,
            kwargs={
                "folder": hparams["rir_folder"],
                "csv_file": hparams["rir_annotation"],
                "ext": hparams["augment_ext"],
            },
        )

    train_data, valid_data, test_data, label_encoder = dataio_prep(hparams)

    sb.core.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    dialect_brain = DialectBrain(
        modules=hparams["modules"],
        opt_class=hparams["opt_class"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )
    dialect_brain.wandb_logger = build_wandb_logger(hparams)
    # Ordered dialect names (index -> label) for the W&B confusion heatmap.
    dialect_brain.dialect_names = [
        label_encoder.ind2lab[i] for i in range(len(label_encoder))
    ]

    # Transfer the VoxCeleb backbone on a FRESH run only. On resume the ADI
    # checkpointer (in on_fit_start) recovers the in-progress weights afterwards,
    # so this init would be harmless anyway -- we skip it to avoid the extra read.
    pretrained = hparams.get("pretrained_ca_mhfa_folder")
    if pretrained and dialect_brain.checkpointer.find_checkpoint() is None:
        run_on_main(
            transfer_from_voxceleb,
            kwargs={
                "folder": pretrained,
                "modules": hparams["modules"],
                "include_classifier": hparams.get(
                    "pretrained_include_classifier", False
                ),
            },
        )

    dialect_brain.fit(
        dialect_brain.hparams.epoch_counter,
        train_data,
        valid_data,
        train_loader_kwargs=hparams["dataloader_options"],
        valid_loader_kwargs=hparams["dataloader_options"],
    )

    # Final dialect-ID error on the held-out combined test split (ADI-17 test +
    # ADI-20 test, best min-ErrorRate ckpt). NOTE: this uses the 5s center-crop
    # (eval_len), same as in-loop valid -- a cheap approximation. For faithful
    # official-format blind predictions, use predict_nadi_test.py with its 30s cap.
    dialect_brain.evaluate(
        test_data,
        min_key="ErrorRate",
        test_loader_kwargs=hparams["dataloader_options"],
    )
