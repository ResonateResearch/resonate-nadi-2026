# ASR data preparation

Run these commands from the repository root. Obtain the NADI source data
separately under its applicable terms, and keep
all generated audio, manifests, embeddings, and model files in the runtime work
directories. No external labeled speech is used by the commands below.

The preparation scripts build the E11/E15/E16 data streams. See
[vc/README.md](vc/README.md) for the ReDimNet voice-conversion workflow.
Supply denoised audio and a NADI noise pool separately; the original
denoising/full-chain package is proprietary. See
[third-party terms](../THIRD_PARTY.md). E17 requires a separately prepared
239,615-row manifest, as described below.

## Environment and input interfaces

Use the ASR environment described in [README.md](README.md). CPU preparation also
needs `numpy`, `scipy`, `soundfile`, `soxr`, and `pyarrow`. The CTC aligner needs
`torch`, `transformers`, and `jiwer`, and uses one visible CUDA device. Voice
conversion uses a separate environment with compatible `torch`/`torchaudio`
versions; see [vc/README.md](vc/README.md).

```bash
export NADI_ASR_WORKSPACE="$PWD/work/asr"
export NADI_VC_WORKSPACE="$PWD/work/vc"
export NADI_SOURCE_ROOT="/path/to/NADI2026_subtask1.1_Robust_ASR"
export NADI_DERIVED_ROOT="/path/to/nadi-derived-versions"
export NADI_NOISE_POOL="/path/to/nadi-noise-pool"
```

The required external interfaces are:

| Input | Layout and contract |
| --- | --- |
| NADI source | `<source>/<Dialect>/train-*.parquet` and `validation-*.parquet`; columns `id`, `transcription`, and `audio` containing `bytes` |
| Denoised NADI train | `<derived>/clean-neural/<Dialect>/train-*.parquet`, same columns and train IDs as the original data; decoded audio must be mono 16 kHz with the original durations |
| Optional recovered full-chain versions | `<derived>/<version>/<Dialect>/train-*.parquet`, with `version` one of `aug-additive`, `aug-fullchain`, or `aug-channel`; these are export inputs, not generators provided here |
| NADI noise pool | `<noise-pool>/segments/<Dialect>/*.flac`, containing suitable NADI-derived noise segments; additive augmentation uses only the selected dialect's files |

`Dialect` is one of Algeria, Egypt, Jordan, Mauritania, Morocco, Palestine, UAE,
and Yemen. The excluded proprietary package produced the denoised/full-chain
versions and noise pool. Its extraction, denoiser selection, residual processing,
and full-chain composition cannot be reproduced by `augment.py` alone. Do not
substitute a different generator and call its output an exact reconstruction.

The denoised training views were produced with ClearerVoice-Studio's
`MossFormerGAN_SE_16K`, selected by a blind listening pilot. The historical noise
pool contains 62,965 segments, with checks for speech leakage. This release
consumes those externally prepared inputs; the excluded wrapper also performed
noise extraction and procedural room-response/full-chain processing.

## Export the NADI audio

```bash
python asr/preparation/prepare_nadi.py \
  --nadi-root "$NADI_SOURCE_ROOT" \
  --derived-root "$NADI_DERIVED_ROOT" --versions clean-neural
```

This creates `audio/original/<Dialect>/*.flac`,
`audio/clean-neural/<Dialect>/*.flac`, `audio/original_val/<Dialect>/*.flac`, and
the corresponding `manifests/nadi_original.jsonl`, `nadi_clean_neural.jsonl`, and
`nadi_val_eval.jsonl`. Training records have `audio` and
`text: "language Arabic<asr_text>TRANSCRIPT"`; evaluation records have `utt_id`,
`audio`, `refs`, `domain`, and `dataset`. Audio paths are absolute. Audio bytes
that already encode FLAC or WAV are preserved, as in the recovered exporter;
therefore a `.flac` filename can contain a WAV header.

`nadi_dev.jsonl` is a deterministic sample of the first 38 nonempty validation
records per dialect. It is a convenient NADI-only probe, not a reconstruction of
the historical duration-matched 512-item probe.

Omit `--derived-root` to export only the original source and validation data. To
export separately obtained full-chain variants, add their names to `--versions`.
Those optional manifests are not used by the E11 recipe below.

## Generate the voice-conversion pool and E11 base

Supply an existing within-dialect pool with the interface
`<vc-root>/<Dialect>/wavs/<source_id>__v<index>.wav`, or follow
[vc/README.md](vc/README.md) to generate it using the clean-neural ReDimNet voice
bank. The source ID must identify the corresponding original training transcript;
the base builder constructs WAV paths from this tree instead of relying on
possibly stale paths in external VC metadata. Point the base builder explicitly
at the pool:

```bash
python asr/preparation/build_e11.py \
  --vc-root "$NADI_VC_WORKSPACE/exp/clean_neural__redimnet__thr0.45_coh0.7/gen/same_n10" \
  --workers 16
```

The builder repeats original and clean-neural rows three times, samples up to
7,500 VC rows per dialect with seed 20260717, and applies an order-10 Butterworth
low-pass filter at 2.5 kHz to the selected VC audio. It writes
`manifests/E11_base.jsonl` and 6,125 original-audio augmentation-source rows per
dialect under `manifests/aug_src_e11/`. Transcript lookup follows the recovered
source-ID rule: the generated filename starts with `<source_id>__`.

## Add the waveform augmentation view

The recovered E11 builder describes dialect-matched noise, 2.5 kHz low-pass
channel augmentation, and speed factors 0.9/1.1. Exactly one of these three
operations is selected per row; every output receives the same 2.5 kHz final
low-pass filter. Historical per-dialect launch commands and SNR overrides were
not recovered. The example below uses the augmenter's implementation defaults,
SNR U(0,20) dB, explicitly as a configurable example; it is not an exact
historical-launch reconstruction. The original E11 docstring's suggested
U(14,30) override is not established as the setting of the final run.

```bash
for dialect in Algeria Egypt Jordan Mauritania Morocco Palestine UAE Yemen; do
  NOISE_DIALECT="$dialect" AUG_SNR_LO=0 AUG_SNR_HI=20 \
    CHANNEL_LP_HZ=2500 FINAL_LP_HZ=2500 \
    python asr/preparation/augment.py \
      --src "$NADI_ASR_WORKSPACE/manifests/aug_src_e11/$dialect.jsonl" \
      --out-audio "$NADI_ASR_WORKSPACE/audio/E11_aug/$dialect" \
      --out-manifest "$NADI_ASR_WORKSPACE/manifests/E11_aug_$dialect.jsonl" \
      --augs additive channel speed --seed 42 --workers 16
done

python asr/preparation/assemble_base.py \
  --inputs "$NADI_ASR_WORKSPACE/manifests/E11_base.jsonl" \
    "$NADI_ASR_WORKSPACE"/manifests/E11_aug_*.jsonl \
  --out "$NADI_ASR_WORKSPACE/manifests/E11_dial.jsonl"
```

The augmentation CLI's seed default is retained; the exact historical per-dialect
launch commands were not recovered. Noise-file enumeration follows the original
filesystem glob order, and multiprocessing writes completed rows in completion
order. Those properties limit byte-for-byte replay across environments.

`assemble_base.py` concatenates inputs and projects each
record to `audio`, `text`, and `dialect` using the trainer's dialect resolver. It
preserves repeated training rows. Supply `--expected-rows 185798` to require the
historical base size. The design total is 185,800; the recovered manifest had
185,798 rows. This package does not discard two arbitrary rows to force agreement.
The recovered augmentation module also contains MUSAN/RIR branches from other
experiments; they require their separate external corpora and are not selected by
the command above. Missing requested pools now fail before augmentation starts.

## Build and filter short clips

```bash
CUDA_VISIBLE_DEVICES=0 python asr/preparation/align_ctc.py \
  --manifest "$NADI_ASR_WORKSPACE/manifests/nadi_original.jsonl" \
  --out "$NADI_ASR_WORKSPACE/manifests/nadi_original.align.jsonl"

python asr/preparation/build_shortclips.py --dry-run
python asr/preparation/build_shortclips.py --workers 16
python asr/preparation/build_e15.py
python asr/preparation/build_e16b.py --mode filtered --check-audio
```

Alignment uses `jonatasgrosman/wav2vec2-large-xlsr-53-arabic`, the recovered NumPy
Viterbi aligner, and the 30 ms timestamp correction. The supplied reference text
is preserved in output word spans. The short-clip builder uses only original
training audio, gates alignment scores at each dialect's tenth percentile,
matches the recovered validation-derived duration/word-count quotas, limits each
source to six spans, and caps word-span Jaccard overlap at 0.65. Its measured
quota constants remain embedded in the source. Validation audio is never a cut
source.

The cut geometry and filters are preserved: 35 ms outward padding around
inter-word gap midpoints, 50 ms at actual utterance edges, whole-source 2.5 kHz
filtering before slicing, and a 6 ms raised-cosine edge fade. `build_e15.py`
assembles the unfiltered short-clip extension. `build_e16b.py` retains a clip only
when the weakest artificial-boundary gap is at least **60 ms**, and both its first
and last words have at least **40 ms** aligned extent. It applies the original
floating-point tolerance of 1e-9; true utterance edges use the 9.0 gap sentinel.

The recovered strict count checks are:

| Manifest | Historical rows |
| --- | ---: |
| `E11_dial.jsonl` | 185,798 |
| `E15_shortclips.jsonl` | 21,381 |
| `E15.jsonl` / `E16_all.jsonl` (`--mode identity`) | 207,179 |
| `E16.jsonl` (`--mode filtered`) | 201,215, including 15,417 retained short clips |
| `E16_noclip.jsonl` (`--mode none`) | 185,798 |

`build_e16b.py` also checks the recovered per-dialect counts. A fresh input pool
may fail those checks; inspect the provenance/count difference before changing
them. They are retained as checks on the historical recipe, not guarantees that
newly generated inputs reproduce identical counts.

The preparation filename `build_e16b.py` builds the gated **E16 data stream**.
The **E16b training run** used the E14/E11 base manifest; it did not train on
`E16.jsonl`. These names refer to different parts of the experimental workflow.

## E17 inputs

The separately reported E17 model used 239,615 training rows. To train E17,
supply its separately prepared manifest; the commands above do not build it.
Neither `E16.jsonl` nor a renamed copy is the E17 training set. E17 is also
distinct from the submitted ROVER ensemble; see the [ASR README](README.md)
for that distinction and training entry points.

Pin external source and model-weight revisions when preparing the environment.
The [VC README](vc/README.md) describes the kNN-VC default pin and ReDimNet
configuration.
