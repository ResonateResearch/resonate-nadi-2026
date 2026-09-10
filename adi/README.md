# Arabic dialect identification

This directory contains the WavLM Base+ / context-aware multi-head factorized
attentive pooling (CA-MHFA) recipe for NADI 2026 Subtask 2. It includes training,
Silero VAD filtering, retained-window manifest generation, and official-format
blind prediction code for the documented **`submission_wavlm`** system. The
configuration preserves its archived seed-420 Base+ numerical settings. Data,
generated speech, manifests, label encoders, and model checkpoints are supplied
locally by the person running the recipe.

The reported official validation result is **94% accuracy and 0.05 C_avg on
10,806 utterances**. The reported blind-test result is **41% accuracy and 0.17
C_avg on 878 utterances**, ranking **7th**. Blind prediction uses the 30-second
inference cap. Prediction requires an explicit `--checkpoint` path and its
matching `dialect_encoder.txt`.

## Model and recorded configuration

WavLM Base+ produces 13 hidden-state streams (the convolutional feature state
and 12 Transformer states), each with 768 features. CA-MHFA learns separate
softmax mixtures across those 13 streams for keys and values. Both streams are
compressed to 128 features; 64 attention heads use a nine-frame context (the
current encoder frame plus four frames on either side, with zero padding) and
produce a 256-dimensional utterance embedding for the 20-class dialect head.
Padding is masked during pooling. The pooling implementation is local in
[`ca_mhfa.py`](ca_mhfa.py).

[`hparams/train_wavlm_base_plus.yaml`](hparams/train_wavlm_base_plus.yaml) retains
these recorded settings:

| Setting | Value |
| --- | --- |
| Seed / input batch size | 420 / 64 |
| Front end | `microsoft/wavlm-base-plus`, supplied as local model files |
| Optimizer | AdamW, peak LR `1e-4`, final LR `1e-7`, weight decay `0.01` |
| LR schedule | Five-epoch linear warmup, then cosine decay; one cycle |
| SSL unfreezing | Three top-down blocks per epoch from epoch 2; all 12 by epoch 5 |
| Convolutional feature extractor | Frozen throughout |
| Pre-encoder parameter bucket | `top` |
| AAM loss | Margin `0.2`, scale `32`; margin ramps from zero through epoch 4 |
| Train windows | 5-second non-overlapping windows; retain tails at least 2 seconds |
| Train start jitter | Uniform in `[-2.5, 2.5]` seconds, clamped to the clip |
| In-loop validation/test crop | One central crop of up to 5 seconds |
| Standalone blind prediction | One central crop of up to 30 seconds |

The documented final training hardware is one NVIDIA H100. The launch below
uses one process and BF16 precision. The configured augmentation concatenates
the original batch with one augmented copy, so the network can see twice the
input batch size. Each augmented copy receives one or two randomly ordered
operations from MUSAN noise (0–15 dB SNR), RIR reverberation, speed perturbation
(90/100/110% in the inspected SpeechBrain implementation), frequency dropping,
and time-chunk dropping. Augmentation runs in FP32 inside BF16 training.

Although the trainer supports optional transfer from a VoxCeleb CA-MHFA
checkpoint, `pretrained_ca_mhfa_folder: false` in this run means that no such
checkpoint is transferred. WavLM starts from Base+ weights; pooling and the
dialect classifier start fresh. Variable-length crop/extra fine-tuning options
remain in the trainer but are disabled in this configuration.

## Dependencies

Use a Python environment with matching PyTorch and torchaudio builds, plus the
packages in [`requirements.txt`](requirements.txt), and SpeechBrain. This file is
a dependency inventory, **not a validated version lock**. The source guide uses
Python 3.10 and an editable SpeechBrain checkout. The inspected SpeechBrain
revision was `e5cb1f65b940634215650aa1171e0440d0808123`.

Use a SpeechBrain checkout supporting `DynamicItemDataset.from_arrow_dataset`,
`seed_everything`, `Brain.optimizers_step`, checkpoint `keep_recent`, and the
checkpoint directory helpers `_is_checkpoint_dir` / `_construct_checkpoint_objects`.
The WavLM wrapper must support `output_all_hiddens=True`; augmentation uses
`replicate_labels`. The custom pooling module is bundled locally.

For an already obtained compatible checkout, a setup pattern is:

```bash
python -m pip install -e "$SPEECHBRAIN_SOURCE"
python -m pip install -r requirements.txt
```

Choose package and CUDA versions for your environment. SoundFile needs working
libsndfile support.
Arrow audio is cast to `Audio(decode=False)` and read with SoundFile, avoiding a
TorchCodec decoder requirement in this path. The optional W&B logger is disabled
and has no account setting. All public entry points set Hugging Face offline
flags and disable implicit Hub-token loading before importing model libraries.
VAD additionally needs `silero-vad`, `matplotlib` and `tqdm`; these are only used
by the preparation script. It loads the installed package's CPU model.

## Prepare local inputs

Acquire ADI-17, the ADI20 corpus, WavLM Base+ assets, MUSAN and RIRS_NOISES under
their respective access and licensing terms. Dataset identifiers used by the
recipe are `ArabicSpeech/ADI17` and `ArabicSpeech/ADI20`. The loader reads
already-prepared Hugging Face Arrow caches directly; it does not download data.
Keep one prepared revision of each corpus in its cache root. It matches:

```text
<cache_dir>/datasets/<Org>___<repo>/default/0.0.0/<hash>/<repo>-<split>.arrow
<cache_dir>/datasets/<Org>___<repo>/default/0.0.0/<hash>/<repo>-<split>-*-of-*.arrow
```

Here `<repo>` is lowercase (`adi17` or `adi20`). ADI-17 uses `train/dev/test`
splits and the `id` column for utterance IDs. ADI20 uses
`train/validation/test`; its IDs are stems of `audio.path`. Both provide
`audio` and string `dialect` fields. The combined label space has 20 classes;
prepare the exact corpus revisions and class mapping intended for the run.
Audio used for the released crop lengths should be 16 kHz mono. The reader
averages multiple channels and can resample, but its non-manifest crop bounds
assume the source sample rate matches the configured 16 kHz.

The default paths are relative to the required `--data_root`:

```text
ADI_DATA_ROOT/
  ADI17/
    datasets/...
    train_problematic.csv
  ADI20/
    datasets/...
    adi20_53h.csv
    train_problematic.csv
    adi20_durations.csv
  voice_conversion/clean/<DIALECT>/
    metadata.jsonl
    wavs/<id>.wav
  musan/...
  RIRS_NOISES/simulated_rirs/...
  adi20_segments_5s.csv
```

The problematic lists and last file are generated by the commands below. Obtain
the published `adi20_53h.csv` from `ArabicSpeech/ADI20`; it supplies the seed IDs
for subset selection. You can override individual YAML paths when your layout
differs. `--ssl_hub` must identify your
local directory of WavLM Base+ model/configuration files; no model is fetched by
the recipe.

The selection/preparation inputs have the following schemas:

| File | Required fields and behavior |
| --- | --- |
| `adi20_53h.csv` | CSV with `ID`, `duration` in seconds; retain the original `dialect` metadata too. IDs must agree with the Arrow data. |
| Each `train_problematic.csv` | CSV with `id`, `reason`; IDs with reasons `empty`, `severe`, or `low` are excluded from training. |
| `adi20_durations.csv` | CSV with `ID`, `duration` in seconds. Missing cache is measured from local ADI20 audio headers and written once; negative/zero cached values are excluded. |
| VC `metadata.jsonl` | One JSON object per generated WAV, with `id` (string without extension), `dialect` (matching directory), `duration_s` (number). |

Real training selection uses `video_topup` with a target of 100 hours per
dialect and seed 420. It can sample from the whole ADI-17 train split
(`subset_restrict_to_manifest_videos: false`), drops the supplied low-speech
IDs, and rejects source clips shorter than one second. ADI20 dialects not
served by ADI-17 are topped up using measured durations. Same-dialect voice
conversion fills the remaining per-dialect budget where generated audio is
available; insufficient real-plus-generated supply can leave a class below
the target. Selection does not alter validation/test splits.

The released [`vad_speech.py`](vad_speech.py) implements the preparation guide's
Silero filtering. It uses speech probability threshold 0.5, minimum speech span
250 ms and minimum silence span 100 ms. Clips are `empty` if speech is under
0.5 seconds or under 10% of duration, `severe` below 30%, and `low` below 50%.
Training excludes all three categories, retaining clips with at least 50%
speech and at least 0.5 seconds of speech. Decoding errors are labelled
`decode_fail`; that label is not among this configuration's exclusion reasons.
The VAD script decodes audio on CPU; subsequent duration measurement reads
audio headers, and window generation uses metadata. No stage writes cropped
copies of the source corpus. Validation and test are not filtered.

The paper reports **1,883.89 nominal retained-segment hours** after filtering
and windowing across all 20 dialects: **1,140.44 hours of
real ADI-17**, **296.98 hours of real ADI20**, and **446.47 hours of
voice-converted speech**. These are retained-window totals, not the sizes of
the original source corpora. The local manifest is rebuilt from your input
metadata and the retained-window rule; reproducing those totals requires the
matching input selection and generated audio.

An illustrative VC metadata row (not a dataset record) is:

```json
{"id":"example__v1","dialect":"ALG","duration_s":5.0,"source_id":"example","target_voice":"voice_example"}
```

The loader uses `id`, `dialect`, and `duration_s`; retaining `source_id` and
`target_voice` supports local generation provenance. IDs must be unique across
real and converted corpora. WAV paths are reconstructed from the chosen root,
variant and ID; stored machine-specific `wav_path` values are ignored. `clean`
and `remix` denote alternate renderings of the same clips and must not be counted
as additional data by combining both variants.

**The original VAD lists, generated speech and an ADI-matched standalone
voice-conversion generator are not bundled.** The VAD lists can be regenerated
below. Exact VC regeneration still requires the matching source/target selection,
preprocessing, conversion model and dependencies. Supply that pipeline's local
outputs. Setting `vc_data_folder: null` runs a real-audio-only variant; it changes
the reported setup and the resulting segment counts and hours.

## Filter speech, build windows and train

Run from this `adi/` directory. Set `ADI_DATA_ROOT` and `WAVLM_BASE_PLUS` in your
shell to the local input/model directories described above.

```bash
python vad_speech.py adi17 adi20 --data_root "$ADI_DATA_ROOT" --num_workers 64

python make_segment_manifest.py hparams/train_wavlm_base_plus.yaml \
  --data_root "$ADI_DATA_ROOT" --ssl_hub "$WAVLM_BASE_PLUS" \
  --out "$ADI_DATA_ROOT/adi20_segments_5s.csv" \
  --seg_len 5 --hop 5 --min_seg 2

python train_ca_mhfa_adi.py hparams/train_wavlm_base_plus.yaml \
  --data_root "$ADI_DATA_ROOT" --ssl_hub "$WAVLM_BASE_PLUS" \
  --device cuda:0 --precision bf16 --eval_precision bf16
```

VAD processes `train` by default, writing `train_problematic.csv`, diagnostic
plots and per-dialect statistics in each corpus directory. Lower
`--num_workers` for smaller CPU hosts. Its offline dataset loader needs the
locally prepared Hugging Face caches described above. Exact original selections
also depend on matching corpus revisions and the original Silero model/version.

The segmenter uses the same real/VC selection as training. It retains complete
5-second windows and tails of at least 2 seconds; clips shorter than 2 seconds
do not yield a training window. It writes `ID,dialect,source,path,start_s,dur_s`
and a `.meta.json` sidecar recording the selection settings. Rebuild when input
metadata, seed, target hours, filtering or VC variant changes. Do not enable
the optional `--balance` or `--max_per_dialect` caps for this configuration.
Those would further trim the retained windows.

Loading this full HyperPyYAML configuration constructs the model and
augmentation objects even for the segmenter; local WavLM files and the listed
dependencies are therefore needed for that script too. It performs no model
forward pass. If the duration cache is absent, it first measures audio headers.
During training, augmentation CSVs are built from the local MUSAN and RIR
folders. Checkpoints, encoder and logs are written below
`results/ca_mhfa_adi20/submission_wavlm/` by default. Reusing an existing output
directory can resume its checkpoints; choose a new `--output_folder` for a fresh run.

The trainer evaluates the combined ADI corpus development/test splits with
5-second central crops and selects by development error rate. These internal
metrics do not replace the official NADI validation/blind evaluations.

## Predict with a supplied checkpoint

Prepare a local Hugging Face `Dataset.save_to_disk` export of the official blind
data (`UBC-NLP/NADI2026_subtask2_adi_test`), preserving its original row order.
It must contain `audio` and `idx`; audio paths/bytes must resolve locally.
For a saved DatasetDict, the script selects `train` by default (the task's blind
data packaging), or accepts `--dataset_split`. Match the official evaluation
revision; the paper's reported blind set contains 878 utterances, and the script
does not silently truncate or enforce that count.

Set `ADI_CHECKPOINT` to the local SpeechBrain checkpoint directory you want to
evaluate, containing `CKPT.yaml` and
the `ssl_model`, `embedding_model`, `classifier`, and epoch-counter recoverables
expected by the configuration. Set `ADI_ENCODER` to that run's
`dialect_encoder.txt`, and `NADI_BLIND_DATASET` to the saved dataset directory.

```bash
python predict_nadi_test.py hparams/train_wavlm_base_plus.yaml \
  --data_root "$ADI_DATA_ROOT" --ssl_hub "$WAVLM_BASE_PLUS" \
  --checkpoint "$ADI_CHECKPOINT" --label_encoder_file "$ADI_ENCODER" \
  --dataset "$NADI_BLIND_DATASET" --output_folder results/blind_prediction \
  --device cuda:0 --eval_precision bf16
```

Prediction does not read the training corpora or run augmentation, although it
constructs their configured objects. It applies one central 30-second cap per
utterance, with shorter clips used whole, and emits scaled cosine logits in
the official order:

```text
MSA BAH TUN ALG EGY IRA JOR KSA KUW LEB LIB MAU MOR OMA PAL QAT SUD SYR UAE YEM
```

Under the requested output directory it writes a checkpoint-named submission
directory. `submission.zip` contains `logits.tsv` (20 tab-separated numbers per
row) and `predictions.tsv` (one zero-based index per row), both without headers
or IDs. `predictions.csv` and `summary.json` are separate local diagnostics.
No official scoring implementation or blind reference labels are included.

## Attribution

The trainer retains its SpeechBrain/VoxCeleb recipe ancestry and inherited
author credits. The CA-MHFA module retains its original reference and
attribution. See the repository's
[`THIRD_PARTY.md`](../THIRD_PARTY.md) and
[`SpeechBrain Apache-2.0 license`](../licenses/SpeechBrain-Apache-2.0.txt).
