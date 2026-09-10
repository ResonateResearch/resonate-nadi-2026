# Robust dialectal Arabic ASR

NADI 2026 Subtask 1.1, eight country varieties. The submitted system is a nine-checkpoint, pivot-anchored ROVER ensemble followed by country-specific orthographic canonicalization. It is distinct from the strongest single model, E17.

## Scope and setup

The ASR workflow includes data preparation, voice conversion, training, decoding, scoring, ROVER combination and canonicalization.

| Directory | Entry points |
|---|---|
| `preparation/` | Export NADI audio; assemble repeated, converted, augmented, and CTC short-clip training streams |
| `vc/` | Prepare the speaker bank and generate within-country conversions with external ReDimNet2 and kNN-VC models |
| `train/` | Fine-tune the ASR model and rank checkpoints using a supplied validation probe |
| `scripts/` | Decode checkpoints, score hypotheses, combine nine voters, canonicalize, and export country files |
| `tests/` | Synthetic text-pipeline checks without competition data or model execution |

Supply trained checkpoints locally. Denoising/full-chain preparation requires separately authorized tooling; see [PREPARATION.md](PREPARATION.md) for the preparation stages and required inputs.

Run the commands below from the repository root. Use a separate environment from ADI:

```sh
python3 -m venv .venv-asr
. .venv-asr/bin/activate
# Install torch for your CUDA environment first.
python -m pip install -r asr/requirements.txt
python -c 'from transformers import CohereAsrForConditionalGeneration'
export NADI_ASR_WORKSPACE="$PWD/work/asr"
```

Saved model configurations specify Transformers 5.12.0; the requirements file is not a complete version lock. Use a compatible Transformers build providing the class checked above. VC requires additional upstream packages/models, described in the preparation guide.

## Data organization and manifests

Obtain the official training, validation and blind-test material through the organizers. Do not redistribute it through this repository. Country identity comes from the released country-organized task structure; it is not predicted from audio. Keep every official split separate.

The workflow uses local JSONL manifests:

- Training: `audio` is a readable local audio path, `text` is `language Arabic<asr_text>` followed by the reference transcript, and `dialect` is the country name. Repeated rows are meaningful sampling weights.
- Evaluation: `utt_id`, `audio`, `refs` (a one-element reference list), `dataset` (country), and optionally `domain`.
- Blind evaluation: the same decoder schema can use `refs: [""]`; this is an empty reference placeholder, **not** a gold transcript. Do not interpret local metrics for these records.
- Submission ordering: additionally preserve a zero-based `row` within each country's original released order. `build_submission.py` consumes this ordering rather than sorting utterance IDs.

The paper uses 12,800 original training clips and 10,808 ASR validation utterances. The advertised 13.6 hours and header-measured 15.3 hours refer to the same training release. A separately supplied duration-matched 512-utterance validation probe is used during training; substituting another probe can change checkpoint rankings.

Audio loading downmixes to mono and resamples to 16 kHz. The ASR loader truncates audio beyond 30 seconds **from the start**; do not confuse it with ADI's central-crop rule.

## Preparation and augmentation

Follow [PREPARATION.md](PREPARATION.md) for runnable supported stages. The submitted training runs use derivatives of the organizer training clips:

- Real clips and one neural-denoised waveform per clip are repeated three times in the base manifest; these are literal repeated rows. The denoised version used ClearerVoice-Studio's `MossFormerGAN_SE_16K`, selected in a blind listening pilot; its historical processing wrapper is external to this release.
- kNN-VC changes speaker characteristics using source and target material from the same country. The selected mixed stream has 7,500 converted clips per country, 60,000 total, low-passed with an order-10 Butterworth filter at 2.5 kHz. The isolated-stream experiment used 124,457 clips instead.
- In-corpus waveform augmentation chooses additive corpus noise, channel filtering or speed perturbation with equal probability; outputs are band-matched. The recorded mixed stream contains 48,998 successful rows.
- Forced CTC alignment supplies timestamps for contiguous words of the original references. The short-clip builder retains 21,381 clips for E15. The boundary-gated stream has 15,417 clips, using at least 60 ms internal gaps and 40 ms edge-word extents.
- E17 additionally uses 38,400 additive/channel/full-chain views. Its total is 239,615 rows. E17 is **not** a submitted ROVER voter, and unavailable full-chain preparation is not silently replaced with the simpler one-transform augmenter.

The general augmenter also supports MUSAN/recorded-RIR modes used in earlier comparisons. Those modes are not equivalent to the in-corpus-only final mixed recipe. Supply authorized local resources for the modes actually selected.

## Training and configuration mapping

The base model is `CohereLabs/cohere-transcribe-arabic-07-2026`: FastConformer encoder plus autoregressive Transformer decoder. The paper fully fine-tunes it, with no adapters. General trainer modes for other workflows remain available in source but are not part of the submitted method.

| Run | Training rows / recipe | Country context | Input batch | Accumulation |
|---|---|---|---:|---:|
| E11 | 185,798-row band-matched base pool | Off | 16 | 1 |
| E14 | Same pool as E11 | On | 16 | 3 |
| E15 | E14 pool plus all CTC short clips: 207,179 | On | 12 | 2 |
| E16b | Rerun of E14's manifest | On | 16 | 3 |
| E17 | Gated short clips plus additional views: 239,615 | On | 24 | 1 |

Training uses two epochs, bf16 precision, and one NVIDIA H100 GPU per run. Effective batch is input batch multiplied by accumulation. The ASR seed is 42. The trainer explicitly sets that seed and does not initialize remote WandB reporting.

Example invocation for the documented E14 configuration, after preparing its manifest and the validation probe:

```sh
CUDA_VISIBLE_DEVICES=0 python asr/train/cohere_asr_sft.py \
  --model_id CohereLabs/cohere-transcribe-arabic-07-2026 \
  --train_files "$NADI_ASR_WORKSPACE/manifests/E11_dial.jsonl" \
  --dev_subset "$NADI_ASR_WORKSPACE/manifests/nadi_dev_only.jsonl" \
  --output_dir "$NADI_ASR_WORKSPACE/runs/E14" \
  --stage full_ft --language ar --dialect_cond 1 \
  --batch_size 16 --grad_acc 3 --epochs 2 \
  --lr 5e-5 --encoder_lr 2e-5 --head_lr 1.25e-5 \
  --label_smoothing 0.1 --weight_decay 0.01 --adam_beta2 0.98 \
  --max_grad_norm 1.0 --lr_scheduler_type cosine --warmup_ratio 0.03 \
  --gradient_checkpointing 0 --sync_bn 0 --num_workers 10 \
  --specaug 1 --specaug_live_bins 80 --specaug_time_ratio 0 \
  --save_steps 250 --eval_steps 250 --select_metric macro_dialect \
  --sample_every 0 --gen_max_new_tokens 448 --probe_nrn6 6
```

Adjust only the documented run-specific inputs for other rows. The checkpoint callback ranks macro per-country WER using the **training-harness normalization**; full-validation comparisons additionally use the local approximation below. The callback keeps a configured top-k plus the latest checkpoint and can prune other checkpoints in its own output directory. Use a fresh output directory and retain any externally selected checkpoints separately.

SpecAugment's optional time-mask ratio cap stays zero: the proposed cap was not enabled in the reported systems.

### Country-context prompts

The exact strings are `Algeria`, `Egypt`, `Jordan`, `Mauritania`, `Morocco`, `Palestine`, `UAE`, `Yemen`. With `--dialect_cond 1`, the collator inserts the country text immediately after `<|startofcontext|>`. Prompt labels are masked with `-100`. Match the training context setting at inference; E11 is unconditioned, while E14/E15/E16b are conditioned.

## Decoding and scoring

Example for E14 checkpoint 6000:

```sh
CUDA_VISIBLE_DEVICES=0 python asr/scripts/nadi_val_eval.py \
  --checkpoint "$NADI_ASR_WORKSPACE/runs/E14/checkpoint-6000" \
  --val "$NADI_ASR_WORKSPACE/manifests/nadi_val_eval.jsonl" \
  --out-dir "$NADI_ASR_WORKSPACE/eval/E14-6000" \
  --num-beams 10 --no-repeat-ngram 6 --dialect-cond 1
python asr/scripts/local_score.py "$NADI_ASR_WORKSPACE/eval/E14-6000"
```

Each decoding directory contains `hyp.shard_0` with `utt_id`, `dataset`, `ref` and `hyp`. Missing audio causes an error rather than silently evaluating a smaller subset. Reuse the decoder with a blind-test manifest and empty references to produce hypotheses; only the organizers can provide official blind-test scores.

| Scorer | Meaning |
|---|---|
| Official scorer | Organizer-returned results. The organizer implementation is not included here. |
| Local approximation: `scripts/local_score.py` | Removes diacritics and punctuation and collapses spaces, without letter folding. Calibrated against E11 checkpoint 6000 at 49.7195% macro WER, using eight per-variety WER and CER comparisons. |
| Training-harness scorer: `train/common.py` | Retains the original trainer's letter-folding normalization. `nadi_val_eval.py --merge` reports this scorer; it is not the official scorer or the local approximation. |

Leaderboard feedback was also used as a coarse signal for prioritizing experimental directions. No exhaustive normalization-grid search is claimed. Scoring functions return fractions; multiply by 100 for percentages. Macro means an unweighted mean of per-country corpus rates. `micro_wer`/`micro_cer` in local JSON reports mean pooled rates across reference words/characters, not utterance-weighted averages.

## Submitted ROVER composition

Decode every voter over the same utterance IDs with beam 10 and no-repeat n-gram size 6. Use country context according to its training run. The first directory is the pivot.

| Position | Run / checkpoint | Weight |
|---:|---|---:|
| 1 | E16b checkpoint-6000, pivot | 1.30 |
| 2 | E14 checkpoint-6000 | 1.10 |
| 3 | E14 checkpoint-7000 | 1.05 |
| 4 | E14 checkpoint-4750 | 1.00 |
| 5 | E14 checkpoint-4500 | 1.00 |
| 6 | E11 checkpoint-6000 | 0.90 |
| 7 | E11 checkpoint-5250 | 0.85 |
| 8 | E15 checkpoint-6000 | 0.80 |
| 9 | E15 checkpoint-6750 | 0.75 |

The weights are a manually assigned descending schedule applied to validation-ranked systems and evaluated during validation development. They are not learned confidence probabilities. The code's equal-1.0 fallback is not the submitted configuration.

```sh
python asr/scripts/rover_combine.py \
  --systems \
    "$NADI_ASR_WORKSPACE/eval/E16b-6000" \
    "$NADI_ASR_WORKSPACE/eval/E14-6000" \
    "$NADI_ASR_WORKSPACE/eval/E14-7000" \
    "$NADI_ASR_WORKSPACE/eval/E14-4750" \
    "$NADI_ASR_WORKSPACE/eval/E14-4500" \
    "$NADI_ASR_WORKSPACE/eval/E11-6000" \
    "$NADI_ASR_WORKSPACE/eval/E11-5250" \
    "$NADI_ASR_WORKSPACE/eval/E15-6000" \
    "$NADI_ASR_WORKSPACE/eval/E15-6750" \
  --weights 1.30 1.10 1.05 1.00 1.00 0.90 0.85 0.80 0.75 \
  --out "$NADI_ASR_WORKSPACE/eval/rover9"
```

ROVER aligns voters to the pivot, drops insertions relative to the pivot, lets deletions vote for the empty word and resolves ties to the pivot. It operates on the intersection of voter IDs; verify the printed common-utterance count against the complete target split before accepting the output.

## Canonicalization and submission output

Build a per-country majority-form lexicon from **training transcripts only**; do not use validation/test references to build it. It groups tokens using alif-maqsura/ya, ta-marbuta/ha, and hamzated/bare-alif equivalences, then chooses the most frequent surface form when the group has at least 30 occurrences and that form accounts for at least 0.60 of them. The implementation also expands the `ﻻ` ligature when forming groups. Counts use the repeated E11 training pool, not deduplicated original transcripts. The public canonicalizer requires explicit country fields instead of assuming a particular directory depth.

```sh
python asr/scripts/ortho_canon.py \
  --in "$NADI_ASR_WORKSPACE/eval/rover9" \
  --out "$NADI_ASR_WORKSPACE/eval/rover9-canonical" \
  --train "$NADI_ASR_WORKSPACE/manifests/E11_dial.jsonl" \
  --minfrac 0.60 --mincount 30
```

Combination and canonicalization write `hyp.shard_0` plus `wer_report_local.json`. Once the same workflow is run on the blind-test manifest, export its canonicalized hypotheses:

```sh
python asr/scripts/build_submission.py \
  --hyp "$NADI_ASR_WORKSPACE/test/rover9-canonical" \
  --manifest "$NADI_ASR_WORKSPACE/manifests/nadi_robust_test.jsonl" \
  --out "$NADI_ASR_WORKSPACE/submission" --expect 500
```

This produces one UTF-8 hypothesis file per country, in the original within-country row order, with 500 lines per country for the released blind split. Never substitute the validation hypothesis directory when packaging a blind submission.

## Expected paper results

| Split / scorer | Macro WER | Macro CER | Pooled WER | Pooled CER |
|---|---:|---:|---:|---:|
| Validation, local approximation | 47.52% | 20.94% | 46.28% | 20.53% |
| Blind test, official scorer | 50.29% | 24.43% | 49.44% | 23.89% |

The final ASR rank is second of nine teams. These values describe the submitted ROVER-9 plus canonicalizer. The strongest single model, E17 checkpoint 8750, scores 47.71% validation macro WER before canonicalization and 47.50% after it; its blind-test WER is available at integer precision, 51%. E17 is absent from the nine-voter pool above.

Run the synthetic checks from the repository root:

```sh
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s asr/tests
```
