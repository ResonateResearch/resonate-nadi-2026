# Resonate at NADI 2026

Code and reproducibility resources for **Resonate at NADI 2026 Shared Task: Robust Speech Recognition and Dialect Identification under Acoustic and Domain Shift**.

This repository separates the two Resonate systems:

- [Robust Dialectal Arabic ASR — NADI 2026 Subtask 1.1](asr/README.md): Cohere Transcribe Arabic fine-tuning, acoustic augmentation, country context, pivot-anchored ROVER and orthographic canonicalization.
- [Spoken Arabic Dialect Identification — NADI 2026 Task 2](adi/README.md): WavLM Base+ with context-aware multi-head factorized attentive pooling (CA-MHFA).

## Reported results

| Task | Result |
|---|---|
| ASR | **50.29% macro WER, 24.43% CER, 2nd place** |
| ADI | **41% blind-test accuracy**, $C_{\mathrm{avg}}=0.17$, **7th place** |

These are the paper's official submitted-system results. $C_{\mathrm{avg}}$ is the LRE average detection cost; lower is better. See the task READMEs for validation results and scoring details.

## Structure

| Path | Purpose |
|---|---|
| `asr/` | ASR training, preparation, decoding, local scoring and combination code |
| `adi/` | ADI data construction, configuration, training, pooling and inference code |
| `CITATION.cff` | Paper citation, author order and repository metadata |
| `LICENSE` | Apache License 2.0 |
| `THIRD_PARTY.md` | Attribution and third-party terms |
| `licenses/` | Retained third-party license texts |

## Setup

Use separate Python environments for ASR and ADI. Follow the task-specific setup instructions: the model frameworks and preprocessing dependencies differ. Install a PyTorch build suitable for your own CUDA environment, then the dependencies listed by the selected task. The recorded final training hardware was a **single NVIDIA H100 GPU** for each task.

ASR requires a Transformers build exposing `CohereAsrForConditionalGeneration`; ADI uses SpeechBrain and the included CA-MHFA module. See each README before installing or running.

## Data and model access

Obtain the NADI 2026 ASR task data and ADI-17/ADI-20 resources through their organizers or authorized dataset providers. ADI uses the `ArabicSpeech/ADI17` and `ArabicSpeech/ADI20` resources. Access conditions may require an account or an organizer-approved release. Use the supplied splits and preserve utterance identity and ordering.

**Organizer datasets are not redistributed in this repository.** Generate derived data locally under the applicable dataset conditions. Voice conversion uses organizer training speech as source and target material; do not substitute evaluation audio into training preparation.

Pretrained models, MUSAN/RIR resources where used, and upstream VC/denoising components must be obtained separately under their own licenses. See the task preparation guides for the required inputs.

## Citation

Use [CITATION.cff](CITATION.cff), or cite the paper as:

```bibtex
@misc{shoaib2026resonate,
  title = {Resonate at NADI 2026 Shared Task: Robust Speech Recognition and Dialect Identification under Acoustic and Domain Shift},
  author = {Shoaib, Omar and Sokar, Mohamed Ashraf and Motawie, Mohamed and AboElFottouh, Omar and Hassouna, Ahmed and Hassan, Al-Amir and Zaki, Nader Essam and Ali, Wael},
  year = {2026}
}
```

Repository: [ResonateResearch/resonate-nadi-2026](https://github.com/ResonateResearch/resonate-nadi-2026).

## License

The Resonate code and documentation are licensed under [Apache License 2.0](LICENSE). Retained third-party attributions and dependency terms are documented in [THIRD_PARTY.md](THIRD_PARTY.md). This license does not grant rights to separately obtained datasets, model weights or other external resources.
