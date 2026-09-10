# Third-party components and licenses

The Resonate code and documentation are licensed under [Apache License 2.0](LICENSE). Inherited third-party source retains its applicable attribution and notices. Separately obtained dependencies, model weights and datasets remain subject to their own terms.

## Included adaptation and notices

- The ADI trainer adapts SpeechBrain's VoxCeleb speaker-training recipe through the team's CA-MHFA recipe. Inherited recipe attribution: **Mirco Ravanelli, Hwidong Na and Nauman Dawalatabad (2020)**. Resonate's modifications cover dialect identification, data preparation, pooling and configurable paths. The upstream base is SpeechBrain revision `e5cb1f65b940634215650aa1171e0440d0808123`.
- The upstream Apache-2.0 text is included in [licenses/SpeechBrain-Apache-2.0.txt](licenses/SpeechBrain-Apache-2.0.txt). See also [SpeechBrain's source license](https://github.com/speechbrain/speechbrain/blob/develop/LICENSE).
- The CA-MHFA implementation retains its method reference and source attribution.
- `adi/vad_speech.py` uses the separately installed Silero VAD package and model.

## External dependencies

| Component | License and access |
|---|---|
| NoiseLab | Proprietary; obtain separately authorized tooling for denoising/full-chain generation. |
| kNN-VC | Obtain externally. Its upstream license is MIT-like **with an additional condition**, not an unmodified MIT text. Read and comply with the exact [upstream license](https://github.com/bshall/knn-vc/blob/c616845c4e309e24d5927f15adbdf277a3d65358/LICENSE). The inspected source revision is `c616845c4e309e24d5927f15adbdf277a3d65358`. |
| kNN-VC's WavLM/fairseq and HiFi-GAN sources | Not vendored here. The upstream checkout includes code derived from Microsoft WavLM, fairseq and HiFi-GAN. Their own copyright/license notices remain applicable; kNN-VC's root license does not erase them. |
| ReDimNet2 | External source/model dependency. Its [source license](https://github.com/PalabraAI/redimnet2/blob/main/LICENSE) is MIT; review the selected model assets' terms separately. |
| ClearerVoice-Studio / ClearVoice | External enhancement dependency under [Apache-2.0](https://github.com/modelscope/ClearerVoice-Studio/blob/main/LICENSE). Check downloaded model assets' terms separately. |
| Transformers, PyTorch, SpeechBrain and other Python packages | Installed separately under their respective licenses. |

## Models and datasets

Obtain data and model assets separately under their applicable access and redistribution conditions. See the root and task READMEs for required inputs.
