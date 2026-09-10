# Within-dialect voice conversion

This directory provides the ReDimNet voice-bank and within-dialect conversion
pipeline using external ReDimNet2 and kNN-VC models. All runtime files live below
`NADI_VC_WORKSPACE`, default
`work/vc` from the current directory. Run commands from the release repository
root after [NADI audio export](../PREPARATION.md).

Use a separate environment with mutually compatible `torch` and `torchaudio`,
plus `numpy`, `soundfile`, `PyYAML`, `scikit-learn`, and `tqdm`, and the runtime
dependencies required by the two upstream model repositories. Their source and
weights are downloaded externally through `torch.hub`; neither is vendored here:

- ReDimNet2: `PalabraAI/redimnet2`, variant `b6/lm/vox2`, 192-dimensional embeddings.
- kNN-VC: `bshall/knn-vc`, pretrained prematched model, `topk=4`.

The upstream kNN-VC license has an additional condition beyond standard MIT;
review the release's [third-party notices](../../THIRD_PARTY.md) and upstream terms before use.
Model weights and dependencies have their own terms. This package grants no
rights to those separate components. `torch.hub` executes the selected upstream
repository code. kNN-VC defaults to the inspected upstream commit
`c616845c4e309e24d5927f15adbdf277a3d65358` as a portability pin; this is not a claim
that the historical run used that exact revision. Prefer setting
`KNNVC_LOCAL_REPO=/path/to/approved/knn-vc-checkout` to use an independently
provisioned checkout without a Hub source fetch. Alternatively set
`KNNVC_HUB_REPO=bshall/knn-vc:<approved-ref>`. Set
`REDIMNET_HUB_REPO=PalabraAI/redimnet2:<approved-ref>` to pin ReDimNet as well;
its default follows upstream. The historical source revisions were not recovered.

```bash
export NADI_ASR_WORKSPACE="$PWD/work/asr"
export NADI_VC_WORKSPACE="$PWD/work/vc"

python asr/vc/prepare_config.py \
  --original-manifest "$NADI_ASR_WORKSPACE/manifests/nadi_original.jsonl" \
  --clean-audio-root "$NADI_ASR_WORKSPACE/audio/clean-neural"

CUDA_VISIBLE_DEVICES=0 python asr/vc/pipeline/embed.py \
  --dataset clean_neural --embed-model redimnet

python asr/vc/pipeline/bank.py \
  --exp clean_neural__redimnet__thr0.45_coh0.7

CUDA_VISIBLE_DEVICES=0 python asr/vc/pipeline/generate.py \
  --exp clean_neural__redimnet__thr0.45_coh0.7 \
  --n-targets 10 --min-target-dur 60
```

`prepare_config.py` constructs the metadata
interface from the exported NADI train manifest and writes `dataset.yaml`, the
experiment `config.yaml`, and train metadata under the work directory. The
configuration uses the recovered `clean_neural` ReDimNet recipe. It requires each
original train clip to have its denoised counterpart.

Embedding is performed per clip, with a 12-second cap and L2 normalization;
ReDimNet2 is not padded across clips because its pooling has no length mask.
The bank uses agglomerative complete-linkage cosine clustering with threshold
0.45, minimum cohesion 0.70, and minimum total cluster duration 30 seconds.
Generation chooses up to ten richest voices of at least 60 seconds from the same
dialect and excludes a source clip's own target cluster. It preserves the source
transcript and dialect. Both source and target metadata must be train-only;
`load_dataset` rejects a configuration containing other splits.

Outputs appear at
`$NADI_VC_WORKSPACE/exp/clean_neural__redimnet__thr0.45_coh0.7/gen/same_n10/<Dialect>/wavs/<source_id>__v<index>.wav`.
This root is the required `--vc-root` for `asr/preparation/build_e11.py`.
Generation resumes from existing WAV files and rebuilds runtime metadata at the
end of each dialect. Use a fresh work directory when changing model revisions,
configuration, input audio, or target count so that cached embeddings and generated
audio are not mixed between runs.
