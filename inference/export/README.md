# Maintainer export

Run from the repository root with a trusted local Lightning checkpoint:

```bash
pip install -e '.[export]'
TORCH_COMPILE_DISABLE=1 tvtsyn-export \
  --checkpoint ckpts/resynthesis_quant_and_tvtimbre_ft.ckpt \
  --output artifacts/TVTSyn
```

The converter instantiates the shared inference network, maps `encoder.net.*`
to `encoder.*`, retains `decoder.*`, and strictly checks all inference weights.
Training-only keys are excluded and recorded. Conversion never instantiates
Lightning trainers, discriminators, or losses. Historical checkpoints containing
Python dataclasses use the retained compatibility module. Use `--pytorch-only`
to omit ONNX generation.

The local bundle contains `config.json`, `model.safetensors`, an ONNX graph and
state-schema sidecar, any external graph weights, and a draft model card.
Configuration records checkpoint and artifact hashes, architecture, fixed audio
and streaming conventions, and pinned speaker-model revisions. The export uses
opset 17, fixed shapes, and tensorized state version 2. Expected fixed-shape trace
warnings do not replace runtime parity testing.

```bash
TVTSYN_TEST_MODEL=artifacts/TVTSyn TORCH_COMPILE_DISABLE=1 pytest -q
python inference/export/validate.py --model artifacts/TVTSyn \
  --reference artifacts/reference --output artifacts/validation.json
```

Before later publication, review the model card and sample/source licensing,
select the Hub repository ID, upload the complete bundle as one version, and pin
that revision in the Space. No upload or Space deployment is performed here.

The original baseline is also reproducible using
`capture_reference.py --legacy-root <original-checkout> --qt-root <TVTSynQT>`,
with the legacy training dependencies installed. It uses the original sample
paths and preprocessed speaker embedding retained locally. Python 3.11+ needs
the legacy `SpecEncoderArgs.spec_args` dataclass default converted to a default
factory in that disposable checkout. Reference data are local validation
fixtures, not required for end-user inference.
