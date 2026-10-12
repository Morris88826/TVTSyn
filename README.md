# TVTSyn: Content-Synchronous Time-Varying Timbre for Streaming Voice Conversion and Anonymization

Waris Quamer, Mu-Ruei Tseng, Ghady Nasrallah, and Ricardo Gutierrez-Osuna

TVTSyn aligns time-varying speech content with time-varying timbre for streaming
voice conversion. A global timbre memory, content-conditioned attention, gating,
and spherical interpolation provide local timbre variation; a factorized
vector-quantized content bottleneck reduces residual speaker information.

[Paper](TVTSyn_ICLR26.pdf) · [Audio demonstrations](https://anonymized0826.github.io/TVTSyn/)

## Inference

| Backend | Whole utterance | Streaming input | Target conditioning |
| --- | --- | --- | --- |
| PyTorch | Yes | 100 ms chunks | Reference audio or saved embedding |
| ONNX Runtime | File through streaming chunks | Fixed 100 ms chunks | Reference audio via optional PyTorch extraction, or saved embedding |

Requires Python 3.10+. Install matching PyTorch and torchaudio builds for your
hardware when using reference-audio conditioning.

```bash
git clone https://github.com/PSI-TAMU/TVTSyn.git
cd TVTSyn
pip install -e '.[pytorch,onnx,speaker]'
```

**Pretrained model:** [warisqr007/TVTSyn](https://huggingface.co/warisqr007/TVTSyn)
contains the PyTorch weights, ONNX graph, state schema and configuration.
TVTSyn weights use CC BY-NC 4.0. This release is pinned to
`2d3147b8a465e1d9aad33a2a0e53956bf8ed32aa`. The Space deployment is being validated.

```python
from tvtsyn_inference import TVTSyn

model = TVTSyn.from_pretrained(
    'warisqr007/TVTSyn', backend='pytorch',
    revision='2d3147b8a465e1d9aad33a2a0e53956bf8ed32aa',
)
speaker = model.encode_speaker('samples/target.wav')
model.convert_file('samples/source.wav', speaker, output='outputs/converted.wav')
```

For ONNX-only conversion with a saved `[1, 704]` speaker embedding:

```bash
pip install -e '.[onnx]'
tvtsyn --model warisqr007/TVTSyn --backend onnx \
  --revision 2d3147b8a465e1d9aad33a2a0e53956bf8ed32aa \
  --source samples/source.wav --embedding target.npy --output outputs/onnx.wav
```

Read the [inference guide](inference/README.md) for PyTorch streaming, ONNX,
live input sessions, offline loading, and notebooks. The first release is mono
16 kHz, with 100 ms streaming steps and 80 ms encoder lookahead. These are model
settings, not a claim of end-to-end latency on arbitrary hardware.

## Listening validation

The author accepted the six-case listening review; physical microphone hardware
was not tested and was explicitly waived for this release. A CPU environment
and listening pack are also available locally to maintainers. Open the
[VS Code listening notebook](inference/notebooks/listening_validation.ipynb)
for embedded playback and saved ratings without a web server. Follow
[the listening guide](docs/LISTENING_VALIDATION.md) to compare all three modes,
test your microphone, and download the review results. The publication sequence
is in [PUBLICATION.md](docs/PUBLICATION.md).

## Local demo

Install FFmpeg (`ffmpeg` and `ffprobe` on PATH) for browser audio playback.

```bash
pip install -e '.[demo]'
TVTSYN_MODEL=artifacts/TVTSyn python spaces/demo/app.py
```

The [Space source](spaces/demo/README.md) supports file conversion and live
microphone input with both backends. It loads weights through the same API.

## Training and research

The original two-stage training workflow, resume support, and experimental
variants are documented in [training/README.md](training/README.md).
[Anonymization](anonymization/README.md) and evaluation remain research tools;
the public conversion API accepts supplied target embeddings.

```text
src/tvtsyn/       Shared networks and components
training/         Lightning modules, data modules, configs, scripts, and training guide
inference/        Public API, backend runtimes, export tools, examples, and notebooks
spaces/demo/      Runnable local Gradio demo
samples/          Small attributed demonstration set
anonymization/    Preserved research anonymization tools
evaluation/       Preserved research evaluation tools
tests/            Training and inference regression tests
artifacts/        Local model bundles and validation outputs (ignored by Git)
```

See [validation results](docs/RELEASE_VALIDATION.md) and
[third-party notices](docs/THIRD_PARTY_NOTICES.md). Existing source license headers
and the repository [LICENSE](LICENSE) are retained.
