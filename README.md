# TVTSyn: Content-Synchronous Time-Varying Timbre for Streaming Voice Conversion and Anonymization (ICLR 2026)

[Waris Quamer](https://github.com/warisqr007), [Mu-Ruei Tseng](https://github.com/Morris88826), [Ghady Nasrallah](https://github.com/Ghadynasrallah), and [Ricardo Gutierrez-Osuna](https://scholar.google.com/citations?user=UnuQfEwAAAAJ&hl=en)


<a href="https://arxiv.org/abs/2602.09389">
  <img src="https://img.shields.io/badge/arXiv-2602.09389-%23B31B1B">
</a>
<a href="https://huggingface.co/warisqr007/TVTSyn">
  <img src="https://img.shields.io/badge/🤗%20Hugging%20Face-Models-FFD21E">
</a>
<a href="https://huggingface.co/spaces/warisqr007/TVTSyn">
  <img src="https://img.shields.io/badge/🤗%20Hugging%20Face-Spaces-FFD21E">
</a>
<a href="https://anonymized0826.github.io/TVTSyn/">
  <img src="https://img.shields.io/badge/Demo%20Page-online-brightgreen">
</a>
<br>


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
`2d3147b8a465e1d9aad33a2a0e53956bf8ed32aa`. Try the [ZeroGPU Space](https://huggingface.co/spaces/warisqr007/TVTSyn)
for file conversion and live microphone input. PyTorch file conversion uses
ZeroGPU; ONNX and live sessions run on CPU.

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
python spaces/demo/app.py
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
