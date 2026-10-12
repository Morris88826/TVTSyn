# Inference

Install from the repository root. PyTorch and ONNX are independent extras:

```bash
pip install -e '.[pytorch,speaker]'  # reference audio + PyTorch
pip install -e '.[onnx]'            # ONNX + saved speaker embeddings, no PyTorch
pip install -e '.[pytorch,onnx,speaker]'  # all inference modes
```

Use Python 3.10+. Install matching torch/torchaudio builds for your hardware.
Validated versions and checks are recorded in [release validation](../docs/RELEASE_VALIDATION.md).

## Obtain a model

Download automatically from [warisqr007/TVTSyn](https://huggingface.co/warisqr007/TVTSyn).
The release uses CC BY-NC 4.0 for model weights. Pin the commit below for
reproducibility; each backend downloads only its required artifacts. A local
exported bundle can also be passed instead of the model ID. Weights are not
stored in Git.

```python
from tvtsyn_inference import TVTSyn

MODEL = 'warisqr007/TVTSyn'
REVISION = '2d3147b8a465e1d9aad33a2a0e53956bf8ed32aa'
model = TVTSyn.from_pretrained(MODEL, backend='pytorch', revision=REVISION, device='cpu')
speaker = model.encode_speaker('samples/target.wav')
y = model.convert_file('samples/source.wav', speaker, output='outputs/whole.wav')
y_stream = model.convert_file('samples/source.wav', speaker,
                              mode='streaming', output='outputs/stream.wav')
```

`encode_speaker()` downloads pinned SpeechBrain x-vector and ECAPA models on first
use. Their embeddings are concatenated in that order into `[1, 704]` float32.
SpeechBrain x-vector pooling adds epsilon noise even in eval mode; save the
embedding when exact repeatability is needed. Use several seconds of clean target speech. Target references shorter than
100 ms are rejected. Save embeddings for later use without speaker dependencies:

```python
import numpy as np
np.save('outputs/target.npy', speaker)
onnx = TVTSyn.from_pretrained(MODEL, backend='onnx', revision=REVISION)
onnx.convert_file('samples/source.wav', 'outputs/target.npy',
                  output='outputs/onnx.wav')
```

ONNX conversion accepts saved embeddings without importing torch, Lightning,
SpeechBrain, or training code. Reference-audio extraction still needs the
`speaker` extra. CPU is the default; pass ONNX Runtime `providers` for a separately
installed GPU runtime. The tested ONNX release uses CPUExecutionProvider.

## Streaming input

```python
import soundfile as sf

source, sr = sf.read('samples/source.wav', dtype='float32')
stream = onnx.stream(speaker, sample_rate=sr)
chunks = []
try:
    for offset in range(0, len(source), 731):  # arbitrary incoming block sizes
        chunks.append(stream.push(source[offset:offset + 731]))
    chunks.append(stream.flush())
finally:
    stream.close()
result = np.concatenate(chunks)
```

The same session interface works with PyTorch. It buffers to 1,600-sample model
steps, retains resampling state, suppresses warmup output, and drains the tail.
`push()` may return an empty array while lookahead fills. `flush()` is idempotent;
call `reset()` before reusing a completed session. `close()` discards unfinished
audio and frees state. Each session is owned by one caller; do not push to the
same session concurrently. Separate sessions may share a loaded model. Public PyTorch inference runs eagerly
without requiring a compiler; calls are serialized across loaded PyTorch models
to protect the shared legacy execution context.

Arrays use `[samples]` or `[samples, channels]` layout. Integer PCM is scaled to
float, stereo is averaged to mono, and inputs are resampled to 16 kHz. Outputs
are 1-D float32 at 16 kHz. Float WAV output preserves model amplitudes.
Empty inputs return empty outputs. PyTorch whole mode pads to the frame hop and
trims the result; ONNX rejects whole mode and processes files through streaming.

## Command line

```bash
tvtsyn --model warisqr007/TVTSyn --revision 2d3147b8a465e1d9aad33a2a0e53956bf8ed32aa --backend pytorch --mode whole \
  --source samples/source.wav --target samples/target.wav \
  --save-embedding outputs/target.npy --output outputs/whole.wav

tvtsyn --model warisqr007/TVTSyn --revision 2d3147b8a465e1d9aad33a2a0e53956bf8ed32aa --backend pytorch --mode streaming \
  --source samples/source.wav --embedding outputs/target.npy --output outputs/stream.wav

tvtsyn --model warisqr007/TVTSyn --revision 2d3147b8a465e1d9aad33a2a0e53956bf8ed32aa --backend onnx --mode streaming \
  --source samples/source.wav --embedding outputs/target.npy --output outputs/onnx.wav
```

## Hugging Face and offline use

```python
model = TVTSyn.from_pretrained(
    'warisqr007/TVTSyn', backend='onnx',
    revision='2d3147b8a465e1d9aad33a2a0e53956bf8ed32aa',
    cache_dir='./model-cache', local_files_only=False,
)
model.save_pretrained('./local-onnx')
model = TVTSyn.from_pretrained('./local-onnx', backend='onnx', local_files_only=True)
```

The loader fetches configuration and only the selected backend's artifacts,
including ONNX external weights when listed. `save_pretrained()` saves the loaded
backend; exporting both formats requires the maintainer export command. Offline
speaker extraction additionally needs cached SpeechBrain files. Keep a saved
embedding when an entirely offline ONNX workflow is required.

## Latency

The first public streaming model uses 100 ms steps and four 20 ms encoder
lookahead frames. The fixed-step scheduler may round input availability up to
another chunk. Chunk size, lookahead, warmup scheduling, computation, resampling,
and browser/network buffering are separate latency contributors. Measure first
output time and steady-state processing on the intended hardware; 100 ms chunks
do not imply 100 ms end-to-end latency. Whole and streaming outputs need not be
identical because their context differs.

See [the notebook](notebooks/inference.ipynb), [array example](examples/stream.py),
and [local Space app](../spaces/demo/README.md).

## Ready-to-run listening review

See [LISTENING_VALIDATION.md](../docs/LISTENING_VALIDATION.md) for the persistent
CPU environment, IDE kernel selection, prepared comparison audio, physical
microphone checks, and downloadable review results. The author accepted the six-case listening review and explicitly waived physical
microphone testing for this release; the model is now published.
