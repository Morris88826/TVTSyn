# Inference

Run these examples from the repository root with Python 3.10 or newer.

## Installation

```bash
pip install -e '.[pytorch,onnx,speaker]'
```

For PyTorch only, install `.[pytorch,speaker]`. For ONNX with a saved speaker
embedding, install `.[onnx]`; that installation does not require PyTorch.

## Convert a file

`from_pretrained()` downloads the selected model from Hugging Face and caches
it for later runs. `encode_speaker()` extracts the target voice from a recording
and downloads the speaker models on its first call.

```python
from tvtsyn_inference import TVTSyn

model = TVTSyn.from_pretrained("warisqr007/TVTSyn", backend="pytorch")
speaker = model.encode_speaker("samples/target.wav")
model.convert_file("samples/source.wav", speaker, output="outputs/whole.wav")
```

Replace the paths with your recordings. Use several seconds of clear target
speech. Output files are mono at 16 kHz; input files are resampled automatically.
Add `device="cuda"` to `from_pretrained()` to use a GPU with PyTorch.

For PyTorch streaming:

```python
model.convert_file(
    "samples/source.wav", speaker,
    mode="streaming", output="outputs/streaming.wav",
)
```

For ONNX:

```python
onnx = TVTSyn.from_pretrained("warisqr007/TVTSyn", backend="onnx")
onnx.convert_file("samples/source.wav", speaker, output="outputs/onnx.wav")
```

ONNX always processes files in 100 ms chunks. PyTorch can process the whole
file at once or use streaming. The two modes can sound slightly different.

## ONNX without PyTorch

First save the speaker embedding in an environment with the `speaker` extra:

```python
from pathlib import Path
import numpy as np
from tvtsyn_inference import TVTSyn

model = TVTSyn.from_pretrained("warisqr007/TVTSyn", backend="onnx")
speaker = model.encode_speaker("samples/target.wav")
Path("outputs").mkdir(exist_ok=True)
np.save("outputs/target.npy", speaker)
```

Copy `target.npy` to the ONNX-only environment. There, install `.[onnx]` and run:

```python
from tvtsyn_inference import TVTSyn

model = TVTSyn.from_pretrained("warisqr007/TVTSyn", backend="onnx")
model.convert_file(
    "samples/source.wav", "outputs/target.npy", output="outputs/onnx.wav",
)
```

Embeddings are NumPy float arrays of shape `[1, 704]`: x-vector first, then
ECAPA. Saving the embedding also lets you reuse the same target voice.

## Incoming audio blocks

Both backends accept audio a block at a time. This example reads a file in
small blocks; a microphone application can pass its incoming blocks instead.

```python
from pathlib import Path
import numpy as np
import soundfile as sf
from tvtsyn_inference import TVTSyn

model = TVTSyn.from_pretrained("warisqr007/TVTSyn", backend="onnx")
speaker = model.encode_speaker("samples/target.wav")
audio, sample_rate = sf.read("samples/source.wav", dtype="float32")
stream = model.stream(speaker, sample_rate=sample_rate)
parts = []
try:
    for offset in range(0, len(audio), 731):
        parts.append(stream.push(audio[offset:offset + 731]))
    parts.append(stream.flush())
finally:
    stream.close()
Path("outputs").mkdir(exist_ok=True)
sf.write("outputs/blocks.wav", np.concatenate(parts), 16000, subtype="FLOAT")
```

Use `backend="pytorch"` for the same interface with PyTorch. Blocks may have
any length. `push()` can return an empty array while the model waits for more
audio. Call `flush()` after the last block to get the remaining output, and
`close()` when done. Call `reset()` to reuse a finished stream. Create a separate
stream for each speaker session.

Audio arrays use `[samples]` or `[samples, channels]` layout. Streaming uses
100 ms model steps and 80 ms encoder lookahead; processing and audio buffering
add to the delay you hear.

## Download for offline use

Download and save the selected backend while connected to the internet:

```python
from tvtsyn_inference import TVTSyn

model = TVTSyn.from_pretrained("warisqr007/TVTSyn", backend="onnx")
model.save_pretrained("models/TVTSyn-onnx")
```

Then load that directory without network access:

```python
model = TVTSyn.from_pretrained(
    "models/TVTSyn-onnx", backend="onnx", local_files_only=True,
)
model.convert_file("samples/source.wav", "outputs/target.npy", output="outputs/offline.wav")
```

Use a saved speaker embedding for fully offline conversion. To pin a model
version, pass `revision="2d3147b8a465e1d9aad33a2a0e53956bf8ed32aa"` to the Hub
load call. Use `cache_dir="model-cache"` to choose where downloads are cached.

## Command line

```bash
tvtsyn --model warisqr007/TVTSyn --backend pytorch --mode whole \
  --source samples/source.wav --target samples/target.wav \
  --save-embedding outputs/target.npy --output outputs/whole.wav

tvtsyn --model warisqr007/TVTSyn --backend pytorch --mode streaming \
  --source samples/source.wav --embedding outputs/target.npy --output outputs/streaming.wav

tvtsyn --model warisqr007/TVTSyn --backend onnx \
  --source samples/source.wav --embedding outputs/target.npy --output outputs/onnx.wav
```

For an interactive walkthrough, open the [inference notebook](notebooks/inference.ipynb).
For microphone input, use the [demo](../spaces/demo/README.md).
