---
title: TVTSyn
sdk: gradio
sdk_version: 5.50.0
app_file: app.py
---

# Local demo

Install the system FFmpeg package first (`ffmpeg` and `ffprobe` must be on PATH),
for example `conda install -c conda-forge ffmpeg` in a conda environment.

From the repository root:

```bash
pip install -e '.[demo]'
TVTSYN_MODEL=artifacts/TVTSyn python spaces/demo/app.py
```

`TVTSYN_MODEL` accepts a local bundle or a Hugging Face model ID. Set
`TVTSYN_REVISION` to a release commit when using the Hub. `TVTSYN_DEVICE`
selects the PyTorch/speaker device (default `cpu`); ONNX uses CPU.
`TVTSYN_THREADS` controls ONNX threads (default one);
`TVTSYN_TORCH_THREADS` controls CPU PyTorch threads (default four); `TVTSYN_OFFLINE=1` disables downloads.
Offline reference-audio conditioning additionally requires cached speaker models.

The file tab supports whole-utterance PyTorch and streaming PyTorch/ONNX.
For live conversion, prepare the target, allow microphone access, record, then
stop to drain the tail. Saved `.npy` embeddings avoid speaker extraction.
Recording is bounded to 60 seconds; prepare again after the limit or a backend
change. Inactive state expires after 120 seconds. Models are shared; stream
caches are per user. Playback uses a continuous generator and a bounded queue per recording,
independent of microphone ingestion. Backend calls are serialized, so concurrent users can
increase latency. PCM playback clips values outside [-1, 1] at a fixed scale.
The Python/file API preserves float samples without this playback clipping.

This is a locally runnable Space source, not a deployed Space. To publish later,
install a pinned TVTSyn package/repository revision in the Space environment,
set its model ID and revision, and copy this app plus its dependencies. The model
repository holds the weights. Browser playback and network latency must be
measured on the intended deployment hardware.

## Browser regression check

```bash
pip install playwright
playwright install chromium
python spaces/demo/validate_browser.py --backend onnx \
  --embedding outputs/target.npy
```

The test uses Chromium's simulated microphone to feed `samples/source.wav`,
asserts actual browser playback progress, stops, and restarts without preparing
the target again. Use `--backend pytorch` to test that path; `TVTSYN_DEVICE`
selects its device. Reports and screenshots go to `artifacts/browser`.
