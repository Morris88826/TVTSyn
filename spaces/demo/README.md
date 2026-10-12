# TVTSyn demo

Try the [online Space](https://huggingface.co/spaces/warisqr007/TVTSyn),
or run the same app locally.

## Run locally

Install FFmpeg so that `ffmpeg` and `ffprobe` are on PATH. For example, in a
conda environment: `conda install -c conda-forge ffmpeg`.

From the repository root:

```bash
pip install -e '.[demo]'
python spaces/demo/app.py
```

Open the URL printed in the terminal. The app downloads the model from
Hugging Face automatically.

## Convert audio

For a file, select a backend, upload the target voice and source speech, then
click **Convert**. PyTorch supports whole-file and streaming conversion;
ONNX uses streaming.

For live input, upload a target recording (or saved `.npy` speaker embedding),
open **Live microphone**, and click **Prepare target**. Allow microphone access,
click **Record**, then **Stop** to finish. You can record again with the same
target. Recordings are limited to 60 seconds; prepare the target again after
reaching the limit or changing the backend.

The online Space uses ZeroGPU for PyTorch file conversion. ONNX and live input
run on CPU. Shared compute and network delays affect playback; PyTorch live
conversion on CPU may be slower than real time.

## Settings

| Variable | Purpose |
| --- | --- |
| `TVTSYN_MODEL` | Hub model ID or local model directory |
| `TVTSYN_REVISION` | Hub model version to load |
| `TVTSYN_DEVICE` | Local PyTorch device, such as `cpu` or `cuda` |
| `TVTSYN_OFFLINE=1` | Load cached/local files without downloads |
| `TVTSYN_THREADS` | ONNX CPU threads (default 1) |
| `TVTSYN_TORCH_THREADS` | PyTorch CPU threads (default 4) |

For offline use, prepare a local model and saved embedding as described in the
[inference guide](../../inference/README.md#download-for-offline-use).

## Deploy your own Space

Build a wheel from a committed checkout and stage the app:

```bash
pip install build
python -m build --wheel --outdir artifacts/dist
python spaces/demo/prepare_space.py \
  --wheel artifacts/dist/tvtsyn-0.1.0-py3-none-any.whl \
  --revision 2d3147b8a465e1d9aad33a2a0e53956bf8ed32aa \
  --source-revision "$(git rev-parse HEAD)" --hardware zero-a10g
```

Upload the contents of `artifacts/space` to a Gradio Space using ZeroGPU and set
its `TVTSYN_ZEROGPU` variable to `1`. The generated requirements install TVTSyn
from the specified GitHub commit, so push that commit first. Leave
`TVTSYN_ZEROGPU` unset for local use.
