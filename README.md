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

Official code for our [ICLR 2026 paper](https://openreview.net/forum?id=Tf4Lfw85lS).
Convert speech into a target voice using a short recording of that voice.

## Inference

### Installation

Requires Python 3.10 or newer. Run the examples from the repository directory.

```bash
git clone https://github.com/PSI-TAMU/TVTSyn.git
cd TVTSyn
pip install -e '.[pytorch,onnx,speaker]'
```

### PyTorch

`from_pretrained()` downloads the model from
[Hugging Face](https://huggingface.co/warisqr007/TVTSyn) automatically.
Run this example with the included recordings, or replace the paths with your own:

```python
from tvtsyn_inference import TVTSyn

# Download the PyTorch model from Hugging Face (cached after the first run).
model = TVTSyn.from_pretrained("warisqr007/TVTSyn", backend="pytorch")

# Use a recording of the voice you want the output to sound like.
speaker = model.encode_speaker("samples/target.wav")
model.convert_file("samples/source.wav", speaker, output="outputs/converted.wav")
```

For PyTorch streaming, process the same file in 100 ms chunks:

```python
model.convert_file(
    "samples/source.wav", speaker,
    mode="streaming", output="outputs/pytorch_streaming.wav",
)
```

### ONNX

ONNX processes audio in 100 ms chunks:

```python
from tvtsyn_inference import TVTSyn

# Download the ONNX model from Hugging Face.
model = TVTSyn.from_pretrained("warisqr007/TVTSyn", backend="onnx")
speaker = model.encode_speaker("samples/target.wav")
model.convert_file("samples/source.wav", speaker, output="outputs/onnx.wav")
```

Reference-audio speaker extraction uses PyTorch. To run ONNX without PyTorch,
use a saved speaker embedding; see the [inference guide](inference/README.md#onnx-without-pytorch).
Outputs are mono WAV files at 16 kHz; input files are resampled automatically.

See the [inference notebook](inference/notebooks/inference.ipynb) and
[inference guide](inference/README.md) for live audio blocks, GPU selection,
saved embeddings, and offline use.

## Local demo

Try the [online demo](https://huggingface.co/spaces/warisqr007/TVTSyn),
or run it locally. Install FFmpeg first, then:

```bash
pip install -e '.[demo]'
python spaces/demo/app.py
```

Open the local URL printed in the terminal. Upload source and target recordings
for file conversion. For live input, prepare the target voice before recording.
See the [demo guide](spaces/demo/README.md) for settings.

## Training

See the [training guide](training/README.md) for data preparation, the two
training stages, and resuming checkpoints. To use your trained model for
inference, follow the [export guide](inference/export/README.md).

Evaluation, anonymization experiments, tests, and release tools are preserved
on the [research-tools branch](https://github.com/PSI-TAMU/TVTSyn/tree/research-tools).

## License

TVTSyn source code and model weights use [CC BY-NC 4.0](LICENSE).
Third-party components and sample recordings retain their own licenses; see
[third-party notices](docs/THIRD_PARTY_NOTICES.md).

## Citation

```bibtex
@inproceedings{quamer2026tvtsyn,
  title={{TVTSyn}: Content-Synchronous Time-Varying Timbre for Streaming Voice Conversion and Anonymization},
  author={Waris Quamer and Mu-Ruei Tseng and Ghady Nasrallah and Ricardo Gutierrez-Osuna},
  booktitle={The Fourteenth International Conference on Learning Representations},
  year={2026},
  url={https://openreview.net/forum?id=Tf4Lfw85lS}
}
```
