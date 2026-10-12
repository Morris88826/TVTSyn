---
library_name: tvtsyn
license: cc-by-nc-4.0
language:
- en
pipeline_tag: audio-to-audio
tags:
- voice-conversion
- streaming
- onnx
- pytorch
model-index: []
---
# TVTSyn (ICLR 2026)

Waris Quamer, Mu-Ruei Tseng, Ghady Nasrallah, and Ricardo Gutierrez-Osuna.

Convert speech into a target voice using a short recording of that voice.
This is the pretrained model for our **ICLR 2026** paper,
[TVTSyn: Content-Synchronous Time-Varying Timbre for Streaming Voice Conversion and Anonymization](https://openreview.net/forum?id=Tf4Lfw85lS).

[Code](https://github.com/PSI-TAMU/TVTSyn) ·
[Try the demo](https://huggingface.co/spaces/warisqr007/TVTSyn) ·
[Audio examples](https://anonymized0826.github.io/TVTSyn/)

## Installation

Requires Python 3.10 or newer:

```bash
git clone https://github.com/PSI-TAMU/TVTSyn.git
cd TVTSyn
pip install -e '.[pytorch,onnx,speaker]'
```

## PyTorch inference

Run from the cloned repository. The first call downloads the weights from this
Hub repository; later runs reuse the cached files.

```python
from tvtsyn_inference import TVTSyn

# Download the PyTorch model from Hugging Face (cached after the first run).
model = TVTSyn.from_pretrained("warisqr007/TVTSyn", backend="pytorch")

# Use a recording of the voice you want the output to sound like.
speaker = model.encode_speaker("samples/target.wav")
model.convert_file("samples/source.wav", speaker, output="outputs/converted.wav")
```

For PyTorch streaming:

```python
model.convert_file(
    "samples/source.wav", speaker,
    mode="streaming", output="outputs/pytorch_streaming.wav",
)
```

## ONNX inference

```python
from tvtsyn_inference import TVTSyn

# Download the ONNX model from Hugging Face.
model = TVTSyn.from_pretrained("warisqr007/TVTSyn", backend="onnx")
speaker = model.encode_speaker("samples/target.wav")
model.convert_file("samples/source.wav", speaker, output="outputs/onnx.wav")
```

ONNX converts audio in 100 ms chunks. Extracting a voice from reference audio
uses PyTorch; saved speaker embeddings let you run ONNX without PyTorch.
See the [inference guide](https://github.com/PSI-TAMU/TVTSyn/blob/main/inference/README.md)
for saved embeddings, incoming audio blocks, and offline loading.

Replace the sample paths with your own recordings. Use several seconds of
clear speech for the target voice. Input files are resampled automatically;
output audio is mono at 16 kHz.

## License

The weights use **CC BY-NC 4.0**: attribution is required and use is limited to
noncommercial purposes. See [LICENSE](https://huggingface.co/warisqr007/TVTSyn/blob/main/LICENSE). Source code and the separate
SpeechBrain speaker models retain their own licenses.

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
