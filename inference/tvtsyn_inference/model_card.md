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
# TVTSyn

Waris Quamer, Mu-Ruei Tseng, Ghady Nasrallah, and Ricardo Gutierrez-Osuna.

Inference weights for [TVTSyn: Content-Synchronous Time-Varying Timbre for
Streaming Voice Conversion and Anonymization](https://arxiv.org/abs/2602.09389).
[Source and installation](https://github.com/PSI-TAMU/TVTSyn) ·
[Try the ZeroGPU Space](https://huggingface.co/spaces/warisqr007/TVTSyn).
PyTorch file conversion uses ZeroGPU; ONNX and live streams run on CPU.

## License and intended use

The TVTSyn weights are released under **CC BY-NC 4.0**. Attribution is required
and use is limited to noncommercial purposes. See `LICENSE` and the
[license terms](https://creativecommons.org/licenses/by-nc/4.0/).
The repository code and external SpeechBrain models retain their own licenses.
Use reference recordings for which you have appropriate permission. This
conversion interface does not establish speaker consent or guarantee anonymity.

## Usage

Install the source package with `pip install -e '.[pytorch,onnx,speaker]'` from
the repository checkout. For conversion with a saved embedding and no PyTorch,
install `.[onnx]` instead.

```python
from tvtsyn_inference import TVTSyn
model = TVTSyn.from_pretrained("warisqr007/TVTSyn", backend="pytorch")
speaker = model.encode_speaker("target.wav")
model.convert_file("source.wav", speaker, output="converted.wav")
```

Pin `revision` to a Hub commit for reproducibility. For ONNX select
`backend="onnx"`; it converts files using the streaming path. Call
`model.stream(speaker)` for incremental `push`, `flush`, and `reset`.
PyTorch supports both `mode="whole"` and `mode="streaming"`.

## Model and artifacts

- Mono 16 kHz float audio; batch size one for streaming.
- Fixed 1,600-sample (100 ms) steps, LA4 (80 ms) encoder lookahead, state version 2.
- The fixed-step scheduler needs 200 ms of input before its first valid output;
  computation, resampling, browser buffering, and network time add latency.
- Speaker embedding: SpeechBrain x-vector (512) followed by ECAPA (192).
  Reference extraction requires optional PyTorch/SpeechBrain dependencies.
  Saved NumPy embeddings allow ONNX-only conversion.
- `model.safetensors` contains conversion networks only; training losses,
  discriminator weights, and optimizer state are deliberately excluded.
- `onnx/tvtsyn_step_100ms.onnx` and its JSON sidecar contain the fixed-step graph
  and explicit state contract. This is a float32 export, not a quantized model.
- `config.json` records architecture, pinned speaker dependencies, artifact
  hashes, and the source checkpoint SHA-256.

## Training and validation

The accompanying paper reports training the content encoder and waveform
synthesizer on approximately 600 hours of English LibriTTS speech. The
separate SpeechBrain speaker encoders were trained on VoxCeleb. The exact
release artifact is converted from `resynthesis_quant_and_tvtimbre_ft.ckpt`;
its checksum in `config.json` identifies this checkpoint, rather than an
assumption that its training step equals the paper's schedule.

Local automated checks reproduce the original whole-utterance output and
streaming prefix exactly on two captured examples. PyTorch/ONNX streaming
maximum absolute error was 0.000166 over a 30-second CPU stream, within the
0.0003 absolute / 0.001 relative tolerance. Whole and streaming modes are
separate reference behaviors and need not produce identical waveforms.

On one AMD EPYC 7543 system, measured mean processing per 100 ms chunk was
148.2 ms with PyTorch (four threads) and 47.0 ms with ONNX Runtime (one thread).
These measurements exclude loading and speaker extraction, and do not promise
real-time operation on other hardware. See the repository validation reports
for setup, limitations, and the author acceptance record.

## Limitations

English read speech is the primary training domain. Quality and target
similarity may vary with accents, languages, noise, microphones, reference
length, and out-of-domain inputs. Speaker pooling includes small stochastic
noise; save embeddings when comparing executions. Conversion alone is not a
privacy guarantee, and the release API does not implement an anonymization
policy. The author accepted the listening review on 2026-10-11. Physical microphone
hardware was not tested and was explicitly waived as a release prerequisite;
automated browser tests used a simulated microphone with both backends.

The expanded 30-second listening case had a localized maximum backend
difference of 0.001117 with mean error near 1e-6. It met the separately
documented extended listening criterion (absolute 0.002, relative 0.001,
mean absolute error at most 0.000005). Original reference tolerances were
kept unchanged.

## Citation

```bibtex
@article{quamer2026tvtsyn,
  title={TVTSyn: Content-Synchronous Time-Varying Timbre for Streaming Voice Conversion and Anonymization},
  author={Quamer, Waris and Tseng, Mu-Ruei and Nasrallah, Ghady and Gutierrez-Osuna, Ricardo},
  journal={arXiv preprint arXiv:2602.09389},
  year={2026}
}
```
