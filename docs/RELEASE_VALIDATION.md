# Release validation — 2026-10-11

The local bundle is `artifacts/TVTSyn`. It contains inference-only safetensors,
a regenerated 100 ms ONNX graph, its version-2 state schema, configuration,
provenance hashes, model card, and license. The same bundle is published at
[warisqr007/TVTSyn](https://huggingface.co/warisqr007/TVTSyn).
The Space was created on the author-selected ZeroGPU tier; hosted validation
is in progress. See [publication status](PUBLICATION.md).

A subsequent repair and CPU-only validation pass is summarized in
[RELEASE_AUDIT.md](RELEASE_AUDIT.md). It adds the ready-to-run listening review,
fixes discovered edge cases, and documents an extended numerical case separately.

## Reference and numerical checks

Release checkpoint: `resynthesis_quant_and_tvtimbre_ft.ckpt`

SHA-256:
`2ba91ff4f6fbe143744980902a0165cb94edd94523c1e53283beb123d3e8e464`

Reference code was captured from commit `bfea2a3` before migration. The snapshot
needed a Python 3.11+ dataclass default-factory compatibility fix; the network
computation was unchanged. Configuration, checkpoint hash, saved speaker
embedding, and two three-second source/reference pairs are in
`artifacts/reference/`. The old implementation loaded with no missing or
unexpected checkpoint weights.

- Extracted and migrated PyTorch whole-utterance outputs: exact match on both clips.
- Tensorized PyTorch streaming: exact match to both original emitted prefixes.
- Final flushing returns all 48,000 input samples; the original notebook omitted
  the final pending streaming chunk.
- The old ONNX artifact was numerically close but had no checkpoint provenance.
  It was regenerated directly from the selected checkpoint.
- Regenerated ONNX versus PyTorch: maximum absolute error 0.000137 on the two
  reference clips; 0.000166 over a 30-second stream, including cache wraparound.
- Float32 acceptance: whole-mode absolute tolerance 0.00005 / relative 0.0001;
  streaming absolute tolerance 0.0003 / relative 0.001. No quantized export is
  advertised as the default release.

See [numerical and timing metrics](validation_metrics.json). Comparison WAVs
are saved under `artifacts/reference/` for listening. Automated checks cover
numerical agreement, finite outputs, lengths, and chunk boundaries through the
reference comparison; the author accepted the six-case listening review. See
[the acceptance record](release_approval.json).

## Runtime and packaging

- 145 tests pass after the follow-up repairs across the existing training suite and new inference/demo tests.
- Real backend tests cover short clips, partial final frames, reset, interleaved
  sessions, saved embeddings, metadata rejection, and offline artifact loading.
- Persistent resampling is checked against whole-file resampling at 8, 44.1,
  and 48 kHz. Empty resampled streams and integer/stereo audio are covered.
- The wheel was installed in two fresh environments: ONNX-only with no torch,
  and CPU PyTorch with no Lightning or SpeechBrain. Actual local/offline
  conversion passed in both environments.
- PyTorch whole and streaming inference also passed on an RTX 5000 Ada GPU.
  The public runtime explicitly uses eager execution; clean-wheel conversion
  also passed with `torch.compile` forbidden, without environment overrides.
- Pinned SpeechBrain reference-audio extraction returned a finite `[1, 704]`
  embedding and completed an ONNX conversion. Speaker cache directories are
  separated by model revision, and offline extraction passed. SpeechBrain x-vector
  pooling adds epsilon noise even in eval mode; seeded replay was identical.
  Save embeddings for exact repeatability. All new notebook code cells executed successfully.
- Hub tests verify backend-selective downloads, inclusion of external ONNX
  weights, and consistent commit resolution across configuration/artifact fetches.
  Actual anonymous downloads at release commit
  `2d3147b8a465e1d9aad33a2a0e53956bf8ed32aa` passed in separate clean environments:
  both PyTorch modes, ONNX streaming without torch installed, backend-selective
  downloads, and cached/offline reload. See [publication results](publication_validation.json).

## Training preservation

Both original checkpoints load through the migrated training classes. CPU
resume tests restore model, optimizer, and scheduler state and execute further
steps on existing local preprocessed examples:

| Stage | Original step | Resumed step | Optimizers |
| --- | ---: | ---: | ---: |
| Content encoder | 679830 | 679832 | 1 |
| Synthesizer | 877200 | 877202 | 2 |

For the direct encoder resume test, the optimizer and ReduceLROnPlateau scheduler
normally injected by LightningCLI were supplied from the stage-1 configuration.
Validation/logging were disabled for these bounded smoke runs. The migrated
training CLI also parses the documented stage-1 configuration successfully.
These checks establish resume compatibility, not a full retraining result.
See [training results](training_validation.json).

## Local demo and performance

The Gradio app starts over HTTP. Its file conversion endpoint was exercised
using the Gradio client with real artifacts. Real PCM streaming callbacks,
flush, restart, concurrent-session isolation, expiry, and recording limits were
checked. A Chromium browser with a simulated microphone streamed the source
WAV through both ONNX and GPU PyTorch in the live UI, verified advancing audio
playback, and flushed on Stop without page errors.
The prepared speaker remains available after stopping; a second recording
started without re-preparation. FFmpeg and ffprobe were present on PATH. This exercises the actual
browser flow; physical microphone hardware was not tested and was explicitly waived by the
author. Human listening was accepted separately.
See [demo results](demo_validation.json).

Measurements below are a local 30-second CPU smoke run on an AMD EPYC 7543
system, excluding model loading and target-speaker extraction. Ten initial
chunks were excluded from steady-state statistics. This was not an isolated,
cross-hardware benchmark.

| Backend | CPU threads | Mean processing / 100 ms chunk | p95 | RTF |
| --- | ---: | ---: | ---: | ---: |
| PyTorch | 4 | 148.2 ms | 164.9 ms | 1.48 |
| ONNX Runtime | 1 | 47.0 ms | 47.4 ms | 0.47 |

The first valid output required 200 ms of input in the fixed-step scheduler,
before computation and any resampling/browser/network delay. The encoder
lookahead is 80 ms. Do not interpret chunk size as end-to-end latency.

Test environment: Python 3.13.5, torch/torchaudio 2.8.0, ONNX Runtime 1.29.0,
ONNX 1.22.0, SpeechBrain 1.1.1, Gradio 5.50.0, Lightning 2.6.5, NumPy 2.5.2,
soxr 1.1.0, safetensors 0.8.0. The combined training environment resolved
huggingface-hub 0.36.2; the clean inference wheels also passed with 1.33.0.
Audiotools was installed from the pinned source revision in training requirements.

## Branch cleanup

`origin/attr_modify` was archived in `release-backups/attr_modify.bundle` and
`git bundle verify` confirmed a complete history. The local and remote tips
both matched `17af7d9f2d70f9ca8673432a11f6851917159f9f`.

After authentication was configured, remote deletion completed on 2026-10-11
using the GitHub CLI from the darkstream environment. The deletion was guarded
by a lease against the verified tip above. A fresh remote lookup confirmed
that `refs/heads/attr_modify` is absent. No history was rewritten.

The backup can restore a local branch without contacting GitHub:

```bash
git fetch release-backups/attr_modify.bundle refs/remotes/origin/attr_modify:refs/heads/attr_modify
```

Generated artifacts removed from Git tracking remain on disk. No repository
history was rewritten. Source changes are committed on main; the Space wheel
records its exact source revision in `release.json`.
