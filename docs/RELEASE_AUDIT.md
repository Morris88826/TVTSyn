# Release preparation status — 2026-10-11

The release implementation and cleanup are complete. The author accepted all
six listening cases and explicitly waived physical microphone testing; see
[release approval](release_approval.json). The public model is
[warisqr007/TVTSyn](https://huggingface.co/warisqr007/TVTSyn), with CC BY-NC 4.0
weights and unchanged source licensing.

Anonymous Hub downloads and real conversion passed for both backends, including
ONNX without PyTorch and cached/offline reload. The Space was created on the author-selected ZeroGPU tier. Hosted validation
is in progress. No paid hardware was requested.

## Resolved findings

| Audit finding | Resolution and evidence |
| --- | --- |
| Notebook import failure | Persistent `.venv-listening` CPU environment and registered kernel; setup script and interpreter check; all notebook cells executed successfully. Source outputs cleared. |
| In-place save loses other backend | Same-directory `save_pretrained()` is a no-op preserving the complete manifest; regression test added. |
| CLI assumes `fit` | Hook uses the selected subcommand; tests cover fit/validate/test/predict. Actual validate command initialized and completed with a local sample manifest. |
| Checkpoint compatibility | Added the exact historical `MimiPasrV2WithQuantModule` alias required by Lightning 2.6.6. Both checkpoints restored optimizer state and executed two more training steps. |
| Personal dataset paths and GPU defaults | Research corpus roots are configurable through environment/JSON and CLI; CPU is the research default, with explicit GPU selection. Personal paths removed from source and cleared notebooks. |
| Sample provenance unknown | Retrieved originals and COPYING from the CMU ARCTIC publisher; release samples now match upstream bytes, with checksums in `samples/provenance.json`. Prior local samples preserved. |
| Draft model card/license | Model card now includes architecture/modes, training data from the paper, citation, limitations, numerical checks, and the selected weight license. Full license included in export and wheel resources. |
| Remote branch cleanup | Verified the complete local bundle and unchanged tip, then deleted `origin/attr_modify` using authenticated GitHub CLI from the darkstream environment. A fresh remote lookup confirmed deletion. |
| Full suite blocked by host GPU driver | Created a persistent CPU-only PyTorch/torchaudio environment. Full suite: **145 passed**. No GPU reset or changes to another research environment. |

## Listening setup and validation

- Six cases: forward/reverse conversion, 80 ms input, partial final chunk,
  silence transitions, and a 30-second repeated stream.
- Each case has source/target plus all three conversion modes. Playback gain
  is shared within a case; original float WAVs remain available.
- Browser review supports per-case results, timestamp notes, persistence, and
  JSON download, with a checkpoint/config-bound review ID.
- A VS Code listening notebook embeds all comparison audio, preserves saved
  ratings, and exports the same review JSON without a web server. It leaves
  physical-microphone checks pending until separately completed.
- Browser automation verified actual playback, rating persistence and download.
  It did not supply human ratings. The author subsequently accepted all six cases.
- Local Gradio microphone playback, stop, and restart were retested using a
  simulated microphone with both ONNX and CPU PyTorch. Physical hardware was not tested; the author
  explicitly waived it as a publication prerequisite.
- CPU PyTorch thread count is bounded/configurable in the demo.

### Extended numerical finding

The original captured references still use their original strict tolerances.
The new 30-second listening case exceeds that stricter streaming criterion in
65 of 480,000 samples, with maximum difference 0.00111679 around 21.2 seconds.
All 300 quantizer code selections match between backends; the affected chunk's
latent difference is 4.77e-7. Disabling ONNX optimizations did not remove it.
This is consistent with decoder floating-point sensitivity. It was not hidden
by changing the original reference tests or altering model weights.

The extended listening set has a separately documented criterion of
`atol=0.002, rtol=0.001` and mean absolute error no more than `0.000005`.
All six current cases meet it. The page displays both criteria and the measured
errors. The author subsequently marked all six listening cases as passing.
See the detailed rationale in the listening guide and the diagnosis JSONs in
`artifacts/`.

## Hosted validation

The Space is being deployed on ZeroGPU. Validate uploads and simulated
microphone playback at its hosted URL. Physical microphone testing remains
explicitly waived, not recorded as a pass. See [PUBLICATION.md](PUBLICATION.md).

Generated model/audio artifacts, environments and branch backups remain
ignored; local historical data/checkpoints and reviewed notebook outputs were
preserved. `origin/attr_modify` is archived and deleted; main history is intact.
