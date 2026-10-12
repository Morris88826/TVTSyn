# Publication — 2026-10-11

Model: [warisqr007/TVTSyn](https://huggingface.co/warisqr007/TVTSyn).
Model revision: `2d3147b8a465e1d9aad33a2a0e53956bf8ed32aa`.
Space: [warisqr007/TVTSyn](https://huggingface.co/spaces/warisqr007/TVTSyn),
created on ZeroGPU; deployment validation is in progress.

The author accepted all six listening cases and explicitly waived physical
microphone hardware testing. See [release approval](release_approval.json).
Weights use **CC BY-NC 4.0**; the source retains CC0. The Hub contains inference
safetensors, the ONNX graph/state schema, configuration, model card, and license.
Training checkpoints and local listening audio were not uploaded.

Anonymous downloads at the pinned commit passed in separate clean environments:
PyTorch whole utterance and streaming, ONNX streaming without PyTorch installed,
backend-selective downloads, and cached/offline reload. See
[publication checks](publication_validation.json).

## Reproduce the Space deployment

Build a wheel from a committed checkout and stage the app:

```bash
python -m build --wheel --outdir artifacts/dist
python spaces/demo/prepare_space.py \
  --wheel artifacts/dist/tvtsyn-0.1.0-py3-none-any.whl \
  --revision 2d3147b8a465e1d9aad33a2a0e53956bf8ed32aa \
  --source-revision "$(git rev-parse HEAD)" --hardware zero-a10g
```

The staging directory `artifacts/space` contains the application, wheel,
dependencies, license, a README derived from the model card, and `release.json`
with the source commit and wheel checksum. Set Space variable
`TVTSYN_ZEROGPU=1` and upload only this directory to the Gradio Space.
`prepare_space.py` also supports `--hardware cpu-basic` for a CPU deployment.

ZeroGPU allocates a GPU for each PyTorch file conversion. ONNX conversion,
reference-speaker preparation, and live sessions run on CPU. Keeping live
state on CPU avoids losing caches across GPU leases. Source clips are bounded
to 60 seconds in the hosted app. ZeroGPU queues and per-user quotas can delay or
reject file conversion; CPU PyTorch live inference can be slower than real time.

The first cpu-basic creation attempt returned HTTP 402. At the author’s request,
ZeroGPU was selected and repository creation succeeded. No paid hardware or
subscription was purchased. See [ZeroGPU hosting requirements](https://huggingface.co/docs/hub/spaces-zerogpu).

## Repository cleanup

Generated data, checkpoints, notebook outputs, environments, and branch backups
remain local and ignored. The reviewed notebook with its ratings is preserved
under `artifacts/publication/`. Source notebooks are clean templates.

`origin/attr_modify` was archived in the verified local
`release-backups/attr_modify.bundle`, then deleted with a lease against tip
`17af7d9f2d70f9ca8673432a11f6851917159f9f`. It was not merged and main history
was not rewritten. Restore a local copy if needed:

```bash
git fetch release-backups/attr_modify.bundle refs/remotes/origin/attr_modify:refs/heads/attr_modify
```
