# Publication after listening validation

Planned model ID: `warisqr007/TVTSyn`.
Planned Space ID: `warisqr007/TVTSyn`.
TVTSyn weight license: **CC BY-NC 4.0**. The repository code retains its existing
license. No repository has been created or uploaded by this preparation step.

## Before uploading

1. Finish the listening and physical-microphone review in
   [LISTENING_VALIDATION.md](LISTENING_VALIDATION.md), including investigation
   of any failed numerical checks. Validate the downloaded review JSON.
2. Record the accepted review, validation results, and source revision.
3. Commit the reviewed source changes so the deployment can use an immutable
   revision. GitHub authentication and the backed-up `attr_modify` branch
   deletion are complete.
4. Authenticate the Hugging Face CLI locally as `warisqr007` with a token
   permitted to create/upload model and Space repositories. Do not commit
   tokens or paste them into source files.

The prepared `artifacts/TVTSyn` bundle contains config, inference safetensors,
ONNX graph/state schema, a complete model card, and the weight license. Review
that directory explicitly; do not upload `artifacts/` wholesale, since it also
contains local validation audio and private/custom review inputs.

## Publication sequence

- Upload `artifacts/TVTSyn` to the model repository and record the resulting
  commit hash. Training Lightning checkpoints are separate from the
  inference weights; the public inference loader expects the exported bundle.
- In fresh PyTorch and ONNX environments, load the actual model ID at that
  commit. Exercise backend-selective downloading, cached/offline loading,
  and a conversion. Keep the saved-embedding ONNX environment free of torch.
- Build a wheel from the committed source and prepare the Gradio Space with
  `spaces/demo/app.py`, FFmpeg system dependencies, and that pinned package.
  Set `TVTSYN_MODEL=warisqr007/TVTSyn` and `TVTSYN_REVISION` to the model commit.
  Choose hardware based on measured latency; CPU PyTorch is slower than
  real time on the tested setup.
- Deploy and test uploads, real microphone playback, stop/restart, and session
  isolation at the deployed URL. Record browser/network latency separately.
- Replace the root README's local release-status text with the working Hub
  and Space links and revision-pinned examples only after these checks pass.

## GitHub authentication and completed branch cleanup

Completed on 2026-10-11 using `gh` from the darkstream environment. The remote
branch is deleted and its complete history remains in
`release-backups/attr_modify.bundle`. The commands below are retained for
reference; deletion does not need to be repeated.

Browser login is separate from command-line credentials. With GitHub CLI:

```bash
gh auth login --hostname github.com --git-protocol https --web
gh auth setup-git
gh auth status
```

If `gh` is unavailable in a conda environment, install it using
`conda install -c conda-forge gh`. Once authentication works:

```bash
git bundle verify release-backups/attr_modify.bundle
git push --force-with-lease=refs/heads/attr_modify:17af7d9f2d70f9ca8673432a11f6851917159f9f origin :refs/heads/attr_modify
```

The lease prevents deletion if someone advances the branch. Its complete
history can be restored from the local bundle, without rewriting main.
