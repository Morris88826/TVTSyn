# Source and artifact attribution

The repository's existing `LICENSE` is retained unchanged. Existing file-level
copyright and attribution notices are preserved, including Kyutai/Mimi,
Meta/Audiocraft, Amphion, and the Lightning template. This reorganization does
not assign a new license to those components, datasets, or external weights.

Speaker extraction downloads these independently distributed models:

- https://huggingface.co/speechbrain/spkrec-xvect-voxceleb
- https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb

Their pinned revisions are recorded in each release bundle. They are referenced
rather than repackaged as TVTSyn weights. See their model cards for attribution,
training data, and license information. See `samples/README.md` for audio provenance.

The TVTSyn weight license selected by the authors is CC BY-NC 4.0, included
in the exported bundle as `LICENSE`. The code license is unchanged. The model
card now documents training data from the accompanying paper, artifact
provenance, numerical validation, limitations, and citation. Human listening
validation and authenticated publication remain separate release steps.
The demonstration recordings are verified publisher originals with their
own `samples/COPYING` notice and SHA-256 provenance.
