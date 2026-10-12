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
training data, and license information. See [samples/README.md](../samples/README.md) for audio provenance.

TVTSyn model weights use CC BY-NC 4.0; the full license is included in the
model repository. Demonstration recordings retain their own
[`samples/COPYING`](../samples/COPYING) notice and
[provenance](../samples/provenance.json).
