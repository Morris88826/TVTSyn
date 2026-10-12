# Export a trained model

Convert your Lightning checkpoint to the format used by `TVTSyn.from_pretrained()`.
Run from the repository root:

```bash
pip install -e '.[export]'
tvtsyn-export --checkpoint ckpts/resynthesis_quant_and_tvtimbre_ft.ckpt \
  --output artifacts/TVTSyn
```

The output contains PyTorch weights (`model.safetensors`), the ONNX streaming
model, `config.json`, a model card, and a license file. Use `--pytorch-only`
to skip ONNX export. Only conversion networks are saved; optimizer state and
training losses are excluded.

Load the result with either backend:

```python
from tvtsyn_inference import TVTSyn
model = TVTSyn.from_pretrained("artifacts/TVTSyn", backend="pytorch")
```

Export accepts trusted local checkpoints. Review the generated model card and
license before distributing a model you trained. The reference comparison and
export checks are available on the
[research-tools branch](https://github.com/PSI-TAMU/TVTSyn/tree/research-tools/inference/export).
