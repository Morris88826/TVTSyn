# Training TVTSyn

Run commands from the repository root using Python 3.10 or newer. Install PyTorch
and matching torchaudio for your hardware, then:

```bash
pip install -r training/requirements.txt
```

Training dependencies include Lightning, logging, loss functions, data tools,
and Audiotools. They are separate from the public inference installation.
Reusable networks live in `tvtsyn`; Lightning modules and data modules live here.

## Data preparation

Follow the preprocessing procedure in [streamVC](https://github.com/PSI-TAMU/streamVC).
Create `data/metadata/train_files.txt` with one waveform path per line, relative
to the repository root or absolute. Override `data.init_args.data_file` for your
own manifests. Preprocessed speaker directories contain `wav/`, `code/` (`.km`),
`pitch/` (`.pit.npy`), `energy/` (`.eng.npy`), and `spkr/` (`.spk.npy`).
The synthesis stage expects x-vector followed by ECAPA, totaling 704 dimensions.
Training datasets and generated features are local files, excluded from Git.

## Stage 1: content encoder

```bash
python -m training.main fit \
  -c training/configs/data/pasr.yaml \
  -c training/configs/model/pasrwithquant_la4.yaml \
  --trainer.logger.name encoder
```

`la4` means four 20 ms lookahead frames. Adjust the model's
`transformer_args.right_context` only when training a different model variant.
Check the metric names of custom training variants before changing callbacks.

## Stage 2: synthesizer

```bash
python -m training.main fit \
  -c training/configs/data/resynthesis.yaml \
  -c training/configs/model/resynthesisquantandtvt_la4.yaml \
  --model.init_args.encoder_path ckpts/mimi_pasrv2_withquant_la4.ckpt \
  --trainer.logger.name synthesizer
```

## Resume

Use the same data/model configs with `--ckpt_path path/to/checkpoint.ckpt`.
Lightning restores optimizer and scheduler state. For stage 2, pass
`--model.init_args.encoder_path null` when resuming a complete checkpoint to
avoid requiring its historical encoder checkpoint path. The `modules.*` alias, historical `MimiPasrV2WithQuantModule`
class name, and old dataclass pickle path are retained for compatibility.

The original training checkpoints remain local under `ckpts/`. They are distinct
from inference-only release weights. To build the release bundle, follow
[the export instructions](../inference/export/README.md).

## Research variants

All existing experimental configurations remain under `configs/`, including
supervised wav2wav, alternate waveform decoders, and excitation conditioning.
See [decoder variants](DECODER_VARIANTS.md). They are not advertised as supported
architectures for the first public streaming bundle. WORLD preprocessing uses
`python -m training.scripts.preprocess_parallel_excitation --help`.

Training logs go to `logs/`. Logging defaults use Weights & Biases; override
`--trainer.logger false` for local checks. Device selection defaults to automatic
with one device and can be overridden with Lightning trainer arguments.
