"""Convert a trusted local Lightning checkpoint into portable inference artifacts."""
import argparse
import hashlib
import json
from pathlib import Path


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def export_checkpoint(checkpoint, output, *, onnx=True):
    import torch
    from safetensors.torch import save_file
    from tvtsyn.models.inference import TVTSynModel, architecture_from_hparams
    from .streaming.tensorized_step import TVTSynStreamingStep
    from .streaming.state_schema import init_tvtsyn_state, make_tvtsyn_state_schema
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    # This command is only for maintainer-owned, trusted Lightning checkpoints.
    data = torch.load(checkpoint, map_location='cpu', weights_only=False)
    architecture = architecture_from_hparams(data['hyper_parameters'])
    model = TVTSynModel(architecture).eval()
    excluded = model.load_training_state(data['state_dict'])
    save_file({k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}, str(out / 'model.safetensors'))
    config = {
        'format_version': 1, 'sample_rate': 16000, 'chunk_samples': 1600,
        'license': 'cc-by-nc-4.0', 'audio_channels': 1,
        'supported_modes': {'pytorch': ['whole', 'streaming'], 'onnx': ['streaming']},
        'lookahead_frames': architecture['transformer_args']['right_context'],
        'speaker_dim': 704, 'tensorized_state_version': 2,
        'architecture': architecture,
        'speaker_models': [
            {'repo_id': 'speechbrain/spkrec-xvect-voxceleb', 'revision': '56895a2df401be4150a159f3a1c653f00051d477'},
            {'repo_id': 'speechbrain/spkrec-ecapa-voxceleb', 'revision': '0f99f2d0ebe89ac095bcc5903c4dd8f72b367286'},
        ],
        'artifacts': {'pytorch': ['model.safetensors']},
        'provenance': {'checkpoint': Path(checkpoint).name, 'checkpoint_sha256': sha256(checkpoint),
                       'torch_version': torch.__version__, 'excluded_training_keys': excluded},
    }
    if onnx:
        graph = out / 'onnx/tvtsyn_step_100ms.onnx'
        graph.parent.mkdir(exist_ok=True)
        step = TVTSynStreamingStep(model, chunk_samples=1600).eval()
        if not step._state_names:
            raise ValueError('Streaming state initialization failed')
        state = init_tvtsyn_state(step, 1, 1600, torch.device('cpu'), torch.float32)
        schema = make_tvtsyn_state_schema(step, 1, 1600, torch.device('cpu'), torch.float32)
        inputs = ['wav_chunk', 'speaker_embed', *[f'state_{i:03d}' for i in range(len(state))]]
        outputs = ['y_chunk', *[f'next_state_{i:03d}' for i in range(len(state))]]
        with torch.inference_mode():
            torch.onnx.export(step, (torch.zeros(1, 1, 1600), torch.zeros(1, 704), *state),
                              str(graph), opset_version=17, input_names=inputs,
                              output_names=outputs, dynamic_axes=None, dynamo=False)
        metadata = {'format': 'onnx', 'opset': 17, 'tensorized_state_version': 2,
                    'input_names': inputs, 'output_names': outputs, 'state_schema': schema.to_jsonable(),
                    'checkpoint_sha256': config['provenance']['checkpoint_sha256']}
        graph.with_suffix('.onnx.json').write_text(json.dumps(metadata, indent=2) + '\n')
        config['artifacts']['onnx'] = [str(p.relative_to(out)) for p in sorted(graph.parent.iterdir()) if p.is_file()]
    config['provenance']['artifact_sha256'] = {name: sha256(out / name)
        for names in config['artifacts'].values() for name in names}
    (out / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
    package = Path(__file__).parent
    (out / 'README.md').write_text((package / 'model_card.md').read_text())
    (out / 'LICENSE').write_text((package / 'model_license.txt').read_text())
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True, help='Trusted local Lightning checkpoint')
    parser.add_argument('--output', required=True)
    parser.add_argument('--pytorch-only', action='store_true')
    args = parser.parse_args()
    import torch
    torch.set_num_threads(4)
    export_checkpoint(args.checkpoint, args.output, onnx=not args.pytorch_only)


if __name__ == '__main__':
    main()
