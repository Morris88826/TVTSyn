from pathlib import Path

from .config import load_config


def resolve_model(model_id, backend, *, revision=None, cache_dir=None, local_files_only=False, token=None):
    if backend not in ('pytorch', 'onnx'):
        raise ValueError('backend must be pytorch or onnx')
    local = Path(model_id).expanduser()
    if local.is_dir():
        root = local
    else:
        from huggingface_hub import hf_hub_download, snapshot_download
        config_path = hf_hub_download(str(model_id), 'config.json', revision=revision,
                                     cache_dir=cache_dir, local_files_only=local_files_only, token=token)
        config = load_config(config_path)
        # Resolve all files against the same commit, even if a branch moves mid-download.
        if backend not in config['artifacts']:
            raise ValueError(f'This bundle does not contain the {backend} backend')
        resolved_revision = Path(config_path).parent.name
        root = Path(snapshot_download(str(model_id), revision=resolved_revision,
                                      allow_patterns=['config.json', *config['artifacts'][backend]],
                                      cache_dir=cache_dir, local_files_only=local_files_only, token=token))
    config = load_config(root / 'config.json')
    if backend not in config['artifacts']:
        raise ValueError(f'This bundle does not contain the {backend} backend')
    for name in config['artifacts'][backend]:
        if not (root / name).is_file():
            raise FileNotFoundError(root / name)
    return root, config
