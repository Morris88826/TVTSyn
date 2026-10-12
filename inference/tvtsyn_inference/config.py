"""Versioned public artifact contract."""
import json
from pathlib import Path


def load_config(path):
    config = json.loads(Path(path).read_text())
    expected = {'format_version': 1, 'sample_rate': 16000, 'chunk_samples': 1600,
                'speaker_dim': 704, 'tensorized_state_version': 2}
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f'Unsupported {key}: {config.get(key)!r}; expected {value}')
    architecture = config['architecture']
    if architecture.get('decoder_type', 'seanet') != 'seanet':
        raise ValueError('The public streaming release currently supports the SEANet decoder only')
    if architecture.get('sample_rate', 16000) != 16000 or architecture.get('frame_rate', 50) != 50:
        raise ValueError('Architecture audio rates disagree with the release contract')
    lookahead = architecture['transformer_args']['right_context']
    if config.get('lookahead_frames') != lookahead or lookahead != 4:
        raise ValueError('The v1 release requires the validated LA4 encoder')
    for paths in config['artifacts'].values():
        for name in paths:
            if not name or Path(name).is_absolute() or '..' in Path(name).parts:
                raise ValueError(f'Artifact must have a relative path: {name!r}')
    return config
