"""Portable research dataset locations and explicit device selection."""
import json
import os
from pathlib import Path


def dataset_roots(root=None):
    base = Path(root or os.environ.get('TVTSYN_DATA_ROOT', 'data/corpora')).expanduser()
    paths = {
        'artic': base / 'L2Artic/all_data_for_ac_vc',
        'l2artic': base / 'L2Artic/all_data_for_ac_vc',
        'vctk': base / 'VCTK-Corpus/wav48',
        'l2en': base / 'Multilingual_dataset/data/L2EN',
        'indictts': base / 'IndicTTS',
        'nptel': base / 'nptel/data',
    }
    for suffix, language in [('eng', 'English'), ('fin', 'Finnish'),
                             ('ger', 'German'), ('man', 'Mandarin')]:
        paths['emime-' + suffix] = base / 'EMIME/UEDIN_bilingual_96kHz_data' / (language + '_talkers')
    override = os.environ.get('TVTSYN_DATASETS_JSON')
    if override:
        values = json.loads(Path(override).read_text())
        paths.update({key: Path(value).expanduser() for key, value in values.items()})
    return {key: str(value) for key, value in paths.items()}


def resolve_device(requested='cpu', local_rank=0, world_size=1):
    if requested == 'cuda' and world_size > 1:
        return f'cuda:{local_rank}'
    return requested
