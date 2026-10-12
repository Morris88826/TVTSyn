"""Notebook review helpers: embedded audio and compatible, resumable JSON results."""
from datetime import datetime, timezone
import hashlib
from html import escape
import json
from pathlib import Path

RESULTS = {'pending', 'pass', 'fail', 'unsure'}


def load_review(pack, model_root, results_path):
    pack, model_root, results_path = map(Path, (pack, model_root, results_path))
    manifest = json.loads((pack / 'manifest.json').read_text())
    if hashlib.sha256((model_root / 'config.json').read_bytes()).hexdigest() != manifest['config_sha256']:
        raise ValueError('The model config changed after the listening pack was generated. Regenerate the pack first.')
    review = {
        'review_id': manifest['review_id'], 'model': manifest['model'],
        'checkpoint_sha256': manifest['checkpoint_sha256'], 'reviewed_at': None,
        'reviewer': '', 'environment': '',
        'cases': [{'id': c['id'], 'result': 'pending', 'notes': ''} for c in manifest['cases']],
        'microphone': {'result': 'pending', 'notes': ''},
    }
    if results_path.exists():
        saved = json.loads(results_path.read_text())
        if (saved.get('review_id'), saved.get('checkpoint_sha256')) != (review['review_id'], review['checkpoint_sha256']):
            raise ValueError('Saved results belong to a different listening pack. Choose a new RESULTS_PATH; keep the old review.')
        if sorted(c['id'] for c in saved['cases']) != sorted(c['id'] for c in review['cases']):
            raise ValueError('Saved results do not contain exactly the expected cases')
        review = saved
    return manifest, review


def rate_case(review, case_id, result=None, notes=None):
    """None preserves an existing rating/note when a notebook cell is rerun."""
    if result is not None and result not in RESULTS:
        raise ValueError(f'Result must be one of {sorted(RESULTS)}')
    case = next(c for c in review['cases'] if c['id'] == case_id)
    if result is not None:
        case['result'] = result
    if notes is not None:
        case['notes'] = notes
    print(f"{case_id}: {case['result']} — {case['notes']}")


def show_case(pack, manifest, case_id, *, start=None, end=None):
    from IPython.display import Audio, HTML, display
    case = next(c for c in manifest['cases'] if c['id'] == case_id)
    display(HTML(f"<h3>{escape(case['title'])}</h3><p>{escape(case['purpose'])}</p>"))
    print(f"Shared playback gain: {case['playback_gain']:.4f}; max backend difference: {case['streaming_max_abs']:.6g}")
    print(f"Original reference tolerance: {case['reference_tolerance_passed']}; extended listening tolerance: {case['release_tolerance_passed']}")
    for label, name in case['audio'].items():
        if start is not None and label == 'Target reference':
            continue  # This reference is shorter than the long-stream excerpt.
        display(HTML(f'<b>{escape(label)}</b>'))
        path = Path(pack) / name
        if start is None:
            # Embed the exact PCM playback file: no HTTP server and no new normalization.
            display(Audio(filename=str(path), embed=True))
        else:
            import soundfile as sf
            audio, sr = sf.read(path, dtype='float32')
            clip = audio[int(start * sr):int(end * sr) if end is not None else None]
            display(Audio(clip, rate=sr, normalize=False))


def show_summary(review):
    from IPython.display import HTML, display
    rows = ''.join('<tr>' + ''.join(f'<td>{escape(str(c[k]))}</td>' for k in ['id', 'result', 'notes']) + '</tr>'
                   for c in review['cases'])
    display(HTML('<table><tr><th>Case</th><th>Result</th><th>Notes</th></tr>' + rows + '</table>'))
    print('Microphone:', review['microphone']['result'])


def save_review(review, path):
    if not review['reviewer'].strip():
        raise ValueError('Enter your reviewer name or initials before saving')
    path = Path(path)
    if path.exists():
        existing = json.loads(path.read_text())
        if existing.get('review_id') != review['review_id']:
            raise ValueError('Refusing to replace a review for a different pack')
    review['reviewed_at'] = datetime.now(timezone.utc).isoformat()
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.json.tmp')
    temp.write_text(json.dumps(review, indent=2) + '\n')
    temp.replace(path)
    print('Saved:', path)
