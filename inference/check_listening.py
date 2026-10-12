"""Check that a human review applies to the current, unchanged listening pack."""
import argparse
import hashlib
import json
from pathlib import Path


def check(results, manifest, model_root):
    errors=[]
    if results.get('review_id') != manifest['review_id']:
        errors.append('Review ID does not match this listening pack')
    current=hashlib.sha256((Path(model_root)/'config.json').read_bytes()).hexdigest()
    if current != manifest['config_sha256']:
        errors.append('Model configuration changed after audio generation; regenerate and review')
    if results.get('checkpoint_sha256') != manifest['checkpoint_sha256']:
        errors.append('Checkpoint differs from the reviewed model')
    if not str(results.get('reviewer','')).strip(): errors.append('Reviewer name is missing')
    expected={c['id'] for c in manifest['cases']}
    cases=results.get('cases',[])
    if len(cases)!=len(expected) or {c.get('id') for c in cases}!=expected:
        errors.append('The review must contain every case exactly once')
    for case in cases:
        if case.get('result')!='pass': errors.append(f"Listening case {case.get('id')} is not passed")
    for case in manifest['cases']:
        if not case['release_tolerance_passed']:
            errors.append(f"Numerical comparison for {case['id']} requires investigation")
    if results.get('microphone',{}).get('result')!='pass':
        errors.append('Physical microphone / stop / restart / isolation check is not passed')
    config=json.loads((Path(model_root)/'config.json').read_text())
    for name,expected_hash in config['provenance']['artifact_sha256'].items():
        with (Path(model_root)/name).open('rb') as f:
            actual=hashlib.file_digest(f,'sha256').hexdigest() if hasattr(hashlib,'file_digest') else hashlib.sha256(f.read()).hexdigest()
        if actual!=expected_hash: errors.append('Model artifact changed: '+name)
    return errors


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('results');p.add_argument('--pack',default='artifacts/listening');p.add_argument('--model',default='artifacts/TVTSyn')
    a=p.parse_args()
    errors=check(json.loads(Path(a.results).read_text()),json.loads((Path(a.pack)/'manifest.json').read_text()),a.model)
    if errors: raise SystemExit('\n'.join(errors))
    print('Listening review is complete and matches the model artifacts. Nothing has been published.')


if __name__=='__main__': main()
