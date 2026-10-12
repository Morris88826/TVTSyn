import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location('listening_notebook', Path(__file__).parents[1] / 'inference/listening_notebook.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


@pytest.fixture
def pack(tmp_path):
    (tmp_path / 'config.json').write_text('{}')
    manifest = {'review_id': 'review-1', 'model': 'local', 'checkpoint_sha256': 'checkpoint',
                'config_sha256': hashlib.sha256(b'{}').hexdigest(), 'cases': [{'id': 'forward'}]}
    (tmp_path / 'manifest.json').write_text(json.dumps(manifest))
    return tmp_path


def test_save_and_resume_does_not_reset_ratings(pack):
    path = pack / 'results.json'
    manifest, review = module.load_review(pack, pack, path)
    assert review['cases'][0]['result'] == 'pending'
    review['reviewer'] = 'Test reviewer'
    module.rate_case(review, 'forward', result='unsure', notes='Click at 1.2 seconds')
    module.save_review(review, path)
    _, resumed = module.load_review(pack, pack, path)
    module.rate_case(resumed, 'forward')
    assert resumed['cases'][0] == {'id': 'forward', 'result': 'unsure', 'notes': 'Click at 1.2 seconds'}
    assert resumed['microphone']['result'] == 'pending'
    assert resumed['review_id'] == manifest['review_id']
    assert resumed['reviewed_at']


def test_preserve_review_for_another_pack(pack):
    path = pack / 'results.json'
    original = '{"review_id": "another-review"}'
    path.write_text(original)
    with pytest.raises(ValueError, match='different listening pack'):
        module.load_review(pack, pack, path)
    _, review = module.load_review(pack, pack, pack / 'unused.json')
    review['reviewer'] = 'Test reviewer'
    with pytest.raises(ValueError, match='different pack'):
        module.save_review(review, path)
    assert path.read_text() == original


def test_missing_reviewer_and_invalid_rating_do_not_save(pack):
    path = pack / 'results.json'
    _, review = module.load_review(pack, pack, path)
    with pytest.raises(ValueError, match='reviewer'):
        module.save_review(review, path)
    with pytest.raises(ValueError, match='Result'):
        module.rate_case(review, 'forward', result='maybe')
    assert not path.exists()
