import importlib.util
import hashlib
import json
from pathlib import Path

spec=importlib.util.spec_from_file_location('review_check',Path(__file__).parents[1]/'inference/check_listening.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def test_review_requires_all_cases_microphone_and_matching_model(tmp_path):
    config={'provenance':{'artifact_sha256':{}}}
    raw=json.dumps(config);(tmp_path/'config.json').write_text(raw)
    manifest={'review_id':'review','checkpoint_sha256':'checkpoint','config_sha256':hashlib.sha256(raw.encode()).hexdigest(),
              'cases':[{'id':'forward','reference_tolerance_passed':True,'release_tolerance_passed':True}]}
    results={'review_id':'review','checkpoint_sha256':'checkpoint','reviewer':'Tester',
             'cases':[{'id':'forward','result':'pass'}],'microphone':{'result':'pass'}}
    assert not module.check(results,manifest,tmp_path)
    results['cases'][0]['result']='pending'
    assert any('not passed' in e for e in module.check(results,manifest,tmp_path))
    results['cases'][0]['result']='pass';results['microphone']['result']='pending'
    assert any('microphone' in e for e in module.check(results,manifest,tmp_path))
    (tmp_path/'config.json').write_text(json.dumps(config,indent=2))
    assert any('configuration changed' in e for e in module.check(results,manifest,tmp_path))
