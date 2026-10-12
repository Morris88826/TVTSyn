"""Public runtime contracts; real-model tests require TVTSYN_TEST_MODEL."""
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from tvtsyn_inference import TVTSyn
from tvtsyn_inference.audio import mono, speaker_array
from tvtsyn_inference.config import load_config
from tvtsyn_inference.streaming.session import StreamingSession


class DelayedEcho:
    chunk_samples = 1600
    lookahead_frames = 4

    def initial_state(self):
        return None

    def step(self, audio, speaker, state):
        return (np.zeros_like(audio) if state is None else state), audio.copy(), state is not None


@pytest.mark.parametrize('length', [0, 1, 319, 1600, 1601, 4799, 32001])
def test_stream_drains_and_trims(length):
    wave = np.random.default_rng(1).normal(size=length).astype('float32')
    stream = StreamingSession(DelayedEcho(), np.zeros(704))
    pieces = [stream.push(wave[i:i+731]) for i in range(0, length, 731)]
    pieces.append(stream.flush())
    np.testing.assert_array_equal(np.concatenate(pieces), wave)
    assert not stream.flush().size
    with pytest.raises(RuntimeError):
        stream.push(np.ones(1))
    stream.reset()
    np.testing.assert_array_equal(np.concatenate([stream.push(wave), stream.flush()]), wave)


def test_independent_sessions():
    backend = DelayedEcho()
    a = StreamingSession(backend, np.zeros(704))
    b = StreamingSession(backend, np.zeros(704))
    assert not a.push(np.ones(1600)).size
    assert not b.push(np.full(1600, 2.0)).size
    np.testing.assert_array_equal(a.flush(), np.ones(1600))
    np.testing.assert_array_equal(b.flush(), np.full(1600, 2.0))


@pytest.mark.parametrize('sr', [8000, 44100, 48000])
def test_persistent_resampling(sr):
    import soxr
    wave = np.sin(np.arange(sr+123) * 2*np.pi*220/sr).astype('float32')
    stream = StreamingSession(DelayedEcho(), np.zeros(704), sample_rate=sr)
    actual = np.concatenate([stream.push(wave[i:i+997]) for i in range(0, len(wave), 997)] + [stream.flush()])
    expected = soxr.resample(wave, sr, 16000, quality='HQ')
    np.testing.assert_allclose(actual, expected, atol=1e-6)


@pytest.mark.parametrize('value', [np.zeros((2,704)), np.zeros(703), np.full(704,np.nan)])
def test_invalid_speaker(value):
    with pytest.raises(ValueError): speaker_array(value)


def test_pcm_and_stereo():
    np.testing.assert_array_equal(mono(np.array([[32767, -32767]],np.int16)), [0])
    np.testing.assert_array_equal(mono(np.array([0,128,255],np.uint8)), [-1,0,127/128])
    with pytest.raises(ValueError): mono(np.array([np.inf]))


def test_import_without_torch():
    code = '''
import sys
class Block:
    def find_spec(self, fullname, *args):
        if fullname.split('.')[0] in ('torch', 'speechbrain', 'pytorch_lightning'):
            raise RuntimeError('Unexpected training/PyTorch import: ' + fullname)
sys.meta_path.insert(0, Block())
from tvtsyn_inference import TVTSyn
from tvtsyn_inference.backends.onnx import ONNXBackend
assert 'torch' not in sys.modules
'''
    subprocess.run([sys.executable, '-c', code], check=True)


@pytest.fixture(scope='module')
def real_models():
    path = os.environ.get('TVTSYN_TEST_MODEL')
    if not path:
        pytest.skip('Set TVTSYN_TEST_MODEL to a local exported model bundle')
    import torch
    torch.set_num_threads(4)
    return [TVTSyn.from_pretrained(path, backend=backend, local_files_only=True) for backend in ['pytorch','onnx']]


@pytest.mark.parametrize('length', [1, 319, 1600, 1601, 4801, 35200])
def test_real_backend_parity(real_models, length):
    rng = np.random.default_rng(4)
    audio = (rng.normal(size=length)*0.02).astype('float32')
    speaker = rng.normal(size=(1,704)).astype('float32')
    pt, ort = real_models
    a = pt.convert(audio,speaker,mode='streaming')
    b = ort.convert(audio,speaker,mode='streaming')
    assert a.shape == b.shape == (length,)
    assert np.isfinite(a).all() and np.isfinite(b).all()
    np.testing.assert_allclose(a,b,atol=3e-4,rtol=1e-3)


def test_real_reset_and_interleaving(real_models):
    model = real_models[0]
    rng = np.random.default_rng(7)
    x = rng.normal(0,.01,6501).astype('float32')
    s = rng.normal(size=704).astype('float32')
    a,b = model.stream(s),model.stream(s)
    ay=[];by=[]
    for i in range(0,len(x),431):
        ay.append(a.push(x[i:i+431])); by.append(b.push(x[i:i+431]))
    ay.append(a.flush());by.append(b.flush())
    np.testing.assert_array_equal(np.concatenate(ay),np.concatenate(by))
    a.reset()
    np.testing.assert_array_equal(np.concatenate([a.push(x),a.flush()]),np.concatenate(ay))


def test_save_pretrained_offline_and_reject_metadata(real_models,tmp_path):
    model=real_models[1]
    model.save_pretrained(tmp_path)
    reloaded=TVTSyn.from_pretrained(tmp_path,backend='onnx',local_files_only=True)
    assert reloaded.config['artifacts'].keys()=={'onnx'}
    config=json.loads((tmp_path/'config.json').read_text())
    config['chunk_samples']=960
    (tmp_path/'config.json').write_text(json.dumps(config))
    with pytest.raises(ValueError,match='chunk_samples'):
        load_config(tmp_path/'config.json')


def test_whole_mode_and_invalid_mode(real_models):
    pt,onnx=real_models
    for n in [1,319,1601]:
        assert pt.convert(np.zeros(n,np.float32),np.zeros(704)).shape==(n,)
    with pytest.raises(ValueError): onnx.convert(np.zeros(10),np.zeros(704),mode='whole')


def test_hub_download_is_selective_and_revision_consistent(tmp_path,monkeypatch):
    import huggingface_hub
    from tvtsyn_inference.hub import resolve_model
    root=tmp_path/'snapshots'/'abc123'
    root.mkdir(parents=True)
    config={'format_version':1,'sample_rate':16000,'chunk_samples':1600,'speaker_dim':704,
            'tensorized_state_version':2,'lookahead_frames':4,
            'architecture':{'transformer_args':{'right_context':4}},
            'artifacts':{'pytorch':['model.safetensors'],
                         'onnx':['onnx/model.onnx','onnx/model.onnx.json','onnx/model.data']}}
    (root/'config.json').write_text(json.dumps(config))
    for name in config['artifacts']['onnx']:
        path=root/name;path.parent.mkdir(exist_ok=True);path.touch()
    calls=[]
    monkeypatch.setattr(huggingface_hub,'hf_hub_download',lambda *a,**kw:str(root/'config.json'))
    def snapshot(*args,**kwargs):calls.append(kwargs);return str(root)
    monkeypatch.setattr(huggingface_hub,'snapshot_download',snapshot)
    actual,_=resolve_model('example/TVTSyn','onnx',revision='release',local_files_only=True)
    assert actual==root
    assert calls[0]['revision']=='abc123'
    assert calls[0]['allow_patterns']==['config.json',*config['artifacts']['onnx']]
    assert calls[0]['local_files_only'] is True


def test_empty_resampled_stream():
    session=StreamingSession(DelayedEcho(),np.zeros(704),sample_rate=48000)
    assert session.flush().shape==(0,)


def test_public_pytorch_does_not_require_compiler(real_models,monkeypatch):
    import torch
    def forbidden(*args,**kwargs):
        raise AssertionError('Public inference should execute eagerly')
    monkeypatch.setattr(torch,'compile',forbidden)
    model=real_models[0]
    for mode in ['whole','streaming']:
        assert model.convert(np.zeros(1601,np.float32),np.zeros(704),mode=mode).shape==(1601,)


def test_save_pretrained_to_source_preserves_other_backends(tmp_path):
    config = {'artifacts': {'pytorch': ['model.safetensors'], 'onnx': ['model.onnx']}}
    path = tmp_path / 'config.json'
    original = json.dumps(config)
    path.write_text(original)
    model = TVTSyn()
    model.root = tmp_path
    model.backend = 'pytorch'
    model.config = config
    # Include an equivalent path spelling to exercise resolved-path comparison.
    model.save_pretrained(tmp_path / '.')
    assert path.read_text() == original
