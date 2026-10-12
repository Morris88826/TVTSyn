"""Demo session lifecycle and independent users, without a browser or model download."""
import importlib.util
from pathlib import Path
import sys
import numpy as np
import pytest


@pytest.fixture
def app(monkeypatch):
    pytest.importorskip('gradio')
    path=Path(__file__).resolve().parents[1]/'spaces/demo/app.py'
    spec=importlib.util.spec_from_file_location('tvtsyn_demo_test',path)
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    from tvtsyn_inference.streaming.session import StreamingSession
    class Backend:
        chunk_samples=1600;lookahead_frames=4
        def initial_state(self):return None
        def step(self,x,s,state):return (np.zeros_like(x) if state is None else state),x.copy(),state is not None
    class Model:
        def stream(self,speaker,sample_rate):return StreamingSession(Backend(),speaker,sample_rate=sample_rate)
    monkeypatch.setattr(module,'get_model',lambda backend:Model())
    return module


def test_live_restart_and_disconnect(app,tmp_path):
    embed=tmp_path/'speaker.npy';np.save(embed,np.zeros(704,np.float32))
    first,_=app.prepare('onnx',None,str(embed),None)
    second,_=app.prepare('pytorch',None,str(embed),None)
    assert app.receive((48000,np.ones(24000,np.float32)*.1),first) is not None
    assert app.receive((16000,np.ones(3200,np.float32)*.2),second) is not None
    assert app.finish(first) is not None
    assert app.SESSIONS[first].stream is None
    assert app.finish(first) == app.gr.skip()
    app.receive((16000,np.zeros(1600,np.float32)),first)
    app.close_session(first)
    assert first not in app.SESSIONS and second in app.SESSIONS
    app.SESSIONS[second].touched-=121
    app.prune()
    assert second not in app.SESSIONS


def test_recording_limit(app,tmp_path):
    embed=tmp_path/'speaker.npy';np.save(embed,np.zeros(704,np.float32))
    key,_=app.prepare('onnx',None,str(embed),None)
    app.receive((16000,np.zeros(1600,np.float32)),key)
    app.SESSIONS[key].started-=61
    with pytest.raises(app.gr.Error):app.receive((16000,np.zeros(1600,np.float32)),key)
    assert key not in app.SESSIONS


def test_gradio_build(app):
    demo=app.build_app()
    assert demo.config['components']
    demo.close()


def test_continuous_playback_queue_and_restart(app,tmp_path):
    embed=tmp_path/'speaker.npy';np.save(embed,np.zeros(704,np.float32))
    key,_=app.prepare('onnx',None,str(embed),None)
    player=app.play_live(key)
    first=app.receive((16000,np.ones(3200,np.float32)*.1),key)
    assert next(player)==first
    tail=app.finish(key)
    assert next(player)==tail
    with pytest.raises(StopIteration):next(player)
    player=app.play_live(key)
    first=app.receive((16000,np.ones(3200,np.float32)*.2),key)
    assert next(player)==first
    app.close_session(key)
    with pytest.raises(StopIteration):next(player)


def test_zerogpu_file_dispatch_preserves_live_cpu_model(app, tmp_path, monkeypatch):
    import soundfile as sf
    source = tmp_path / 'source.wav'
    embedding = tmp_path / 'speaker.npy'
    sf.write(source, np.zeros(3200, np.float32), 16000)
    np.save(embedding, np.zeros(704, np.float32))
    calls = []
    monkeypatch.setattr(app, 'ZERO_GPU', True)
    def gpu(source, speaker, mode):
        calls.append((source, speaker.shape, mode))
        return 'gpu-output.wav'
    monkeypatch.setattr(app, 'convert_on_gpu', gpu, raising=False)
    assert app.convert('pytorch', 'streaming', str(source), None, str(embedding)) == 'gpu-output.wav'
    assert calls == [(str(source), (1, 704), 'streaming')]
    key, _ = app.prepare('pytorch', None, str(embedding), None)
    assert app.receive((16000, np.zeros(3200, np.float32)), key) is not None
    assert len(calls) == 1
    sf.write(source, np.zeros(16000 * 61, np.float32), 16000)
    with pytest.raises(app.gr.Error, match='60 seconds'):
        app.convert('pytorch', 'whole', str(source), None, str(embedding))
