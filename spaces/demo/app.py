"""Run from the repository root: python spaces/demo/app.py."""
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
import io
import os
import queue
import shutil
import tempfile
import threading
import time
import uuid

ZERO_GPU = os.environ.get('TVTSYN_ZEROGPU') == '1'
if ZERO_GPU:
    import spaces  # Must precede torch imports so ZeroGPU can install its CUDA hooks.

import gradio as gr
import numpy as np
import soundfile as sf

from tvtsyn_inference import TVTSyn
from tvtsyn_inference.audio import speaker_array

MODEL = os.environ.get('TVTSYN_MODEL', 'warisqr007/TVTSyn')
REVISION = os.environ.get('TVTSYN_REVISION', '2d3147b8a465e1d9aad33a2a0e53956bf8ed32aa')
DEVICE = 'cpu' if ZERO_GPU else os.environ.get('TVTSYN_DEVICE', 'cpu')
MAX_SECONDS = 60
LOCK = threading.RLock()


@dataclass
class Session:
    model: object
    speaker: np.ndarray
    stream: object = None
    started: float = 0
    touched: float = 0
    samples: int = 0
    output_queue: object = field(default_factory=lambda: queue.Queue(maxsize=128))
    output_done: bool = False


SESSIONS = {}

# ZeroGPU leases end after each decorated call. Live sessions therefore stay
# on CPU; the file path owns all GPU state inside one call. Load CUDA weights
# at startup as required by ZeroGPU's CUDA emulation.
if ZERO_GPU:
    GPU_MODEL = TVTSyn.from_pretrained(MODEL, backend='pytorch', revision=REVISION, device='cuda')

    @spaces.GPU(duration=60)
    def convert_on_gpu(source, speaker, mode):
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            output = f.name
        GPU_MODEL.convert_file(source, speaker, output=output, mode=mode)
        return output



@lru_cache(maxsize=2)
def get_model(backend):
    if backend == 'pytorch' and DEVICE == 'cpu':
        import torch
        torch.set_num_threads(int(os.environ.get('TVTSYN_TORCH_THREADS', '4')))
    return TVTSyn.from_pretrained(MODEL, backend=backend, revision=REVISION, device=DEVICE,
                                  threads=int(os.environ.get('TVTSYN_THREADS', '1')),
                                  local_files_only=os.environ.get('TVTSYN_OFFLINE') == '1')


def close_session(key):
    with LOCK:
        session = SESSIONS.pop(key, None)
        if session:
            if session.stream:
                session.stream.close()
            # Wake a disconnected playback generator without blocking ingestion.
            while not session.output_queue.empty():
                try:
                    session.output_queue.get_nowait()
                except queue.Empty:
                    break
            session.output_queue.put_nowait(None)


def prune():
    now = time.monotonic()
    for key, session in list(SESSIONS.items()):
        if now - session.touched > 120:
            close_session(key)


def condition(model, target, embedding):
    if embedding:
        return speaker_array(embedding)
    if not target:
        raise gr.Error('Provide target reference audio or a saved speaker embedding.')
    if DEVICE == 'cpu':
        import torch
        torch.set_num_threads(int(os.environ.get('TVTSYN_TORCH_THREADS', '4')))
    return model.encode_speaker(target)


def prepare(backend, target, embedding, old_key):
    with LOCK:
        prune()
        close_session(old_key)
        model = get_model(backend)
        speaker = condition(model, target, embedding)
        key = str(uuid.uuid4())
        SESSIONS[key] = Session(model, speaker, touched=time.monotonic())
        return key, 'Target ready. Record up to 60 seconds; stop to flush the final audio.'


def wav_bytes(audio):
    if not len(audio):
        return None
    buf = io.BytesIO()
    # Fixed PCM scale avoids per-chunk normalization and resulting volume jumps.
    sf.write(buf, np.clip(audio, -1, 1), 16000, format='WAV', subtype='PCM_16')
    return buf.getvalue()


def ensure_output(session):
    if session.output_done:
        session.output_queue = queue.Queue(maxsize=128)
        session.output_done = False
    return session.output_queue


def enqueue(session, data):
    if data is not None:
        try:
            session.output_queue.put_nowait(data)
        except queue.Full:
            raise gr.Error('Playback cannot keep up. Stop and prepare a new session.')


def play_live(key):
    """One continuous generator per recording; ingestion uses a separate queue."""
    with LOCK:
        session = SESSIONS.get(key)
        if session is None:
            raise gr.Error('Prepare a target before recording.')
        output_queue = ensure_output(session)
    deadline = time.monotonic() + MAX_SECONDS + 15
    while time.monotonic() < deadline:
        try:
            audio = output_queue.get(timeout=1)
        except queue.Empty:
            continue
        if audio is None:
            return
        yield audio
    close_session(key)


def receive(audio, key):
    with LOCK:
        prune()
        session = SESSIONS.get(key)
        if session is None:
            raise gr.Error('Prepare a target before recording.')
        sr, wave = audio
        if session.stream is None:
            ensure_output(session)
            session.stream = session.model.stream(session.speaker, sample_rate=sr)
            session.started = time.monotonic()
            session.samples = 0
        if session.stream.sample_rate != sr:
            close_session(key)
            raise gr.Error('Input sample rate changed. Prepare the target again.')
        session.samples += len(wave)
        if time.monotonic() - session.started > MAX_SECONDS or session.samples > sr * MAX_SECONDS:
            close_session(key)
            raise gr.Error('Recording limit reached. Prepare the target to start another session.')
        session.touched = time.monotonic()
        audio = wav_bytes(session.stream.push(wave))
        enqueue(session, audio)
        return audio


def finish(key):
    with LOCK:
        session = SESSIONS.get(key)
        if session is None or session.stream is None:
            return gr.skip()
        try:
            audio = wav_bytes(session.stream.flush())
            enqueue(session, audio)
            return audio
        finally:
            session.stream.close()
            session.stream = None
            session.touched = time.monotonic()
            session.output_done = True
            try:
                session.output_queue.put_nowait(None)
            except queue.Full:
                close_session(key)


def convert(backend, mode, source, target, embedding):
    if not source:
        raise gr.Error('Provide source audio.')
    if backend == 'onnx' and mode == 'whole':
        raise gr.Error('Select streaming mode for ONNX.')
    if ZERO_GPU and sf.info(source).duration > MAX_SECONDS:
        raise gr.Error('This hosted demo accepts source clips up to 60 seconds.')
    with LOCK:
        model = get_model(backend)
        speaker = condition(model, target, embedding)
        if ZERO_GPU and backend == 'pytorch':
            return convert_on_gpu(source, speaker, mode)
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            output = f.name
        model.convert_file(source, speaker, output=output, mode=mode)
        return output


def ingest(audio, key):
    receive(audio, key)


def end_recording(key):
    finish(key)


def build_app():
    with gr.Blocks(title='TVTSyn voice conversion', delete_cache=(60, 3600)) as demo:
        gr.Markdown('# TVTSyn\nConvert speech using a target voice. [Model card](https://huggingface.co/warisqr007/TVTSyn) · [Source](https://github.com/PSI-TAMU/TVTSyn)\n\nTVTSyn weights: CC BY-NC 4.0 (noncommercial use with attribution). Streaming uses 100 ms model chunks; browser and network buffering add latency. Live PyTorch audio may process slower than real time on CPU.')
        if ZERO_GPU:
            gr.Markdown('PyTorch file conversion uses ZeroGPU (subject to its queue and quota). ONNX, speaker preparation, and live sessions use CPU. Clips are limited to 60 seconds.')
        backend = gr.Radio(['pytorch', 'onnx'], value='onnx', label='Backend')
        with gr.Row():
            target = gr.Audio(sources=['upload', 'microphone'], type='filepath', label='Target reference voice')
            embedding = gr.File(file_types=['.npy'], type='filepath', label='Saved speaker embedding (optional; overrides reference)')
        with gr.Tab('File conversion'):
            source = gr.Audio(sources=['upload', 'microphone'], type='filepath', label='Source speech')
            mode = gr.Radio(['whole', 'streaming'], value='streaming', label='Processing mode')
            button = gr.Button('Convert')
            output = gr.Audio(label='Converted speech', type='filepath')
            button.click(convert, [backend, mode, source, target, embedding], output,
                         concurrency_id='model', concurrency_limit=1)
        with gr.Tab('Live microphone'):
            key = gr.State(None, time_to_live=120, delete_callback=close_session)
            prepare_button = gr.Button('Prepare target')
            status = gr.Textbox(label='Status', interactive=False)
            mic = gr.Audio(sources=['microphone'], streaming=True, type='numpy', label='Live source')
            playback = gr.Audio(streaming=True, autoplay=True, label='Converted stream')
            prepare_button.click(prepare, [backend, target, embedding, key], [key, status],
                                 concurrency_id='model', concurrency_limit=1)
            mic.start_recording(play_live, key, playback, concurrency_id='playback', concurrency_limit=8)
            mic.stream(ingest, [mic, key], None, stream_every=0.1, time_limit=60,
                       concurrency_id='model', concurrency_limit=1, trigger_mode='multiple')
            mic.stop_recording(end_recording, key, None, concurrency_id='model', concurrency_limit=1)
            mic.clear(end_recording, key, None, concurrency_id='model', concurrency_limit=1)
            backend.change(close_session, key, concurrency_id='model', concurrency_limit=1)
            target.change(close_session, key, concurrency_id='model', concurrency_limit=1)
            embedding.change(close_session, key, concurrency_id='model', concurrency_limit=1)
    return demo.queue()


if __name__ == '__main__':
    if not all(shutil.which(name) for name in ('ffmpeg', 'ffprobe')):
        raise SystemExit('Live playback requires ffmpeg and ffprobe on PATH. See spaces/demo/README.md.')
    build_app().launch()
