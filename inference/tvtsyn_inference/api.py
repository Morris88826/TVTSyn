from pathlib import Path
import shutil
import threading
import numpy as np

from .audio import load_audio, resample, speaker_array
from .hub import resolve_model
from .streaming.session import StreamingSession


class TVTSyn:
    @classmethod
    def from_pretrained(cls, model_id, *, backend='pytorch', revision=None, cache_dir=None,
                        local_files_only=False, token=None, device='cpu', providers=None, threads=1):
        root, config = resolve_model(model_id, backend, revision=revision, cache_dir=cache_dir,
                                     local_files_only=local_files_only, token=token)
        if backend == 'pytorch':
            from .backends.pytorch import PyTorchBackend
            runtime = PyTorchBackend(root, config, device=device)
        else:
            from .backends.onnx import ONNXBackend
            runtime = ONNXBackend(root, config, providers=providers, threads=threads)
        instance = cls()
        instance.backend = backend
        instance.config = config
        instance.root = root
        instance.runtime = runtime
        instance.device = device
        instance.cache_dir = cache_dir
        instance.local_files_only = local_files_only
        instance._speaker = None
        instance._speaker_lock = threading.Lock()
        return instance

    def encode_speaker(self, audio, *, sample_rate=16000):
        with self._speaker_lock:
            if self._speaker is None:
                from .speaker import SpeakerEncoder
                self._speaker = SpeakerEncoder(device=self.device, cache_dir=self.cache_dir,
                                               local_files_only=self.local_files_only,
                                               models=self.config['speaker_models'])
            return self._speaker.encode(audio, sample_rate)

    def stream(self, speaker, *, sample_rate=16000):
        return StreamingSession(self.runtime, speaker, sample_rate=sample_rate)

    def convert(self, audio, speaker, *, sample_rate=16000, mode=None):
        mode = mode or ('whole' if self.backend == 'pytorch' else 'streaming')
        if mode not in ('whole', 'streaming') or (mode == 'whole' and self.backend != 'pytorch'):
            raise ValueError('Whole-utterance inference requires PyTorch; use mode="streaming" for ONNX')
        x = resample(audio, sample_rate)
        s = speaker_array(speaker)
        if not x.size:
            return x
        if mode == 'whole':
            return self.runtime.whole(x, s)
        session = self.stream(s)
        try:
            # Keep backend buffering bounded even for long files.
            parts = [session.push(x[i:i+1600]) for i in range(0, len(x), 1600)]
            parts.append(session.flush())
            return np.concatenate(parts)
        finally:
            session.close()

    def convert_file(self, source, speaker, *, output=None, mode=None):
        audio, sr = load_audio(source)
        result = self.convert(audio, speaker, sample_rate=sr, mode=mode)
        if output is not None:
            import soundfile as sf
            path = Path(output)
            path.parent.mkdir(parents=True, exist_ok=True)
            sf.write(path, result, self.config['sample_rate'], subtype='FLOAT')
        return result

    def save_pretrained(self, directory):
        """Copy the loaded backend and its configuration into a portable local bundle."""
        import json
        dest = Path(directory)
        # Saving back to the source is a no-op: preserve all backend manifests.
        if dest.resolve() == self.root.resolve():
            return
        dest.mkdir(parents=True, exist_ok=True)
        for name in self.config['artifacts'][self.backend]:
            target = dest / name
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.resolve() != (self.root / name).resolve():
                shutil.copy2(self.root / name, target)
        config = dict(self.config)
        config['artifacts'] = {self.backend: self.config['artifacts'][self.backend]}
        (dest / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
