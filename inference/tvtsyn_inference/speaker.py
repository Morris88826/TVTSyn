"""Optional reference-audio conditioning, separate from the conversion backend."""
from pathlib import Path
import hashlib

from .audio import load_audio, resample, speaker_array


class SpeakerEncoder:
    def __init__(self, *, device='cpu', cache_dir=None, local_files_only=False, models=None):
        import torch
        from speechbrain.inference.classifiers import EncoderClassifier
        from speechbrain.utils.fetching import FetchConfig
        self.torch = torch
        self.device = device
        self.encoders = []
        cache = Path(cache_dir) if cache_dir else Path.home() / '.cache' / 'tvtsyn' / 'speaker'
        for model in models:
            # SpeechBrain reuses savedir files before checking the Hub revision.
            # Give each pinned dependency revision its own destination.
            identity = model['repo_id'] + '@' + str(model.get('revision', 'main'))
            destination = cache / hashlib.sha256(identity.encode()).hexdigest()[:24]
            self.encoders.append(EncoderClassifier.from_hparams(
                source=model['repo_id'], savedir=str(destination),
                run_opts={'device': device},
                fetch_config=FetchConfig(revision=model.get('revision'), allow_network=not local_files_only),
            ))
        for encoder in self.encoders:
            encoder.eval()

    def encode(self, audio, sample_rate=16000):
        if isinstance(audio, (str, Path)):
            audio, sample_rate = load_audio(audio)
        x = resample(audio, sample_rate)
        if len(x) < 1600:
            raise ValueError('Target reference must contain at least 100 ms of audio; several seconds are recommended')
        torch = self.torch
        with torch.inference_mode():
            wave = torch.from_numpy(x)[None].to(self.device)
            result = torch.cat([e.encode_batch(wave) for e in self.encoders], dim=-1).squeeze(1)
        return speaker_array(result)
