import threading
import numpy as np
import torch
from safetensors.torch import load_file

from tvtsyn.models.inference import TVTSynModel
from tvtsyn.utils.compile import no_compile
from ..streaming.tensorized_step import TVTSynStreamingStep
from ..streaming.state_schema import init_tvtsyn_state


# Legacy compile suppression is process-global, so serialize public PyTorch
# calls across loaded models as well as across sessions of one model.
_RUNTIME_LOCK = threading.RLock()


class PyTorchBackend:
    def __init__(self, root, config, *, device='cpu', **kwargs):
        self.config = config
        self.device = torch.device(device)
        self.chunk_samples = config['chunk_samples']
        self.lookahead_frames = config['lookahead_frames']
        self.lock = _RUNTIME_LOCK
        self.model = TVTSynModel(config['architecture']).eval()
        self.model.load_state_dict(load_file(str(root / 'model.safetensors')), strict=True)
        self.model.to(self.device)
        self.streaming_step = TVTSynStreamingStep(self.model, self.chunk_samples).eval().to(self.device)
        if not self.streaming_step._state_names or self.streaming_step._encoder_transformer_layout is None:
            raise ValueError('Could not construct the tensorized streaming state')

    def initial_state(self):
        return init_tvtsyn_state(self.streaming_step, 1, self.chunk_samples, self.device, torch.float32)

    def step(self, audio, speaker, state):
        with self.lock, torch.inference_mode(), no_compile():
            x = torch.from_numpy(audio.copy())[None, None].to(self.device)
            s = torch.from_numpy(speaker).to(self.device)
            y, *next_state = self.streaming_step(x, s, *state)
            return y.flatten().cpu().numpy(), tuple(next_state), bool(next_state[1].item())

    def whole(self, audio, speaker):
        with self.lock, torch.inference_mode(), no_compile():
            x = torch.from_numpy(audio)[None, None].to(self.device)
            # The encoder/decoder operate at 320 samples per frame.
            x = torch.nn.functional.pad(x, (0, (-len(audio)) % 320))
            s = torch.from_numpy(speaker).to(self.device)
            return self.model(x, s).flatten().cpu().numpy()[:len(audio)]
