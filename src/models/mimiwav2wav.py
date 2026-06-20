"""End-to-end streaming waveform-to-waveform model wrapper."""

from contextlib import nullcontext

import torch
from torch import nn
from torch.nn import functional as F

from src.components.streaming import StreamingContainer
from src.utils.compile import no_compile


class MimiWav2WavModel(StreamingContainer):
    """Compose a waveform frontend, frame transformer, and waveform decoder."""

    def __init__(
        self,
        encoder: nn.Module,
        transformer: nn.Module,
        decoder: nn.Module,
        sample_rate: int = 16000,
        frame_rate: int = 50,
        channels: int = 1,
        torch_compile_encoder_decoder: bool = False,
    ):
        super().__init__()
        if sample_rate % frame_rate:
            raise ValueError("sample_rate must be divisible by frame_rate")
        self.encoder = encoder
        self.transformer = transformer
        self.decoder = decoder
        self._sample_rate = sample_rate
        self._frame_rate = frame_rate
        self._channels = channels
        self._hop_length = sample_rate // frame_rate
        self.torch_compile_encoder_decoder = torch_compile_encoder_decoder

        dimension = getattr(encoder, "dimension", None)
        if not isinstance(dimension, int):
            raise ValueError("encoder must expose an integer dimension attribute")
        self.dimension = dimension

    @property
    def channels(self) -> int:
        return self._channels

    @property
    def frame_rate(self) -> int:
        return self._frame_rate

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @property
    def _component_context(self):
        if self.torch_compile_encoder_decoder:
            return nullcontext()
        return no_compile()

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        if waveform.dim() != 3 or waveform.shape[1] != self.channels:
            raise ValueError(
                f"Expected waveform [B, {self.channels}, T], got {tuple(waveform.shape)}"
            )
        input_length = waveform.shape[-1]
        analysis_waveform = waveform
        if not self.is_streaming and input_length % self._hop_length:
            padding = self._hop_length - input_length % self._hop_length
            analysis_waveform = F.pad(waveform, (0, padding))
        with self._component_context:
            features = self.encoder(analysis_waveform)
        (features,) = self.transformer(features)
        with self._component_context:
            output = self.decoder(features)

        if self.is_streaming:
            return output
        if output.shape[-1] < input_length:
            raise RuntimeError(
                f"Decoder returned {output.shape[-1]} samples for "
                f"{input_length} input samples"
            )
        return output[..., :input_length]

    def decode(self, waveform: torch.Tensor) -> torch.Tensor:
        return self.forward(waveform)
