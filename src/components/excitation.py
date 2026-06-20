"""Causal prosody and harmonic-source conditioning components."""

from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F

from .conv import StreamingConv1d
from .streaming import StreamingContainer, StreamingModule


class _ProsodyBlock(StreamingContainer):
    def __init__(self, channels: int, kernel_size: int):
        super().__init__()
        self.conv = StreamingConv1d(
            channels,
            channels,
            kernel_size,
            causal=True,
            pad_mode="constant",
        )
        self.norm = nn.LayerNorm(channels)
        self.activation = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv(x)
        y = self.norm(y.transpose(1, 2)).transpose(1, 2)
        return x + self.activation(y)


class ProsodyPredictor(StreamingContainer):
    def __init__(
        self,
        input_dim: int = 512,
        hidden_dim: int = 256,
        num_layers: int = 2,
        kernel_size: int = 3,
        predict_energy: bool = True,
        predict_vuv: bool = True,
        predict_coded_ap: bool = False,
        coded_ap_dim: int = 0,
    ):
        super().__init__()
        if predict_coded_ap and coded_ap_dim < 1:
            raise ValueError("coded_ap_dim must be positive when prediction is enabled")
        self.input_projection = nn.Conv1d(input_dim, hidden_dim, 1)
        self.hidden_dim = hidden_dim
        self.predict_vuv = predict_vuv
        self.predict_energy = predict_energy
        self.coded_ap_dim = coded_ap_dim if predict_coded_ap else 0
        self.blocks = nn.ModuleList(
            [_ProsodyBlock(hidden_dim, kernel_size) for _ in range(num_layers)]
        )
        self.log_f0_head = nn.Conv1d(hidden_dim, 1, 1)
        self.vuv_head = nn.Conv1d(hidden_dim, 1, 1) if predict_vuv else None
        self.energy_head = nn.Conv1d(hidden_dim, 1, 1) if predict_energy else None
        self.coded_ap_head = (
            nn.Conv1d(hidden_dim, coded_ap_dim, 1) if predict_coded_ap else None
        )

    def forward(self, z: torch.Tensor) -> dict[str, torch.Tensor]:
        if z.shape[-1] == 0:
            output = {"log_f0": z.new_empty(z.shape[0], 1, 0)}
            if self.predict_vuv:
                output["vuv_logits"] = z.new_empty(z.shape[0], 1, 0)
            if self.predict_energy:
                output["log_energy"] = z.new_empty(z.shape[0], 1, 0)
            if self.coded_ap_dim:
                output["coded_ap"] = z.new_empty(z.shape[0], self.coded_ap_dim, 0)
            return output
        x = self.input_projection(z)
        for block in self.blocks:
            x = block(x)
        output = {"log_f0": self.log_f0_head(x)}
        if self.vuv_head is not None:
            output["vuv_logits"] = self.vuv_head(x)
        if self.energy_head is not None:
            output["log_energy"] = self.energy_head(x)
        if self.coded_ap_head is not None:
            output["coded_ap"] = self.coded_ap_head(x)
        return output


class FeatureConditioner(nn.Module):
    def __init__(self, input_dim: int, cond_dim: int, mode: str = "add"):
        super().__init__()
        if mode not in {"add", "concat", "film"}:
            raise ValueError("conditioning mode must be 'add', 'concat', or 'film'")
        self.mode = mode
        output_dim = 2 * input_dim if mode == "film" else input_dim
        projection_input = input_dim + cond_dim if mode == "concat" else cond_dim
        self.projection = nn.Conv1d(projection_input, output_dim, 1)

    def forward(self, z: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        frames = min(z.shape[-1], condition.shape[-1])
        z = z[..., :frames]
        condition = condition[..., :frames]
        if self.mode == "add":
            return z + self.projection(condition)
        if self.mode == "concat":
            return self.projection(torch.cat([z, condition], dim=1))
        gamma, beta = self.projection(condition).chunk(2, dim=1)
        return z * (1.0 + gamma) + beta


class ProsodyConditioner(FeatureConditioner):
    def __init__(
        self,
        input_dim: int = 512,
        cond_dim: int = 3,
        hidden_dim: int = 128,
        mode: str = "add",
    ):
        super().__init__(input_dim, hidden_dim, mode)
        self.feature_projection = nn.Sequential(
            nn.Conv1d(cond_dim, hidden_dim, 1),
            nn.SiLU(),
        )

    def forward(self, z: torch.Tensor, prosody_features: torch.Tensor) -> torch.Tensor:
        return super().forward(z, self.feature_projection(prosody_features))


@dataclass
class _HarmonicSourceState:
    phase: torch.Tensor | None = None
    sample_offset: int = 0

    def reset(self):
        self.phase = None
        self.sample_offset = 0


class HarmonicSourceGenerator(StreamingModule[_HarmonicSourceState]):
    def __init__(
        self,
        sample_rate: int = 16000,
        hop_length: int = 320,
        num_harmonics: int = 8,
        noise_std: float = 0.003,
        use_energy_scaling: bool = True,
        max_f0: float = 800.0,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.num_harmonics = num_harmonics
        self.noise_std = noise_std
        self.use_energy_scaling = use_energy_scaling
        self.max_f0 = max_f0

    def _init_streaming_state(self, batch_size: int) -> _HarmonicSourceState:
        return _HarmonicSourceState()

    def _noise(self, reference: torch.Tensor, offset: int) -> torch.Tensor:
        if self.training:
            return torch.randn_like(reference)
        indices = torch.arange(
            offset,
            offset + reference.shape[-1],
            device=reference.device,
            dtype=reference.dtype,
        )
        noise = torch.sin(indices * 12.9898 + 78.233) * 43758.5453
        return ((noise - noise.floor()) * 2.0 - 1.0).expand_as(reference)

    def forward(
        self,
        f0_hz: torch.Tensor,
        vuv: torch.Tensor,
        log_energy: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if f0_hz.dim() != 2 or vuv.shape != f0_hz.shape:
            raise ValueError("Expected f0_hz and vuv with shape [B, T]")
        f0 = f0_hz.clamp(0.0, self.max_f0).repeat_interleave(
            self.hop_length, dim=-1
        )
        voiced = vuv.clamp(0.0, 1.0).repeat_interleave(self.hop_length, dim=-1)
        phase_increment = 2.0 * math.pi * f0 / self.sample_rate
        state = self._streaming_state
        initial_phase = (
            state.phase
            if state is not None and state.phase is not None
            else f0.new_zeros(f0.shape[0], 1)
        )
        phase = initial_phase + phase_increment.cumsum(dim=-1)
        harmonics = torch.arange(
            1, self.num_harmonics + 1, device=f0.device, dtype=f0.dtype
        ).view(1, -1, 1)
        source = (
            torch.sin(phase.unsqueeze(1) * harmonics) / harmonics
        ).sum(dim=1) / max(1.0, math.sqrt(self.num_harmonics))
        offset = state.sample_offset if state is not None else 0
        noise = self._noise(source, offset) * self.noise_std
        source = voiced * source + (1.0 - voiced) * noise
        if self.use_energy_scaling and log_energy is not None:
            energy = log_energy.exp().clamp_max(10.0)
            source = source * energy.repeat_interleave(self.hop_length, dim=-1)
        if state is not None:
            state.phase = phase[..., -1:].remainder(2.0 * math.pi)
            state.sample_offset += source.shape[-1]
        return source.unsqueeze(1)


@dataclass
class _StreamingMagnitudeState:
    previous: torch.Tensor | None = None
    padding_to_add: int = 0
    original_padding: int = 0

    def reset(self):
        self.previous = None
        self.padding_to_add = self.original_padding


class _StreamingLogMagnitude(StreamingModule[_StreamingMagnitudeState]):
    def __init__(self, n_fft: int, hop_length: int, win_length: int):
        super().__init__()
        if not 0 < hop_length <= win_length <= n_fft:
            raise ValueError("Expected 0 < hop_length <= win_length <= n_fft")
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.left_padding = win_length - hop_length
        self.register_buffer("window", torch.hann_window(win_length), persistent=False)

    def _init_streaming_state(self, batch_size: int) -> _StreamingMagnitudeState:
        return _StreamingMagnitudeState(
            padding_to_add=self.left_padding,
            original_padding=self.left_padding,
        )

    def _features(self, samples: torch.Tensor) -> torch.Tensor:
        if samples.shape[-1] < self.win_length:
            return samples.new_empty(samples.shape[0], self.n_fft // 2 + 1, 0)
        frames = samples.unfold(-1, self.win_length, self.hop_length)
        compute_dtype = (
            torch.float32
            if samples.dtype in {torch.float16, torch.bfloat16}
            else samples.dtype
        )
        frames = frames.to(compute_dtype)
        window = self.window.to(frames.device, frames.dtype)
        spectrum = torch.fft.rfft(frames * window, n=self.n_fft, dim=-1)
        return spectrum.abs().clamp_min(1e-7).log().transpose(1, 2).to(samples.dtype)

    def forward(self, source_wave: torch.Tensor) -> torch.Tensor:
        samples = source_wave[:, 0]
        state = self._streaming_state
        if state is None:
            samples = F.pad(samples, (self.left_padding, 0))
            return self._features(samples)
        if state.previous is not None:
            samples = torch.cat([state.previous, samples], dim=-1)
        if state.padding_to_add:
            samples = F.pad(samples, (state.padding_to_add, 0))
            state.padding_to_add = 0
        frames = max(0, 1 + (samples.shape[-1] - self.win_length) // self.hop_length)
        consumed = frames * self.hop_length
        state.previous = samples[..., consumed:]
        if frames == 0:
            return samples.new_empty(samples.shape[0], self.n_fft // 2 + 1, 0)
        used = (frames - 1) * self.hop_length + self.win_length
        return self._features(samples[..., :used])


class HarmonicSourceConditioner(StreamingContainer):
    def __init__(
        self,
        input_dim: int = 512,
        sample_rate: int = 16000,
        n_fft: int = 512,
        hop_length: int = 320,
        win_length: int = 512,
        source_feature_dim: int | None = None,
        mode: str = "add",
    ):
        super().__init__()
        del sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        feature_dim = n_fft // 2 + 1
        if source_feature_dim is not None and source_feature_dim != feature_dim:
            raise ValueError("source_feature_dim must equal n_fft // 2 + 1")
        self.conditioner = FeatureConditioner(input_dim, feature_dim, mode)
        self.analysis = _StreamingLogMagnitude(n_fft, hop_length, win_length)

    def forward(self, z: torch.Tensor, source_wave: torch.Tensor) -> torch.Tensor:
        return self.conditioner(z, self.analysis(source_wave))


def target_prosody_features(excitation: dict[str, torch.Tensor]) -> torch.Tensor:
    return torch.stack(
        [excitation["log_f0"], excitation["vuv"], excitation["log_energy"]],
        dim=1,
    )


def predicted_prosody_features(
    prediction: dict[str, torch.Tensor],
) -> torch.Tensor:
    return torch.cat(
        [
            prediction["log_f0"],
            torch.sigmoid(prediction["vuv_logits"]),
            prediction["log_energy"],
        ],
        dim=1,
    )
