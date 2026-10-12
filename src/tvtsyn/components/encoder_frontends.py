"""Selectable streaming waveform frontends for the PASR content encoder."""

from dataclasses import asdict, dataclass, is_dataclass
import typing as tp

import torch
from torch import nn

from .conv import StreamingConv1d
from .seanet import SEANetEncoder
from .spectrogram import StreamingLogMelSpectrogram
from .streaming import StreamingContainer


def _activation(name: str) -> nn.Module:
    try:
        activation = getattr(nn, name)
    except AttributeError as exc:
        raise ValueError(f"Unsupported activation: {name}") from exc
    return activation()


class CausalFrameConvBlock(StreamingContainer):
    """Residual causal frame-rate convolution block."""

    def __init__(
        self,
        channels: int,
        kernel_size: int,
        activation: str,
        use_depthwise: bool = True,
        use_residual: bool = True,
    ):
        super().__init__()
        groups = channels if use_depthwise else 1
        self.temporal = StreamingConv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            groups=groups,
            causal=True,
            pad_mode="constant",
        )
        self.pointwise = (
            nn.Conv1d(channels, channels, 1) if use_depthwise else nn.Identity()
        )
        self.activation = _activation(activation)
        self.use_residual = use_residual

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.activation(self.pointwise(self.temporal(x)))
        return x + y if self.use_residual else y


class ChannelLayerNorm(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x.transpose(1, 2)).transpose(1, 2)


class MelTransformerFrontend(StreamingContainer):
    """Causal log-mel analysis followed by a frame-rate projection."""

    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 1024,
        win_length: int = 1024,
        hop_length: int = 320,
        n_mels: int = 80,
        f_min: float = 0.0,
        f_max: float | None = None,
        dimension: int = 512,
        mel_projection_type: str = "linear",
        **kwargs,
    ):
        super().__init__()
        if mel_projection_type not in {"linear", "conv1x1"}:
            raise ValueError("mel_projection_type must be 'linear' or 'conv1x1'")
        self.dimension = dimension
        self.hop_length = hop_length
        self.mel = StreamingLogMelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            n_mels=n_mels,
            f_min=f_min,
            f_max=f_max,
            center=False,
        )
        self.mel_projection_type = mel_projection_type
        if mel_projection_type == "linear":
            self.projection: nn.Module = nn.Linear(n_mels, dimension)
        else:
            self.projection = nn.Conv1d(n_mels, dimension, 1)

    def _project(self, mel: torch.Tensor) -> torch.Tensor:
        if mel.shape[-1] == 0:
            return mel.new_empty(mel.shape[0], self.dimension, 0)
        if self.mel_projection_type == "linear":
            return self.projection(mel.transpose(1, 2)).transpose(1, 2)
        return self.projection(mel)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        return self._project(self.mel(waveform))


class MelCNNTransformerFrontend(MelTransformerFrontend):
    """Mel frontend with lightweight causal frame-rate refinement."""

    def __init__(
        self,
        mel_cnn_layers: int = 3,
        mel_cnn_kernel_size: int = 5,
        mel_cnn_use_residual: bool = True,
        mel_cnn_activation: str = "ELU",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.blocks = nn.ModuleList(
            [
                CausalFrameConvBlock(
                    self.dimension,
                    mel_cnn_kernel_size,
                    mel_cnn_activation,
                    use_depthwise=True,
                    use_residual=mel_cnn_use_residual,
                )
                for _ in range(mel_cnn_layers)
            ]
        )

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        x = super().forward(waveform)
        if x.shape[-1] == 0:
            return x
        for block in self.blocks:
            x = block(x)
        return x



@dataclass
class _RawPatchFrontendState:
    previous_raw_samples: torch.Tensor | None = None

    def reset(self):
        self.previous_raw_samples = None


class RawPatchTransformerFrontend(StreamingContainer):
    """Encode non-overlapping raw waveform patches at the frame rate."""

    def __init__(
        self,
        hop_length: int = 320,
        dimension: int = 512,
        hybrid_raw_frame_samples: int = 320,
        hybrid_raw_dim: int = 512,
        hybrid_raw_conv_layers: int = 1,
        hybrid_raw_conv_kernel_size: int = 3,
        hybrid_raw_use_depthwise: bool = True,
        mel_cnn_activation: str = "ELU",
        **kwargs,
    ):
        super().__init__()
        if hybrid_raw_frame_samples != hop_length:
            raise ValueError("hybrid_raw_frame_samples must equal hop_length")
        if hybrid_raw_dim < 1 or dimension < 1:
            raise ValueError("hybrid_raw_dim and dimension must be positive")
        if hybrid_raw_conv_layers < 0:
            raise ValueError("hybrid_raw_conv_layers must be non-negative")
        if hybrid_raw_conv_kernel_size < 1:
            raise ValueError("hybrid_raw_conv_kernel_size must be positive")

        self.dimension = dimension
        self.hop_length = hop_length
        self.raw_frame_samples = hybrid_raw_frame_samples
        self.raw_projection = nn.Linear(hybrid_raw_frame_samples, hybrid_raw_dim)
        self.raw_blocks = nn.ModuleList(
            [
                CausalFrameConvBlock(
                    hybrid_raw_dim,
                    hybrid_raw_conv_kernel_size,
                    mel_cnn_activation,
                    use_depthwise=hybrid_raw_use_depthwise,
                )
                for _ in range(hybrid_raw_conv_layers)
            ]
        )
        self.output_projection: nn.Module
        if hybrid_raw_dim == dimension:
            self.output_projection = nn.Identity()
        else:
            self.output_projection = nn.Conv1d(hybrid_raw_dim, dimension, 1)

    def _init_streaming_state(self, batch_size: int) -> _RawPatchFrontendState:
        return _RawPatchFrontendState()

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        if waveform.dim() != 3 or waveform.shape[1] != 1:
            raise ValueError(
                f"Expected mono waveform [B, 1, T], got {tuple(waveform.shape)}"
            )

        raw = waveform[:, 0]
        state = self._streaming_state
        if state is not None and state.previous_raw_samples is not None:
            raw = torch.cat([state.previous_raw_samples, raw], dim=-1)

        num_frames = raw.shape[-1] // self.raw_frame_samples
        consumed = num_frames * self.raw_frame_samples
        if state is not None:
            state.previous_raw_samples = raw[..., consumed:]
        if num_frames == 0:
            return waveform.new_empty(waveform.shape[0], self.dimension, 0)

        patches = raw[..., :consumed].reshape(
            raw.shape[0], num_frames, self.raw_frame_samples
        )
        features = self.raw_projection(patches).transpose(1, 2)
        for block in self.raw_blocks:
            features = block(features)
        return self.output_projection(features)


@dataclass
class _HybridFrontendState:
    previous_raw_samples: torch.Tensor | None = None
    pending_raw_features: torch.Tensor | None = None
    pending_mel_features: torch.Tensor | None = None

    def reset(self):
        self.previous_raw_samples = None
        self.pending_raw_features = None
        self.pending_mel_features = None


class HybridStemMelTransformerFrontend(StreamingContainer):
    """Fuse non-overlapping raw waveform patches with causal mel features."""

    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 1024,
        win_length: int = 1024,
        hop_length: int = 320,
        n_mels: int = 80,
        f_min: float = 0.0,
        f_max: float | None = None,
        dimension: int = 512,
        mel_projection_type: str = "linear",
        hybrid_raw_frame_samples: int = 320,
        hybrid_raw_dim: int = 512,
        hybrid_raw_conv_layers: int = 1,
        hybrid_raw_conv_kernel_size: int = 3,
        hybrid_raw_use_depthwise: bool = True,
        hybrid_fusion_type: str = "concat",
        mel_cnn_activation: str = "ELU",
        **kwargs,
    ):
        super().__init__()
        if hybrid_raw_frame_samples != hop_length:
            raise ValueError("hybrid_raw_frame_samples must equal mel hop_length")
        if hybrid_fusion_type not in {"sum", "concat", "gated"}:
            raise ValueError("hybrid_fusion_type must be 'sum', 'concat', or 'gated'")
        if hybrid_raw_dim < 1 or dimension < 1:
            raise ValueError("hybrid_raw_dim and dimension must be positive")
        self.dimension = dimension
        self.hop_length = hop_length
        self.raw_frame_samples = hybrid_raw_frame_samples
        self.fusion_type = hybrid_fusion_type

        self.raw_projection = nn.Linear(hybrid_raw_frame_samples, hybrid_raw_dim)
        self.raw_blocks = nn.ModuleList(
            [
                CausalFrameConvBlock(
                    hybrid_raw_dim,
                    hybrid_raw_conv_kernel_size,
                    mel_cnn_activation,
                    use_depthwise=hybrid_raw_use_depthwise,
                )
                for _ in range(hybrid_raw_conv_layers)
            ]
        )
        self.raw_output_projection: nn.Module
        if hybrid_raw_dim == dimension:
            self.raw_output_projection = nn.Identity()
        else:
            self.raw_output_projection = nn.Conv1d(hybrid_raw_dim, dimension, 1)

        self.mel_frontend = MelTransformerFrontend(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            n_mels=n_mels,
            f_min=f_min,
            f_max=f_max,
            dimension=dimension,
            mel_projection_type=mel_projection_type,
        )
        if hybrid_fusion_type == "sum":
            self.fusion: nn.Module = ChannelLayerNorm(dimension)
        else:
            self.fusion = nn.Conv1d(2 * dimension, dimension, 1)

    def _init_streaming_state(self, batch_size: int) -> _HybridFrontendState:
        return _HybridFrontendState()

    def _raw_features(self, waveform: torch.Tensor) -> torch.Tensor:
        raw = waveform[:, 0]
        state = self._streaming_state
        if state is not None and state.previous_raw_samples is not None:
            raw = torch.cat([state.previous_raw_samples, raw], dim=-1)
        num_frames = raw.shape[-1] // self.raw_frame_samples
        consumed = num_frames * self.raw_frame_samples
        if state is not None:
            state.previous_raw_samples = raw[..., consumed:]
        if num_frames == 0:
            return waveform.new_empty(waveform.shape[0], self.dimension, 0)

        patches = raw[..., :consumed].reshape(
            raw.shape[0], num_frames, self.raw_frame_samples
        )
        features = self.raw_projection(patches).transpose(1, 2)
        for block in self.raw_blocks:
            features = block(features)
        return self.raw_output_projection(features)

    @staticmethod
    def _append_pending(
        pending: torch.Tensor | None, current: torch.Tensor
    ) -> torch.Tensor:
        if pending is None or pending.shape[-1] == 0:
            return current
        if current.shape[-1] == 0:
            return pending
        return torch.cat([pending, current], dim=-1)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        if waveform.dim() != 3 or waveform.shape[1] != 1:
            raise ValueError(
                f"Expected mono waveform [B, 1, T], got {tuple(waveform.shape)}"
            )
        raw = self._raw_features(waveform)
        mel = self.mel_frontend(waveform)
        state = self._streaming_state
        if state is not None:
            raw = self._append_pending(state.pending_raw_features, raw)
            mel = self._append_pending(state.pending_mel_features, mel)

        aligned = min(raw.shape[-1], mel.shape[-1])
        if state is not None:
            state.pending_raw_features = raw[..., aligned:]
            state.pending_mel_features = mel[..., aligned:]
        if aligned == 0:
            return waveform.new_empty(waveform.shape[0], self.dimension, 0)

        raw = raw[..., :aligned]
        mel = mel[..., :aligned]
        if self.fusion_type == "sum":
            return self.fusion(raw + mel)
        joined = torch.cat([raw, mel], dim=1)
        if self.fusion_type == "concat":
            return self.fusion(joined)
        gate = torch.sigmoid(self.fusion(joined))
        return gate * raw + (1.0 - gate) * mel


FRONTEND_TYPES = {
    "mel_transformer": MelTransformerFrontend,
    "mel_cnn_transformer": MelCNNTransformerFrontend,
    "raw_patch_transformer": RawPatchTransformerFrontend,
    "hybrid_stem_mel_transformer": HybridStemMelTransformerFrontend,
}


def _config_dict(config: tp.Any) -> dict[str, tp.Any]:
    if config is None:
        return {}
    if is_dataclass(config):
        return asdict(config)
    return dict(config)


def create_encoder_frontend(
    frontend_type: str = "seanet",
    *,
    seanet_args: tp.Any = None,
    frontend_args: tp.Any = None,
) -> nn.Module:
    if frontend_type == "seanet":
        return SEANetEncoder(**_config_dict(seanet_args))
    try:
        frontend_class = FRONTEND_TYPES[frontend_type]
    except KeyError as exc:
        valid = ", ".join(["seanet", *FRONTEND_TYPES])
        raise ValueError(
            f"Unknown encoder_frontend_type={frontend_type!r}. "
            f"Expected one of: {valid}"
        ) from exc
    return frontend_class(**_config_dict(frontend_args))
