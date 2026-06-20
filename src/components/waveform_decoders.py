"""Lightweight frame-rate waveform decoders.

All decoders accept frame features in ``[B, C, T]`` layout. The direct and
hybrid variants emit exactly ``T * frame_hop_samples`` samples in both offline
and streaming mode. Streaming state is owned by the existing transformer and
causal convolution modules and is activated with ``decoder.streaming(B)``.
"""

from dataclasses import asdict, is_dataclass
import inspect
import typing as tp

import torch
from torch import nn

from .conv import StreamingConv1d, StreamingConvTranspose1d
from .periodic_activations import Snake, SnakeBeta
from .seanet import SEANetDecoder
from .spectrogram import StreamingISTFT
from .streaming import StreamingContainer
from .transformer import StreamingTransformer


def _activation(name: str, channels: int | None = None) -> nn.Module:
    if name == "GELU":
        return nn.GELU()
    if name == "SiLU":
        return nn.SiLU()
    if name == "ELU":
        return nn.ELU()
    if name == "Snake":
        if channels is None:
            raise ValueError("Snake activation requires channels")
        return Snake(channels)
    if name == "SnakeBeta":
        if channels is None:
            raise ValueError("SnakeBeta activation requires channels")
        return SnakeBeta(channels)
    raise ValueError(f"Unsupported activation: {name}")


class CausalDepthwiseSeparableConv1d(StreamingContainer):
    """Residual causal depthwise-separable block for ``[B, C, T]`` inputs."""

    def __init__(self, channels: int, kernel_size: int = 7, activation: str = "SiLU"):
        super().__init__()
        if kernel_size < 1:
            raise ValueError("kernel_size must be positive")
        self.depthwise = StreamingConv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            groups=channels,
            causal=True,
            pad_mode="constant",
        )
        self.pointwise = nn.Conv1d(channels, channels, kernel_size=1)
        self.activation = _activation(activation, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.activation(self.pointwise(self.depthwise(x)))


class StreamingCausalPostFilter(StreamingContainer):
    """Small residual waveform post-filter with persistent causal conv state."""

    def __init__(
        self,
        channels: int = 64,
        layers: int = 2,
        kernel_size: int = 7,
        activation: str = "SiLU",
    ):
        super().__init__()
        if layers < 1:
            raise ValueError("post-filter layers must be positive")
        self.input_projection = nn.Conv1d(1, channels, kernel_size=1)
        self.blocks = nn.ModuleList(
            [
                CausalDepthwiseSeparableConv1d(
                    channels, kernel_size=kernel_size, activation=activation
                )
                for _ in range(layers)
            ]
        )
        self.output_projection = nn.Conv1d(channels, 1, kernel_size=1)

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        x = self.input_projection(wav)
        for block in self.blocks:
            x = block(x)
        return wav + self.output_projection(x)


class SubframeCausalRefiner(StreamingContainer):
    """Causal residual CNN operating on a low-rate subframe sequence."""

    def __init__(
        self,
        channels: int,
        layers: int = 4,
        kernel_size: int = 7,
        activation: str = "SiLU",
    ):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                CausalDepthwiseSeparableConv1d(
                    channels, kernel_size=kernel_size, activation=activation
                )
                for _ in range(layers)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return x


class _FrameContextDecoder(StreamingContainer):
    def __init__(
        self,
        input_dimension: int = 512,
        decoder_dim: int = 512,
        decoder_layers: int = 4,
        decoder_heads: int = 8,
        decoder_ffn_dim: int = 2048,
        decoder_context_frames: int = 100,
        decoder_right_context_frames: int = 0,
        activation: str = "SiLU",
    ):
        super().__init__()
        self.input_dimension = input_dimension
        self.decoder_dim = decoder_dim
        if input_dimension == decoder_dim:
            self.input_projection: nn.Module = nn.Identity()
        else:
            self.input_projection = nn.Conv1d(
                input_dimension, decoder_dim, kernel_size=1
            )
        self.context = StreamingTransformer(
            d_model=decoder_dim,
            num_heads=decoder_heads,
            num_layers=decoder_layers,
            dim_feedforward=decoder_ffn_dim,
            causal=True,
            context=decoder_context_frames,
            right_context=decoder_right_context_frames,
            positional_embedding="rope",
            layer_scale=0.01,
        )
        self.activation_name = activation

    def _contextualize(self, z: torch.Tensor) -> torch.Tensor:
        if z.dim() != 3:
            raise ValueError(f"Expected [B, C, T] input, got {tuple(z.shape)}")
        if z.shape[1] != self.input_dimension:
            raise ValueError(
                f"Expected {self.input_dimension} input channels, got {z.shape[1]}"
            )
        return self.context(self.input_projection(z).transpose(1, 2))


class FrameLinearWaveformDecoder(_FrameContextDecoder):
    """Predict one contiguous waveform segment for each 50 Hz input frame."""

    def __init__(
        self,
        frame_hop_samples: int = 320,
        linear_hidden_dim: int = 1024,
        use_linear_hidden: bool = True,
        use_post_filter: bool = True,
        post_filter_kernel_size: int = 7,
        post_filter_layers: int = 2,
        post_filter_channels: int = 64,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.frame_hop_samples = frame_hop_samples
        if use_linear_hidden:
            self.waveform_projection = nn.Sequential(
                nn.Linear(self.decoder_dim, linear_hidden_dim),
                _activation(self.activation_name, linear_hidden_dim),
                nn.Linear(linear_hidden_dim, frame_hop_samples),
            )
        else:
            self.waveform_projection = nn.Linear(self.decoder_dim, frame_hop_samples)
        if use_post_filter:
            self.post_filter: nn.Module = StreamingCausalPostFilter(
                channels=post_filter_channels,
                layers=post_filter_layers,
                kernel_size=post_filter_kernel_size,
                activation=self.activation_name,
            )
        else:
            self.post_filter = nn.Identity()

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        frames = self._contextualize(z)
        wav = self.waveform_projection(frames).flatten(1).unsqueeze(1)
        return self.post_filter(wav)


class SubframeLinearWaveformDecoder(_FrameContextDecoder):
    """Predict ordered waveform subframes directly or with learned tokens."""

    def __init__(
        self,
        frame_hop_samples: int = 320,
        num_subframes: int = 4,
        subframe_samples: int = 80,
        subframe_mode: str = "learned_tokens",
        subframe_hidden_dim: int = 1024,
        use_post_filter: bool = True,
        post_filter_kernel_size: int = 7,
        post_filter_layers: int = 2,
        post_filter_channels: int = 64,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if num_subframes * subframe_samples != frame_hop_samples:
            raise ValueError(
                "num_subframes * subframe_samples must equal frame_hop_samples"
            )
        if subframe_mode not in {"direct", "learned_tokens"}:
            raise ValueError("subframe_mode must be 'direct' or 'learned_tokens'")
        self.frame_hop_samples = frame_hop_samples
        self.num_subframes = num_subframes
        self.subframe_samples = subframe_samples
        self.subframe_mode = subframe_mode
        if subframe_mode == "direct":
            self.subframe_projection = nn.Sequential(
                nn.Linear(self.decoder_dim, subframe_hidden_dim),
                _activation(self.activation_name, subframe_hidden_dim),
                nn.Linear(subframe_hidden_dim, num_subframes * subframe_samples),
            )
            self.subframe_tokens = None
        else:
            self.subframe_tokens = nn.Parameter(
                torch.empty(num_subframes, self.decoder_dim)
            )
            nn.init.normal_(self.subframe_tokens, std=0.02)
            self.subframe_projection = nn.Sequential(
                nn.Linear(self.decoder_dim, subframe_hidden_dim),
                _activation(self.activation_name, subframe_hidden_dim),
                nn.Linear(subframe_hidden_dim, subframe_samples),
            )
        if use_post_filter:
            self.post_filter: nn.Module = StreamingCausalPostFilter(
                channels=post_filter_channels,
                layers=post_filter_layers,
                kernel_size=post_filter_kernel_size,
                activation=self.activation_name,
            )
        else:
            self.post_filter = nn.Identity()

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        frames = self._contextualize(z)
        if self.subframe_tokens is None:
            segments = self.subframe_projection(frames)
        else:
            subframes = frames.unsqueeze(2) + self.subframe_tokens[None, None]
            segments = self.subframe_projection(subframes)
        wav = segments.flatten(1).unsqueeze(1)
        return self.post_filter(wav)


class HybridLinearCNNDecoder(_FrameContextDecoder):
    """Expand to a 200 Hz latent sequence and refine it before synthesis."""

    def __init__(
        self,
        frame_hop_samples: int = 320,
        hybrid_mode: str = "linear_200hz",
        hybrid_subframes: int = 4,
        hybrid_subframe_dim: int = 256,
        hybrid_refiner_layers: int = 4,
        hybrid_refiner_kernel_size: int = 7,
        use_post_filter: bool = False,
        post_filter_kernel_size: int = 7,
        post_filter_layers: int = 2,
        post_filter_channels: int = 64,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if hybrid_mode not in {"linear_200hz", "partial_seanet"}:
            raise ValueError(
                "hybrid_mode must be 'linear_200hz' or 'partial_seanet'"
            )
        if frame_hop_samples % hybrid_subframes:
            raise ValueError("frame_hop_samples must be divisible by hybrid_subframes")
        self.frame_hop_samples = frame_hop_samples
        self.hybrid_mode = hybrid_mode
        self.hybrid_subframes = hybrid_subframes
        self.subframe_samples = frame_hop_samples // hybrid_subframes
        self.subframe_dim = hybrid_subframe_dim
        self.expansion = nn.Linear(
            self.decoder_dim, hybrid_subframes * hybrid_subframe_dim
        )
        self.refiner = SubframeCausalRefiner(
            channels=hybrid_subframe_dim,
            layers=hybrid_refiner_layers,
            kernel_size=hybrid_refiner_kernel_size,
            activation=self.activation_name,
        )
        if hybrid_mode == "linear_200hz":
            self.waveform_projection: nn.Module = nn.Linear(
                hybrid_subframe_dim, self.subframe_samples
            )
        else:
            self.waveform_projection = StreamingConvTranspose1d(
                hybrid_subframe_dim,
                1,
                kernel_size=2 * self.subframe_samples,
                stride=self.subframe_samples,
                causal=True,
            )
        if use_post_filter:
            self.post_filter: nn.Module = StreamingCausalPostFilter(
                channels=post_filter_channels,
                layers=post_filter_layers,
                kernel_size=post_filter_kernel_size,
                activation=self.activation_name,
            )
        else:
            self.post_filter = nn.Identity()

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        frames = self._contextualize(z)
        batch, time, _ = frames.shape
        subframes = self.expansion(frames).view(
            batch, time * self.hybrid_subframes, self.subframe_dim
        )
        subframes = self.refiner(subframes.transpose(1, 2))
        if self.hybrid_mode == "linear_200hz":
            wav = self.waveform_projection(
                subframes.transpose(1, 2)
            ).flatten(1).unsqueeze(1)
        else:
            wav = self.waveform_projection(subframes)
        return self.post_filter(wav)


class ISTFTDecoder(_FrameContextDecoder):
    """Complex-spectrum decoder with stateful normalized overlap-add synthesis."""

    def __init__(
        self,
        frame_hop_samples: int = 320,
        istft_n_fft: int = 512,
        istft_hop_length: int = 320,
        istft_win_length: int = 512,
        istft_predict: str = "complex",
        istft_center: bool = False,
        linear_hidden_dim: int = 1024,
        use_snake_activation: bool = False,
        snake_type: str = "snakebeta",
        snake_location: str = "decoder",
        **kwargs,
    ):
        super().__init__(**kwargs)
        if istft_predict != "complex":
            raise ValueError("Only istft_predict='complex' is currently supported")
        if istft_hop_length != frame_hop_samples:
            raise ValueError("istft_hop_length must equal frame_hop_samples")
        if istft_center:
            raise ValueError("Streaming ISTFT requires istft_center=False")
        self.frame_hop_samples = frame_hop_samples
        self.n_fft = istft_n_fft
        self.hop_length = istft_hop_length
        self.win_length = istft_win_length
        self.center = istft_center
        if snake_type not in {"snake", "snakebeta"}:
            raise ValueError("snake_type must be snake or snakebeta")
        if snake_location != "decoder":
            raise ValueError("snake_location must be decoder")
        projection_activation = (
            "SnakeBeta" if snake_type == "snakebeta" else "Snake"
        ) if use_snake_activation else self.activation_name
        bins = istft_n_fft // 2 + 1
        self.spectrum_projection = nn.Sequential(
            nn.Linear(self.decoder_dim, linear_hidden_dim),
            _activation(projection_activation, linear_hidden_dim),
            nn.Linear(linear_hidden_dim, 2 * bins),
        )
        self.istft = StreamingISTFT(
            n_fft=istft_n_fft,
            hop_length=istft_hop_length,
            win_length=istft_win_length,
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        frames = self._contextualize(z)
        spectrum = self.spectrum_projection(frames)
        real, imag = spectrum.chunk(2, dim=-1)
        complex_spectrum = torch.complex(real.float(), imag.float()).transpose(1, 2)
        wav = self.istft(complex_spectrum).to(real.dtype)
        return wav.unsqueeze(1)


DECODER_TYPES = {
    "frame_linear": FrameLinearWaveformDecoder,
    "subframe_linear": SubframeLinearWaveformDecoder,
    "hybrid_linear_cnn": HybridLinearCNNDecoder,
    "istft": ISTFTDecoder,
}


def _config_dict(config: tp.Any) -> dict[str, tp.Any]:
    if config is None:
        return {}
    if is_dataclass(config):
        return asdict(config)
    return dict(config)


def _decoder_kwargs(decoder_class: type[nn.Module], config: tp.Any) -> dict[str, tp.Any]:
    values = _config_dict(config)
    accepted = set()
    for cls in decoder_class.mro():
        if cls is nn.Module:
            break
        accepted.update(
            name
            for name, parameter in inspect.signature(cls.__init__).parameters.items()
            if name != "self"
            and parameter.kind
            in {parameter.POSITIONAL_OR_KEYWORD, parameter.KEYWORD_ONLY}
        )
    return {name: value for name, value in values.items() if name in accepted}


def create_waveform_decoder(
    decoder_type: str = "seanet",
    *,
    seanet_args: tp.Any = None,
    decoder_args: tp.Any = None,
) -> nn.Module:
    """Construct a waveform decoder while preserving old SEANet configs."""

    if decoder_type == "seanet":
        return SEANetDecoder(**_config_dict(seanet_args))
    try:
        decoder_class = DECODER_TYPES[decoder_type]
    except KeyError as exc:
        valid = ", ".join(["seanet", *DECODER_TYPES])
        raise ValueError(
            f"Unknown decoder_type={decoder_type!r}. Expected one of: {valid}"
        ) from exc
    return decoder_class(**_decoder_kwargs(decoder_class, decoder_args))
