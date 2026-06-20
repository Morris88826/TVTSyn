"""Causal streaming spectral analysis and synthesis utilities."""

from dataclasses import dataclass
import math

import torch
from torch import nn

from .streaming import StreamingModule


def _hz_to_mel(freq: torch.Tensor) -> torch.Tensor:
    min_log_hz = 1000.0
    min_log_mel = 15.0
    logstep = math.log(6.4) / 27.0
    linear = 3.0 * freq / 200.0
    log = min_log_mel + torch.log(freq.clamp_min(min_log_hz) / min_log_hz) / logstep
    return torch.where(freq >= min_log_hz, log, linear)


def _mel_to_hz(mels: torch.Tensor) -> torch.Tensor:
    min_log_hz = 1000.0
    min_log_mel = 15.0
    logstep = math.log(6.4) / 27.0
    linear = 200.0 * mels / 3.0
    log = min_log_hz * torch.exp(logstep * (mels - min_log_mel))
    return torch.where(mels >= min_log_mel, log, linear)


def _slaney_mel_filterbank(
    n_fft: int,
    n_mels: int,
    sample_rate: int,
    f_min: float,
    f_max: float,
) -> torch.Tensor:
    fft_freqs = torch.linspace(0.0, sample_rate / 2, n_fft // 2 + 1)
    mel_min = _hz_to_mel(torch.tensor(float(f_min)))
    mel_max = _hz_to_mel(torch.tensor(float(f_max)))
    mel_points = torch.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = _mel_to_hz(mel_points)

    lower = hz_points[:-2, None]
    center = hz_points[1:-1, None]
    upper = hz_points[2:, None]
    down = (fft_freqs[None] - lower) / (center - lower).clamp_min(1e-12)
    up = (upper - fft_freqs[None]) / (upper - center).clamp_min(1e-12)
    filters = torch.minimum(down, up).clamp_min(0.0)
    filters *= 2.0 / (upper - lower).clamp_min(1e-12)
    return filters


@dataclass
class _StreamingLogMelState:
    previous: torch.Tensor | None = None
    padding_to_add: int = 0
    original_padding: int = 0

    def reset(self):
        self.previous = None
        self.padding_to_add = self.original_padding


class StreamingLogMelSpectrogram(StreamingModule[_StreamingLogMelState]):
    """Causal, center-free log-mel frontend with persistent sample buffering."""

    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 1024,
        win_length: int = 1024,
        hop_length: int = 320,
        n_mels: int = 80,
        f_min: float = 0.0,
        f_max: float | None = None,
        center: bool = False,
        eps: float = 1e-5,
    ):
        super().__init__()
        if center:
            raise ValueError("StreamingLogMelSpectrogram requires center=False")
        if not 0 < hop_length <= win_length <= n_fft:
            raise ValueError("Expected 0 < hop_length <= win_length <= n_fft")
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.f_min = f_min
        self.f_max = float(sample_rate // 2) if f_max is None else f_max
        self.center = center
        self.eps = eps
        self.left_padding = win_length - hop_length

        self.register_buffer("window", torch.hann_window(win_length), persistent=False)
        self.register_buffer(
            "mel_filters",
            _slaney_mel_filterbank(n_fft, n_mels, sample_rate, f_min, self.f_max),
            persistent=False,
        )

    def _init_streaming_state(self, batch_size: int) -> _StreamingLogMelState:
        return _StreamingLogMelState(
            padding_to_add=self.left_padding,
            original_padding=self.left_padding,
        )

    def _spectrogram(self, waveform: torch.Tensor) -> torch.Tensor:
        batch, samples = waveform.shape
        if samples < self.win_length:
            return waveform.new_empty(batch, self.n_mels, 0)

        frames = waveform.unfold(-1, self.win_length, self.hop_length)
        compute_dtype = (
            torch.float32
            if waveform.dtype in {torch.float16, torch.bfloat16}
            else waveform.dtype
        )
        frames = frames.to(compute_dtype)
        window = self.window.to(device=waveform.device, dtype=compute_dtype)
        spectrum = torch.fft.rfft(frames * window, n=self.n_fft, dim=-1)
        filters = self.mel_filters.to(device=waveform.device, dtype=compute_dtype)
        mel = torch.matmul(spectrum.abs(), filters.transpose(0, 1))
        return mel.clamp_min(self.eps).log().transpose(1, 2).to(waveform.dtype)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        if waveform.dim() != 3 or waveform.shape[1] != 1:
            raise ValueError(
                f"Expected mono waveform [B, 1, T], got {tuple(waveform.shape)}"
            )
        x = waveform[:, 0]
        state = self._streaming_state
        if state is None:
            if self.left_padding:
                x = nn.functional.pad(x, (self.left_padding, 0))
            return self._spectrogram(x)

        if state.previous is not None:
            x = torch.cat([state.previous, x], dim=-1)
        if state.padding_to_add:
            x = nn.functional.pad(x, (state.padding_to_add, 0))
            state.padding_to_add = 0

        num_frames = max(0, 1 + (x.shape[-1] - self.win_length) // self.hop_length)
        consumed = num_frames * self.hop_length
        state.previous = x[..., consumed:]
        if num_frames == 0:
            return waveform.new_empty(waveform.shape[0], self.n_mels, 0)
        used = (num_frames - 1) * self.hop_length + self.win_length
        return self._spectrogram(x[..., :used])


@dataclass
class _StreamingISTFTState:
    previous_signal: torch.Tensor | None = None
    previous_norm: torch.Tensor | None = None

    def reset(self):
        self.previous_signal = None
        self.previous_norm = None


class StreamingISTFT(StreamingModule[_StreamingISTFTState]):
    """Rectangular-window ISTFT with normalized overlap-add streaming state."""

    def __init__(
        self,
        n_fft: int,
        hop_length: int,
        win_length: int | None = None,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = n_fft if win_length is None else win_length
        if not 0 < self.hop_length <= self.win_length <= self.n_fft:
            raise ValueError("Expected 0 < hop_length <= win_length <= n_fft")
        self.tail = self.win_length - self.hop_length
        self.eps = eps
        self.register_buffer("window", torch.ones(self.win_length), persistent=False)

    def _init_streaming_state(self, batch_size: int) -> _StreamingISTFTState:
        return _StreamingISTFTState()

    def forward(self, spectrum: torch.Tensor) -> torch.Tensor:
        if spectrum.dim() != 3 or not spectrum.is_complex():
            raise ValueError("Expected complex spectrum [B, n_fft // 2 + 1, F]")
        batch, bins, num_frames = spectrum.shape
        if bins != self.n_fft // 2 + 1:
            raise ValueError(
                f"Expected {self.n_fft // 2 + 1} frequency bins, got {bins}"
            )
        if num_frames == 0:
            return spectrum.real.new_empty(batch, 0)

        frames = torch.fft.irfft(spectrum.transpose(1, 2), n=self.n_fft)
        frames = frames[..., : self.win_length]
        window = self.window.to(device=frames.device, dtype=frames.dtype)
        frames = frames * window

        output_length = num_frames * self.hop_length + self.tail
        output = frames.new_zeros(batch, output_length)
        norm = frames.new_zeros(batch, output_length)
        window_sq = window.square()
        for frame_index in range(num_frames):
            start = frame_index * self.hop_length
            end = start + self.win_length
            output[:, start:end] += frames[:, frame_index]
            norm[:, start:end] += window_sq

        state = self._streaming_state
        if state is not None and state.previous_signal is not None:
            output[:, : self.tail] += state.previous_signal
            norm[:, : self.tail] += state.previous_norm

        ready_length = num_frames * self.hop_length
        ready = output[:, :ready_length] / norm[:, :ready_length].clamp_min(self.eps)
        if state is not None:
            state.previous_signal = output[:, ready_length:]
            state.previous_norm = norm[:, ready_length:]
        return ready
