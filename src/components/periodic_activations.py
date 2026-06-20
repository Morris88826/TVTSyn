"""Learnable periodic activations for waveform-proximal decoder blocks."""

import torch
from torch import nn


class _PeriodicActivation(nn.Module):
    def __init__(self, channels: int, alpha: float = 1.0):
        super().__init__()
        if channels < 1:
            raise ValueError("channels must be positive")
        self.channels = channels
        self.log_alpha = nn.Parameter(torch.full((channels,), float(alpha)).log())

    def _parameter(self, value: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] == self.channels:
            return value.view(*([1] * (x.dim() - 1)), self.channels)
        if x.dim() >= 2 and x.shape[1] == self.channels:
            return value.view(1, self.channels, *([1] * (x.dim() - 2)))
        raise ValueError(
            f"Expected channel dimension of size {self.channels}, got {tuple(x.shape)}"
        )


class Snake(_PeriodicActivation):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        alpha = self._parameter(self.log_alpha.exp().clamp_min(1e-6), x)
        return x + torch.sin(alpha * x).square() / alpha


class SnakeBeta(_PeriodicActivation):
    def __init__(self, channels: int, alpha: float = 1.0, beta: float = 1.0):
        super().__init__(channels, alpha)
        self.log_beta = nn.Parameter(torch.full((channels,), float(beta)).log())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        alpha = self._parameter(self.log_alpha.exp().clamp_min(1e-6), x)
        beta = self._parameter(self.log_beta.exp().clamp_min(1e-6), x)
        return x + torch.sin(alpha * x).square() / beta
