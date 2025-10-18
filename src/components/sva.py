import typing as tp

import numpy as np
import torch
import torch.nn as nn

from .conv import StreamingConv1d, StreamingConvTranspose1d, TransposedLayerNorm
from .streaming import StreamingContainer, StreamingAdd
from ..utils.compile import torch_compile_lazy
from .speaker_adapter import SpeakerAdapter



class VariancePredictor(StreamingContainer):
    def __init__(
        self,
        in_channels, 
        out_channels, 
        kernel_size,
        activation: str = "ELU",
        activation_params: dict = {"alpha": 1.0},
        dropout: float = 0.0,
        norm: str = "none",
        norm_params: tp.Dict[str, tp.Any] = {},
        causal: bool = False,
        pad_mode: str = "reflect",
    ):
        super().__init__()
        act = getattr(nn, activation)

        self.model = nn.Sequential(
            StreamingConv1d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                norm=norm,
                norm_kwargs=norm_params,
                causal=causal,
                pad_mode=pad_mode,
            ),
            act(**activation_params),
            nn.Dropout(dropout),
            TransposedLayerNorm(normalized_shape=out_channels),
            StreamingConv1d(
                out_channels,
                out_channels,
                kernel_size=kernel_size,
                norm=norm,
                norm_kwargs=norm_params,
                causal=causal,
                pad_mode=pad_mode,
            ),
            act(**activation_params),
            nn.Dropout(dropout),
            TransposedLayerNorm(normalized_shape=out_channels),
            StreamingConv1d(
                out_channels,
                1,
                kernel_size=1,
                norm=norm,
                norm_kwargs=norm_params,
                causal=causal,
                pad_mode=pad_mode,
            ),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    

class SpeakerVarianceAdapter(StreamingContainer):
    def __init__(
        self,
        in_channels: int,
        v_channels: int,
        out_channels: int,
        speaker_dim: int,
        var_kernel_size: int = 3,
        pitch_kernel_size: int = 3,
        energy_kernel_size: int = 3,
        activation: str = "ELU",
        activation_params: dict = {"alpha": 1.0},
        dropout=0.0,
        norm: str = "none",
        norm_params: tp.Dict[str, tp.Any] = {},
        causal: bool = False,
        pad_mode: str = "reflect",
    ):
        super().__init__()
        act = getattr(nn, activation)


        self.spk_adapter = SpeakerAdapter(
            speaker_dim,
            out_channels,
        )

        self.pitchPred = VariancePredictor(
            in_channels=in_channels,
            out_channels=v_channels,
            kernel_size=var_kernel_size,
            activation=activation,
            activation_params=activation_params,
            dropout=dropout,
            norm=norm,
            norm_params=norm_params,
            causal=causal,
            pad_mode=pad_mode,
        )
        
        self.energyPred = VariancePredictor(
            in_channels=in_channels,
            out_channels=v_channels,
            kernel_size=var_kernel_size,
            activation=activation,
            activation_params=activation_params,
            dropout=dropout,
            norm=norm,
            norm_params=norm_params,
            causal=causal,
            pad_mode=pad_mode,
        )

        self.pitchEmbed = StreamingConv1d(
                1,
                out_channels,
                kernel_size=pitch_kernel_size,
                norm=norm,
                norm_kwargs=norm_params,
                causal=causal,
                pad_mode=pad_mode,
            )

        self.energyEmbed = StreamingConv1d(
                1,
                out_channels,
                kernel_size=energy_kernel_size,
                norm=norm,
                norm_kwargs=norm_params,
                causal=causal,
                pad_mode=pad_mode,
            )

        self.pitch_add = StreamingAdd()
        self.energy_add = StreamingAdd()

    def _average_over_durations(self, values, durs):
        """Average values over durations.
        Arguments
        ---------
        values: torch.Tensor
            shape: [B, 1, T_de]
        durs: torch.Tensor
            shape: [B, T_en]
        Returns
        ---------
        avg: torch.Tensor
            shape: [B, 1, T_en]
        """
        avg_batch = []
        for i, (d, v) in enumerate(zip(durs, values)):
            v = v.unsqueeze(0)
            durs_cums_ends = torch.cumsum(d.unsqueeze(0), dim=1).long()
            durs_cums_starts = torch.nn.functional.pad(durs_cums_ends[:, :-1], (1, 0))
            values_nonzero_cums = torch.nn.functional.pad(
                torch.cumsum(v != 0.0, dim=2), (1, 0)
            )
            values_cums = torch.nn.functional.pad(torch.cumsum(v, dim=2), (1, 0))

            bs, length = durs_cums_ends.size()
            n_formants = v.size(1)
            dcs = durs_cums_starts[:, None, :].expand(bs, n_formants, length)
            dce = durs_cums_ends[:, None, :].expand(bs, n_formants, length)

            values_sums = (
                torch.gather(values_cums, 2, dce) - torch.gather(values_cums, 2, dcs)
            ).float()
            values_nelems = (
                torch.gather(values_nonzero_cums, 2, dce)
                - torch.gather(values_nonzero_cums, 2, dcs)
            ).float()

            avg = torch.where(
                values_nelems == 0.0, values_nelems, values_sums / values_nelems
            )
            avg_batch.append(avg.squeeze(0).repeat_interleave(d, dim=-1))
        return torch.stack(avg_batch, dim=0)


    def _get_durations(self, tokens):
        durations = [
            torch.unique_consecutive(tokens[i], return_counts=True)[1]
            for i in range(len(tokens))
        ]
        return durations

    @torch_compile_lazy
    def forward(
        self,
        x: torch.Tensor,
        embeds: torch.Tensor, 
        pitch: torch.Tensor=None, 
        energy: torch.Tensor=None, 
        tokens: torch.Tensor=None,
        return_variance: bool = False,
        pitch_rate: float = 1.0,
        energy_rate: float = 1.0,
    ):
        """Forward pass of the speaker variance adapter.

        Args:
            x (torch.Tensor): Input tensor of shape [B, C, T].
            embeds (torch.Tensor): Speaker embedding of shape [B, speaker_dim].
            pitch (torch.Tensor): Pitch tensor of shape [B, 1, T].
            energy (torch.Tensor): Energy tensor of shape [B, 1, T].
            tokens (torch.Tensor): Token tensor of shape [B, T].
            return_variance (bool): Whether to return the variance predictions.

        Returns:
            torch.Tensor: Output tensor of shape [B, C, T].
        """
        # Apply speaker adapter
        x = self.spk_adapter(x, embeds)

        # Get pitch and energy predictions
        predicted_pitch = self.pitchPred(x) * pitch_rate
        predicted_energy = self.energyPred(x) * energy_rate

        avg_pitch, avg_energy = None, None
        # Apply pitch and energy prediction
        if pitch is not None and energy is not None:
            if tokens is not None:
                durs = self._get_durations(tokens)
                # Get the average over durations
                avg_pitch = self._average_over_durations(pitch.unsqueeze(1), durs)
                avg_energy = self._average_over_durations(energy.unsqueeze(1), durs)
            else:
                avg_pitch, avg_energy = pitch, energy

            pitch = self.pitchEmbed(avg_pitch)
            energy = self.energyEmbed(avg_energy)
        else:
            pitch = self.pitchEmbed(predicted_pitch)
            energy = self.energyEmbed(predicted_energy)

        
        # Add pitch and energy to the input
        x = self.pitch_add(x, pitch)
        x = self.energy_add(x, energy)

        if return_variance:
            return x, (avg_pitch, predicted_pitch, avg_energy, predicted_energy)
        
        return x
        
     

