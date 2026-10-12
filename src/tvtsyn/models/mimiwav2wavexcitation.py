"""Excitation-aware streaming waveform-to-waveform model wrapper."""

from contextlib import nullcontext

import torch
from torch import nn
from torch.nn import functional as F

from tvtsyn.components.excitation import (
    HarmonicSourceConditioner,
    HarmonicSourceGenerator,
    ProsodyConditioner,
    ProsodyPredictor,
    predicted_prosody_features,
    target_prosody_features,
)
from tvtsyn.components.streaming import StreamingContainer
from tvtsyn.utils.compile import no_compile


class MimiWav2WavExcitationModel(StreamingContainer):
    def __init__(
        self,
        encoder: nn.Module,
        transformer: nn.Module,
        decoder: nn.Module,
        sample_rate: int = 16000,
        frame_rate: int = 50,
        channels: int = 1,
        use_prosody_conditioning: bool = True,
        prosody_hidden_dim: int = 256,
        prosody_num_layers: int = 2,
        prosody_kernel_size: int = 3,
        prosody_conditioning_mode: str = "add",
        predict_coded_ap: bool = False,
        coded_ap_dim: int = 0,
        use_harmonic_source_conditioning: bool = True,
        harmonic_num_harmonics: int = 8,
        harmonic_noise_std: float = 0.003,
        harmonic_use_energy_scaling: bool = True,
        harmonic_conditioning_mode: str = "add",
        harmonic_source_n_fft: int = 512,
        harmonic_source_win_length: int = 512,
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
        self.use_prosody_conditioning = use_prosody_conditioning
        self.use_harmonic_source_conditioning = (
            use_prosody_conditioning and use_harmonic_source_conditioning
        )
        if use_prosody_conditioning:
            self.prosody_predictor = ProsodyPredictor(
                input_dim=dimension,
                hidden_dim=prosody_hidden_dim,
                num_layers=prosody_num_layers,
                kernel_size=prosody_kernel_size,
                predict_coded_ap=predict_coded_ap,
                coded_ap_dim=coded_ap_dim,
            )
            condition_dim = 3 + (coded_ap_dim if predict_coded_ap else 0)
            self.prosody_conditioner = ProsodyConditioner(
                input_dim=dimension,
                cond_dim=condition_dim,
                hidden_dim=prosody_hidden_dim,
                mode=prosody_conditioning_mode,
            )
        if self.use_harmonic_source_conditioning:
            self.harmonic_source_generator = HarmonicSourceGenerator(
                sample_rate=sample_rate,
                hop_length=self._hop_length,
                num_harmonics=harmonic_num_harmonics,
                noise_std=harmonic_noise_std,
                use_energy_scaling=harmonic_use_energy_scaling,
            )
            self.harmonic_source_conditioner = HarmonicSourceConditioner(
                input_dim=dimension,
                sample_rate=sample_rate,
                n_fft=harmonic_source_n_fft,
                hop_length=self._hop_length,
                win_length=harmonic_source_win_length,
                mode=harmonic_conditioning_mode,
            )

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
        return nullcontext() if self.torch_compile_encoder_decoder else no_compile()

    @staticmethod
    def _target_source(excitation: dict, frames: int):
        return (
            excitation["f0_hz"][:, :frames],
            excitation["vuv"][:, :frames],
            excitation["log_energy"][:, :frames],
        )

    @staticmethod
    def _predicted_source(prediction: dict, frames: int):
        return (
            prediction["log_f0"][:, 0, :frames].clamp(-10, 7).exp(),
            torch.sigmoid(prediction["vuv_logits"][:, 0, :frames]),
            prediction["log_energy"][:, 0, :frames],
        )

    def forward(
        self,
        waveform: torch.Tensor,
        excitation: dict | None = None,
        use_target_prosody: bool = False,
        return_aux: bool = False,
    ):
        if waveform.dim() != 3 or waveform.shape[1] != self.channels:
            raise ValueError(
                f"Expected waveform [B, {self.channels}, T], got {tuple(waveform.shape)}"
            )
        input_length = waveform.shape[-1]
        analysis_waveform = waveform
        if not self.is_streaming and input_length % self._hop_length:
            analysis_waveform = F.pad(
                waveform, (0, self._hop_length - input_length % self._hop_length)
            )
        with self._component_context:
            z = self.encoder(analysis_waveform)
        (z,) = self.transformer(z)
        aux: dict[str, torch.Tensor | dict] = {}
        if z.shape[-1] == 0:
            output = waveform.new_empty(waveform.shape[0], 1, 0)
            if return_aux:
                return output, aux
            return output
        if self.use_prosody_conditioning:
            prediction = self.prosody_predictor(z)
            aux["prosody_pred"] = prediction
            if use_target_prosody and excitation is not None:
                frames = min(z.shape[-1], excitation["f0_hz"].shape[-1])
                condition = target_prosody_features(excitation)[..., :frames]
                if "coded_ap" in excitation and "coded_ap" in prediction:
                    condition = torch.cat(
                        [condition, excitation["coded_ap"][:, :frames].transpose(1, 2)],
                        dim=1,
                    )
                elif "coded_ap" in prediction:
                    condition = torch.cat(
                        [condition, prediction["coded_ap"][..., :frames]],
                        dim=1,
                    )
                f0, vuv, log_energy = self._target_source(excitation, frames)
            else:
                frames = z.shape[-1]
                condition = predicted_prosody_features(prediction)
                if "coded_ap" in prediction:
                    condition = torch.cat([condition, prediction["coded_ap"]], dim=1)
                f0, vuv, log_energy = self._predicted_source(prediction, frames)
            z = self.prosody_conditioner(z[..., :frames], condition)
            if self.use_harmonic_source_conditioning:
                source = self.harmonic_source_generator(f0, vuv, log_energy)
                aux["harmonic_source"] = source
                z = self.harmonic_source_conditioner(z, source)
        with self._component_context:
            output = self.decoder(z)
        if not self.is_streaming:
            output = output[..., :input_length]
        if return_aux:
            return output, aux
        return output

    def decode(self, waveform: torch.Tensor, **kwargs):
        return self.forward(waveform, **kwargs)
