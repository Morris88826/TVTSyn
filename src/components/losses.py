import contextlib
import typing
from typing import List

import torch
import torch.nn.functional as F
from audiotools import AudioSignal
from audiotools import STFTParams
from torch import nn
# from src.models import SpeakerEncoder


class L1Loss(nn.L1Loss):
    """L1 Loss between AudioSignals. Defaults
    to comparing ``audio_data``, but any
    attribute of an AudioSignal can be used.

    Parameters
    ----------
    attribute : str, optional
        Attribute of signal to compare, defaults to ``audio_data``.
    weight : float, optional
        Weight of this loss, defaults to 1.0.

    Implementation copied from: https://github.com/descriptinc/lyrebird-audiotools/blob/961786aa1a9d628cca0c0486e5885a457fe70c1a/audiotools/metrics/distance.py
    """

    def __init__(self, attribute: str = "audio_data", weight: float = 1.0, **kwargs):
        self.attribute = attribute
        self.weight = weight
        super().__init__(**kwargs)

    def forward(self, x: AudioSignal, y: AudioSignal):
        """
        Parameters
        ----------
        x : AudioSignal
            Estimate AudioSignal
        y : AudioSignal
            Reference AudioSignal

        Returns
        -------
        torch.Tensor
            L1 loss between AudioSignal attributes.
        """
        if isinstance(x, AudioSignal):
            x = getattr(x, self.attribute)
            y = getattr(y, self.attribute)
        return super().forward(x, y)


class SISDRLoss(nn.Module):
    """
    Computes the Scale-Invariant Source-to-Distortion Ratio between a batch
    of estimated and reference audio signals or aligned features.

    Parameters
    ----------
    scaling : int, optional
        Whether to use scale-invariant (True) or
        signal-to-noise ratio (False), by default True
    reduction : str, optional
        How to reduce across the batch (either 'mean',
        'sum', or none).], by default ' mean'
    zero_mean : int, optional
        Zero mean the references and estimates before
        computing the loss, by default True
    clip_min : int, optional
        The minimum possible loss value. Helps network
        to not focus on making already good examples better, by default None
    weight : float, optional
        Weight of this loss, defaults to 1.0.

    Implementation copied from: https://github.com/descriptinc/lyrebird-audiotools/blob/961786aa1a9d628cca0c0486e5885a457fe70c1a/audiotools/metrics/distance.py
    """

    def __init__(
        self,
        scaling: int = True,
        reduction: str = "mean",
        zero_mean: int = True,
        clip_min: int = None,
        weight: float = 1.0,
    ):
        self.scaling = scaling
        self.reduction = reduction
        self.zero_mean = zero_mean
        self.clip_min = clip_min
        self.weight = weight
        super().__init__()

    def forward(self, x: AudioSignal, y: AudioSignal):
        eps = 1e-8
        # nb, nc, nt
        if isinstance(x, AudioSignal):
            references = x.audio_data
            estimates = y.audio_data
        else:
            references = x
            estimates = y

        nb = references.shape[0]
        references = references.reshape(nb, 1, -1).permute(0, 2, 1)
        estimates = estimates.reshape(nb, 1, -1).permute(0, 2, 1)

        # samples now on axis 1
        if self.zero_mean:
            mean_reference = references.mean(dim=1, keepdim=True)
            mean_estimate = estimates.mean(dim=1, keepdim=True)
        else:
            mean_reference = 0
            mean_estimate = 0

        _references = references - mean_reference
        _estimates = estimates - mean_estimate

        references_projection = (_references**2).sum(dim=-2) + eps
        references_on_estimates = (_estimates * _references).sum(dim=-2) + eps

        scale = (
            (references_on_estimates / references_projection).unsqueeze(1)
            if self.scaling
            else 1
        )

        e_true = scale * _references
        e_res = _estimates - e_true

        signal = (e_true**2).sum(dim=1)
        noise = (e_res**2).sum(dim=1)
        sdr = -10 * torch.log10(signal / noise + eps)

        if self.clip_min is not None:
            sdr = torch.clamp(sdr, min=self.clip_min)

        if self.reduction == "mean":
            sdr = sdr.mean()
        elif self.reduction == "sum":
            sdr = sdr.sum()
        return sdr


class MultiScaleSTFTLoss(nn.Module):
    """Computes the multi-scale STFT loss from [1].

    Parameters
    ----------
    window_lengths : List[int], optional
        Length of each window of each STFT, by default [2048, 512]
    loss_fn : typing.Callable, optional
        How to compare each loss, by default nn.L1Loss()
    clamp_eps : float, optional
        Clamp on the log magnitude, below, by default 1e-5
    mag_weight : float, optional
        Weight of raw magnitude portion of loss, by default 1.0
    log_weight : float, optional
        Weight of log magnitude portion of loss, by default 1.0
    pow : float, optional
        Power to raise magnitude to before taking log, by default 2.0
    weight : float, optional
        Weight of this loss, by default 1.0
    match_stride : bool, optional
        Whether to match the stride of convolutional layers, by default False

    References
    ----------

    1.  Engel, Jesse, Chenjie Gu, and Adam Roberts.
        "DDSP: Differentiable Digital Signal Processing."
        International Conference on Learning Representations. 2019.

    Implementation copied from: https://github.com/descriptinc/lyrebird-audiotools/blob/961786aa1a9d628cca0c0486e5885a457fe70c1a/audiotools/metrics/spectral.py
    """

    def __init__(
        self,
        window_lengths: List[int] = [2048, 512],
        loss_fn: typing.Callable = nn.L1Loss(),
        clamp_eps: float = 1e-5,
        mag_weight: float = 1.0,
        log_weight: float = 1.0,
        pow: float = 2.0,
        weight: float = 1.0,
        match_stride: bool = False,
        window_type: str = None,
    ):
        super().__init__()
        self.stft_params = [
            STFTParams(
                window_length=w,
                hop_length=w // 4,
                match_stride=match_stride,
                window_type=window_type,
            )
            for w in window_lengths
        ]
        self.loss_fn = loss_fn
        self.log_weight = log_weight
        self.mag_weight = mag_weight
        self.clamp_eps = clamp_eps
        self.weight = weight
        self.pow = pow

    def forward(self, x: AudioSignal, y: AudioSignal):
        """Computes multi-scale STFT between an estimate and a reference
        signal.

        Parameters
        ----------
        x : AudioSignal
            Estimate signal
        y : AudioSignal
            Reference signal

        Returns
        -------
        torch.Tensor
            Multi-scale STFT loss.
        """
        loss = 0.0
        for s in self.stft_params:
            x.stft(s.window_length, s.hop_length, s.window_type)
            y.stft(s.window_length, s.hop_length, s.window_type)
            loss += self.log_weight * self.loss_fn(
                x.magnitude.clamp(self.clamp_eps).pow(self.pow).log10(),
                y.magnitude.clamp(self.clamp_eps).pow(self.pow).log10(),
            )
            loss += self.mag_weight * self.loss_fn(x.magnitude, y.magnitude)
        return loss


class MelSpectrogramLoss(nn.Module):
    """Compute distance between mel spectrograms. Can be used
    in a multi-scale way.

    Parameters
    ----------
    n_mels : List[int]
        Number of mels per STFT, by default [150, 80],
    window_lengths : List[int], optional
        Length of each window of each STFT, by default [2048, 512]
    loss_fn : typing.Callable, optional
        How to compare each loss, by default nn.L1Loss()
    clamp_eps : float, optional
        Clamp on the log magnitude, below, by default 1e-5
    mag_weight : float, optional
        Weight of raw magnitude portion of loss, by default 1.0
    log_weight : float, optional
        Weight of log magnitude portion of loss, by default 1.0
    pow : float, optional
        Power to raise magnitude to before taking log, by default 2.0
    weight : float, optional
        Weight of this loss, by default 1.0
    match_stride : bool, optional
        Whether to match the stride of convolutional layers, by default False

    Implementation copied from: https://github.com/descriptinc/lyrebird-audiotools/blob/961786aa1a9d628cca0c0486e5885a457fe70c1a/audiotools/metrics/spectral.py
    """

    def __init__(
        self,
        n_mels: List[int] = [5, 10, 20, 40, 80, 160, 320],
        window_lengths: List[int] = [32, 64, 128, 256, 512, 1024, 2048],
        loss_fn: typing.Callable = nn.L1Loss(),
        clamp_eps: float = 1e-5,
        mag_weight: float = 0.0,
        log_weight: float = 1.0,
        pow: float = 1.0,
        weight: float = 1.0,
        match_stride: bool = False,
        mel_fmin: List[float] = [0, 0, 0, 0, 0, 0, 0],
        mel_fmax: List[float] = [None, None, None, None, None, None, None],
        window_type: str = None,
    ):
        super().__init__()
        self.stft_params = [
            STFTParams(
                window_length=w,
                hop_length=w // 4,
                match_stride=match_stride,
                window_type=window_type,
            )
            for w in window_lengths
        ]
        self.n_mels = n_mels
        self.loss_fn = loss_fn
        self.clamp_eps = clamp_eps
        self.log_weight = log_weight
        self.mag_weight = mag_weight
        self.weight = weight
        self.mel_fmin = mel_fmin
        self.mel_fmax = mel_fmax
        self.pow = pow

    def forward(self, x: AudioSignal, y: AudioSignal):
        """Computes mel loss between an estimate and a reference
        signal.

        Parameters
        ----------
        x : AudioSignal
            Estimate signal
        y : AudioSignal
            Reference signal

        Returns
        -------
        torch.Tensor
            Mel loss.
        """
        loss = 0.0
        for n_mels, fmin, fmax, s in zip(
            self.n_mels, self.mel_fmin, self.mel_fmax, self.stft_params
        ):
            kwargs = {
                "window_length": s.window_length,
                "hop_length": s.hop_length,
                "window_type": s.window_type,
            }
            x_mels = x.mel_spectrogram(n_mels, mel_fmin=fmin, mel_fmax=fmax, **kwargs)
            y_mels = y.mel_spectrogram(n_mels, mel_fmin=fmin, mel_fmax=fmax, **kwargs)

            loss += self.log_weight * self.loss_fn(
                x_mels.clamp(self.clamp_eps).pow(self.pow).log10(),
                y_mels.clamp(self.clamp_eps).pow(self.pow).log10(),
            )
            loss += self.mag_weight * self.loss_fn(x_mels, y_mels)
        return loss


class GANLoss(nn.Module):
    """
    Computes a discriminator loss, given a discriminator on
    generated waveforms/spectrograms compared to ground truth
    waveforms/spectrograms. Computes the loss for both the
    discriminator and the generator in separate functions.
    """

    def __init__(self, discriminator):
        super().__init__()
        self.discriminator = discriminator

    def forward(self, fake, real):
        d_fake = self.discriminator(fake.audio_data)
        d_real = self.discriminator(real.audio_data)
        return d_fake, d_real

    def discriminator_loss(self, fake, real):
        d_fake, d_real = self.forward(fake.clone().detach(), real)

        loss_d = 0
        for x_fake, x_real in zip(d_fake, d_real):
            loss_d += torch.mean(x_fake[-1] ** 2)
            loss_d += torch.mean((1 - x_real[-1]) ** 2)
        return loss_d

    def generator_loss(self, fake, real):
        d_fake, d_real = self.forward(fake, real)

        loss_g = 0
        for x_fake in d_fake:
            loss_g += torch.mean((1 - x_fake[-1]) ** 2)

        loss_feature = 0

        for i in range(len(d_fake)):
            for j in range(len(d_fake[i]) - 1):
                loss_feature += F.l1_loss(d_fake[i][j], d_real[i][j].detach())
        return loss_g, loss_feature


class LossPED(nn.Module):
    """Loss Computation
    Arguments
    ---------
    log_scale_durations: bool
       applies logarithm to target durations
    duration_loss_weight: int
       weight for the duration loss
    pitch_loss_weight: int
       weight for the pitch loss
    energy_loss_weight: int
       weight for the energy loss
    """

    def __init__(self, pitch_loss_weight=1, energy_loss_weight=1):
        super().__init__()

        self.pitch_loss = nn.MSELoss()
        self.energy_loss = nn.MSELoss()
        self.pitch_loss_weight = pitch_loss_weight
        self.energy_loss_weight = energy_loss_weight

    def forward(self, predicted_pitch, average_pitch, 
                predicted_energy, average_energy):
        """Computes the value of the loss function and updates stats
        Arguments
        ---------
        predictions: tuple
            model predictions
        targets: tuple
            ground truth data
        Returns
        -------
        loss: torch.Tensor
            the loss value
        """

        predicted_pitch = predicted_pitch.squeeze(-1)
        predicted_energy = predicted_energy.squeeze(-1)

        target_pitch = average_pitch.squeeze(-1)
        target_energy = average_energy.squeeze(-1)

        # change this to perform batch level using padding mask
        pitch_loss = self.pitch_loss(
            predicted_pitch,
            target_pitch.to(torch.float32),
        )
        energy_loss = self.energy_loss(
            predicted_energy,
            target_energy.to(torch.float32),
        )
    
        total_loss = (
            pitch_loss * self.pitch_loss_weight
            + energy_loss * self.energy_loss_weight
        )
        
        loss = {
            "total_loss": total_loss,
            "pitch_loss": pitch_loss * self.pitch_loss_weight,
            "energy_loss": energy_loss * self.energy_loss_weight,
        }
        return loss


class HuBERTLoss(nn.Module):
    """Frozen, fairseq-free HuBERT perceptual feature loss.

    By default this matches the convolutional feature encoder used by LLVC.
    Set ``feature_layer`` to an integer to compare transformer hidden states.
    HuBERT parameters stay frozen, but gradients are retained for ``inputs``.
    """

    def __init__(
        self,
        model_name_or_path: str = "facebook/hubert-base-ls960",
        feature_layer: int | None = None,
        loss_type: str = "mse",
        normalize_waveform: bool = False,
        local_files_only: bool = False,
        revision: str | None = None,
    ):
        super().__init__()
        if loss_type not in {"mse", "l1"}:
            raise ValueError("loss_type must be mse or l1")

        try:
            from transformers import HubertModel
        except ImportError as exc:
            raise ImportError(
                "HuBERTLoss requires the transformers package. "
                "Install transformers without fairseq."
            ) from exc

        load_kwargs = {"local_files_only": local_files_only}
        if revision is not None:
            load_kwargs["revision"] = revision
        model = HubertModel.from_pretrained(model_name_or_path, **load_kwargs)
        model.requires_grad_(False)
        model.eval()

        # The fixed auxiliary network should not enter optimizers or checkpoints.
        object.__setattr__(self, "_hubert_model", model)
        self.feature_layer = feature_layer
        self.loss_type = loss_type
        self.normalize_waveform = normalize_waveform

    @property
    def model(self) -> nn.Module:
        return self._hubert_model

    @staticmethod
    def _waveform(value: AudioSignal | torch.Tensor) -> torch.Tensor:
        if isinstance(value, AudioSignal):
            value = value.audio_data
        if value.dim() == 3:
            if value.shape[1] != 1:
                raise ValueError(f"Expected mono waveform, got {tuple(value.shape)}")
            value = value[:, 0]
        if value.dim() != 2:
            raise ValueError(
                f"Expected waveform [B, T] or [B, 1, T], got {tuple(value.shape)}"
            )
        return value

    def _normalize(self, waveform: torch.Tensor) -> torch.Tensor:
        if not self.normalize_waveform:
            return waveform
        mean = waveform.mean(dim=-1, keepdim=True)
        variance = waveform.var(dim=-1, keepdim=True, unbiased=False)
        return (waveform - mean) * torch.rsqrt(variance + 1e-7)

    def _features(self, waveform: torch.Tensor) -> torch.Tensor:
        if self.feature_layer is None:
            return self.model.feature_extractor(waveform)
        output = self.model(
            waveform,
            output_hidden_states=True,
            return_dict=True,
        )
        hidden_states = output.hidden_states
        if not -len(hidden_states) <= self.feature_layer < len(hidden_states):
            raise ValueError(
                f"feature_layer={self.feature_layer} is out of range for "
                f"{len(hidden_states)} HuBERT hidden states"
            )
        return hidden_states[self.feature_layer]

    def forward(
        self,
        inputs: AudioSignal | torch.Tensor,
        targets: AudioSignal | torch.Tensor,
    ) -> torch.Tensor:
        inputs = self._normalize(self._waveform(inputs).float())
        targets = self._normalize(self._waveform(targets).float())
        if inputs.shape != targets.shape:
            length = min(inputs.shape[-1], targets.shape[-1])
            inputs = inputs[..., :length]
            targets = targets[..., :length]

        model_device = next(self.model.parameters()).device
        if model_device != inputs.device:
            self.model.to(inputs.device)
        self.model.eval()

        autocast = (
            torch.autocast(device_type=inputs.device.type, enabled=False)
            if inputs.device.type in {"cpu", "cuda"}
            else contextlib.nullcontext()
        )
        with autocast:
            input_features = self._features(inputs)
            with torch.no_grad():
                target_features = self._features(targets)

            if self.feature_layer is None:
                frames = min(input_features.shape[-1], target_features.shape[-1])
                input_features = input_features[..., :frames]
                target_features = target_features[..., :frames]
            else:
                frames = min(input_features.shape[1], target_features.shape[1])
                input_features = input_features[:, :frames]
                target_features = target_features[:, :frames]
            if self.loss_type == "mse":
                return F.mse_loss(input_features, target_features)
            return F.l1_loss(input_features, target_features)


# class SpeakerLoss(nn.Module):
#     """Loss Computation for Speaker model using CosineEmbeddingLoss"""

#     def __init__(self, loss_fn: typing.Callable = nn.CosineEmbeddingLoss(), device = "cuda"):
#         super().__init__()
#         self.loss_fn = loss_fn
#         self.model = SpeakerEncoder(device=device)
#         self.device = device

#     def forward(self, inputs, target_embeddings):
#         if isinstance(inputs, AudioSignal):
#             inputs = getattr(inputs, "audio_data").to(self.device)
            
#         input_embeds = self.model(inputs.squeeze())
#         loss = self.loss_fn(input_embeds, target_embeddings, torch.ones(input_embeds.shape[0]).to(input_embeds.device))
#         return loss