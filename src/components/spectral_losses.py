"""Differentiable spectral and excitation-aware training losses."""

import math

import torch
from torch import nn
from torch.nn import functional as F


def _waveform(x: torch.Tensor) -> torch.Tensor:
    if hasattr(x, "audio_data"):
        x = x.audio_data
    if x.dim() == 3:
        x = x[:, 0]
    if x.dim() != 2:
        raise ValueError(f"Expected [B, T] or [B, 1, T], got {tuple(x.shape)}")
    return x


def _align(x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    length = min(x.shape[-1], y.shape[-1])
    return x[..., :length], y[..., :length]


def _stft(
    x: torch.Tensor, n_fft: int, hop: int, win_length: int, window: torch.Tensor
) -> torch.Tensor:
    x = _waveform(x).float()
    left = win_length - hop
    x = F.pad(x, (left, 0))
    with torch.autocast(device_type=x.device.type, enabled=False):
        return torch.stft(
            x,
            n_fft=n_fft,
            hop_length=hop,
            win_length=win_length,
            window=window.to(x.device, x.dtype),
            center=False,
            return_complex=True,
        )


class MultiResolutionSTFTLoss(nn.Module):
    def __init__(
        self,
        fft_sizes=(256, 512, 1024, 2048),
        hop_sizes=(64, 160, 320, 640),
        win_lengths=(256, 512, 1024, 2048),
        log_mag_weight: float = 1.0,
        spectral_convergence_weight: float = 1.0,
        eps: float = 1e-7,
    ):
        super().__init__()
        if not len(fft_sizes) == len(hop_sizes) == len(win_lengths):
            raise ValueError("STFT resolution lists must have equal lengths")
        self.resolutions = list(zip(fft_sizes, hop_sizes, win_lengths))
        self.log_mag_weight = log_mag_weight
        self.spectral_convergence_weight = spectral_convergence_weight
        self.eps = eps
        for index, win_length in enumerate(win_lengths):
            self.register_buffer(
                f"window_{index}", torch.hann_window(win_length), persistent=False
            )

    def forward(self, estimate: torch.Tensor, target: torch.Tensor):
        estimate, target = _align(_waveform(estimate), _waveform(target))
        logmag = estimate.new_zeros(())
        convergence = estimate.new_zeros(())
        for index, (n_fft, hop, win_length) in enumerate(self.resolutions):
            window = getattr(self, f"window_{index}")
            est_mag = _stft(estimate, n_fft, hop, win_length, window).abs()
            tgt_mag = _stft(target, n_fft, hop, win_length, window).abs()
            logmag = logmag + F.l1_loss(
                (est_mag + self.eps).log(), (tgt_mag + self.eps).log()
            )
            numerator = torch.linalg.vector_norm(est_mag - tgt_mag, dim=(-2, -1))
            denominator = torch.linalg.vector_norm(tgt_mag, dim=(-2, -1))
            convergence = convergence + (
                numerator / denominator.clamp_min(self.eps)
            ).mean()
        count = len(self.resolutions)
        return (
            self.log_mag_weight * logmag / count,
            self.spectral_convergence_weight * convergence / count,
        )


class ComplexSTFTLoss(nn.Module):
    def __init__(
        self,
        fft_size: int = 512,
        hop_size: int = 320,
        win_length: int = 512,
        real_imag_weight: float = 1.0,
        mag_weight: float = 1.0,
        eps: float = 1e-7,
    ):
        super().__init__()
        self.fft_size, self.hop_size, self.win_length = (
            fft_size,
            hop_size,
            win_length,
        )
        self.real_imag_weight, self.mag_weight, self.eps = (
            real_imag_weight,
            mag_weight,
            eps,
        )
        self.register_buffer("window", torch.hann_window(win_length), persistent=False)

    def forward(self, estimate: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        estimate, target = _align(_waveform(estimate), _waveform(target))
        est = _stft(
            estimate, self.fft_size, self.hop_size, self.win_length, self.window
        )
        tgt = _stft(target, self.fft_size, self.hop_size, self.win_length, self.window)
        complex_loss = F.l1_loss(est.real, tgt.real) + F.l1_loss(est.imag, tgt.imag)
        magnitude_loss = F.l1_loss(
            (est.abs() + self.eps).log(), (tgt.abs() + self.eps).log()
        )
        return self.real_imag_weight * complex_loss + self.mag_weight * magnitude_loss


class ProsodyLoss(nn.Module):
    def __init__(
        self,
        f0_loss: str = "smooth_l1",
        f0_weight: float = 1.0,
        vuv_weight: float = 1.0,
        energy_weight: float = 1.0,
        coded_ap_weight: float = 0.0,
    ):
        super().__init__()
        if f0_loss not in {"smooth_l1", "l1", "mse"}:
            raise ValueError("Unsupported f0_loss")
        self.f0_loss = f0_loss
        self.weights = (f0_weight, vuv_weight, energy_weight, coded_ap_weight)

    @staticmethod
    def _masked(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return value[mask].mean() if mask.any() else value.new_zeros(())

    def forward(self, prediction: dict, excitation: dict) -> dict[str, torch.Tensor]:
        frames = min(prediction["log_f0"].shape[-1], excitation["frame_mask"].shape[-1])
        valid = excitation["frame_mask"][:, :frames]
        voiced = valid & (excitation["vuv"][:, :frames] > 0.5)
        pred_f0 = prediction["log_f0"][:, 0, :frames]
        target_f0 = excitation["log_f0"][:, :frames]
        if self.f0_loss == "smooth_l1":
            f0_values = F.smooth_l1_loss(pred_f0, target_f0, reduction="none")
        elif self.f0_loss == "l1":
            f0_values = F.l1_loss(pred_f0, target_f0, reduction="none")
        else:
            f0_values = F.mse_loss(pred_f0, target_f0, reduction="none")
        f0 = self._masked(f0_values, voiced)
        vuv = self._masked(
            F.binary_cross_entropy_with_logits(
                prediction["vuv_logits"][:, 0, :frames],
                excitation["vuv"][:, :frames],
                reduction="none",
            ),
            valid,
        )
        energy = self._masked(
            F.smooth_l1_loss(
                prediction["log_energy"][:, 0, :frames],
                excitation["log_energy"][:, :frames],
                reduction="none",
            ),
            valid,
        )
        coded_ap = f0.new_zeros(())
        if "coded_ap" in prediction and "coded_ap" in excitation:
            channels = min(
                prediction["coded_ap"].shape[1], excitation["coded_ap"].shape[-1]
            )
            values = F.l1_loss(
                prediction["coded_ap"][:, :channels, :frames].transpose(1, 2),
                excitation["coded_ap"][:, :frames, :channels],
                reduction="none",
            ).mean(dim=-1)
            coded_ap = self._masked(values, valid)
        wf0, wvuv, wenergy, wap = self.weights
        return {
            "prosody_loss": wf0 * f0 + wvuv * vuv + wenergy * energy + wap * coded_ap,
            "f0_loss": f0,
            "vuv_loss": vuv,
            "energy_loss": energy,
            "coded_ap_loss": coded_ap,
            "voiced_frames": voiced.sum().to(f0.dtype),
        }


class _SingleResolutionLoss(nn.Module):
    def __init__(self, fft_size: int, hop_size: int, win_length: int, eps: float):
        super().__init__()
        self.fft_size, self.hop_size, self.win_length, self.eps = (
            fft_size,
            hop_size,
            win_length,
            eps,
        )
        self.register_buffer("window", torch.hann_window(win_length), persistent=False)

    def spectra(self, estimate, target):
        estimate, target = _align(_waveform(estimate), _waveform(target))
        return (
            _stft(
                estimate,
                self.fft_size,
                self.hop_size,
                self.win_length,
                self.window,
            ),
            _stft(
                target,
                self.fft_size,
                self.hop_size,
                self.win_length,
                self.window,
            ),
        )


class HarmonicMagnitudeLoss(_SingleResolutionLoss):
    def __init__(
        self,
        fft_size=1024,
        hop_size=320,
        win_length=1024,
        sample_rate=16000,
        max_freq=7600.0,
        max_harmonics=80,
        eps=1e-7,
    ):
        super().__init__(fft_size, hop_size, win_length, eps)
        self.sample_rate, self.max_freq, self.max_harmonics = (
            sample_rate,
            max_freq,
            max_harmonics,
        )

    def forward(self, estimate, target, excitation):
        est, tgt = self.spectra(estimate, target)
        frames = min(est.shape[-1], excitation["f0_hz"].shape[-1])
        f0 = excitation["f0_hz"][:, :frames]
        valid = excitation["frame_mask"][:, :frames] & (
            excitation["vuv"][:, :frames] > 0.5
        )
        harmonics = torch.arange(
            1, self.max_harmonics + 1, device=f0.device, dtype=f0.dtype
        )[None, None]
        frequencies = f0[..., None] * harmonics
        harmonic_mask = valid[..., None] & (frequencies <= self.max_freq)
        bins = (frequencies * self.fft_size / self.sample_rate).round().long()
        bins = bins.clamp(0, est.shape[1] - 1)
        est_values = torch.gather(
            (est[..., :frames].abs() + self.eps).log().transpose(1, 2), 2, bins
        )
        tgt_values = torch.gather(
            (tgt[..., :frames].abs() + self.eps).log().transpose(1, 2), 2, bins
        )
        if not harmonic_mask.any():
            return est.real.new_zeros(())
        return (est_values - tgt_values).abs()[harmonic_mask].mean()


class CepstralEnvelopeLoss(_SingleResolutionLoss):
    def __init__(
        self, fft_size=1024, hop_size=320, win_length=1024, num_ceps=40, eps=1e-7
    ):
        super().__init__(fft_size, hop_size, win_length, eps)
        self.num_ceps = num_ceps
        bins = fft_size // 2 + 1
        n = torch.arange(bins).float()
        k = torch.arange(num_ceps).float()[:, None]
        basis = torch.cos(math.pi / bins * (n + 0.5) * k)
        self.register_buffer("dct_basis", basis, persistent=False)

    def forward(self, estimate, target):
        est, tgt = self.spectra(estimate, target)
        basis = self.dct_basis.to(est.device, est.real.dtype)
        est_ceps = torch.einsum("kf,bft->bkt", basis, (est.abs() + self.eps).log())
        tgt_ceps = torch.einsum("kf,bft->bkt", basis, (tgt.abs() + self.eps).log())
        return F.l1_loss(est_ceps, tgt_ceps)


class BandWeightedSTFTLoss(_SingleResolutionLoss):
    def __init__(
        self,
        fft_size=1024,
        hop_size=320,
        win_length=1024,
        sample_rate=16000,
        bands=((0, 1000, 1.0), (1000, 4000, 2.0), (4000, 8000, 2.0)),
        eps=1e-7,
    ):
        super().__init__(fft_size, hop_size, win_length, eps)
        self.sample_rate, self.bands = sample_rate, bands

    def forward(self, estimate, target):
        est, tgt = self.spectra(estimate, target)
        est_log, tgt_log = (est.abs() + self.eps).log(), (tgt.abs() + self.eps).log()
        total = est.real.new_zeros(())
        weight_sum = 0.0
        for low, high, weight in self.bands:
            start = int(low * self.fft_size / self.sample_rate)
            end = min(est.shape[1], math.ceil(high * self.fft_size / self.sample_rate))
            if end > start:
                total = total + weight * F.l1_loss(
                    est_log[:, start:end], tgt_log[:, start:end]
                )
                weight_sum += weight
        return total / max(weight_sum, 1e-8)
