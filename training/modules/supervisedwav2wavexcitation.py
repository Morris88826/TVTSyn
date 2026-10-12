"""Excitation-aware supervised parallel waveform training."""

from dataclasses import asdict
import warnings
from typing import Any

from audiotools import AudioSignal
import numpy as np
import torch
from torch.nn import functional as F
from torchmetrics import MeanMetric

from tvtsyn.components.data_classes import (
    DecSeanetNCArgs,
    EncoderFrontendArgs,
    HuBERTLossArgs,
    SeanetArgs,
    SupervisedWav2WavLambdas,
    TransformerArgs,
    WaveformDecoderArgs,
)
from tvtsyn.components.spectral_losses import (
    BandWeightedSTFTLoss,
    CepstralEnvelopeLoss,
    ComplexSTFTLoss,
    HarmonicMagnitudeLoss,
    MultiResolutionSTFTLoss,
    ProsodyLoss,
)
from tvtsyn.models import MimiWav2WavExcitationModel
from training.modules.supervisedwav2wav import SupervisedWav2WavModule


class SupervisedWav2WavExcitationModule(SupervisedWav2WavModule):
    def __init__(
        self,
        seanet_args: SeanetArgs = SeanetArgs(),
        encoder_frontend_type: str = "seanet",
        encoder_frontend_args: EncoderFrontendArgs = EncoderFrontendArgs(),
        transformer_args: TransformerArgs = TransformerArgs(),
        decoder_type: str = "seanet",
        decoder_seanet_args: DecSeanetNCArgs = DecSeanetNCArgs(),
        waveform_decoder_args: WaveformDecoderArgs = WaveformDecoderArgs(),
        sample_rate: int = 16000,
        frame_rate: int = 50,
        lambdas: SupervisedWav2WavLambdas = SupervisedWav2WavLambdas(),
        hubert_loss_args: HuBERTLossArgs = HuBERTLossArgs(),
        discriminator_args: dict | None = None,
        learning_rate: float = 1e-4,
        lr_betas: list[float] = [0.8, 0.99],
        lr_sch_gamma: float = 0.999996,
        validation_audio_samples: int = 4,
        use_prosody_conditioning: bool = True,
        use_target_prosody_teacher_forcing: bool = True,
        prosody_teacher_forcing_prob: float = 1.0,
        prosody_conditioning_mode: str = "add",
        prosody_hidden_dim: int = 256,
        prosody_num_layers: int = 2,
        prosody_kernel_size: int = 3,
        predict_coded_ap: bool = False,
        coded_ap_dim: int = 0,
        use_harmonic_source_conditioning: bool = True,
        harmonic_num_harmonics: int = 8,
        harmonic_noise_std: float = 0.003,
        harmonic_use_energy_scaling: bool = True,
        harmonic_conditioning_mode: str = "add",
        harmonic_source_n_fft: int = 512,
        harmonic_source_win_length: int = 512,
        mrstft_loss_args: dict | None = None,
        complex_stft_loss_args: dict | None = None,
        prosody_loss_args: dict | None = None,
        harmonic_loss_args: dict | None = None,
        cepstral_envelope_loss_args: dict | None = None,
        band_stft_loss_args: dict | None = None,
    ):
        super().__init__(
            seanet_args=seanet_args,
            encoder_frontend_type=encoder_frontend_type,
            encoder_frontend_args=encoder_frontend_args,
            transformer_args=transformer_args,
            decoder_type=decoder_type,
            decoder_seanet_args=decoder_seanet_args,
            waveform_decoder_args=waveform_decoder_args,
            sample_rate=sample_rate,
            frame_rate=frame_rate,
            lambdas=lambdas,
            hubert_loss_args=hubert_loss_args,
            discriminator_args=discriminator_args,
            learning_rate=learning_rate,
            lr_betas=lr_betas,
            lr_sch_gamma=lr_sch_gamma,
            validation_audio_samples=validation_audio_samples,
        )
        base = self.generator
        self.generator = MimiWav2WavExcitationModel(
            encoder=base.encoder,
            transformer=base.transformer,
            decoder=base.decoder,
            sample_rate=sample_rate,
            frame_rate=frame_rate,
            channels=1,
            use_prosody_conditioning=use_prosody_conditioning,
            prosody_hidden_dim=prosody_hidden_dim,
            prosody_num_layers=prosody_num_layers,
            prosody_kernel_size=prosody_kernel_size,
            prosody_conditioning_mode=prosody_conditioning_mode,
            predict_coded_ap=predict_coded_ap,
            coded_ap_dim=coded_ap_dim,
            use_harmonic_source_conditioning=use_harmonic_source_conditioning,
            harmonic_num_harmonics=harmonic_num_harmonics,
            harmonic_noise_std=harmonic_noise_std,
            harmonic_use_energy_scaling=harmonic_use_energy_scaling,
            harmonic_conditioning_mode=harmonic_conditioning_mode,
            harmonic_source_n_fft=harmonic_source_n_fft,
            harmonic_source_win_length=harmonic_source_win_length,
        )
        self.save_hyperparameters(
            {
                "use_target_prosody_teacher_forcing": use_target_prosody_teacher_forcing,
                "prosody_teacher_forcing_prob": prosody_teacher_forcing_prob,
            }
        )
        if not 0.0 <= prosody_teacher_forcing_prob <= 1.0:
            raise ValueError("prosody_teacher_forcing_prob must be in [0, 1]")
        self.mrstft_loss = MultiResolutionSTFTLoss(**(mrstft_loss_args or {}))
        self.complex_stft_loss = ComplexSTFTLoss(**(complex_stft_loss_args or {}))
        self.prosody_loss = ProsodyLoss(**(prosody_loss_args or {}))
        self.harmonic_loss = HarmonicMagnitudeLoss(**(harmonic_loss_args or {}))
        self.cepstral_loss = CepstralEnvelopeLoss(
            **(cepstral_envelope_loss_args or {})
        )
        self.band_stft_loss = BandWeightedSTFTLoss(**(band_stft_loss_args or {}))
        self._warned_missing_excitation = False
        metric = MeanMetric()
        for name in (
            "mrstft_logmag",
            "mrstft_sc",
            "complex_stft",
            "f0",
            "vuv",
            "energy",
            "coded_ap",
            "harmonic",
            "cepstral",
            "band_stft",
            "wav_l1",
        ):
            setattr(self, f"train_{name}_loss", metric.clone())

    @staticmethod
    def unpack_batch(batch):
        if len(batch) == 3:
            wav_x, wav_y, file_name = batch
            excitation = None
        elif len(batch) == 4:
            wav_x, wav_y, file_name, excitation = batch
        else:
            raise ValueError(f"Expected batch of length 3 or 4, got {len(batch)}")
        return wav_x, wav_y, file_name, excitation

    def _teacher_forcing(self, excitation) -> bool:
        return bool(
            self.training
            and excitation is not None
            and self.hparams.use_target_prosody_teacher_forcing
            and torch.rand((), device=self.device)
            < self.hparams.prosody_teacher_forcing_prob
        )

    def model_step(self, batch: Any, use_target_prosody: bool | None = None):
        wav_x, wav_y, file_name, excitation = self.unpack_batch(batch)
        wav_x, wav_y = self._prepare_waveform(wav_x), self._prepare_waveform(wav_y)
        if use_target_prosody is None:
            use_target_prosody = self._teacher_forcing(excitation)
        generated, aux = self.generator(
            wav_x,
            excitation=excitation,
            use_target_prosody=use_target_prosody,
            return_aux=True,
        )
        length = min(generated.shape[-1], wav_y.shape[-1])
        return (
            AudioSignal(generated[..., :length], self.hparams.sample_rate),
            AudioSignal(wav_y[..., :length], self.hparams.sample_rate),
            wav_x[..., :length],
            file_name,
            excitation,
            aux,
        )

    def _zero(self, reference: torch.Tensor) -> torch.Tensor:
        return reference.new_zeros(())

    def generator_objective(self, reconstruction, target, excitation=None, aux=None):
        mel = self.mel_loss(reconstruction, target)
        adv_gen, adv_feat = self.gan_loss.generator_loss(reconstruction, target)
        hubert = (
            self.hubert_loss(reconstruction, target)
            if self.hubert_loss is not None
            else self._zero(mel)
        )
        losses = {
            "mel_loss": mel,
            "adv_gen_loss": adv_gen,
            "adv_feat_loss": adv_feat,
            "hubert_loss": hubert,
        }
        audio_est, audio_tgt = reconstruction.audio_data, target.audio_data
        if self.loss_weights["mrstft_logmag_loss"] or self.loss_weights["mrstft_sc_loss"]:
            losses["mrstft_logmag_loss"], losses["mrstft_sc_loss"] = self.mrstft_loss(
                audio_est, audio_tgt
            )
        if self.loss_weights["complex_stft_loss"]:
            losses["complex_stft_loss"] = self.complex_stft_loss(audio_est, audio_tgt)
        if self.loss_weights["cepstral_envelope_loss"]:
            losses["cepstral_envelope_loss"] = self.cepstral_loss(audio_est, audio_tgt)
        if self.loss_weights["band_stft_loss"]:
            losses["band_stft_loss"] = self.band_stft_loss(audio_est, audio_tgt)
        if self.loss_weights["wav_l1_loss"]:
            losses["wav_l1_loss"] = F.l1_loss(audio_est, audio_tgt)

        excitation_weights = (
            "f0_loss", "vuv_loss", "energy_loss", "coded_ap_loss", "harmonic_loss"
        )
        if excitation is not None and aux and "prosody_pred" in aux:
            prosody = self.prosody_loss(aux["prosody_pred"], excitation)
            losses.update({name: prosody[name] for name in excitation_weights[:-1]})
            losses["voiced_frames"] = prosody["voiced_frames"]
            if self.loss_weights["harmonic_loss"]:
                losses["harmonic_loss"] = self.harmonic_loss(
                    audio_est, audio_tgt, excitation
                )
        elif any(self.loss_weights[name] for name in excitation_weights):
            if not self._warned_missing_excitation:
                warnings.warn(
                    "Excitation-dependent loss weights are nonzero, but excitation "
                    "targets are unavailable; those losses are skipped."
                )
                self._warned_missing_excitation = True

        for name in self.loss_weights:
            losses.setdefault(name, self._zero(mel))
        total = sum(self.loss_weights[name] * losses[name] for name in self.loss_weights)
        losses["total_loss"] = total
        return losses

    def training_step(self, batch: Any, batch_idx: int):
        opt_g, opt_d = self.optimizers()
        reconstruction, target, _, _, excitation, aux = self.model_step(batch)
        batch_size = reconstruction.audio_data.shape[0]
        self.toggle_optimizer(opt_d)
        opt_d.zero_grad()
        fake_detached = AudioSignal(
            reconstruction.audio_data.detach(), self.hparams.sample_rate
        )
        discriminator_loss = self.gan_loss.discriminator_loss(fake_detached, target)
        self.manual_backward(discriminator_loss)
        opt_d.step()
        self.untoggle_optimizer(opt_d)

        self.toggle_optimizer(opt_g)
        opt_g.zero_grad()
        losses = self.generator_objective(reconstruction, target, excitation, aux)
        self.manual_backward(losses["total_loss"])
        opt_g.step()
        self.untoggle_optimizer(opt_g)
        self._log_losses("train", losses, batch_size)
        self.log(
            "train/adv/d_loss", discriminator_loss, on_epoch=True,
            batch_size=batch_size, sync_dist=True
        )
        self.log(
            "train/lr", opt_g.param_groups[0]["lr"], on_step=True,
            on_epoch=False, sync_dist=True
        )
        return {"loss": losses["total_loss"], "discriminator_loss": discriminator_loss}

    def _log_losses(self, prefix: str, losses: dict, batch_size: int):
        names = {
            "mel_loss": "mel/loss",
            "hubert_loss": "hubert/loss",
            "adv_gen_loss": "adv/gen_loss",
            "adv_feat_loss": "adv/feat_loss",
            "mrstft_logmag_loss": "mrstft/logmag_loss",
            "mrstft_sc_loss": "mrstft/sc_loss",
            "complex_stft_loss": "complex_stft/loss",
            "f0_loss": "prosody/f0_loss",
            "vuv_loss": "prosody/vuv_loss",
            "energy_loss": "prosody/energy_loss",
            "coded_ap_loss": "prosody/coded_ap_loss",
            "voiced_frames": "prosody/voiced_frames",
            "harmonic_loss": "harmonic/loss",
            "cepstral_envelope_loss": "envelope/cepstral_loss",
            "band_stft_loss": "band_stft/loss",
            "wav_l1_loss": "wav_l1/loss",
            "total_loss": "total_gen_loss",
        }
        for key, metric_name in names.items():
            if key in losses:
                self.log(
                    f"{prefix}/{metric_name}", losses[key],
                    on_step=prefix == "train" and key == "total_loss",
                    on_epoch=True, batch_size=batch_size, sync_dist=True,
                )

    def validation_step(self, batch: Any, batch_idx: int):
        predicted = self.model_step(batch, use_target_prosody=False)
        reconstruction, target, source, names, excitation, aux = predicted
        losses = self.generator_objective(reconstruction, target, excitation, aux)
        batch_size = reconstruction.audio_data.shape[0]
        self.val_mel_loss(losses["mel_loss"])
        self._log_losses("val", losses, batch_size)
        teacher_audio = None
        if excitation is not None:
            teacher_audio = self.model_step(batch, use_target_prosody=True)[0].audio_data
        if batch_idx == 0:
            self._log_excitation_audio(
                source,
                target.audio_data,
                reconstruction.audio_data,
                teacher_audio,
                aux.get("harmonic_source"),
                names,
            )
        return losses

    def test_step(self, batch: Any, batch_idx: int):
        reconstruction, target, _, _, excitation, aux = self.model_step(
            batch, use_target_prosody=False
        )
        losses = self.generator_objective(reconstruction, target, excitation, aux)
        batch_size = reconstruction.audio_data.shape[0]
        self.test_mel_loss(losses["mel_loss"])
        self.test_hubert_loss(losses["hubert_loss"])
        self._log_losses("test", losses, batch_size)
        return losses

    def _log_excitation_audio(
        self, source, target, predicted, teacher, harmonic, file_names
    ):
        if self._validation_audio_logged or self.hparams.validation_audio_samples <= 0:
            return
        trainer = getattr(self, "_trainer", None)
        if trainer is None or trainer.sanity_checking or not trainer.is_global_zero:
            return
        from pytorch_lightning.loggers import WandbLogger
        if not isinstance(trainer.logger, WandbLogger):
            return
        import wandb
        count = min(self.hparams.validation_audio_samples, source.shape[0])
        payload = {}
        entries = [("source", source), ("target", target), ("predicted", predicted)]
        if teacher is not None:
            entries.append(("teacher_forced", teacher))
        if harmonic is not None:
            entries.append(("harmonic_source", harmonic))
        for index in range(count):
            caption = str(file_names[index]).rsplit("/", 1)[-1]
            for label, audio in entries:
                samples = audio[index].detach().float().cpu().squeeze().clamp(-1, 1)
                payload[f"validation/audio_{index}/{label}"] = wandb.Audio(
                    np.asarray(samples), sample_rate=self.hparams.sample_rate,
                    caption=f"{caption} {label}",
                )
        trainer.logger.experiment.log(payload, step=self.global_step)
        self._validation_audio_logged = True
