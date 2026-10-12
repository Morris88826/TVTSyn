"""Supervised speaker-independent parallel waveform-to-waveform training."""

from dataclasses import asdict
from typing import Any

from audiotools import AudioSignal
import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import WandbLogger
from torchmetrics import MeanMetric, MinMetric

from tvtsyn.components import create_encoder_frontend, create_waveform_decoder, transformer
from tvtsyn.components.data_classes import (
    DecSeanetNCArgs,
    EncoderFrontendArgs,
    HuBERTLossArgs,
    SeanetArgs,
    SupervisedWav2WavLambdas,
    TransformerArgs,
    WaveformDecoderArgs,
)
from training.components.discriminator import Discriminator
from training.components.losses import GANLoss, HuBERTLoss, MelSpectrogramLoss
from tvtsyn.models import MimiWav2WavModel


class SupervisedWav2WavModule(pl.LightningModule):
    """Train an end-to-end wav2wav generator on aligned parallel audio."""

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
    ):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False

        encoder = create_encoder_frontend(
            encoder_frontend_type,
            seanet_args=seanet_args,
            frontend_args=encoder_frontend_args,
        )
        frame_transformer = transformer.ProjectedTransformer(
            device=self.device, **asdict(transformer_args)
        )
        decoder = create_waveform_decoder(
            decoder_type=decoder_type,
            seanet_args=decoder_seanet_args,
            decoder_args=waveform_decoder_args,
        )
        self.generator = MimiWav2WavModel(
            encoder=encoder,
            transformer=frame_transformer,
            decoder=decoder,
            sample_rate=sample_rate,
            frame_rate=frame_rate,
            channels=1,
        )

        discriminator_kwargs = dict(discriminator_args or {})
        discriminator_kwargs.setdefault("sample_rate", sample_rate)
        self.discriminator = Discriminator(**discriminator_kwargs)
        self.mel_loss = MelSpectrogramLoss()
        self.gan_loss = GANLoss(self.discriminator)
        self.loss_weights = asdict(lambdas)
        self.hubert_loss = (
            HuBERTLoss(**asdict(hubert_loss_args))
            if self.loss_weights["hubert_loss"] > 0
            else None
        )

        metric = MeanMetric()
        self.train_mel_loss = metric.clone()
        self.train_adv_gen_loss = metric.clone()
        self.train_adv_feat_loss = metric.clone()
        self.train_hubert_loss = metric.clone()
        self.train_adv_d_loss = metric.clone()
        self.train_total_loss = metric.clone()
        self.val_mel_loss = metric.clone()
        self.val_hubert_loss = metric.clone()
        self.test_mel_loss = metric.clone()
        self.test_hubert_loss = metric.clone()
        self.val_loss_best = MinMetric()
        self._validation_audio_logged = False

    def configure_optimizers(self):
        opt_g = torch.optim.AdamW(
            self.generator.parameters(),
            self.hparams.learning_rate,
            betas=tuple(self.hparams.lr_betas),
        )
        opt_d = torch.optim.AdamW(
            self.discriminator.parameters(),
            self.hparams.learning_rate,
            betas=tuple(self.hparams.lr_betas),
        )
        sch_g = torch.optim.lr_scheduler.ExponentialLR(
            opt_g, gamma=self.hparams.lr_sch_gamma
        )
        sch_d = torch.optim.lr_scheduler.ExponentialLR(
            opt_d, gamma=self.hparams.lr_sch_gamma
        )
        return [opt_g, opt_d], [sch_g, sch_d]

    @staticmethod
    def _prepare_waveform(waveform: torch.Tensor) -> torch.Tensor:
        if waveform.dim() == 2:
            waveform = waveform.unsqueeze(1)
        if waveform.dim() != 3 or waveform.shape[1] != 1:
            raise ValueError(f"Expected [B, T] or [B, 1, T], got {waveform.shape}")
        return waveform

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        return self.generator(self._prepare_waveform(waveform))

    def model_step(self, batch: Any):
        wav_x, wav_y, file_name = batch
        wav_x = self._prepare_waveform(wav_x)
        wav_y = self._prepare_waveform(wav_y)
        generated = self.generator(wav_x)
        length = min(generated.shape[-1], wav_y.shape[-1])
        generated = generated[..., :length]
        wav_y = wav_y[..., :length]
        return (
            AudioSignal(generated, self.hparams.sample_rate),
            AudioSignal(wav_y, self.hparams.sample_rate),
            wav_x[..., :length],
            file_name,
        )

    def generator_objective(self, reconstruction: AudioSignal, target: AudioSignal):
        mel_loss = self.mel_loss(reconstruction, target)
        adv_gen_loss, adv_feat_loss = self.gan_loss.generator_loss(
            reconstruction, target
        )
        hubert_loss = (
            self.hubert_loss(reconstruction, target)
            if self.hubert_loss is not None
            else mel_loss.new_zeros(())
        )
        total = (
            self.loss_weights["mel_loss"] * mel_loss
            + self.loss_weights["adv_gen_loss"] * adv_gen_loss
            + self.loss_weights["adv_feat_loss"] * adv_feat_loss
            + self.loss_weights["hubert_loss"] * hubert_loss
        )
        return total, mel_loss, adv_gen_loss, adv_feat_loss, hubert_loss

    def on_train_start(self):
        self.val_loss_best.reset()

    def training_step(self, batch: Any, batch_idx: int):
        opt_g, opt_d = self.optimizers()
        reconstruction, target, _, _ = self.model_step(batch)
        batch_size = reconstruction.audio_data.shape[0]

        self.toggle_optimizer(opt_d)
        opt_d.zero_grad()
        discriminator_loss = self.gan_loss.discriminator_loss(reconstruction, target)
        self.manual_backward(discriminator_loss)
        opt_d.step()
        self.untoggle_optimizer(opt_d)

        self.toggle_optimizer(opt_g)
        opt_g.zero_grad()
        total, mel_loss, adv_gen_loss, adv_feat_loss, hubert_loss = (
            self.generator_objective(reconstruction, target)
        )
        self.manual_backward(total)
        opt_g.step()
        self.untoggle_optimizer(opt_g)

        self.train_mel_loss(mel_loss)
        self.train_adv_gen_loss(adv_gen_loss)
        self.train_adv_feat_loss(adv_feat_loss)
        self.train_hubert_loss(hubert_loss)
        self.train_adv_d_loss(discriminator_loss)
        self.train_total_loss(total)
        self.log("train/mel/loss", self.train_mel_loss, on_epoch=True, batch_size=batch_size, sync_dist=True)
        self.log("train/adv/gen_loss", self.train_adv_gen_loss, on_epoch=True, batch_size=batch_size, sync_dist=True)
        self.log("train/adv/feat_loss", self.train_adv_feat_loss, on_epoch=True, batch_size=batch_size, sync_dist=True)
        self.log("train/hubert/loss", self.train_hubert_loss, on_epoch=True, batch_size=batch_size, sync_dist=True)
        self.log("train/adv/d_loss", self.train_adv_d_loss, on_epoch=True, batch_size=batch_size, sync_dist=True)
        self.log("train/total_gen_loss", self.train_total_loss, on_step=True, on_epoch=True, batch_size=batch_size, sync_dist=True)
        self.log("train/lr", opt_g.param_groups[0]["lr"], on_step=True, on_epoch=False, sync_dist=True)
        return {"loss": total, "discriminator_loss": discriminator_loss}

    def on_train_epoch_end(self):
        sch_g, sch_d = self.lr_schedulers()
        sch_g.step()
        sch_d.step()

    def on_validation_epoch_start(self):
        self._validation_audio_logged = False

    def _log_validation_audio(
        self,
        source: torch.Tensor,
        target: torch.Tensor,
        generated: torch.Tensor,
        file_names: Any,
    ) -> None:
        if self._validation_audio_logged or self.hparams.validation_audio_samples <= 0:
            return
        trainer = getattr(self, "_trainer", None)
        if trainer is None or trainer.sanity_checking or not trainer.is_global_zero:
            return
        logger = trainer.logger
        if not isinstance(logger, WandbLogger):
            return

        import wandb

        count = min(self.hparams.validation_audio_samples, source.shape[0])
        payload = {}
        for index in range(count):
            name = str(file_names[index]) if file_names is not None else str(index)
            caption = name.rsplit("/", 1)[-1]
            for label, audio in (
                ("source", source[index]),
                ("target", target[index]),
                ("generated", generated[index]),
            ):
                samples = audio.detach().float().cpu().squeeze().clamp(-1, 1).numpy()
                payload[f"validation/audio_{index}/{label}"] = wandb.Audio(
                    np.asarray(samples),
                    sample_rate=self.hparams.sample_rate,
                    caption=f"{caption} {label}",
                )
        logger.experiment.log(payload, step=self.global_step)
        self._validation_audio_logged = True

    def validation_step(self, batch: Any, batch_idx: int):
        reconstruction, target, source, file_names = self.model_step(batch)
        mel_loss = self.mel_loss(reconstruction, target)
        hubert_loss = (
            self.hubert_loss(reconstruction, target)
            if self.hubert_loss is not None
            else mel_loss.new_zeros(())
        )
        batch_size = reconstruction.audio_data.shape[0]
        self.val_mel_loss(mel_loss)
        self.val_hubert_loss(hubert_loss)
        self.log("val/mel/loss", self.val_mel_loss, on_epoch=True, batch_size=batch_size, sync_dist=True)
        self.log("val/hubert/loss", self.val_hubert_loss, on_epoch=True, batch_size=batch_size, sync_dist=True)
        if batch_idx == 0:
            self._log_validation_audio(
                source,
                target.audio_data,
                reconstruction.audio_data,
                file_names,
            )
        return {"loss": mel_loss, "mel/loss": mel_loss, "hubert/loss": hubert_loss}

    def on_validation_epoch_end(self):
        self.val_loss_best(self.val_mel_loss.compute())
        self.log("val/loss_best", self.val_loss_best.compute(), sync_dist=True)

    def test_step(self, batch: Any, batch_idx: int):
        reconstruction, target, _, _ = self.model_step(batch)
        mel_loss = self.mel_loss(reconstruction, target)
        hubert_loss = (
            self.hubert_loss(reconstruction, target)
            if self.hubert_loss is not None
            else mel_loss.new_zeros(())
        )
        self.test_mel_loss(mel_loss)
        self.test_hubert_loss(hubert_loss)
        self.log("test/mel/loss", self.test_mel_loss, on_epoch=True, sync_dist=True)
        self.log("test/hubert/loss", self.test_hubert_loss, on_epoch=True, sync_dist=True)
        return {"loss": mel_loss, "hubert/loss": hubert_loss}
