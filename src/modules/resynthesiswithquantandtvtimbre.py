# Adapted from: https://github.com/ashleve/lightning-hydra-template/blob/main/src/models/mnist_module.py
from typing import Any
from dataclasses import asdict

import pytorch_lightning as pl
import torch
from audiotools import AudioSignal
from torchmetrics import MinMetric, MeanMetric

from src.modules import PasrWithQuantModule
from src.models import MimiDecoderWithTVTimbreModel
from src.components import SEANetDecoder
from src.components.frame_synthesizer import FrameSynthesizerWOExciter
from src.components.discriminator import Discriminator
from src.components.losses import L1Loss, MultiScaleSTFTLoss, MelSpectrogramLoss, GANLoss, LossPED
from src.components.data_classes import SeanetArgs, TransformerArgs, FVQuantizerArgs, DecSeanetNCArgs, Lambdas, FrameSynthesizerArgs, TVTimbreArgs



class ResynthesisWithQuantAndTVTimbreModule(pl.LightningModule):
    """
    LightningModule for resynthesis with Vector Quantization content encoder and TV-Timbre decoder.
    Combines pretrained PASR encoder with a SEANet decoder enhanced with TV-Timbre features.
    Uses multiple loss functions including adversarial, STFT, mel-spectrogram, waveform, and perceptual losses.
    """

    def __init__(
        self,
        seanet_args: SeanetArgs = SeanetArgs(),
        transformer_args: TransformerArgs = TransformerArgs(),
        quantizer_args: FVQuantizerArgs = FVQuantizerArgs(),
        sample_rate: int = 16000,
        frame_rate: int = 50,
        num_codes: int = 200,
        encoder_dim: int = 512,
        decoder_seanet_args: DecSeanetNCArgs = DecSeanetNCArgs(),
        frame_synthesizer_args: FrameSynthesizerArgs = FrameSynthesizerArgs(),
        tvtimbre_args: TVTimbreArgs = TVTimbreArgs(),
        lambdas: Lambdas = Lambdas(),
        learning_rate: float = 0.0001,
        lr_betas: list = [0.8, 0.99],
        lr_sch_gamma: float = 0.999996,
        encoder_path: str = None,
        decoder_pretrained: bool = False,
        decoder_path: str = None,
    ):
        super().__init__()

        self.save_hyperparameters()

        # Multiple optimizers need manual optimization
        self.automatic_optimization = False

        ## Init pretrained encoder
        if encoder_path is None:
            self.encoder = PasrWithQuantModule(
                seanet_args=seanet_args,
                transformer_args=transformer_args,
                quantizer_args=quantizer_args,
                sample_rate=sample_rate,
                frame_rate=frame_rate,
                num_codes=num_codes,
                encoder_dim=encoder_dim,
            )
        else:
            self.encoder = PasrWithQuantModule.load_from_checkpoint(self.hparams.encoder_path, map_location=self.device)
            self.encoder.freeze()

        ## Init decoder
        decoder_transformer = FrameSynthesizerWOExciter(
            device=self.device, **asdict(frame_synthesizer_args), tvtimbre_args=asdict(tvtimbre_args)
        )
        decoder = SEANetDecoder(**asdict(decoder_seanet_args))

        self.decoder = MimiDecoderWithTVTimbreModel(
            decoder,
            frame_rate=self.hparams.frame_rate,
            sample_rate=self.hparams.sample_rate,
            channels=1,
            encoder_dim=self.hparams.encoder_dim,
            causal = True,
            decoder_transformer=decoder_transformer,
        )
        
        ## Discriminators
        self.discriminator = Discriminator()

        # Load pretrained decoder/discriminator
        if self.hparams.decoder_pretrained:
            model_state_dict = torch.load(self.hparams.decoder_path, map_location=self.device)["state_dict"]
            # Filter out the keys that belong to the decoder.
            decoder_state = {
                k.replace("decoder.", ""): v
                for k, v in model_state_dict.items()
                if k.startswith("decoder.")
            }
            self.decoder.load_state_dict(decoder_state, strict=False)

            # Filter out the keys that belong to the discriminator.
            discriminator_state = {
                k.replace("discriminator.", ""): v
                for k, v in model_state_dict.items()
                if k.startswith("discriminator.")
            }
            self.discriminator.load_state_dict(discriminator_state, strict=False)

        # loss function
        self.lambdas = asdict(lambdas)
        self.waveform_loss = L1Loss()
        self.stft_loss = MultiScaleSTFTLoss()
        self.mel_loss = MelSpectrogramLoss()
        self.gan_loss = GANLoss(self.discriminator)
        self.loss_ped = LossPED()

        # for averaging loss across batches
        loss_metric = MeanMetric()
        self.train_stft_loss = loss_metric.clone()
        self.train_mel_loss = loss_metric.clone()
        self.train_waveform_loss = loss_metric.clone()
        self.train_adv_gen_loss = loss_metric.clone()
        self.train_adv_feat_loss = loss_metric.clone()
        self.train_adv_d_loss = loss_metric.clone()

        self.val_stft_loss = loss_metric.clone()
        self.val_mel_loss = loss_metric.clone()
        self.val_waveform_loss = loss_metric.clone()

        self.test_loss = loss_metric.clone()

        # for tracking best so far validation accuracy
        self.val_loss_best = MinMetric()

    def configure_optimizers(self):
        opt_g = torch.optim.AdamW(self.decoder.parameters(), self.hparams.learning_rate, betas=self.hparams.lr_betas)
        opt_d = torch.optim.AdamW(self.discriminator.parameters(), self.hparams.learning_rate, betas=self.hparams.lr_betas)

        sch_g = torch.optim.lr_scheduler.ExponentialLR(opt_g, gamma=self.hparams.lr_sch_gamma)
        sch_d = torch.optim.lr_scheduler.ExponentialLR(opt_d, gamma=self.hparams.lr_sch_gamma)
        return [opt_g, opt_d], [sch_g, sch_d]


    def forward(self, x: torch.Tensor, spkr_embed: torch.Tensor, pitch_rate: float = 1.0, energy_rate: float = 1.0):
        _, z_embed, _ = self.encoder(x)
        audio = self.decoder.decode(z_embed, spkr_embed, pitch_rate=pitch_rate, energy_rate=energy_rate)
        return audio

    def on_train_start(self):
        self.val_loss_best.reset()

    def model_step(self, batch: Any):
        y, tokens, pitch, energy, embeds, _ = batch
        y = y.unsqueeze(1)

        content = self.get_features(y)
        
        # content = self.projector(content_logits.transpose(1, 2)).transpose(1, 2)
        y_g_hat, variances = self.decoder(content, embeds, pitch=pitch, energy=energy, tokens=tokens)
        avg_pitch, predict_pitch, avg_energy, predict_energy = variances

        assert y_g_hat.shape == y.shape, f"Mismatch in vocoder output shape - {y_g_hat.shape} != {y.shape}"

        recons = AudioSignal(y_g_hat, self.hparams.sample_rate)
        target = AudioSignal(y, self.hparams.sample_rate)

        return recons, target, (predict_pitch, avg_pitch, predict_energy, avg_energy), embeds

    @torch.no_grad()
    def get_features(self, x: torch.Tensor):
        logits, z_embed, extra_metrics = self.encoder(x)
        return z_embed


    def training_step(self, batch: Any, batch_idx: int):
        # Retrieve optimizers and lr schedulers
        opt_g, opt_d = self.optimizers()
        

        # Forward pass
        recons, signal, ped, spkr_embed = self.model_step(batch)
        batch_size = recons.shape[0]

        output = {}
        #### Discriminator Update ####
        opt_d.zero_grad()

        disc_loss = self.gan_loss.discriminator_loss(recons, signal)
        output["adv/d_loss"] = disc_loss
        
        # Manually backprop for discriminator
        self.manual_backward(disc_loss)
        opt_d.step()
        

        #### Generator Update ####
        opt_g.zero_grad()

        output["stft_loss"] = self.stft_loss(recons, signal)
        output["mel_loss"] = self.mel_loss(recons, signal)
        output["waveform_loss"] = self.waveform_loss(recons, signal)
        (
            output["adv_gen_loss"],
            output["adv_feat_loss"],
        ) = self.gan_loss.generator_loss(recons, signal)
        output["ped_loss"] = self.loss_ped(*ped)["total_loss"]
        gen_loss = sum([v * output[k] for k, v in self.lambdas.items() if k in output])
        output["loss"] = gen_loss
        
        # Manually backprop for generator
        self.manual_backward(gen_loss)
        opt_g.step()

        
        # update and log metrics
        self.train_stft_loss(output["stft_loss"])
        self.train_mel_loss(output["mel_loss"])
        self.train_waveform_loss(output["waveform_loss"])
        self.train_adv_gen_loss(output["adv_gen_loss"])
        self.train_adv_feat_loss(output["adv_feat_loss"])
        self.train_adv_d_loss(output["adv/d_loss"])


        self.log("train/stft/loss", self.train_stft_loss, on_step=False, on_epoch=True, batch_size=batch_size)
        self.log("train/mel/loss", output["mel_loss"], on_step=True, on_epoch=True, batch_size=batch_size)
        self.log("train/waveform/loss", self.train_waveform_loss, on_step=False, on_epoch=True, batch_size=batch_size)
        self.log("train/adv/gen_loss", self.train_adv_gen_loss, on_step=False, on_epoch=True, batch_size=batch_size)
        self.log("train/adv/feat_loss", self.train_adv_feat_loss, on_step=False, on_epoch=True, batch_size=batch_size)
        self.log("train/total_gen_loss", output["loss"], on_step=True, on_epoch=True, batch_size=batch_size)
        self.log("train/adv/d_loss", self.train_adv_d_loss, on_step=False, on_epoch=True, batch_size=batch_size)
        return output


    def on_train_epoch_end(self):
        sch_g, sch_d = self.lr_schedulers()
        sch_g.step()
        sch_d.step()


    def validation_step(self, batch: Any, batch_idx: int):
        recons, signal, _, _ = self.model_step(batch)
        batch_size = recons.shape[0]

        # update and log metrics
        stft_loss = self.stft_loss(recons, signal)
        mel_loss = self.mel_loss(recons, signal)
        waveform_loss = self.waveform_loss(recons, signal)
        loss = mel_loss

        self.val_stft_loss(stft_loss)
        self.val_mel_loss(mel_loss)
        self.val_waveform_loss(waveform_loss)

        self.log("val/stft/loss", self.val_stft_loss, on_step=False, on_epoch=True, batch_size=batch_size)
        self.log("val/mel/loss", self.val_mel_loss, on_step=False, on_epoch=True, batch_size=batch_size)
        self.log("val/waveform/loss", self.val_waveform_loss, on_step=False, on_epoch=True, batch_size=batch_size)

        return {
            "loss": loss,
            "stft/loss": stft_loss,
            "mel/loss": mel_loss,
            "waveform/loss": waveform_loss
        }

    def on_validation_epoch_end(self):
        mel_loss = self.val_mel_loss.compute()  # get current val acc
        self.val_loss_best(mel_loss)  # update best so far val acc
        self.log("val/loss_best", self.val_loss_best.compute())

    def test_step(self, batch: Any, batch_idx: int):
        recons, signal, _, _ = self.model_step(batch)

        loss = self.mel_loss(recons, signal)
        # update and log metrics
        self.test_loss(loss)
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True)
        return {"loss": loss}

    def on_test_epoch_end(self):
        pass


if __name__ == "__main__":
    m = ResynthesisWithQuantAndTVTimbreModule()
    print(m)