# Adapted from: https://github.com/ashleve/lightning-hydra-template/blob/main/src/models/mnist_module.py
from typing import Any, Optional
from dataclasses import asdict

import pytorch_lightning as pl
import torch
from audiotools import AudioSignal
from torchmetrics import MinMetric, MeanMetric

from src.modules import PasrWithQuantModule
from src.models import MimiDecoderWithTVTimbreModel
from src.components import create_waveform_decoder
from src.components.frame_synthesizer import FrameSynthesizerWOExciter
from src.components.discriminator import Discriminator
from src.components.losses import L1Loss, MultiScaleSTFTLoss, MelSpectrogramLoss, GANLoss, LossPED
from src.components.data_classes import SeanetArgs, TransformerArgs, FVQuantizerArgs, DecSeanetNCArgs, Lambdas, FrameSynthesizerArgs, TVTimbreArgs, WaveformDecoderArgs



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
        decoder_type: str = "seanet",
        decoder_seanet_args: DecSeanetNCArgs = DecSeanetNCArgs(),
        waveform_decoder_args: WaveformDecoderArgs = WaveformDecoderArgs(),
        frame_synthesizer_args: FrameSynthesizerArgs = FrameSynthesizerArgs(),
        tvtimbre_args: TVTimbreArgs = TVTimbreArgs(),
        use_pitch_energy_predictor: bool = True,
        lambdas: Lambdas = Lambdas(),
        add_spec_augment_content: bool = False,
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

        self.use_pitch_energy_predictor = use_pitch_energy_predictor
        ## Init decoder
        decoder_transformer = FrameSynthesizerWOExciter(
            device=self.device, 
            **asdict(frame_synthesizer_args), 
            tvtimbre_args=asdict(tvtimbre_args),
            use_pitch_energy_predictor=use_pitch_energy_predictor,
        )
        decoder = create_waveform_decoder(
            decoder_type=decoder_type,
            seanet_args=decoder_seanet_args,
            decoder_args=waveform_decoder_args,
        )

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
            print(f"Loaded pretrained decoder and discriminator from {self.hparams.decoder_path}")

        self.add_spec_augment_content = add_spec_augment_content

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
    
    def spec_augment_content(
        self,
        content: torch.Tensor,
        *,
        time_mask_prob: float = 0.5,
        feature_mask_prob: float = 0.5,
        num_time_masks: int = 2,
        num_feature_masks: int = 2,
        max_time_mask_frac: float = 0.04,
        max_feature_mask_frac: float = 0.04,
        replace_with: str = "mean",
        p: float = 0.5,
    ) -> torch.Tensor:
        """
        SpecAugment-style masking for continuous content features.

        Args:
            content:
                Tensor of shape [B, C, T].
            time_mask_prob:
                Probability of applying each time mask per sample.
            feature_mask_prob:
                Probability of applying each feature/channel mask per sample.
            num_time_masks:
                Number of time masks attempted per sample.
            num_feature_masks:
                Number of feature/channel masks attempted per sample.
            max_time_mask_frac:
                Maximum width of each time mask as a fraction of T.
                Example: 0.08 means each mask can cover up to 8% of frames.
            max_feature_mask_frac:
                Maximum width of each feature mask as a fraction of C.
            replace_with:
                "zero"  -> replace masked region with 0.
                "mean"  -> replace masked region with per-sample content mean.
                "noise" -> replace masked region with small Gaussian noise around mean.
            p:
                Global probability of applying augmentation.

        Returns:
            Augmented content tensor of shape [B, C, T].
        """

        if content.ndim != 3:
            raise ValueError(f"Expected content shape [B, C, T], got {content.shape}")

        if p <= 0.0 or torch.rand((), device=content.device).item() > p:
            return content

        B, C, T = content.shape
        augmented = content.clone()

        max_time_width = max(1, int(T * max_time_mask_frac))
        max_feature_width = max(1, int(C * max_feature_mask_frac))

        # Per-sample mean/std used for stable replacement.
        content_mean = content.detach().mean(dim=(1, 2), keepdim=True)
        content_std = content.detach().std(dim=(1, 2), keepdim=True).clamp_min(1e-5)

        def replacement_value(b: int, shape: tuple[int, ...]) -> torch.Tensor:
            if replace_with == "zero":
                return torch.zeros(shape, device=content.device, dtype=content.dtype)
            elif replace_with == "mean":
                return content_mean[b].expand(shape)
            elif replace_with == "noise":
                return content_mean[b].expand(shape) + 0.01 * content_std[b].expand(shape) * torch.randn(
                    shape, device=content.device, dtype=content.dtype
                )
            else:
                raise ValueError(f"Unsupported replace_with={replace_with}. Use 'zero', 'mean', or 'noise'.")

        for b in range(B):
            # Time masking: mask consecutive frames across all channels.
            for _ in range(num_time_masks):
                if torch.rand((), device=content.device).item() > time_mask_prob:
                    continue

                width = int(torch.randint(1, max_time_width + 1, (), device=content.device).item())
                if width >= T:
                    start = 0
                    width = T
                else:
                    start = int(torch.randint(0, T - width + 1, (), device=content.device).item())

                augmented[b, :, start : start + width] = replacement_value(b, (C, width))

            # Feature masking: mask consecutive channels across all time frames.
            for _ in range(num_feature_masks):
                if torch.rand((), device=content.device).item() > feature_mask_prob:
                    continue

                width = int(torch.randint(1, max_feature_width + 1, (), device=content.device).item())
                if width >= C:
                    start = 0
                    width = C
                else:
                    start = int(torch.randint(0, C - width + 1, (), device=content.device).item())

                augmented[b, start : start + width, :] = replacement_value(b, (width, T))

        return augmented


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
        
        ## Add little noise to the content features for better generalization (similar to VQ-VAE2 paper)
        # noise = torch.randn_like(content) * 0.01
        # content = content + noise
        if self.training:
            content_std = content.detach().std(dim=(1, 2), keepdim=True).clamp_min(1e-5)
            noise = torch.randn_like(content) * content_std * 0.02
            content = content + noise
            
            if self.add_spec_augment_content:
                content = self.spec_augment_content(content)
        
        # if not self.use_pitch_energy_predictor:
        #     pitch = None
        #     energy = None
        #     tokens = None
        y_g_hat, variances = self.decoder(content.detach(), embeds, pitch=pitch, energy=energy, tokens=tokens)
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
        # output["ped_loss"] = torch.tensor(0.0, device=recons.device)
        # if self.use_pitch_energy_predictor:
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


        self.log("train/stft/loss", self.train_stft_loss, on_step=False, on_epoch=True, batch_size=batch_size, sync_dist=True)
        self.log("train/mel/loss", output["mel_loss"], on_step=True, on_epoch=True, batch_size=batch_size, sync_dist=True)
        self.log("train/waveform/loss", self.train_waveform_loss, on_step=False, on_epoch=True, batch_size=batch_size, sync_dist=True)
        self.log("train/adv/gen_loss", self.train_adv_gen_loss, on_step=False, on_epoch=True, batch_size=batch_size, sync_dist=True)
        self.log("train/adv/feat_loss", self.train_adv_feat_loss, on_step=False, on_epoch=True, batch_size=batch_size, sync_dist=True)
        self.log("train/total_gen_loss", output["loss"], on_step=True, on_epoch=True, batch_size=batch_size, sync_dist=True)
        self.log("train/adv/d_loss", self.train_adv_d_loss, on_step=False, on_epoch=True, batch_size=batch_size, sync_dist=True)
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

        self.log("val/stft/loss", self.val_stft_loss, on_step=False, on_epoch=True, batch_size=batch_size, sync_dist=True)
        self.log("val/mel/loss", self.val_mel_loss, on_step=False, on_epoch=True, batch_size=batch_size, sync_dist=True)
        self.log("val/waveform/loss", self.val_waveform_loss, on_step=False, on_epoch=True, batch_size=batch_size, sync_dist=True)

        return {
            "loss": loss,
            "stft/loss": stft_loss,
            "mel/loss": mel_loss,
            "waveform/loss": waveform_loss
        }

    def on_validation_epoch_end(self):
        mel_loss = self.val_mel_loss.compute()  # get current val acc
        self.val_loss_best(mel_loss)  # update best so far val acc
        self.log("val/loss_best", self.val_loss_best.compute(), sync_dist=True)  # log best so far val acc

    def test_step(self, batch: Any, batch_idx: int):
        recons, signal, _, _ = self.model_step(batch)

        loss = self.mel_loss(recons, signal)
        # update and log metrics
        self.test_loss(loss)
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, sync_dist=True)
        return {"loss": loss}

    def on_test_epoch_end(self):
        pass


if __name__ == "__main__":
    m = ResynthesisWithQuantAndTVTimbreModule()
    print(m)