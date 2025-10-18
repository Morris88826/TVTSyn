# Adapted from: https://github.com/ashleve/lightning-hydra-template/blob/main/src/models/mnist_module.py
from typing import Any
from dataclasses import asdict

import pytorch_lightning as pl
import torch
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import Accuracy

from src.components import SEANetEncoder, transformer
from src.models import MimiEncoderWithQuantModel
from src.components.code_predictor import CodePredictor
from src.components.quantizer.factorized_vector_quantize import FactorizedVectorQuantize
from src.components.data_classes import SeanetArgs, TransformerArgs, FVQuantizerArgs
# from src.components.losses import FocalLoss


class PasrWithQuantModule(pl.LightningModule):
    """
    LightningModule for streaming pseudo ASR (PASR) with Vector Quantization.
    Combines SEANet encoder, transformer, and factorized vector quantizer.
    Predicts discrete codes using a code predictor and computes cross-entropy loss.
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
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters()

        encoder = SEANetEncoder(**asdict(seanet_args))
        encoder_transformer = transformer.ProjectedTransformer(
            device=self.device, **asdict(transformer_args)
        )
        quantizer = FactorizedVectorQuantize(**asdict(quantizer_args))
        self.net = MimiEncoderWithQuantModel(
            encoder,
            quantizer,
            channels=1,
            sample_rate=sample_rate,
            frame_rate=frame_rate,
            causal=True,
            encoder_transformer=encoder_transformer,
        )
        self.code_predictor = CodePredictor(
            embed_dim=self.hparams.encoder_dim,
            num_codes=self.hparams.num_codes,
            apply_ln=False,
        )

        # loss function
        self.criterion = torch.nn.CrossEntropyLoss() #FocalLoss() #

        # metric objects for calculating and averaging accuracy across batches
        metric = Accuracy(task="multiclass", num_classes=self.hparams.num_codes)
        self.train_acc = metric.clone()
        self.val_acc = metric.clone()
        self.test_acc = metric.clone()

        # for averaging loss across batches
        loss_metric = MeanMetric()
        self.train_loss = loss_metric.clone()
        self.commit_loss = loss_metric.clone()
        self.codebook_loss = loss_metric.clone()
        self.val_loss = loss_metric.clone()
        self.test_loss = loss_metric.clone()

        # for tracking best so far validation accuracy
        self.val_acc_best = MaxMetric()

    def forward(self, x: torch.Tensor):
        z_embed, extra_metrics = self.net(x)
        logits = self.code_predictor(z_embed)
        return logits, z_embed, extra_metrics

    def on_train_start(self):
        self.val_acc_best.reset()

    def model_step(self, batch: Any):
        x, y, _ = batch
        x = x.unsqueeze(1)

        logits, _, extra_metrics = self.forward(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        return loss, extra_metrics, preds, y

    def training_step(self, batch: Any, batch_idx: int):
        ce_loss, extra_metrics, preds, targets = self.model_step(batch)

        loss = ce_loss + extra_metrics['commit_loss'] + extra_metrics['codebook_loss']

        # update and log metrics
        self.train_loss(ce_loss)
        self.commit_loss(extra_metrics['commit_loss'])
        self.codebook_loss(extra_metrics['codebook_loss'])
        self.train_acc(preds, targets)
        self.log("train/loss", self.train_loss, on_step=True, on_epoch=True)
        self.log("train/commit_loss", self.commit_loss, on_step=True, on_epoch=True)
        self.log("train/codebook_loss", self.codebook_loss, on_step=True, on_epoch=True)
        self.log("train/acc", self.train_acc, on_step=True, on_epoch=True)

        return {"loss": loss, "preds": preds, "targets": targets}

    def on_train_epoch_end(self):
        pass

    def validation_step(self, batch: Any, batch_idx: int):
        loss, _, preds, targets = self.model_step(batch)

        # update and log metrics
        self.val_loss(loss)
        self.val_acc(preds, targets)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True)
        self.log("val/acc", self.val_acc, on_step=False, on_epoch=True)

        return {"loss": loss, "preds": preds, "targets": targets}

    def on_validation_epoch_end(self):
        acc = self.val_acc.compute()  # get current val acc
        self.val_acc_best(acc)  # update best so far val acc
        self.log("val/acc_best", self.val_acc_best.compute(), sync_dist=True)

    def test_step(self, batch: Any, batch_idx: int):
        loss, _, preds, targets = self.model_step(batch)

        # update and log metrics
        self.test_loss(loss)
        self.test_acc(preds, targets)
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True)
        self.log("test/acc", self.test_acc, on_step=False, on_epoch=True)

        return {"loss": loss, "preds": preds, "targets": targets}

    def on_test_epoch_end(self):
        pass


if __name__ == "__main__":
    m = PasrWithQuantModule()
    print(m)
