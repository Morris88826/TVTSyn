# Adapted from: https://github.com/ashleve/lightning-hydra-template/blob/main/src/models/mnist_module.py
from typing import Any

import pytorch_lightning as pl
import torch
import torch.nn as nn
from speechbrain.inference import EncoderClassifier


class SpeakerModule(pl.LightningModule):
    """Example of LightningModule for MNIST classification.
    A LightningModule organizes your PyTorch code into 6 sections:
        - Computations (init)
        - Train loop (training_step)
        - Validation loop (validation_step)
        - Test loop (test_step)
        - Prediction Loop (predict_step)
        - Optimizers and LR Schedulers (configure_optimizers)
    Docs:
        https://lightning.ai/docs/pytorch/latest/common/lightning_module.html
    """

    def __init__(
        self,
        embed_type="xvector+ecapa",
        device="cuda",
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters()

        self.speaker_encoders = nn.ModuleList()
        if "xvector" in embed_type:
            _xvec_model = EncoderClassifier.from_hparams(
                source='speechbrain/spkrec-xvect-voxceleb',
                savedir='pretrained_models/speaker/spkrec-xvect-voxceleb',
                run_opts={'device': device}
            )
            self.speaker_encoders.append(_xvec_model)
        if "ecapa" in embed_type:
            _tdnn_model = EncoderClassifier.from_hparams(
                source='speechbrain/spkrec-ecapa-voxceleb',
                savedir='pretrained_models/speaker/spkrec-ecapa-voxceleb',
                run_opts={'device': device}
            )
            self.speaker_encoders.append(_tdnn_model)

    def forward(self, x: torch.Tensor):
        embeddings = []
        for encoder in self.speaker_encoders:
            embeddings.append(encoder.encode_batch(x))
        
        # for i, emb in enumerate(embeddings):
        #     print(f"Embedding {i}: {emb.shape}")
        embeddings = torch.cat(embeddings, dim=-1).squeeze(1)
        return embeddings

    def on_train_start(self):
        # by default lightning executes validation step sanity checks before training starts,
        # so we need to make sure val_acc_best doesn't store accuracy from these checks
        pass

    def model_step(self, batch: Any):
        pass

    def training_step(self, batch: Any, batch_idx: int):
        pass

    def on_train_epoch_end(self):
        # `outputs` is a list of dicts returned from `training_step()`

        # Warning: when overriding `training_epoch_end()`, lightning accumulates outputs from all batches of the epoch
        # this may not be an issue when training on mnist
        # but on larger datasets/models it's easy to run into out-of-memory errors

        # consider detaching tensors before returning them from `training_step()`
        # or using `on_train_epoch_end()` instead which doesn't accumulate outputs
        pass

    def validation_step(self, batch: Any, batch_idx: int):
        pass

    def on_validation_epoch_end(self):
        pass

    def test_step(self, batch: Any, batch_idx: int):
        pass

    def on_test_epoch_end(self):
        pass


if __name__ == "__main__":
    m = SpeakerModule()
    print(m)
