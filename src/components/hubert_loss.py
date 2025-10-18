import torch
import torch.nn as nn
import torch.nn.functional as F
import fairseq
from typing import Dict, Iterable, Tuple, Optional


class HuBERTPerceptualLoss(nn.Module):
    """
    Multi-layer HuBERT perceptual loss.
    - Freezes HuBERT weights (eval mode), but keeps grad through inputs.
    - Compares hidden states at specified layers with given weights.
    - Optionally cosine + L1/MSE mixture and per-channel norm.
    """
    def __init__(
        self,
        ckpt_path: str,
        device: torch.device,
        layer_weights: Dict[int, float] = (None),
        loss_type: str = "l1",                 # 'l1' or 'mse'
        add_cosine: bool = False,              # also include cosine distance term
        cosine_weight: float = 0.25,
        per_channel_norm: bool = False,        # normalize features before loss
        max_chunk_samples: int = 1600000,      # chunk long waveforms to avoid OOM
        expect_sr: int = 16000,                # HuBERT’s SR; resample upstream if needed
    ):
        super().__init__()
        if layer_weights is None:
            # sensible default for base; you’ll pass your dict below anyway
            layer_weights = {3: 0.1, 6: 0.2, 9: 0.5, 12: 0.2}
        self.layer_weights = dict(layer_weights)
        self.layers = tuple(sorted(self.layer_weights.keys()))
        self.loss_type = loss_type
        self.add_cosine = add_cosine
        self.cosine_weight = cosine_weight
        self.per_channel_norm = per_channel_norm
        self.max_chunk = max_chunk_samples
        self.expect_sr = expect_sr

        models, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([ckpt_path])
        self.hubert = models[0].to(device).eval()
        # Freeze params (no updates), but keep autograd through input
        for p in self.hubert.parameters():
            p.requires_grad_(False)
        self.task = task
        # honor dataset normalization convention
        self.do_ln = getattr(task.cfg, "normalize", True)

    @torch.no_grad()
    def _prep(self, wav: torch.Tensor) -> torch.Tensor:
        """(no grad) Per-utterance layer-norm if task.cfg.normalize==True. Expects mono (B,T)."""
        x = wav
        if self.do_ln:
            x = F.layer_norm(x, x.shape[-1:])
        return x

    def _extract_features_layer(self, x: torch.Tensor, layer: int) -> torch.Tensor:
        """
        Extract HuBERT features for a single layer, chunking waveform if needed.
        x: (B, T) on same device as model
        returns: (B, T_frames, C)
        """
        B, T = x.shape
        outs = []
        for start in range(0, T, self.max_chunk):
            x_chunk = x[:, start:start+self.max_chunk]
            feats, _ = self.hubert.extract_features(
                source=x_chunk, padding_mask=None, mask=False, output_layer=layer
            )
            outs.append(feats)
        return torch.cat(outs, dim=1)

    def _maybe_norm_feats(self, f: torch.Tensor) -> torch.Tensor:
        if not self.per_channel_norm:
            return f
        # channel-wise LN over time: (B, T, C) -> normalize along C
        mu = f.mean(dim=-1, keepdim=True)
        std = f.std(dim=-1, keepdim=True).clamp_min(1e-6)
        return (f - mu) / std

    def forward(
        self,
        y_hat: torch.Tensor,  # (B, 1, T) or (B, T) @ 16k mono in [-1,1]
        y: torch.Tensor,      # (B, 1, T) or (B, T)
    ) -> torch.Tensor:
        # ensure (B, T)
        if y_hat.dim() == 3: y_hat = y_hat.squeeze(1)
        if y.dim() == 3:     y = y.squeeze(1)

        # IMPORTANT: y branch no grad (GT), y_hat branch needs grad for TTS
        with torch.no_grad():
            y_prep = self._prep(y.to(next(self.hubert.parameters()).device).float())

        yhat_prep = self._prep(y_hat.to(next(self.hubert.parameters()).device).float())

        total = 0.0
        wsum = 0.0
        for L, w in self.layer_weights.items():
            # GT features (no grad)
            with torch.no_grad():
                f_gt = self._extract_features_layer(y_prep, L)     # (B, T', C)
                f_gt = self._maybe_norm_feats(f_gt)

            # Y_hat features (grad flows to y_hat)
            f_out = self._extract_features_layer(yhat_prep, L)     # (B, T', C)
            f_out = self._maybe_norm_feats(f_out)

            # align time (T' could differ by 1 frame)
            Tm = min(f_gt.size(1), f_out.size(1))
            f_gt  = f_gt[:, :Tm]
            f_out = f_out[:, :Tm]

            if self.loss_type == "l1":
                L_rec = F.l1_loss(f_out, f_gt)
            else:
                L_rec = F.mse_loss(f_out, f_gt)

            if self.add_cosine:
                # cosine distance (1 - cos) averaged over time then batch
                cos = 1.0 - F.cosine_similarity(f_out, f_gt, dim=-1).mean()
                L_rec = L_rec + self.cosine_weight * cos

            total = total + w * L_rec
            wsum = wsum + w

        return total / (wsum + 1e-8)
