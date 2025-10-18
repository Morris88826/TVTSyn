import torch
import torch.nn.functional as F
from torch.nn.utils import weight_norm, remove_weight_norm

from src.components.duration_predictor import DurationPredictor
from src.components.causal_generator import Generator
from src.components.causal_conv import CausalConv
from src.components.speaker_adapter import SpeakerAdapter

from src.components.streaming import StreamingContainer, StreamingAdd
from src.utils.compile import torch_compile_lazy


LRELU_SLOPE = 0.1

class HiFiDecoder(Generator):
    def __init__(self, h):
        super().__init__(h)
        
        self.h = h
        
        self.spk_adapter = SpeakerAdapter(h.spkr_embed_dim, h.upsample_initial_channel)
        
        self.pitchPred = DurationPredictor(
            in_channels=h.model_in_dim,
            out_channels=h.model_in_dim,
            kernel_size=h.dur_pred_kernel_size,
            dropout=h.variance_predictor_dropout,
        )
        self.energyPred = DurationPredictor(
            in_channels=h.model_in_dim,
            out_channels=h.model_in_dim,
            kernel_size=h.dur_pred_kernel_size,
            dropout=h.variance_predictor_dropout,
        )
        self.pitchEmbed = CausalConv(
            in_channel=1,
            out_channel=h.model_in_dim,
            kernel_size=h.pitch_pred_kernel_size,
            stride=1
        )

        self.energyEmbed = CausalConv(
            in_channel=1,
            out_channel=h.model_in_dim,
            kernel_size=h.energy_pred_kernel_size,
            stride=1
        )

        self.pitch_add = StreamingAdd()
        self.energy_add = StreamingAdd()

    def _average_over_durations(self, values, durs):
        """Average values over durations.
        Arguments
        ---------
        values: torch.Tensor
            shape: [B, 1, T_de]
        durs: torch.Tensor
            shape: [B, T_en]
        Returns
        ---------
        avg: torch.Tensor
            shape: [B, 1, T_en]
        """
        avg_batch = []
        for i, (d, v) in enumerate(zip(durs, values)):
            v = v.unsqueeze(0)
            durs_cums_ends = torch.cumsum(d.unsqueeze(0), dim=1).long()
            durs_cums_starts = torch.nn.functional.pad(durs_cums_ends[:, :-1], (1, 0))
            values_nonzero_cums = torch.nn.functional.pad(
                torch.cumsum(v != 0.0, dim=2), (1, 0)
            )
            values_cums = torch.nn.functional.pad(torch.cumsum(v, dim=2), (1, 0))

            bs, length = durs_cums_ends.size()
            n_formants = v.size(1)
            dcs = durs_cums_starts[:, None, :].expand(bs, n_formants, length)
            dce = durs_cums_ends[:, None, :].expand(bs, n_formants, length)

            values_sums = (
                torch.gather(values_cums, 2, dce) - torch.gather(values_cums, 2, dcs)
            ).float()
            values_nelems = (
                torch.gather(values_nonzero_cums, 2, dce)
                - torch.gather(values_nonzero_cums, 2, dcs)
            ).float()

            avg = torch.where(
                values_nelems == 0.0, values_nelems, values_sums / values_nelems
            )
            avg_batch.append(avg.squeeze(0).repeat_interleave(d, dim=-1))
        return torch.stack(avg_batch, dim=0)


    def _get_durations(self, tokens):
        durations = [
            torch.unique_consecutive(tokens[i], return_counts=True)[1]
            for i in range(len(tokens))
        ]
        return durations

    # @torch_compile_lazy
    def forward(self, x, embeds, pitch=None, energy=None, code=None, inference=False):
        # Perturbation
        # x = x + torch.randn_like(x)
        durations = None
        if not inference:
            durations = self._get_durations(code)

        # Add speaker info
        token_feats = self.spk_adapter(x, embeds)


        # pitch predictor
        avg_pitch = None
        predict_pitch = self.pitchPred(token_feats)
        # use a pitch rate to adjust the pitch
        predict_pitch = predict_pitch * self.h.pitch_rate

        if pitch is not None:
            avg_pitch = self._average_over_durations(pitch.unsqueeze(1), durations)
            pitch = self.pitchEmbed(avg_pitch)
            #avg_pitch = avg_pitch.permute(0, 2, 1)
        else:
            pitch = self.pitchEmbed(predict_pitch) #.permute(0, 2, 1))
        #pitch = pitch.permute(0, 2, 1)
        token_feats = self.pitch_add(token_feats, pitch)

        # energy predictor
        avg_energy = None
        predict_energy = self.energyPred(token_feats)
        # use an energy rate to adjust the energy
        predict_energy = predict_energy * self.h.energy_rate
        if energy is not None:
            avg_energy = self._average_over_durations(energy.unsqueeze(1), durations)
            energy = self.energyEmbed(avg_energy)
            # avg_energy = avg_energy.permute(0, 2, 1)
        else:
            energy = self.energyEmbed(predict_energy)#.permute(0, 2, 1))
        # energy = energy.permute(0, 2, 1)
        token_feats = self.energy_add(token_feats, energy)

        
        y = super().forward(token_feats, embeds)

        if inference:
            return y
        
        return y, predict_pitch, avg_pitch, predict_energy, avg_energy

    # @torch_compile_lazy
    def decode(self, x, embeds, pitch_rate=1.0, energy_rate=1.0):
        # Add speaker info
        token_feats = self.spk_adapter(x, embeds)

        # pitch predictor
        predict_pitch = self.pitchPred(token_feats)
        # use a pitch rate to adjust the pitch
        predict_pitch = predict_pitch * pitch_rate
        pitch = self.pitchEmbed(predict_pitch) #.permute(0, 2, 1))
        #pitch = pitch.permute(0, 2, 1)
        token_feats = self.pitch_add(token_feats, pitch)

        # energy predictor
        predict_energy = self.energyPred(token_feats)
        # use an energy rate to adjust the energy
        predict_energy = predict_energy * energy_rate
        energy = self.energyEmbed(predict_energy)#.permute(0, 2, 1))
        # energy = energy.permute(0, 2, 1)
        token_feats = self.energy_add(token_feats, energy)

        
        y = super().forward(token_feats, embeds)

        return y