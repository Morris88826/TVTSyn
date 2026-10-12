"""Inference network with an explicit, lossless mapping from training weights."""
from dataclasses import asdict

import torch
from torch import nn

from tvtsyn.components import create_encoder_frontend, create_waveform_decoder, transformer
from tvtsyn.components import data_classes as dc
from tvtsyn.components.frame_synthesizer import FrameSynthesizerWOExciter
from tvtsyn.components.quantizer.factorized_vector_quantize import FactorizedVectorQuantize
from tvtsyn.models.mimiencoderwithquant import MimiEncoderWithQuantModel
from tvtsyn.models.mimidecoderwithtvtimbre import MimiDecoderWithTVTimbreModel

ARCHITECTURE_KEYS = (
    'seanet_args', 'transformer_args', 'quantizer_args', 'sample_rate', 'frame_rate',
    'encoder_dim', 'decoder_type', 'decoder_seanet_args', 'waveform_decoder_args',
    'frame_synthesizer_args', 'tvtimbre_args', 'use_pitch_energy_predictor',
)


def architecture_from_hparams(hparams):
    return {key: asdict(value) if hasattr(value, '__dataclass_fields__') else value
            for key, value in hparams.items() if key in ARCHITECTURE_KEYS}


class TVTSynModel(nn.Module):
    def __init__(self, architecture):
        super().__init__()
        self.architecture = dict(architecture)
        def args(key, cls):
            return cls(**architecture.get(key, {}))
        sample_rate = architecture.get('sample_rate', 16000)
        frame_rate = architecture.get('frame_rate', 50)
        self.encoder = MimiEncoderWithQuantModel(
            create_encoder_frontend('seanet', seanet_args=args('seanet_args', dc.SeanetArgs)),
            FactorizedVectorQuantize(**asdict(args('quantizer_args', dc.FVQuantizerArgs))),
            channels=1, sample_rate=sample_rate, frame_rate=frame_rate, causal=True,
            encoder_transformer=transformer.ProjectedTransformer(
                device='cpu', **asdict(args('transformer_args', dc.TransformerArgs))),
        )
        self.decoder = MimiDecoderWithTVTimbreModel(
            create_waveform_decoder(
                decoder_type=architecture.get('decoder_type', 'seanet'),
                seanet_args=args('decoder_seanet_args', dc.DecSeanetNCArgs),
                decoder_args=args('waveform_decoder_args', dc.WaveformDecoderArgs)),
            frame_rate=frame_rate, sample_rate=sample_rate, channels=1,
            encoder_dim=architecture.get('encoder_dim', 512), causal=True,
            decoder_transformer=FrameSynthesizerWOExciter(
                device='cpu', **asdict(args('frame_synthesizer_args', dc.FrameSynthesizerArgs)),
                tvtimbre_args=asdict(args('tvtimbre_args', dc.TVTimbreArgs)),
                use_pitch_energy_predictor=architecture.get('use_pitch_energy_predictor', True)),
        )

    def forward(self, audio, speaker):
        latent, _ = self.encoder(audio)
        return self.decoder.decode(latent, speaker)

    def load_training_state(self, state):
        mapped = {}
        excluded = []
        for key, value in state.items():
            if key.startswith('encoder.net.'):
                mapped['encoder.' + key[len('encoder.net.'):]] = value
            elif key.startswith('decoder.'):
                mapped[key] = value
            else:
                excluded.append(key)
        self.load_state_dict(mapped, strict=True)
        return excluded
