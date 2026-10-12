from pathlib import Path

import torch
from torch import nn
import yaml

from tvtsyn.components.data_classes import (
    EncoderFrontendArgs,
    SupervisedWav2WavLambdas,
    TransformerArgs,
    WaveformDecoderArgs,
)
from tvtsyn.models import MimiWav2WavExcitationModel
import training.modules.supervisedwav2wav as base_module
from training.modules.supervisedwav2wavexcitation import (
    SupervisedWav2WavExcitationModule,
)


class TinyDiscriminator(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.conv = nn.Conv1d(1, 1, 1)

    def forward(self, waveform):
        feature = self.conv(waveform)
        return [[feature, feature.mean(dim=-1, keepdim=True)]]


def _module(monkeypatch):
    monkeypatch.setattr(base_module, "Discriminator", TinyDiscriminator)
    return SupervisedWav2WavExcitationModule(
        encoder_frontend_type="mel_transformer",
        encoder_frontend_args=EncoderFrontendArgs(
            sample_rate=1600,
            n_fft=64,
            win_length=64,
            hop_length=32,
            n_mels=8,
            dimension=16,
        ),
        transformer_args=TransformerArgs(
            d_model=16,
            num_heads=4,
            num_layers=1,
            dim_feedforward=32,
            context=16,
            input_dimension=16,
            output_dimensions=[16],
        ),
        decoder_type="frame_linear",
        waveform_decoder_args=WaveformDecoderArgs(
            input_dimension=16,
            frame_hop_samples=32,
            decoder_dim=16,
            decoder_layers=1,
            decoder_heads=4,
            decoder_ffn_dim=32,
            decoder_context_frames=16,
            use_post_filter=False,
        ),
        sample_rate=1600,
        frame_rate=50,
        prosody_hidden_dim=16,
        prosody_num_layers=1,
        harmonic_source_n_fft=64,
        harmonic_source_win_length=64,
        lambdas=SupervisedWav2WavLambdas(
            mel_loss=1.0,
            f0_loss=1.0,
            vuv_loss=1.0,
            energy_loss=1.0,
        ),
        discriminator_args={},
    )


def test_excitation_module_batch_and_objective(monkeypatch):
    module = _module(monkeypatch)
    assert isinstance(module.generator, MimiWav2WavExcitationModel)
    waveform = torch.randn(2, 2048)
    frames = 64
    excitation = {
        "f0_hz": torch.full((2, frames), 120.0),
        "log_f0": torch.full((2, frames), 120.0).log(),
        "vuv": torch.ones(2, frames),
        "energy": torch.ones(2, frames),
        "log_energy": torch.zeros(2, frames),
        "frame_mask": torch.ones(2, frames, dtype=torch.bool),
    }
    batch = (waveform, waveform, ["a.wav", "b.wav"], excitation)
    reconstruction, target, _, _, loaded, aux = module.model_step(
        batch, use_target_prosody=True
    )
    losses = module.generator_objective(reconstruction, target, loaded, aux)
    assert reconstruction.audio_data.shape == target.audio_data.shape
    assert torch.isfinite(losses["total_loss"])
    assert losses["voiced_frames"] == 2 * frames


def test_excitation_module_accepts_legacy_batch(monkeypatch):
    module = _module(monkeypatch)
    waveform = torch.randn(1, 160)
    result = module.model_step((waveform, waveform, ["a.wav"]))
    assert result[4] is None


def test_combined_excitation_config_parses():
    root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load(
        (root / "training/configs/train_supervised_wav2wav_excitation.yaml").read_text()
    )
    assert config["model"]["class_path"] == "training.modules.SupervisedWav2WavExcitationModule"
    assert config["data"]["init_args"]["return_excitation"] is True
    assert config["model"]["init_args"]["discriminator_args"]["cleanup"] is True
    assert (
        config["model"]["init_args"]["waveform_decoder_args"]["snake_type"]
        == "snakebeta"
    )
