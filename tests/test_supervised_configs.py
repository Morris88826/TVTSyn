from pathlib import Path

import yaml

from tvtsyn.components.data_classes import DecSeanetNCArgs, EncoderFrontendArgs, SeanetArgs, WaveformDecoderArgs
from tvtsyn.components.encoder_frontends import HybridStemMelTransformerFrontend, create_encoder_frontend
from tvtsyn.components.seanet import SEANetDecoder, SEANetEncoder
from tvtsyn.components.waveform_decoders import ISTFTDecoder, create_waveform_decoder
from training.data_modules.parallel_audio import ParallelAudioDataModule


ROOT = Path(__file__).resolve().parents[1]


def _load(path):
    with (ROOT / path).open() as handle:
        return yaml.safe_load(handle)


def test_supervised_seanet_config_selects_baseline_components():
    config = _load("training/configs/model/supervised_wav2wav_seanet.yaml")
    args = config["model"]["init_args"]
    assert config["model"]["class_path"] == "training.modules.SupervisedWav2WavModule"
    encoder = create_encoder_frontend(
        args["encoder_frontend_type"],
        seanet_args=SeanetArgs(dimension=16, n_filters=4, ratios=[2, 2]),
    )
    decoder = create_waveform_decoder(
        args["decoder_type"],
        seanet_args=DecSeanetNCArgs(dimension=16, n_filters=4, ratios=[2, 2]),
    )
    assert isinstance(encoder, SEANetEncoder)
    assert isinstance(decoder, SEANetDecoder)


def test_supervised_hybrid_istft_config_selects_experimental_components():
    config = _load("training/configs/model/supervised_wav2wav_hybrid_istft_nola.yaml")
    args = config["model"]["init_args"]
    frontend_args = dict(args["encoder_frontend_args"])
    frontend_args.update(
        n_fft=64,
        win_length=64,
        hop_length=32,
        n_mels=8,
        dimension=16,
        hybrid_raw_frame_samples=32,
        hybrid_raw_dim=16,
    )
    decoder_args = dict(args["waveform_decoder_args"])
    decoder_args.update(
        input_dimension=16,
        frame_hop_samples=32,
        decoder_dim=16,
        decoder_layers=1,
        decoder_heads=4,
        decoder_ffn_dim=32,
        istft_n_fft=64,
        istft_hop_length=32,
        istft_win_length=64,
    )
    encoder = create_encoder_frontend(
        args["encoder_frontend_type"],
        frontend_args=EncoderFrontendArgs(**frontend_args),
    )
    decoder = create_waveform_decoder(
        args["decoder_type"],
        decoder_args=WaveformDecoderArgs(**decoder_args),
    )
    assert isinstance(encoder, HybridStemMelTransformerFrontend)
    assert isinstance(decoder, ISTFTDecoder)


def test_parallel_audio_data_config_instantiates():
    config = _load("training/configs/data/parallel_audio.yaml")
    assert config["data"]["class_path"] == "training.data_modules.ParallelAudioDataModule"
    datamodule = ParallelAudioDataModule(**config["data"]["init_args"])
    assert datamodule.hparams.segment_size == 48000
    assert datamodule.hparams.validation_split == 0.002
