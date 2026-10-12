from types import SimpleNamespace
import sys

from audiotools import AudioSignal
import pytest
import torch
from torch import nn

from tvtsyn.components.data_classes import (
    EncoderFrontendArgs,
    HuBERTLossArgs,
    SupervisedWav2WavLambdas,
    TransformerArgs,
    WaveformDecoderArgs,
)
from tvtsyn.components.encoder_frontends import MelTransformerFrontend
from training.components.losses import HuBERTLoss
from tvtsyn.components.transformer import ProjectedTransformer
from tvtsyn.components.waveform_decoders import FrameLinearWaveformDecoder
from tvtsyn.models import MimiWav2WavModel
import training.modules.supervisedwav2wav as supervised_module


def _build_wrapper():
    encoder = MelTransformerFrontend(
        sample_rate=1600,
        n_fft=64,
        win_length=64,
        hop_length=32,
        n_mels=8,
        dimension=16,
    )
    frame_transformer = ProjectedTransformer(
        input_dimension=16,
        output_dimensions=[16],
        d_model=16,
        conv_layout=True,
        num_heads=4,
        num_layers=1,
        dim_feedforward=32,
        causal=True,
        context=16,
        right_context=0,
    )
    decoder = FrameLinearWaveformDecoder(
        input_dimension=16,
        decoder_dim=16,
        decoder_layers=1,
        decoder_heads=4,
        decoder_ffn_dim=32,
        decoder_context_frames=16,
        frame_hop_samples=32,
        use_post_filter=False,
    )
    return MimiWav2WavModel(
        encoder, frame_transformer, decoder, sample_rate=1600, frame_rate=50
    )


class TinyDiscriminator(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.conv = nn.Conv1d(1, 1, 1)

    def forward(self, waveform):
        feature = self.conv(waveform)
        return [[feature, feature.mean(dim=-1, keepdim=True)]]


def _build_module(monkeypatch, **module_kwargs):
    monkeypatch.setattr(supervised_module, "Discriminator", TinyDiscriminator)
    return supervised_module.SupervisedWav2WavModule(
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
        discriminator_args={},
        validation_audio_samples=2,
        **module_kwargs,
    )


def test_wrapper_output_length_backward_and_streaming_equivalence():
    model = _build_wrapper().eval()
    unaligned = torch.randn(2, 1, 173, requires_grad=True)
    output = model(unaligned)
    assert output.shape == unaligned.shape
    output.square().mean().backward()
    assert unaligned.grad is not None

    waveform = torch.randn(1, 1, 32 * 7)
    with torch.no_grad():
        offline = model(waveform)
        with model.streaming(1):
            streamed = torch.cat(
                [model(waveform[..., :64]), model(waveform[..., 64:160]), model(waveform[..., 160:])],
                dim=-1,
            )
    torch.testing.assert_close(streamed, offline, atol=1e-5, rtol=1e-4)


def test_module_batch_contract_and_optimizer_coverage(monkeypatch):
    module = _build_module(monkeypatch)
    batch = (torch.randn(2, 160), torch.randn(2, 160), ["a.wav", "b.wav"])
    reconstruction, target, source, names = module.model_step(batch)
    assert reconstruction.audio_data.shape == target.audio_data.shape == (2, 1, 160)
    assert source.shape == (2, 1, 160)
    assert names == ["a.wav", "b.wav"]
    assert not hasattr(module.generator, "quantizer")

    optimizers, _ = module.configure_optimizers()
    generator_ids = {id(parameter) for parameter in module.generator.parameters()}
    discriminator_ids = {id(parameter) for parameter in module.discriminator.parameters()}
    opt_generator_ids = {id(parameter) for group in optimizers[0].param_groups for parameter in group["params"]}
    opt_discriminator_ids = {id(parameter) for group in optimizers[1].param_groups for parameter in group["params"]}
    assert generator_ids == opt_generator_ids
    assert discriminator_ids == opt_discriminator_ids


def test_hubert_loss_loads_pretrained_weights_and_backpropagates(tmp_path):
    from transformers import HubertConfig, HubertModel

    checkpoint = tmp_path / "tiny-hubert"
    config = HubertConfig(
        hidden_size=8,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=16,
        conv_dim=(4, 4),
        conv_stride=(2, 2),
        conv_kernel=(3, 3),
        num_conv_pos_embeddings=4,
        num_conv_pos_embedding_groups=2,
        feat_extract_norm="layer",
    )
    HubertModel(config).save_pretrained(checkpoint)

    loss_fn = HuBERTLoss(
        model_name_or_path=str(checkpoint),
        local_files_only=True,
    )
    generated = torch.randn(2, 1, 64, requires_grad=True)
    target = torch.randn(2, 1, 64)
    loss = loss_fn(generated, target)
    loss.backward()

    assert torch.isfinite(loss)
    assert generated.grad is not None
    assert generated.grad.abs().sum() > 0
    assert all(not parameter.requires_grad for parameter in loss_fn.model.parameters())
    assert not loss_fn.state_dict()



def test_module_constructs_hubert_loss_only_when_enabled(monkeypatch):
    disabled = _build_module(monkeypatch)
    assert disabled.hubert_loss is None

    class DummyHuBERTLoss(nn.Module):
        def __init__(self, **kwargs):
            super().__init__()
            self.kwargs = kwargs

        def forward(self, generated, target):
            return generated.audio_data.new_zeros(())

    monkeypatch.setattr(supervised_module, "HuBERTLoss", DummyHuBERTLoss)
    enabled = _build_module(
        monkeypatch,
        lambdas=SupervisedWav2WavLambdas(hubert_loss=1.0),
        hubert_loss_args=HuBERTLossArgs(
            model_name_or_path="local-checkpoint",
            local_files_only=True,
        ),
    )
    assert isinstance(enabled.hubert_loss, DummyHuBERTLoss)
    assert enabled.hubert_loss.kwargs["model_name_or_path"] == "local-checkpoint"
    assert enabled.hubert_loss.kwargs["local_files_only"] is True



class ConstantLoss(nn.Module):
    def __init__(self, value):
        super().__init__()
        self.value = value

    def forward(self, *args):
        return torch.tensor(self.value)


class ConstantGAN(nn.Module):
    def generator_loss(self, fake, real):
        return torch.tensor(3.0), torch.tensor(4.0)


def test_generator_loss_composition(monkeypatch):
    module = _build_module(monkeypatch)
    module.mel_loss = ConstantLoss(2.0)
    module.gan_loss = ConstantGAN()
    module.hubert_loss = ConstantLoss(5.0)
    module.loss_weights = {
        "mel_loss": 20.0,
        "adv_gen_loss": 1.0,
        "adv_feat_loss": 2.0,
        "hubert_loss": 6.0,
    }
    signal = AudioSignal(torch.zeros(1, 1, 64), 1600)
    total, mel, adversarial, feature, hubert = module.generator_objective(signal, signal)
    assert total.item() == 81.0
    assert (
        mel.item(),
        adversarial.item(),
        feature.item(),
        hubert.item(),
    ) == (2.0, 3.0, 4.0, 5.0)


def test_wandb_audio_logging_and_guards(monkeypatch):
    module = _build_module(monkeypatch)

    class FakeExperiment:
        def __init__(self):
            self.calls = []

        def log(self, payload, step):
            self.calls.append((payload, step))

    class FakeWandbLogger:
        def __init__(self):
            self.experiment = FakeExperiment()

    class FakeAudio:
        def __init__(self, samples, sample_rate, caption):
            self.samples = samples
            self.sample_rate = sample_rate
            self.caption = caption

    monkeypatch.setattr(supervised_module, "WandbLogger", FakeWandbLogger)
    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(Audio=FakeAudio))
    logger = FakeWandbLogger()
    module._trainer = SimpleNamespace(
        sanity_checking=False,
        is_global_zero=True,
        logger=logger,
        global_step=12,
    )
    audio = torch.zeros(2, 1, 64)
    module._log_validation_audio(audio, audio, audio, ["a.wav", "b.wav"])
    assert len(logger.experiment.calls) == 1
    payload, step = logger.experiment.calls[0]
    assert step == 12
    assert len(payload) == 6

    module._validation_audio_logged = False
    module._trainer.sanity_checking = True
    module._log_validation_audio(audio, audio, audio, ["a.wav", "b.wav"])
    assert len(logger.experiment.calls) == 1


def test_module_has_expected_metric_names(monkeypatch):
    module = _build_module(monkeypatch)
    for name in (
        "train_mel_loss",
        "train_adv_gen_loss",
        "train_adv_feat_loss",
        "train_hubert_loss",
        "train_adv_d_loss",
        "train_total_loss",
        "val_mel_loss",
        "val_hubert_loss",
        "test_mel_loss",
        "test_hubert_loss",
        "val_loss_best",
    ):
        assert hasattr(module, name)
