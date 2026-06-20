import torch

from src.components.excitation import (
    HarmonicSourceGenerator,
    ProsodyConditioner,
    ProsodyPredictor,
)
from src.components.periodic_activations import Snake, SnakeBeta
from src.components.spectral_losses import (
    BandWeightedSTFTLoss,
    CepstralEnvelopeLoss,
    ComplexSTFTLoss,
    HarmonicMagnitudeLoss,
    MultiResolutionSTFTLoss,
    ProsodyLoss,
)
from src.components.encoder_frontends import MelTransformerFrontend
from src.components.transformer import ProjectedTransformer
from src.components.waveform_decoders import FrameLinearWaveformDecoder
from src.models import MimiWav2WavExcitationModel


def _model():
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
        decoder_right_context_frames=0,
        frame_hop_samples=32,
        use_post_filter=False,
    )
    return MimiWav2WavExcitationModel(
        encoder,
        frame_transformer,
        decoder,
        sample_rate=1600,
        frame_rate=50,
        prosody_hidden_dim=16,
        prosody_num_layers=1,
        harmonic_source_n_fft=64,
        harmonic_source_win_length=64,
    )


def test_periodic_activations_are_finite_and_differentiable():
    for activation in (Snake(8), SnakeBeta(8)):
        x = torch.randn(2, 8, 11, requires_grad=True)
        activation(x).mean().backward()
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()


def test_prosody_predictor_streaming_and_conditioner_modes():
    predictor = ProsodyPredictor(16, 12, num_layers=2).eval()
    z = torch.randn(1, 16, 9)
    with torch.no_grad():
        offline = predictor(z)
        with predictor.streaming(1):
            streamed = [
                predictor(z[..., :3]),
                predictor(z[..., 3:7]),
                predictor(z[..., 7:]),
            ]
    for key in offline:
        torch.testing.assert_close(
            torch.cat([part[key] for part in streamed], dim=-1),
            offline[key],
            atol=1e-5,
            rtol=1e-4,
        )
    condition = torch.randn(1, 3, 9)
    for mode in ("add", "concat", "film"):
        assert ProsodyConditioner(16, 3, 8, mode)(z, condition).shape == z.shape


def test_harmonic_source_streaming_reset_and_unvoiced_safety():
    generator = HarmonicSourceGenerator(
        sample_rate=1600, hop_length=32, num_harmonics=4
    ).eval()
    f0 = torch.tensor([[100.0, 110.0, 0.0, 120.0]])
    vuv = torch.tensor([[1.0, 1.0, 0.0, 1.0]])
    energy = torch.zeros_like(f0)
    offline = generator(f0, vuv, energy)
    with generator.streaming(1):
        streamed = torch.cat(
            [
                generator(f0[:, :2], vuv[:, :2], energy[:, :2]),
                generator(f0[:, 2:], vuv[:, 2:], energy[:, 2:]),
            ],
            dim=-1,
        )
        generator.reset_streaming()
        reset = generator(f0, vuv, energy)
    torch.testing.assert_close(streamed, offline, atol=1e-5, rtol=1e-4)
    torch.testing.assert_close(reset, offline, atol=1e-5, rtol=1e-4)
    assert torch.isfinite(offline).all()


def test_excitation_model_predicted_and_teacher_forced_streaming():
    model = _model().eval()
    waveform = torch.randn(1, 1, 32 * 7)
    excitation = {
        "f0_hz": torch.full((1, 7), 120.0),
        "log_f0": torch.full((1, 7), 120.0).log(),
        "vuv": torch.ones(1, 7),
        "energy": torch.ones(1, 7),
        "log_energy": torch.zeros(1, 7),
        "frame_mask": torch.ones(1, 7, dtype=torch.bool),
    }

    def excitation_slice(start, stop):
        return {key: value[:, start:stop] for key, value in excitation.items()}

    with torch.no_grad():
        predicted, aux = model(waveform, return_aux=True)
        teacher = model(waveform, excitation, True)
        with model.streaming(1):
            streamed = torch.cat(
                [
                    model(waveform[..., :64]),
                    model(waveform[..., 64:160]),
                    model(waveform[..., 160:]),
                ],
                dim=-1,
            )
        with model.streaming(1):
            teacher_streamed = torch.cat(
                [
                    model(waveform[..., :64], excitation_slice(0, 2), True),
                    model(waveform[..., 64:160], excitation_slice(2, 5), True),
                    model(waveform[..., 160:], excitation_slice(5, 7), True),
                ],
                dim=-1,
            )
        with model.streaming(1):
            first_streaming = model(waveform)
            model.reset_streaming()
            reset = model(waveform)
    assert predicted.shape == teacher.shape == waveform.shape
    assert "prosody_pred" in aux and "harmonic_source" in aux
    torch.testing.assert_close(streamed, predicted, atol=2e-4, rtol=1e-4)
    torch.testing.assert_close(teacher_streamed, teacher, atol=4e-4, rtol=1e-4)
    torch.testing.assert_close(first_streaming, predicted, atol=2e-4, rtol=1e-4)
    torch.testing.assert_close(reset, first_streaming, atol=1e-6, rtol=1e-5)


def test_spectral_and_prosody_losses_have_finite_gradients():
    estimate = torch.randn(2, 1, 2048, requires_grad=True)
    target = torch.randn(2, 1, 2048)
    excitation = {
        "f0_hz": torch.full((2, 64), 120.0),
        "log_f0": torch.full((2, 64), 120.0).log(),
        "vuv": torch.ones(2, 64),
        "energy": torch.ones(2, 64),
        "log_energy": torch.zeros(2, 64),
        "frame_mask": torch.ones(2, 64, dtype=torch.bool),
    }
    prediction = {
        "log_f0": torch.randn(2, 1, 64, requires_grad=True),
        "vuv_logits": torch.randn(2, 1, 64, requires_grad=True),
        "log_energy": torch.randn(2, 1, 64, requires_grad=True),
    }
    values = list(
        MultiResolutionSTFTLoss((64, 128), (32, 64), (64, 128))(
            estimate, target
        )
    )
    values += [
        ComplexSTFTLoss(64, 32, 64)(estimate, target),
        HarmonicMagnitudeLoss(
            64, 32, 64, sample_rate=1600, max_freq=700, max_harmonics=5
        )(estimate, target, excitation),
        CepstralEnvelopeLoss(64, 32, 64, 8)(estimate, target),
        BandWeightedSTFTLoss(
            64, 32, 64, sample_rate=1600, bands=((0, 800, 1.0),)
        )(estimate, target),
        ProsodyLoss()(prediction, excitation)["prosody_loss"],
    ]
    total = sum(values)
    total.backward()
    assert all(torch.isfinite(value) for value in values)
    assert estimate.grad is not None

