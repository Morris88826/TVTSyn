import pytest
import torch

from src.components.data_classes import (
    EncoderFrontendArgs,
    FVQuantizerArgs,
    SeanetArgs,
    TransformerArgs,
)
from src.components.encoder_frontends import (
    HybridStemMelTransformerFrontend,
    MelCNNTransformerFrontend,
    MelTransformerFrontend,
    RawPatchTransformerFrontend,
    create_encoder_frontend,
)
from src.components.seanet import SEANetEncoder
from src.modules.pasrwithquant import PasrWithQuantModule


FRONTEND_ARGS = {
    "sample_rate": 16000,
    "n_fft": 1024,
    "win_length": 1024,
    "hop_length": 320,
    "n_mels": 80,
    "dimension": 32,
}


@pytest.mark.parametrize("time", [1, 2, 3, 5, 10, 17])
@pytest.mark.parametrize(
    "frontend_class,extra_args",
    [
        (MelTransformerFrontend, {}),
        (MelCNNTransformerFrontend, {"mel_cnn_layers": 2}),
        (
            RawPatchTransformerFrontend,
            {"hybrid_raw_dim": 32, "hybrid_raw_conv_layers": 1},
        ),
        (
            HybridStemMelTransformerFrontend,
            {"hybrid_raw_dim": 32, "hybrid_raw_conv_layers": 1},
        ),
    ],
)
def test_frontend_shapes(frontend_class, extra_args, time):
    frontend = frontend_class(**FRONTEND_ARGS, **extra_args)
    waveform = torch.randn(2, 1, time * 320)
    features = frontend(waveform)
    assert features.shape == (2, 32, time)


@pytest.mark.parametrize(
    "frontend_class", [MelTransformerFrontend, MelCNNTransformerFrontend]
)
def test_mel_frontend_streaming_matches_offline(frontend_class):
    frontend = frontend_class(**FRONTEND_ARGS).eval()
    waveform = torch.randn(1, 1, 320 * 7)
    with torch.no_grad():
        offline = frontend(waveform)
        with frontend.streaming(1):
            streamed = torch.cat(
                [frontend(waveform[..., :777]), frontend(waveform[..., 777:])],
                dim=-1,
            )
    torch.testing.assert_close(streamed, offline, atol=1e-5, rtol=1e-4)


@pytest.mark.parametrize("fusion_type", ["sum", "concat", "gated"])
def test_hybrid_streaming_matches_offline_and_resets(fusion_type):
    frontend = HybridStemMelTransformerFrontend(
        **FRONTEND_ARGS,
        hybrid_raw_dim=32,
        hybrid_raw_conv_layers=2,
        hybrid_fusion_type=fusion_type,
    ).eval()
    waveform = torch.randn(1, 1, 320 * 8)

    with torch.no_grad():
        offline = frontend(waveform)
        with frontend.streaming(1):
            chunks = [
                frontend(waveform[..., :111]),
                frontend(waveform[..., 111:960]),
                frontend(waveform[..., 960:1277]),
                frontend(waveform[..., 1277:]),
            ]
            streamed = torch.cat(chunks, dim=-1)
            frontend.reset_streaming()
            reset = frontend(waveform)

    assert streamed.shape == offline.shape
    torch.testing.assert_close(streamed, offline, atol=1e-5, rtol=1e-4)
    torch.testing.assert_close(reset, offline, atol=1e-5, rtol=1e-4)


def test_hybrid_960_sample_chunks_emit_three_frames():
    frontend = HybridStemMelTransformerFrontend(
        **FRONTEND_ARGS, hybrid_raw_dim=32
    ).eval()
    waveform = torch.randn(1, 1, 1920)
    with torch.no_grad(), frontend.streaming(1):
        first = frontend(waveform[..., :960])
        second = frontend(waveform[..., 960:])
    assert first.shape == (1, 32, 3)
    assert second.shape == (1, 32, 3)



def test_raw_patch_streaming_buffers_residual_samples_and_resets():
    frontend = RawPatchTransformerFrontend(
        **FRONTEND_ARGS,
        hybrid_raw_dim=32,
        hybrid_raw_conv_layers=2,
    ).eval()
    waveform = torch.randn(1, 1, 320 * 8)

    with torch.no_grad():
        offline = frontend(waveform)
        with frontend.streaming(1):
            chunks = [
                frontend(waveform[..., :111]),
                frontend(waveform[..., 111:960]),
                frontend(waveform[..., 960:1277]),
                frontend(waveform[..., 1277:]),
            ]
            streamed = torch.cat(chunks, dim=-1)
            frontend.reset_streaming()
            reset = frontend(waveform)

    assert chunks[0].shape == (1, 32, 0)
    torch.testing.assert_close(streamed, offline, atol=1e-5, rtol=1e-4)
    torch.testing.assert_close(reset, offline, atol=1e-5, rtol=1e-4)


def test_raw_patch_960_sample_chunks_emit_three_frames():
    frontend = RawPatchTransformerFrontend(
        **FRONTEND_ARGS, hybrid_raw_dim=32
    ).eval()
    waveform = torch.randn(1, 1, 1920)
    with torch.no_grad(), frontend.streaming(1):
        first = frontend(waveform[..., :960])
        second = frontend(waveform[..., 960:])
    assert first.shape == (1, 32, 3)
    assert second.shape == (1, 32, 3)



def test_frontend_backward():
    for frontend_class in (
        RawPatchTransformerFrontend,
        HybridStemMelTransformerFrontend,
    ):
        frontend = frontend_class(**FRONTEND_ARGS, hybrid_raw_dim=32)
        waveform = torch.randn(2, 1, 320 * 3, requires_grad=True)
        frontend(waveform).square().mean().backward()
        assert waveform.grad is not None


def test_frontend_factory_and_validation():
    seanet = create_encoder_frontend(
        "seanet",
        seanet_args=SeanetArgs(dimension=16, n_filters=4, ratios=[2, 2]),
    )
    assert isinstance(seanet, SEANetEncoder)
    assert isinstance(
        create_encoder_frontend(
            "mel_transformer", frontend_args={**FRONTEND_ARGS}
        ),
        MelTransformerFrontend,
    )
    assert isinstance(
        create_encoder_frontend(
            "raw_patch_transformer", frontend_args={**FRONTEND_ARGS}
        ),
        RawPatchTransformerFrontend,
    )
    with pytest.raises(ValueError, match="hybrid_raw_frame_samples"):
        RawPatchTransformerFrontend(
            **FRONTEND_ARGS, hybrid_raw_frame_samples=160
        )
    with pytest.raises(ValueError, match="hybrid_raw_frame_samples"):
        HybridStemMelTransformerFrontend(
            **FRONTEND_ARGS, hybrid_raw_frame_samples=160
        )
    with pytest.raises(ValueError, match="hybrid_fusion_type"):
        HybridStemMelTransformerFrontend(
            **FRONTEND_ARGS, hybrid_fusion_type="invalid"
        )


@pytest.mark.parametrize(
    "frontend_type",
    ["raw_patch_transformer", "hybrid_stem_mel_transformer"],
)
def test_frontend_integrates_with_transformer_and_vq_path(frontend_type):
    module = PasrWithQuantModule(
        encoder_frontend_type=frontend_type,
        encoder_frontend_args=EncoderFrontendArgs(
            dimension=16,
            hybrid_raw_dim=16,
            hybrid_raw_conv_layers=1,
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
        quantizer_args=FVQuantizerArgs(
            input_dim=16,
            codebook_size=16,
            codebook_dim=4,
        ),
        encoder_dim=16,
        num_codes=8,
    )
    logits, latent, metrics = module(torch.randn(1, 1, 960))
    assert logits.shape == (1, 8, 3)
    assert latent.shape == (1, 16, 3)
    assert "commit_loss" in metrics
    assert "codebook_loss" in metrics


def test_default_pasr_encoder_remains_seanet():
    module = PasrWithQuantModule(
        seanet_args=SeanetArgs(dimension=16, n_filters=4, ratios=[2, 2]),
        transformer_args=TransformerArgs(
            d_model=16,
            num_heads=4,
            num_layers=1,
            dim_feedforward=32,
            input_dimension=16,
            output_dimensions=[16],
        ),
        quantizer_args=FVQuantizerArgs(input_dim=16, codebook_size=16, codebook_dim=4),
        encoder_dim=16,
    )
    assert isinstance(module.net.encoder, SEANetEncoder)
