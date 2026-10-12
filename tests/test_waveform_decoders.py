import pytest
import torch

from tvtsyn.components.data_classes import DecSeanetNCArgs
from tvtsyn.components.waveform_decoders import (
    FrameLinearWaveformDecoder,
    HybridLinearCNNDecoder,
    ISTFTDecoder,
    SubframeLinearWaveformDecoder,
    create_waveform_decoder,
)
from tvtsyn.components.seanet import SEANetDecoder


COMMON_ARGS = {
    "input_dimension": 16,
    "decoder_dim": 16,
    "decoder_layers": 1,
    "decoder_heads": 4,
    "decoder_ffn_dim": 32,
    "decoder_context_frames": 32,
    "frame_hop_samples": 32,
}


@pytest.mark.parametrize("time", [1, 2, 3, 5, 10, 17])
@pytest.mark.parametrize(
    "decoder_class,extra_args",
    [
        (
            FrameLinearWaveformDecoder,
            {
                "linear_hidden_dim": 32,
                "post_filter_channels": 8,
                "post_filter_layers": 1,
            },
        ),
        (
            SubframeLinearWaveformDecoder,
            {
                "num_subframes": 4,
                "subframe_samples": 8,
                "subframe_hidden_dim": 32,
                "post_filter_channels": 8,
                "post_filter_layers": 1,
            },
        ),
        (
            HybridLinearCNNDecoder,
            {
                "hybrid_subframes": 4,
                "hybrid_subframe_dim": 16,
                "hybrid_refiner_layers": 2,
            },
        ),
        (
            HybridLinearCNNDecoder,
            {
                "hybrid_mode": "partial_seanet",
                "hybrid_subframes": 4,
                "hybrid_subframe_dim": 16,
                "hybrid_refiner_layers": 1,
            },
        ),
    ],
)
def test_decoder_shape_and_backward(decoder_class, extra_args, time):
    decoder = decoder_class(**COMMON_ARGS, **extra_args)
    x = torch.randn(2, 16, time, requires_grad=True)
    wav = decoder(x)
    assert wav.shape == (2, 1, time * 32)
    wav.abs().mean().backward()
    assert x.grad is not None


@pytest.mark.parametrize(
    "decoder",
    [
        FrameLinearWaveformDecoder(
            **COMMON_ARGS,
            linear_hidden_dim=32,
            post_filter_channels=8,
            post_filter_layers=1,
        ),
        SubframeLinearWaveformDecoder(
            **COMMON_ARGS,
            num_subframes=4,
            subframe_samples=8,
            subframe_hidden_dim=32,
            post_filter_channels=8,
            post_filter_layers=1,
        ),
        HybridLinearCNNDecoder(
            **COMMON_ARGS,
            hybrid_subframes=4,
            hybrid_subframe_dim=16,
            hybrid_refiner_layers=2,
        ),
        HybridLinearCNNDecoder(
            **COMMON_ARGS,
            hybrid_mode="partial_seanet",
            hybrid_subframes=4,
            hybrid_subframe_dim=16,
            hybrid_refiner_layers=1,
        ),
    ],
)
def test_streaming_matches_offline(decoder):
    decoder.eval()
    x = torch.randn(1, 16, 11)
    with torch.no_grad():
        offline = decoder(x)
        chunks = []
        with decoder.streaming(1):
            for start in range(0, x.shape[-1], 3):
                chunks.append(decoder(x[..., start : start + 3]))
        streamed = torch.cat(chunks, dim=-1)
    assert streamed.shape == offline.shape
    torch.testing.assert_close(streamed, offline, atol=1e-5, rtol=1e-4)


def test_subframe_configuration_validation():
    with pytest.raises(ValueError, match="must equal"):
        SubframeLinearWaveformDecoder(
            **COMMON_ARGS, num_subframes=3, subframe_samples=8
        )


def test_istft_shape_backward_and_streaming_equivalence():
    decoder = ISTFTDecoder(
        **COMMON_ARGS,
        istft_n_fft=64,
        istft_hop_length=32,
        istft_win_length=64,
        linear_hidden_dim=32,
    )
    x = torch.randn(2, 16, 5, requires_grad=True)
    wav = decoder(x)
    assert wav.shape == (2, 1, 5 * 32)
    wav.square().mean().backward()
    decoder.eval()
    with torch.no_grad():
        offline = decoder(x.detach())
        with decoder.streaming(2):
            streamed = torch.cat(
                [decoder(x.detach()[..., :2]), decoder(x.detach()[..., 2:])],
                dim=-1,
            )
    assert streamed.shape == offline.shape
    torch.testing.assert_close(streamed, offline, atol=1e-5, rtol=1e-4)

    with decoder.streaming(2):
        first = decoder(x.detach())
        decoder.reset_streaming()
        reset = decoder(x.detach())
    torch.testing.assert_close(reset, first, atol=1e-5, rtol=1e-4)


@pytest.mark.parametrize(
    "decoder_type,decoder_class,extra_args",
    [
        ("frame_linear", FrameLinearWaveformDecoder, {}),
        (
            "subframe_linear",
            SubframeLinearWaveformDecoder,
            {"num_subframes": 4, "subframe_samples": 8},
        ),
        ("hybrid_linear_cnn", HybridLinearCNNDecoder, {}),
        (
            "istft",
            ISTFTDecoder,
            {"istft_n_fft": 64, "istft_hop_length": 32, "istft_win_length": 64},
        ),
    ],
)
def test_factory_constructs_new_decoder_types(
    decoder_type, decoder_class, extra_args
):
    decoder = create_waveform_decoder(
        decoder_type,
        decoder_args={
            **COMMON_ARGS,
            **extra_args,
            "use_post_filter": False,
        },
    )
    assert isinstance(decoder, decoder_class)


def test_factory_and_seanet_backward_compatibility():
    seanet = create_waveform_decoder(
        "seanet",
        seanet_args=DecSeanetNCArgs(
            dimension=16,
            n_filters=4,
            ratios=[2, 2],
        ),
    )
    assert isinstance(seanet, SEANetDecoder)
    linear = create_waveform_decoder(
        "frame_linear",
        decoder_args={
            **COMMON_ARGS,
            "linear_hidden_dim": 32,
            "use_post_filter": False,
        },
    )
    assert isinstance(linear, FrameLinearWaveformDecoder)
