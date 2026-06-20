import torch

from src.components.discriminator import Discriminator, MPD, MRD, MSD


def test_mpd_does_not_pad_already_aligned_input():
    discriminator = MPD(4)
    x = torch.randn(1, 1, 16)
    assert discriminator.pad_to_period(x).shape == x.shape
    assert discriminator.pad_to_period(x[..., :2]).shape[-1] == 4


def test_cleanup_preprocess_preserves_amplitude_difference():
    legacy = Discriminator(periods=[], rates=[], fft_sizes=[], cleanup=False)
    cleanup = Discriminator(periods=[], rates=[], fft_sizes=[], cleanup=True)
    x = torch.tensor([[[0.0, 1.0, -1.0]]])
    torch.testing.assert_close(legacy.preprocess(x).abs().max(), torch.tensor(0.8))
    assert cleanup.preprocess(x).abs().max() == 1.0


def test_cleanup_msd_and_mrd_are_finite():
    waveform = torch.randn(1, 1, 2048)
    for discriminator in (MSD(2, cleanup=True), MRD(256, cleanup=True)):
        features = discriminator(waveform)
        assert features
        assert all(torch.isfinite(feature).all() for feature in features)
