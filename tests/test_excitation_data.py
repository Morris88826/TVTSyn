import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

from src.data_modules import parallel_audio
from src.data_modules.parallel_audio import (
    ParallelAudioDataset,
    excitation_path_for,
    parallel_audio_collate,
)


def _pair(root: Path):
    source = root / "source" / "speaker" / "wav" / "a.wav"
    target = root / "target" / "speaker" / "wav" / "a.wav"
    source.parent.mkdir(parents=True)
    target.parent.mkdir(parents=True)
    source.touch()
    target.touch()
    return source, target


def _save_excitation(target: Path, frames: int = 8, coded_ap: bool = False):
    path = excitation_path_for(target)
    path.parent.mkdir(parents=True)
    values = np.arange(frames, dtype=np.float32)
    arrays = {
        "f0_hz": values + 100,
        "log_f0": values,
        "vuv": np.ones(frames, dtype=np.float32),
        "energy": values + 1,
        "log_energy": values,
    }
    if coded_ap:
        arrays["coded_ap"] = values[:, None]
    np.savez(path, **arrays)
    return path


def test_excitation_crop_and_collate_align_with_audio(tmp_path, monkeypatch):
    pair = _pair(tmp_path)
    _save_excitation(pair[1], frames=8, coded_ap=True)
    arrays = {
        pair[0]: np.arange(8 * 4, dtype=np.float32),
        pair[1]: np.arange(8 * 4, dtype=np.float32) + 100,
    }
    monkeypatch.setattr(parallel_audio, "load_audio", lambda path, rate: arrays[path])
    monkeypatch.setattr(parallel_audio.random, "randint", lambda start, end: 2)
    dataset = ParallelAudioDataset(
        [pair],
        segment_size=12,
        sampling_rate=16,
        random_crop=True,
        return_excitation=True,
        excitation_hop_length=4,
        excitation_required=True,
    )
    source, target, _, excitation = dataset[0]
    torch.testing.assert_close(source, torch.arange(8, 20).float())
    torch.testing.assert_close(excitation["log_f0"], torch.arange(2, 5).float())
    batch = parallel_audio_collate([(source, target, "a", excitation)])
    assert batch[3]["f0_hz"].shape == (1, 3)
    assert batch[3]["coded_ap"].shape == (1, 3, 1)
    assert batch[3]["frame_mask"].all()


def test_optional_missing_excitation_disables_whole_batch(tmp_path, monkeypatch):
    first = _pair(tmp_path / "first")
    second = _pair(tmp_path / "second")
    _save_excitation(first[1], frames=4)
    arrays = {
        path: np.arange(16, dtype=np.float32)
        for path in (*first, *second)
    }
    monkeypatch.setattr(parallel_audio, "load_audio", lambda path, rate: arrays[path])
    dataset = ParallelAudioDataset(
        [first, second],
        segment_size=16,
        sampling_rate=16,
        random_crop=False,
        return_excitation=True,
        excitation_hop_length=4,
    )
    with pytest.warns(UserWarning):
        batch = parallel_audio_collate([dataset[0], dataset[1]])
    assert batch[3] is None


def test_preprocessing_helpers_and_validation_output(tmp_path, capsys):
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "preprocess_parallel_excitation.py"
    )
    spec = importlib.util.spec_from_file_location("preprocess_excitation", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    target = tmp_path / "speaker" / "wav" / "sample.wav"
    expected = tmp_path / "speaker" / "excitation" / "sample.npz"
    assert module.output_path(target) == expected
    rms = module.frame_rms(np.ones(7, dtype=np.float64), 4)
    assert rms.shape == (2,)
    expected.parent.mkdir(parents=True)
    np.savez(
        expected,
        f0_hz=np.array([0, 100, 120], dtype=np.float32),
        vuv=np.array([0, 1, 1], dtype=np.float32),
        energy=np.array([0.1, 0.2, 0.3], dtype=np.float32),
        log_energy=np.log(np.array([0.1, 0.2, 0.3], dtype=np.float32)),
    )
    module.validate_npz(expected)
    output = capsys.readouterr().out
    assert "voiced percentage" in output
    assert "100.00/110.00/120.00" in output
