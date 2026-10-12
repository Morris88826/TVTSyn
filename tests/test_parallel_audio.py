from pathlib import Path

import numpy as np
import pytest
import torch

from training.data_modules import parallel_audio
from training.data_modules.parallel_audio import (
    ParallelAudioDataModule,
    ParallelAudioDataset,
    read_parallel_manifests,
)


def _touch_pair(root: Path, index: int):
    source = root / f"source_{index}.wav"
    target = root / f"target_{index}.wav"
    source.touch()
    target.touch()
    return source, target


def test_manifest_parsing_supports_blank_and_relative_paths(tmp_path):
    source, target = _touch_pair(tmp_path, 0)
    manifest = tmp_path / "pairs.txt"
    manifest.write_text(f"\n{source.name}|{target.name}\n")
    assert read_parallel_manifests(str(manifest)) == [(source, target)]


def test_manifest_reports_malformed_and_missing_files(tmp_path):
    malformed = tmp_path / "malformed.txt"
    malformed.write_text("only_one_path.wav\n")
    with pytest.raises(ValueError, match="malformed.txt:1"):
        read_parallel_manifests(str(malformed))

    missing = tmp_path / "missing.txt"
    missing.write_text("missing_source.wav|missing_target.wav\n")
    with pytest.raises(FileNotFoundError, match="missing.txt:1"):
        read_parallel_manifests(str(missing))


def test_dataset_trims_common_length_and_uses_shared_center_crop(tmp_path, monkeypatch):
    pair = _touch_pair(tmp_path, 0)
    arrays = {
        pair[0]: np.arange(12, dtype=np.float32),
        pair[1]: np.arange(10, dtype=np.float32) + 100,
    }
    monkeypatch.setattr(parallel_audio, "load_audio", lambda path, rate: arrays[path])
    dataset = ParallelAudioDataset([pair], segment_size=6, random_crop=False)
    source, target, _ = dataset[0]
    torch.testing.assert_close(source, torch.arange(2, 8).float())
    torch.testing.assert_close(target - source, torch.full((6,), 100.0))


def test_dataset_repeats_short_pairs_and_shares_random_offset(tmp_path, monkeypatch):
    pair = _touch_pair(tmp_path, 0)
    arrays = {
        pair[0]: np.arange(5, dtype=np.float32),
        pair[1]: np.arange(5, dtype=np.float32) + 10,
    }
    monkeypatch.setattr(parallel_audio, "load_audio", lambda path, rate: arrays[path])
    monkeypatch.setattr(parallel_audio.random, "randint", lambda start, end: 1)
    dataset = ParallelAudioDataset([pair], segment_size=8, random_crop=True)
    source, target, _ = dataset[0]
    assert source.shape == target.shape == (8,)
    torch.testing.assert_close(target - source, torch.full((8,), 10.0))


def test_datamodule_split_is_configurable_and_deterministic(tmp_path):
    pairs = [_touch_pair(tmp_path, index) for index in range(10)]
    manifest = tmp_path / "pairs.txt"
    manifest.write_text("\n".join(f"{source}|{target}" for source, target in pairs))
    kwargs = dict(
        data_file=str(manifest),
        batch_size=2,
        num_workers=0,
        segment_size=320,
        validation_split=0.2,
        test_split=0.2,
        split_seed=7,
    )
    first = ParallelAudioDataModule(**kwargs)
    second = ParallelAudioDataModule(**kwargs)
    first.setup()
    second.setup()
    assert len(first.training_pairs) == 6
    assert len(first.validation_pairs) == 2
    assert len(first.test_pairs) == 2
    assert first.training_pairs == second.training_pairs
    assert first.validation_pairs == second.validation_pairs
    assert first.test_pairs == second.test_pairs


def test_datamodule_validates_frame_aligned_segments():
    with pytest.raises(ValueError, match="divisible by 320"):
        ParallelAudioDataModule(data_file="unused", segment_size=321)
