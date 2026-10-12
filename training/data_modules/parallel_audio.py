"""Aligned parallel-audio dataset with optional target excitation tracks."""

from pathlib import Path
import math
import random
from typing import Any, Iterable
import warnings

import librosa
import numpy as np
import pytorch_lightning as pl
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, default_collate


AudioPair = tuple[Path, Path]
_WARNED_MISSING_EXCITATION = False


def load_audio(path: Path, sampling_rate: int) -> np.ndarray:
    audio, _ = librosa.load(path, sr=sampling_rate, mono=True)
    return np.asarray(audio, dtype=np.float32)


def excitation_path_for(
    target_path: Path, directory_name: str = "excitation", extension: str = ".npz"
) -> Path:
    return target_path.parent.parent / directory_name / f"{target_path.stem}{extension}"


def read_parallel_manifests(data_file: str | list[str]) -> list[AudioPair]:
    manifest_paths = [data_file] if isinstance(data_file, str) else list(data_file)
    if not manifest_paths:
        raise ValueError("At least one parallel audio manifest is required")

    pairs: list[AudioPair] = []
    for manifest_value in manifest_paths:
        manifest = Path(manifest_value)
        if not manifest.is_file():
            raise FileNotFoundError(f"Parallel audio manifest not found: {manifest}")
        with manifest.open("r") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                line = raw_line.strip()
                if not line:
                    continue
                parts = [part.strip() for part in line.split("|")]
                if len(parts) != 2 or not all(parts):
                    raise ValueError(
                        f"{manifest}:{line_number}: expected 'source.wav|target.wav'"
                    )
                source, target = Path(parts[0]), Path(parts[1])
                if not source.is_absolute():
                    source = manifest.parent / source
                if not target.is_absolute():
                    target = manifest.parent / target
                if not source.is_file():
                    raise FileNotFoundError(
                        f"{manifest}:{line_number}: source audio not found: {source}"
                    )
                if not target.is_file():
                    raise FileNotFoundError(
                        f"{manifest}:{line_number}: target audio not found: {target}"
                    )
                pairs.append((source, target))
    if not pairs:
        raise ValueError("Parallel audio manifests did not contain any pairs")
    return pairs


def _load_excitation(path: Path) -> dict[str, torch.Tensor | str]:
    with np.load(path, allow_pickle=False) as data:
        required = ("f0_hz", "log_f0", "vuv", "energy", "log_energy")
        missing = [name for name in required if name not in data]
        if missing:
            raise ValueError(f"{path}: missing excitation arrays: {', '.join(missing)}")
        output: dict[str, torch.Tensor | str] = {
            name: torch.from_numpy(np.asarray(data[name], dtype=np.float32))
            for name in required
        }
        if "coded_ap" in data:
            output["coded_ap"] = torch.from_numpy(
                np.asarray(data["coded_ap"], dtype=np.float32)
            )
    output["excitation_path"] = str(path)
    return output


def parallel_audio_collate(batch):
    if len(batch[0]) == 3:
        return default_collate(batch)
    sources, targets, names, excitations = zip(*batch)
    audio = (default_collate(sources), default_collate(targets), list(names))
    if any(excitation is None for excitation in excitations):
        return (*audio, None)

    tensor_keys = ["f0_hz", "log_f0", "vuv", "energy", "log_energy"]
    collated: dict[str, Any] = {}
    lengths = torch.tensor([excitation["f0_hz"].shape[0] for excitation in excitations])
    for key in tensor_keys:
        collated[key] = pad_sequence(
            [excitation[key] for excitation in excitations], batch_first=True
        )
    max_frames = int(lengths.max())
    if all("frame_mask" in excitation for excitation in excitations):
        collated["frame_mask"] = pad_sequence(
            [excitation["frame_mask"] for excitation in excitations],
            batch_first=True,
        ).bool()
    else:
        collated["frame_mask"] = (
            torch.arange(max_frames)[None] < lengths[:, None]
        )
    if all("coded_ap" in excitation for excitation in excitations):
        collated["coded_ap"] = pad_sequence(
            [excitation["coded_ap"] for excitation in excitations], batch_first=True
        )
    collated["excitation_path"] = [
        excitation["excitation_path"] for excitation in excitations
    ]
    return (*audio, collated)


class ParallelAudioDataset(Dataset):
    def __init__(
        self,
        audio_pairs: Iterable[AudioPair],
        segment_size: int,
        sampling_rate: int = 16000,
        random_crop: bool = True,
        return_excitation: bool = False,
        excitation_dir_name: str = "excitation",
        excitation_ext: str = ".npz",
        excitation_required: bool = False,
        excitation_hop_length: int = 320,
    ):
        self.audio_pairs = list(audio_pairs)
        self.segment_size = segment_size
        self.sampling_rate = sampling_rate
        self.random_crop = random_crop
        self.return_excitation = return_excitation
        self.excitation_dir_name = excitation_dir_name
        self.excitation_ext = excitation_ext
        self.excitation_required = excitation_required
        self.excitation_hop_length = excitation_hop_length
        if segment_size <= 0:
            raise ValueError("segment_size must be positive")
        if return_excitation and segment_size % excitation_hop_length:
            raise ValueError("segment_size must be divisible by excitation_hop_length")

    def __len__(self) -> int:
        return len(self.audio_pairs)

    def _excitation(self, target_path: Path):
        global _WARNED_MISSING_EXCITATION
        if not self.return_excitation:
            return None
        path = excitation_path_for(
            target_path, self.excitation_dir_name, self.excitation_ext
        )
        if path.is_file():
            return _load_excitation(path)
        if self.excitation_required:
            raise FileNotFoundError(f"Target excitation file not found: {path}")
        if not _WARNED_MISSING_EXCITATION:
            warnings.warn(
                f"Target excitation file not found; batch excitation will be disabled: {path}"
            )
            _WARNED_MISSING_EXCITATION = True
        return None

    def __getitem__(self, index: int):
        source_path, target_path = self.audio_pairs[index]
        source = load_audio(source_path, self.sampling_rate)
        target = load_audio(target_path, self.sampling_rate)
        excitation = self._excitation(target_path)
        common_length = min(source.shape[-1], target.shape[-1])
        if common_length == 0:
            raise ValueError(f"Empty aligned audio pair: {source_path} | {target_path}")
        source, target = source[:common_length], target[:common_length]

        repeats = 1
        if common_length < self.segment_size:
            repeats = math.ceil(self.segment_size / common_length)
            source, target = np.tile(source, repeats), np.tile(target, repeats)
            common_length = source.shape[-1]
            if excitation is not None:
                for key, value in list(excitation.items()):
                    if isinstance(value, torch.Tensor):
                        excitation[key] = value.repeat((repeats,) + (1,) * (value.dim() - 1))

        max_start = common_length - self.segment_size
        if self.return_excitation:
            max_start_frame = max_start // self.excitation_hop_length
            start_frame = (
                random.randint(0, max_start_frame)
                if self.random_crop
                else max_start_frame // 2
            )
            start = start_frame * self.excitation_hop_length
        else:
            start = random.randint(0, max_start) if self.random_crop else max_start // 2
            start_frame = 0
        end = start + self.segment_size
        source_segment = torch.from_numpy(source[start:end].copy()).float()
        target_segment = torch.from_numpy(target[start:end].copy()).float()

        if not self.return_excitation:
            return source_segment, target_segment, str(source_path)
        if excitation is not None:
            num_frames = self.segment_size // self.excitation_hop_length
            valid_frames = min(
                num_frames, max(0, excitation["f0_hz"].shape[0] - start_frame)
            )
            for key, value in list(excitation.items()):
                if not isinstance(value, torch.Tensor):
                    continue
                cropped = value[start_frame : start_frame + num_frames]
                if cropped.shape[0] < num_frames:
                    pad_shape = (num_frames - cropped.shape[0], *cropped.shape[1:])
                    cropped = torch.cat([cropped, cropped.new_zeros(pad_shape)], dim=0)
                excitation[key] = cropped
            excitation["frame_mask"] = (
                torch.arange(num_frames) < valid_frames
            )
        return source_segment, target_segment, str(source_path), excitation


class ParallelAudioDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_file: str | list[str],
        batch_size: int = 16,
        num_workers: int = 4,
        pin_memory: bool = True,
        segment_size: int = 48000,
        sampling_rate: int = 16000,
        frame_rate: int = 50,
        validation_split: float = 0.001,
        test_split: float = 0.001,
        split_seed: int = 42,
        return_excitation: bool = False,
        excitation_dir_name: str = "excitation",
        excitation_ext: str = ".npz",
        excitation_required: bool = False,
        excitation_hop_length: int = 320,
    ):
        super().__init__()
        if sampling_rate % frame_rate:
            raise ValueError("sampling_rate must be divisible by frame_rate")
        hop_length = sampling_rate // frame_rate
        if segment_size <= 0 or segment_size % hop_length:
            raise ValueError(f"segment_size must be positive and divisible by {hop_length}")
        if return_excitation and excitation_hop_length != hop_length:
            raise ValueError("excitation_hop_length must equal sampling_rate / frame_rate")
        if validation_split < 0 or test_split < 0:
            raise ValueError("validation_split and test_split must be non-negative")
        if validation_split + test_split >= 1:
            raise ValueError("validation_split + test_split must be less than 1")
        self.save_hyperparameters()
        self.data_train: Dataset | None = None
        self.data_val: Dataset | None = None
        self.data_test: Dataset | None = None
        self.training_pairs: list[AudioPair] = []
        self.validation_pairs: list[AudioPair] = []
        self.test_pairs: list[AudioPair] = []

    @staticmethod
    def _split_count(total: int, ratio: float) -> int:
        return 0 if ratio == 0 else max(1, int(round(total * ratio)))

    def setup(self, stage: str | None = None):
        if self.data_train is not None:
            return
        pairs = read_parallel_manifests(self.hparams.data_file)
        rng = random.Random(self.hparams.split_seed)
        rng.shuffle(pairs)
        val_count = self._split_count(len(pairs), self.hparams.validation_split)
        test_count = self._split_count(len(pairs), self.hparams.test_split)
        if val_count + test_count >= len(pairs):
            raise ValueError("Parallel dataset is too small for configured splits")
        self.validation_pairs = pairs[:val_count]
        self.test_pairs = pairs[val_count : val_count + test_count]
        self.training_pairs = pairs[val_count + test_count :]
        common = {
            "segment_size": self.hparams.segment_size,
            "sampling_rate": self.hparams.sampling_rate,
            "return_excitation": self.hparams.return_excitation,
            "excitation_dir_name": self.hparams.excitation_dir_name,
            "excitation_ext": self.hparams.excitation_ext,
            "excitation_required": self.hparams.excitation_required,
            "excitation_hop_length": self.hparams.excitation_hop_length,
        }
        self.data_train = ParallelAudioDataset(self.training_pairs, random_crop=True, **common)
        self.data_val = ParallelAudioDataset(self.validation_pairs, random_crop=False, **common)
        self.data_test = ParallelAudioDataset(self.test_pairs, random_crop=False, **common)

    def _loader(self, dataset, shuffle):
        return DataLoader(
            dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=shuffle,
            collate_fn=parallel_audio_collate if self.hparams.return_excitation else None,
        )

    def train_dataloader(self):
        return self._loader(self.data_train, True)

    def val_dataloader(self):
        return self._loader(self.data_val, False)

    def test_dataloader(self):
        return self._loader(self.data_test, False)

    def state_dict(self) -> dict[str, Any]:
        return {}

    def load_state_dict(self, state_dict: dict[str, Any]):
        pass
