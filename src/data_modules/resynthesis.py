# Adapted from: https://github.com/ashleve/lightning-hydra-template/blob/main/src/data/mnist_datamodule.py
from typing import Any
import random
from pathlib import Path
import librosa
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from librosa.util import normalize
from tqdm import tqdm

import pytorch_lightning as pl
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split

MAX_WAV_VALUE = 32768.0


def load_audio(full_path, sampling_rate=16000):
    data, sampling_rate = librosa.load(full_path, sr = sampling_rate)
    return data, sampling_rate


class ResynthesisDataset(Dataset):
    def __init__(
        self, 
        training_files,
        segment_size,
        code_hop_size, 
        sampling_rate,
        embed_type="xvector+ecapa",
        split=True
    ):

        self.audio_files = training_files
        self.segment_size = segment_size
        self.code_hop_size = code_hop_size
        self.sampling_rate = sampling_rate
        self.embed_type = embed_type
        self.split = split

        random.seed(1234)

    def _sample_interval(self, seqs, seq_len=None):
        N = max([v.shape[-1] for v in seqs])
        if seq_len is None:
            seq_len = self.segment_size if self.segment_size > 0 else N

        hops = [N // v.shape[-1] for v in seqs]
        lcm = np.lcm.reduce(hops)

        # Randomly pickup with the batch_max_steps length of the part
        interval_start = 0
        interval_end = N // lcm - seq_len // lcm

        start_step = random.randint(interval_start, interval_end)

        new_seqs = []
        for i, v in enumerate(seqs):
            start = start_step * (lcm // hops[i])
            end = (start_step + seq_len // lcm) * (lcm // hops[i])
            new_seqs += [v[..., start:end]]

        return new_seqs

    def __getitem__(self, index):
        wav_fpath = self.audio_files[index]
        spkr_embed_fpath = wav_fpath.parent.parent / "spkr" / f"{wav_fpath.stem}.spk.npy"
        pitch = wav_fpath.parent.parent / "pitch" / f"{wav_fpath.stem}.pit.npy"
        energy = wav_fpath.parent.parent / "energy" / f"{wav_fpath.stem}.eng.npy"
        tokens = wav_fpath.parent.parent / "code" / f"{wav_fpath.stem}.km"

        tokens = open(tokens, 'r').readlines()[0].strip()
        tokens = np.array([int(i) for i in tokens.split(" ")])

        spkr_embed = np.load(spkr_embed_fpath)
        spkr_embed_loss = np.load(spkr_embed_fpath)
        if self.embed_type == "xvector":
            spkr_embed = spkr_embed[:512]
            assert spkr_embed.shape[0] == 512
        elif self.embed_type == "ecapa":
            spkr_embed = spkr_embed[512:]
            assert spkr_embed.shape[0] == 192
        else:
            assert spkr_embed.shape[0] == 704
        spkr_embed = torch.FloatTensor(spkr_embed)

        pitch = np.load(pitch)
        energy = np.load(energy)


        audio, sampling_rate = load_audio(wav_fpath, self.sampling_rate)
        # if sampling_rate != self.sampling_rate:
        #     import resampy
        #     audio = resampy.resample(audio, sampling_rate, self.sampling_rate)

        # # audio = audio / MAX_WAV_VALUE
        # # audio = normalize(audio) * 0.95
        # audio = audio / (max(abs(audio)) + 0.00001) * 0.8

        # Trim audio ending
        code_length = min(audio.shape[0] // self.code_hop_size, tokens.shape[-1])
        tokens = tokens[:code_length]
        pitch = pitch[:code_length]
        energy = energy[:code_length]
        tokens = tokens[:code_length]        
        audio = audio[:code_length * self.code_hop_size]
        
        assert audio.shape[0] // self.code_hop_size == tokens.shape[-1], "Code audio mismatch"

        while audio.shape[0] < self.segment_size:
            audio = np.hstack([audio, audio])
            tokens = np.hstack([tokens, tokens])
            pitch = np.hstack([pitch, pitch])
            energy = np.hstack([energy, energy])

        audio = torch.FloatTensor(audio)
        audio = audio.unsqueeze(0)

        pitch = torch.FloatTensor(pitch)
        energy = torch.FloatTensor(energy)
        tokens = torch.tensor(tokens)

        assert audio.size(1) >= self.segment_size, "Padding not supported!!"
        
        audio, pitch, energy, tokens = self._sample_interval([audio, pitch, energy, tokens])

        return audio.squeeze(0), tokens, pitch, energy, spkr_embed, str(wav_fpath)

    def __len__(self):
        return len(self.audio_files)


class ResynthesisDataModule(pl.LightningDataModule):
    """Example of LightningDataModule for MNIST dataset.
    A DataModule implements 5 key methods:
        def prepare_data(self):
            # things to do on 1 GPU/TPU (not on every GPU/TPU in DDP)
            # download data, pre-process, split, save to disk, etc...
        def setup(self, stage):
            # things to do on every process in DDP
            # load data, set variables, etc...
        def train_dataloader(self):
            # return train dataloader
        def val_dataloader(self):
            # return validation dataloader
        def test_dataloader(self):
            # return test dataloader
        def teardown(self):
            # called on every process in DDP
            # clean up after fit or test
    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.
    Read the docs:
        https://lightning.ai/docs/pytorch/latest/data/datamodule.html
    """

    def __init__(
        self,
        data_dir: str = None,
        data_indictts: str = None,
        data_file: str | list = None,
        batch_size: int = 16,
        num_workers: int = 4,
        pin_memory: bool = True,
        segment_size: int = 20480,
        code_hop_size: int = 320,
        sampling_rate: int = 16000,
        embed_type: str = "xvector+ecapa",
        split: bool = True,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters()

        # data transformations
        # self.transforms = T.Compose([T.ToTensor(), T.Normalize((0.1307,), (0.3081,))])

        self.data_train: Dataset = None
        self.data_val: Dataset = None
        self.data_test: Dataset = None

    @property
    def num_classes(self):
        return self.hparams.num_codes

    def prepare_data(self):
        """Download data if needed.
        Do not use it to assign state (self.x = y).
        """
        # training_files = list(Path(self.hparams.data_dir).rglob("*.wav"))

        # self.training_files = training_files
        # # self.validation_files = training_files
        # # self.test_files = training_files
        # training_files, self.validation_files, _, _ = train_test_split(training_files, training_files, test_size=0.01, random_state=42)
        # self.training_files, self.test_files, _, _ = train_test_split(training_files, training_files, test_size=0.001, random_state=42)
        pass

    def setup(self, stage: str = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.
        This method is called by lightning with both `trainer.fit()` and `trainer.test()`, so be
        careful not to execute things like random split twice!
        """
        assert self.hparams.data_dir is not None or \
            self.hparams.data_indictts is not None or \
                self.hparams.data_file is not None, "Either data_dir, data_indictts or data_file must be provided"
                
        if self.hparams.data_dir is not None:
            training_files = list(Path(self.hparams.data_dir).rglob("*.wav"))
        else:
            training_files = []

        if self.hparams.data_indictts is not None:
            indic_tts_files = list(Path(self.hparams.data_indictts).rglob("**/english/wav/*.wav"))
            training_files.extend(indic_tts_files)
            
        if self.hparams.data_file is not None:
            if isinstance(self.hparams.data_file, str):
                self.hparams.data_file = [self.hparams.data_file]
            
            for file_path in self.hparams.data_file:
                with open(file_path, 'r') as f:
                    data_files = f.readlines()
                data_files = [Path(x.strip()) for x in data_files]
                training_files.extend(data_files)
            
        filtered_training_files = []
        for f in tqdm(training_files, desc="Filtering training files"):
            pitch_fpath = f.parent.parent / "pitch" / f"{f.stem}.pit.npy"
            energy_fpath = f.parent.parent / "energy" / f"{f.stem}.eng.npy"
            spkr_embed_fpath = f.parent.parent / "spkr" / f"{f.stem}.spk.npy"
            code_fpath = f.parent.parent / "code" / f"{f.stem}.km"
            if pitch_fpath.exists() and energy_fpath.exists() and spkr_embed_fpath.exists() and code_fpath.exists():
                filtered_training_files.append(f)
            
        print(f"Total files before filtering: {len(training_files)}")
        print(f"Total files after filtering: {len(filtered_training_files)}")

        self.training_files = filtered_training_files
        # self.validation_files = training_files
        # self.test_files = training_files
        self.training_files, self.validation_files, _, _ = train_test_split(self.training_files, self.training_files, test_size=0.001, random_state=42)
        self.training_files, self.test_files, _, _ = train_test_split(self.training_files, self.training_files, test_size=0.001, random_state=42)
        
        print(f"Total training files: {len(self.training_files)}")
        print(f"Total validation files: {len(self.validation_files)}")
        print(f"Total test files: {len(self.test_files)}")
        
        
        # load and split datasets only if not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            self.data_train = ResynthesisDataset(
                training_files=self.training_files,
                segment_size=self.hparams.segment_size,
                code_hop_size=self.hparams.code_hop_size,
                sampling_rate=self.hparams.sampling_rate,
                embed_type=self.hparams.embed_type,
                split=True,
            )
            self.data_val = ResynthesisDataset(
                training_files=self.validation_files,
                segment_size=self.hparams.segment_size,
                code_hop_size=self.hparams.code_hop_size,
                sampling_rate=self.hparams.sampling_rate,
                embed_type=self.hparams.embed_type,
                split=True,
            )
            self.data_test = ResynthesisDataset(
                training_files=self.test_files,
                segment_size=self.hparams.segment_size,
                code_hop_size=self.hparams.code_hop_size,
                sampling_rate=self.hparams.sampling_rate,
                embed_type=self.hparams.embed_type,
                split=False,
            )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def teardown(self, stage: str = None):
        """Clean up after fit or test."""
        pass

    def state_dict(self):
        """Extra things to save to checkpoint."""
        return {}

    def load_state_dict(self, state_dict: dict[str, Any]):
        """Things to do when loading checkpoint."""
        pass


if __name__ == "__main__":
    dm = ResynthesisDataModule()
    dm.prepare_data()
    dm.setup()
    for batch in dm.train_dataloader():
        print(batch[0].shape)
        print(batch[1].shape)
        break
