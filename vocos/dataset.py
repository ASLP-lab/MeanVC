from dataclasses import dataclass

import os
import numpy as np
import torch
import torchaudio
import librosa
from pytorch_lightning import LightningDataModule
from torch.utils.data import Dataset, DataLoader

torch.set_num_threads(1)


@dataclass
class DataConfig:
    filelist_path: str
    sampling_rate: int
    num_samples: int
    batch_size: int
    num_workers: int
    hop_size: int
    condition_window: int


class VocosDataModule(LightningDataModule):
    def __init__(self, train_params: DataConfig, val_params: DataConfig):
        super().__init__()
        self.train_config = train_params
        self.val_config = val_params

    def _get_dataloder(self, cfg: DataConfig, train: bool):
        dataset = VocosDataset_wmel(cfg, train=train)
        dataloader = DataLoader(
            dataset, batch_size=cfg.batch_size, num_workers=cfg.num_workers, shuffle=train, pin_memory=True,
        )
        return dataloader

    def train_dataloader(self) -> DataLoader:
        return self._get_dataloder(self.train_config, train=True)

    def val_dataloader(self) -> DataLoader:
        return self._get_dataloder(self.val_config, train=False)


class VocosDataset(Dataset):
    def __init__(self, cfg: DataConfig, train: bool):
        with open(cfg.filelist_path) as f:
            self.filelist = f.read().splitlines()
        self.sampling_rate = cfg.sampling_rate
        self.num_samples = cfg.num_samples
        self.train = train

    def __len__(self) -> int:
        return len(self.filelist)

    def __getitem__(self, index: int) -> torch.Tensor:
        audio_path = self.filelist[index]
        y, sr = torchaudio.load(audio_path)
        if y.size(0) > 1:
            # mix to mono
            y = y.mean(dim=0, keepdim=True)
        gain = np.random.uniform(-1, -6) if self.train else -3
        y, _ = torchaudio.sox_effects.apply_effects_tensor(y, sr, [["norm", f"{gain:.2f}"]])
        if sr != self.sampling_rate:
            y = torchaudio.functional.resample(y, orig_freq=sr, new_freq=self.sampling_rate)
        if y.size(-1) < self.num_samples:
            pad_length = self.num_samples - y.size(-1)
            padding_tensor = y.repeat(1, 1 + pad_length // y.size(-1))
            y = torch.cat((y, padding_tensor[:, :pad_length]), dim=1)
        elif self.train:
            start = np.random.randint(low=0, high=y.size(-1) - self.num_samples + 1)
            y = y[:, start : start + self.num_samples]
        else:
            # During validation, take always the first segment for determinism
            y = y[:, : self.num_samples]

        return y[0]



class VocosDataset_wmel(Dataset):
    def __init__(self, cfg: DataConfig, train: bool):
        with open(cfg.filelist_path) as f:
            self.filelist = f.read().splitlines()
        self.sampling_rate = cfg.sampling_rate
        self.num_samples = cfg.num_samples
        self.hop_size = cfg.hop_size
        self.train = train
        self.condition_window = cfg.condition_window
        

    def __len__(self) -> int:
        return len(self.filelist)

    def __getitem__(self, index: int) -> torch.Tensor:
        audio_path, mel_path = self.filelist[index].split('|')
        mel = np.load(mel_path)
        utt = os.path.splitext(os.path.basename(mel_path))[0]
        # if "large_data_10ms_wpcpc_all_ft_large" in mel_path:
        if "large_data_10ms_wpcpc_all_ft_handpicked_large" in mel_path:
        # if utt.endswith("gta"):
            mel = mel.T / 4
        # gta
        # mel = mel.T / 4
        y, sr = torchaudio.load(audio_path)
        if y.size(0) > 1:
            # mix to mono
            y = y.mean(dim=0, keepdim=True)
        gain = np.random.uniform(-1, -6) if self.train else -3
        y, _ = torchaudio.sox_effects.apply_effects_tensor(y, sr, [["norm", f"{gain:.2f}"]])
        mel = torch.tensor(mel)
        if sr != self.sampling_rate:
            y = torchaudio.functional.resample(y, orig_freq=sr, new_freq=self.sampling_rate)
        if y.size(-1) <= self.num_samples or mel.shape[-1] <= self.condition_window:

            pad_length = self.num_samples - y.size(-1)
            padding_tensor = y.repeat(1, 1 + pad_length // y.size(-1))
            y = torch.cat((y, padding_tensor[:, :pad_length]), dim=1)
            
            pad_mel_length = self.condition_window - mel.shape[-1]
            padding_mel_tensor = mel.repeat(1, 1 + pad_mel_length // mel.size(-1))
            mel = torch.cat((mel, padding_mel_tensor[:, :pad_mel_length]), dim=1)
        elif self.train:
            if mel.shape[-1] - self.condition_window - 1 <= 0:
                eta = 0
            else:
                eta = np.random.randint(0, mel.shape[-1] - self.condition_window - 1)
            # start = np.random.randint(low=0, high=y.size(-1) - self.num_samples + 1)
            mel = mel[:,eta:eta+self.condition_window]
            y = y[:, eta * self.hop_size: eta * self.hop_size + self.num_samples]
        else:
            # During validation, take always the first segment for determinism
            y = y[:, : self.num_samples]
            mel = mel[:,:self.condition_window]
        if y[0].shape[-1] != 32000:
            return self.__getitem__(index+1)
        return (y[0], mel)
