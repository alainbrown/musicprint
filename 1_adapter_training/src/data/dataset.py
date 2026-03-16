import os
import glob
import random
import torch
import torchaudio
from torch.utils.data import Dataset

SAMPLE_RATE = 24000
WINDOW_SAMPLES = 120000  # 5 seconds
STRIDE_SAMPLES = 24000   # 1 second


def load_audio(path):
    """Load audio, downmix to mono, resample to 24kHz."""
    audio, sr = torchaudio.load(path)
    if audio.shape[0] > 1:
        audio = audio.mean(dim=0, keepdim=True)
    if sr != SAMPLE_RATE:
        audio = torchaudio.transforms.Resample(sr, SAMPLE_RATE)(audio)
    return audio.squeeze(0)


def make_windows(audio):
    """Split audio into overlapping 5s windows with 1s stride."""
    if audio.shape[0] < WINDOW_SAMPLES:
        return [torch.nn.functional.pad(audio, (0, WINDOW_SAMPLES - audio.shape[0]))]
    windows = []
    start = 0
    while start + WINDOW_SAMPLES <= audio.shape[0]:
        windows.append(audio[start : start + WINDOW_SAMPLES])
        start += STRIDE_SAMPLES
    return windows


class SongWindowDataset(Dataset):
    """Returns all 5s windows for a song. Each item is (windows_tensor, label).

    windows_tensor shape: (num_windows, 120000)
    """

    def __init__(self, file_label_pairs):
        self.files = file_label_pairs

    def __getitem__(self, idx):
        path, label = self.files[idx]
        audio = load_audio(path)
        windows = make_windows(audio)
        return torch.stack(windows), torch.tensor(label, dtype=torch.long)

    def __len__(self):
        return len(self.files)


def collate_songs(batch):
    """Custom collation — each song has a different number of windows.

    Returns list of (windows_tensor, label) rather than stacking,
    since window counts vary per song.
    """
    return batch
