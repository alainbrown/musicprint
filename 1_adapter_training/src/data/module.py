import pytorch_lightning as pl
import os
import glob
import random
from collections import defaultdict
from torch.utils.data import DataLoader
from .dataset import ContrastiveAudioDataset, AudioDataset


def discover_song_windows(windows_dir):
    """Find all .pt window files and group by song index.

    Files are named {song_idx:06d}_{window_idx:04d}.pt.
    Returns list of (song_idx, [path1, path2, ...]) sorted by song_idx.
    """
    files = sorted(glob.glob(os.path.join(windows_dir, "*.pt")))
    songs = defaultdict(list)
    for fp in files:
        name = os.path.splitext(os.path.basename(fp))[0]
        parts = name.split("_")
        if len(parts) == 2:
            song_idx = int(parts[0])
            songs[song_idx].append(fp)

    return [(idx, songs[idx]) for idx in sorted(songs.keys())]


class MusicDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size=32, val_split=0.05, noise_dir=None):
        super().__init__()
        self.batch_size = batch_size
        self.noise_dir = noise_dir

        all_songs = discover_song_windows(data_dir)
        self.all_files = all_songs  # for len() compatibility with train.py

        random.seed(42)
        shuffled = list(all_songs)
        random.shuffle(shuffled)

        split_idx = int(len(shuffled) * (1 - val_split))
        self.train_songs = shuffled[:split_idx]
        self.val_songs = shuffled[split_idx:]

    def train_dataloader(self):
        ds = ContrastiveAudioDataset(self.train_songs, noise_dir=self.noise_dir)
        return DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=os.cpu_count(),
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=2,
        )

    def val_dataloader(self):
        if not self.val_songs:
            return None
        ds = AudioDataset(self.val_songs)
        return DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=os.cpu_count(),
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=2,
        )
