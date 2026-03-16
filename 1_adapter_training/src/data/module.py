import pytorch_lightning as pl
import os
import glob
import random
from torch.utils.data import DataLoader
from .dataset import SongDataset, collate_songs


def discover_files(data_dir):
    for ext in ("*.flac", "*.wav", "*.mp3"):
        files = sorted(glob.glob(os.path.join(data_dir, "**", ext), recursive=True))
        if files:
            return files
    return []


def build_file_label_pairs(files, data_dir):
    pairs = []
    for idx, fp in enumerate(sorted(files)):
        pairs.append((fp, idx))
    return pairs


class MusicDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size=2, val_split=0.05):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size

        self.all_files = discover_files(data_dir)

        random.seed(42)
        shuffled = list(self.all_files)
        random.shuffle(shuffled)

        split_idx = int(len(shuffled) * (1 - val_split))
        self.train_pairs = build_file_label_pairs(shuffled[:split_idx], data_dir)
        self.val_pairs = build_file_label_pairs(shuffled[split_idx:], data_dir)

    def train_dataloader(self):
        ds = SongDataset(self.train_pairs)
        return DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=collate_songs,
        )

    def val_dataloader(self):
        if not self.val_pairs:
            return None
        ds = SongDataset(self.val_pairs)
        return DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=collate_songs,
        )
