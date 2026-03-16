import pytorch_lightning as pl
import os
import glob
from torch.utils.data import DataLoader
from isrc_utils import pack_isrc
from .dataset import InferenceAudioDataset, collate_variable_length


def discover_files(data_dir):
    for ext in ("*.flac", "*.wav", "*.mp3"):
        files = sorted(glob.glob(os.path.join(data_dir, "**", ext), recursive=True))
        if files:
            return files
    return []


def build_file_label_pairs(files, data_dir):
    pairs = []
    for fp in files:
        rel_path = os.path.relpath(fp, data_dir)
        name = os.path.splitext(rel_path)[0]
        label = pack_isrc(name) if len(name) == 12 else hash(name) & 0xFFFFFFFFFFFFFFFF
        pairs.append((fp, label))
    return pairs


class MusicDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size=1):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.all_files = discover_files(data_dir)
        self.pairs = build_file_label_pairs(self.all_files, data_dir)

    def predict_dataloader(self):
        if not self.pairs:
            return None
        ds = InferenceAudioDataset(self.pairs)
        return DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=os.cpu_count(),
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=2,
            collate_fn=collate_variable_length,
        )
