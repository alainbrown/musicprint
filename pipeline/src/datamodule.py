import pytorch_lightning as pl
from data.dali_loader import DALIGPULoader
import torch
import os
import glob
import random

class MusicDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size=32, val_split=0.05):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.val_split = val_split
        
        # Scrape files once to ensure train/val don't overlap
        self.all_files = sorted(glob.glob(os.path.join(data_dir, "**/*.mp3"), recursive=True) + 
                               glob.glob(os.path.join(data_dir, "**/*.wav"), recursive=True))
        
        # Shuffle with a fixed seed for reproducibility
        random.seed(42)
        random.shuffle(self.all_files)
        
        split_idx = int(len(self.all_files) * (1 - val_split))
        self.train_files = self.all_files[:split_idx]
        self.val_files = self.all_files[split_idx:]
        
        # Create temp directory for split lists (DALI needs a file_root or file_list)
        # For simplicity with DALI fn.readers.file, we'll use a temporary file list if needed,
        # but the easiest way is to use fn.readers.file(file_list=...)
        # I will update dali_loader to accept a file_list.
        self._write_file_list("train_list.txt", self.train_files)
        self._write_file_list("val_list.txt", self.val_files)

    def _write_file_list(self, name, files):
        with open(name, "w") as f:
            for i, fp in enumerate(files):
                f.write(f"{fp} {i}\n") # DALI format: path label

    def train_dataloader(self):
        device_id = self.trainer.local_rank if self.trainer else 0
        shard_id = self.trainer.global_rank if self.trainer else 0
        num_shards = self.trainer.world_size if self.trainer else 1

        return DALIGPULoader(
            file_list="train_list.txt", # Use the split list
            batch_size=self.batch_size,
            device_id=device_id,
            shard_id=shard_id,
            num_shards=num_shards,
            augment=True
        )

    def val_dataloader(self):
        # Validation is usually clean and non-augmented
        device_id = self.trainer.local_rank if self.trainer else 0
        shard_id = self.trainer.global_rank if self.trainer else 0
        num_shards = self.trainer.world_size if self.trainer else 1

        if not self.val_files:
            return None

        return DALIGPULoader(
            file_list="val_list.txt",
            batch_size=self.batch_size,
            device_id=device_id,
            shard_id=shard_id,
            num_shards=num_shards,
            augment=False
        )
