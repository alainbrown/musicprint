import pytorch_lightning as pl
from dali_loader import DALIGPULoader
import torch
import os
import glob
import random
from isrc_utils import pack_isrc

class MusicDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size=32, val_split=0.05, window_secs=5.0):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.val_split = val_split
        self.window_secs = window_secs
        
        # Scrape files
        # Priority: FLAC (Pre-processed) > WAV > MP3
        self.all_files = sorted(glob.glob(os.path.join(data_dir, "**/*.flac"), recursive=True))
        
        if not self.all_files:
            self.all_files = sorted(glob.glob(os.path.join(data_dir, "**/*.wav"), recursive=True))
            
        if not self.all_files:
            self.all_files = sorted(glob.glob(os.path.join(data_dir, "**/*.mp3"), recursive=True))
        
        # Shuffle with a fixed seed
        random.seed(42)
        random.shuffle(self.all_files)
        
        split_idx = int(len(self.all_files) * (1 - val_split))
        self.train_files = self.all_files[:split_idx]
        self.val_files = self.all_files[split_idx:]
        
        # Write file lists with ISRCs as labels
        self.train_list_path = os.path.join("/tmp", "train_list.txt")
        self.val_list_path = os.path.join("/tmp", "val_list.txt")
        
        self._write_file_list(self.train_list_path, self.train_files)
        self._write_file_list(self.val_list_path, self.val_files)

    def _write_file_list(self, name, files):
        with open(name, "w") as f:
            for fp in files:
                # Extract ISRC from filename (e.g., /path/to/USRC11234567.wav)
                filename = os.path.basename(fp).split('.')[0]
                # If filename is 12 chars, assume it's an ISRC
                if len(filename) == 12:
                    isrc_val = pack_isrc(filename)
                else:
                    # Fallback for testing with sample1.wav
                    # We'll just hash the filename for a stable numeric ID
                    isrc_val = hash(filename) & 0xFFFFFFFFFFFFFFFF
                
                f.write(f"{fp} {isrc_val}\n") # DALI format: path label

    def train_dataloader(self):
        device_id = self.trainer.local_rank if self.trainer else 0
        shard_id = self.trainer.global_rank if self.trainer else 0
        num_shards = self.trainer.world_size if self.trainer else 1

        return DALIGPULoader(
            file_list=self.train_list_path, # Use the split list
            batch_size=self.batch_size,
            device_id=device_id,
            shard_id=shard_id,
            num_shards=num_shards,
            window_secs=self.window_secs,
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
            file_list=self.val_list_path,
            batch_size=self.batch_size,
            device_id=device_id,
            shard_id=shard_id,
            num_shards=num_shards,
            window_secs=self.window_secs,
            augment=False
        )
