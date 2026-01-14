import pytorch_lightning as pl
import os
import glob
import random
from isrc_utils import pack_isrc
from .loader import DALILoader
from .pipelines.contrastive import contrastive_pipeline
from .pipelines.inference import inference_pipeline

class MusicDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size=32, val_split=0.05, window_secs=5.0):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.val_split = val_split
        self.window_secs = window_secs
        
        # Discover files
        self.all_files = sorted(glob.glob(os.path.join(data_dir, "**/*.flac"), recursive=True))
        if not self.all_files:
            self.all_files = sorted(glob.glob(os.path.join(data_dir, "**/*.wav"), recursive=True))
        if not self.all_files:
            self.all_files = sorted(glob.glob(os.path.join(data_dir, "**/*.mp3"), recursive=True))
        
        random.seed(42)
        random.shuffle(self.all_files)
        
        split_idx = int(len(self.all_files) * (1 - val_split))
        self.train_files = self.all_files[:split_idx]
        self.val_files = self.all_files[split_idx:]
        
        # Generate manifests
        self.train_list = os.path.join("/tmp", "train_list.txt")
        self.val_list = os.path.join("/tmp", "val_list.txt")
        self._write_list(self.train_list, self.train_files)
        self._write_list(self.val_list, self.val_files)

    def _write_list(self, path, files):
        with open(path, "w") as f:
            for fp in files:
                # Use relative path to ensure uniqueness across subfolders (e.g. 1920/01.mp3 vs 1930/01.mp3)
                rel_path = os.path.relpath(fp, self.data_dir)
                name = os.path.splitext(rel_path)[0]
                isrc = pack_isrc(name) if len(name) == 12 else hash(name) & 0xFFFFFFFFFFFFFFFF
                f.write(f"{fp} {isrc}\n")

    def train_dataloader(self):
        return DALILoader(
            pipeline_fn=contrastive_pipeline,
            output_map=["audio_1", "audio_2", "label"],
            batch_size=self.batch_size,
            window_samples=int(self.window_secs * 24000),
            sample_rate=24000,
            file_list=self.train_list,
            device_id=self.trainer.local_rank if self.trainer else 0,
            shard_id=self.trainer.global_rank if self.trainer else 0,
            num_shards=self.trainer.world_size if self.trainer else 1
        )

    def val_dataloader(self):
        if not self.val_files: return None
        return DALILoader(
            pipeline_fn=inference_pipeline,
            output_map=["audio", "label"],
            batch_size=self.batch_size,
            window_samples=int(self.window_secs * 24000),
            sample_rate=24000,
            file_list=self.val_list,
            device_id=self.trainer.local_rank if self.trainer else 0,
            shard_id=self.trainer.global_rank if self.trainer else 0,
            num_shards=self.trainer.world_size if self.trainer else 1
        )
