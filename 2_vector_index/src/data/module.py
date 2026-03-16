import pytorch_lightning as pl
import os
import glob
from isrc_utils import pack_isrc
from .loader import DALILoader
from .pipelines.inference import inference_pipeline

class MusicDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size=32, window_secs=5.0):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.window_secs = window_secs
        
        # Discover files
        self.all_files = sorted(glob.glob(os.path.join(data_dir, "**/*.flac"), recursive=True))
        if not self.all_files:
            self.all_files = sorted(glob.glob(os.path.join(data_dir, "**/*.wav"), recursive=True))
        if not self.all_files:
            self.all_files = sorted(glob.glob(os.path.join(data_dir, "**/*.mp3"), recursive=True))
        
        # Generate manifest for inference
        self.predict_list = os.path.join("/tmp", "predict_list.txt")
        self._write_list(self.predict_list, self.all_files)

    def _write_list(self, path, files):
        with open(path, "w") as f:
            for fp in files:
                # Use relative path to ensure uniqueness across subfolders
                rel_path = os.path.relpath(fp, self.data_dir)
                name = os.path.splitext(rel_path)[0]
                isrc = pack_isrc(name) if len(name) == 12 else hash(name) & 0xFFFFFFFFFFFFFFFF
                f.write(f"{fp} {isrc}\n")

    def predict_dataloader(self):
        if not self.all_files: return None
        return DALILoader(
            pipeline_fn=inference_pipeline,
            output_map=["audio", "label"],
            batch_size=self.batch_size,
            window_samples=int(self.window_secs * 24000),
            sample_rate=24000,
            file_list=self.predict_list,
            device_id=self.trainer.local_rank if self.trainer else 0,
            shard_id=self.trainer.global_rank if self.trainer else 0,
            num_shards=self.trainer.world_size if self.trainer else 1
        )
