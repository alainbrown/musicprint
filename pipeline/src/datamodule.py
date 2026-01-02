import pytorch_lightning as pl
from data.dali_loader import DALIGPULoader
import torch

class MusicDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size=32):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size

    def train_dataloader(self):
        # When running on 8 GPUs, Lightning sets self.trainer.global_rank (0 to 7)
        # and self.trainer.world_size (8).
        if self.trainer:
            device_id = self.trainer.local_rank
            shard_id = self.trainer.global_rank
            num_shards = self.trainer.world_size
        else:
            device_id = 0
            shard_id = 0
            num_shards = 1

        return DALIGPULoader(
            file_root=self.data_dir,
            batch_size=self.batch_size,
            device_id=device_id,
            shard_id=shard_id,
            num_shards=num_shards
        )

    # DALI usually handles validation similarly, but for now we focus on train
    def val_dataloader(self):
        return None
