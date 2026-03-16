from pytorch_lightning.callbacks import BasePredictionWriter
import torch
import os

class IndexWriter(BasePredictionWriter):
    def __init__(self, output_dir, write_interval="batch"):
        super().__init__(write_interval)
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def write_on_batch_end(self, trainer, pl_module, prediction, batch_indices, batch, batch_idx, dataloader_idx):
        # prediction is the output of system.predict_step
        # It contains {'hashes': ..., 'times': ..., 'id': ...} for a batch of songs
        
        # We will append to a file specific to this GPU rank
        rank = trainer.global_rank
        filename = os.path.join(self.output_dir, f"index_shard_{rank}.pt")
        
        # Load existing (or append mode logic - simplified here as Torch save/load)
        # For 100M files, real "append" is needed (e.g., streaming JSONL or appending to a list on disk).
        # PyTorch .pt files don't support appending easily.
        # Recommendation: Write 1 file per batch, or use a custom binary format.
        
        # For robust H200 scale: Write "batch_{rank}_{batch_idx}.pt"
        # Then a separate job merges them. This is the safest, lock-free way.
        
        batch_filename = os.path.join(self.output_dir, f"batch_{rank}_{batch_idx}.pt")
        torch.save(prediction, batch_filename)
