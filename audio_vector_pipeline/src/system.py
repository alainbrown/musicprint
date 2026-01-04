import pytorch_lightning as pl
import torch
import torch.optim as optim
from models.mert_adapter import MERTAdapter
from models.loss import SupervisedContrastiveLoss

class MusicPrintSystem(pl.LightningModule):
    def __init__(self, lr=1e-4, output_dim=64):
        super().__init__()
        self.save_hyperparameters()
        
        self.model = MERTAdapter(output_dim=output_dim)
        self.criterion = SupervisedContrastiveLoss(temperature=0.07)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        # DALI output: [{'audio': tensor, 'label': tensor}]
        # Lightning auto-moves batch to GPU, but DALI already puts it there.
        # We just need to unpack.
        
        # NOTE: When using DALI iterator, the batch is a list of dicts
        data_dict = batch[0]
        audio = data_dict["audio"] # (B, Time, 1)
        labels = data_dict["label"].squeeze().long() # (B,)
        
        # Squeeze channel dim
        if audio.dim() == 3:
            audio = audio.squeeze(-1)
            
        embeddings = self(audio)
        loss = self.criterion(embeddings, labels)
        
        # Logs to TensorBoard/WandB automatically
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        data_dict = batch[0]
        audio = data_dict["audio"]
        labels = data_dict["label"].squeeze().long()
        
        if audio.dim() == 3:
            audio = audio.squeeze(-1)
            
        embeddings = self(audio)
        loss = self.criterion(embeddings, labels)
        
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def predict_step(self, batch, batch_idx):
        # Batch is (B, Time, 1) from DALI
        # We need to process each song in the batch
        
        data_dict = batch[0]
        audio_batch = data_dict["audio"] # (B, Time, 1)
        # DALI labels might be file indices; we ideally need filenames for IDs.
        # For now, we'll return the numeric label.
        labels = data_dict["label"]
        
        if audio_batch.dim() == 3:
            audio_batch = audio_batch.squeeze(-1)
            
        results = []
        
        # We process the batch song-by-song because adaptive indexing is variable-length output
        # (Some songs have 3 hashes, some have 10)
        for i in range(audio_batch.shape[0]):
            audio = audio_batch[i] # (Time,)
            song_id = labels[i].item()
            
            # --- Adaptive Indexing Logic ---
            # 1. Sliding Window (on GPU)
            # Create windows: 0-5s, 1-6s...
            # Stride=24000 (1s), Window=120000 (5s)
            
            # Unfold creates sliding windows efficiently
            # Input: (T,) -> (N_windows, Window_Size)
            # step=24000, size=120000
            if audio.shape[0] < 120000:
                continue
                
            windows = audio.unfold(0, 120000, 24000) 
            # Note: unfold might view same memory. MERT expects (Batch, Time).
            
            # 2. Bulk Inference
            # Get all hashes for the song at once
            with torch.no_grad():
                # windows is (N, 120000)
                # Ensure we don't blow up VRAM if song is super long (unlikely for 3-5min)
                hashes = self.model.get_hash(windows) # (N, 64)
                
            # 3. Greedy Deduplication (Sphere Method)
            # Perform on CPU to avoid complex boolean masking on GPU
            hashes_cpu = hashes.cpu()
            timestamps = [j * 1.0 for j in range(len(hashes_cpu))] # 1.0s stride
            
            selected_hashes = []
            selected_times = []
            
            for j, h in enumerate(hashes_cpu):
                if not selected_hashes:
                    selected_hashes.append(h)
                    selected_times.append(timestamps[j])
                    continue
                
                # Stack & Dist
                stack = torch.stack(selected_hashes)
                sim = torch.matmul(stack, h) / 64.0
                dist = 0.5 * (1.0 - sim)
                
                if torch.min(dist) > 0.15: # Threshold
                    selected_hashes.append(h)
                    selected_times.append(timestamps[j])
                    if len(selected_hashes) >= 10: break
            
            results.append({
                "id": song_id,
                "hashes": torch.stack(selected_hashes),
                "times": selected_times
            })
            
        return results

    def configure_optimizers(self):
        # Only optimize the adapter, not the frozen backbone
        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()), 
            lr=self.hparams.lr
        )
        return optimizer
