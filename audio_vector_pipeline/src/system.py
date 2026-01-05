import pytorch_lightning as pl
import torch
import torch.optim as optim
import torch.nn.functional as F
from models.mert_adapter import MERTAdapter
from models.loss import SupervisedContrastiveLoss

class MusicPrintSystem(pl.LightningModule):
    def __init__(self, lr=1e-4, output_dim=64, pq_path=None):
        super().__init__()
        self.save_hyperparameters()
        
        self.model = MERTAdapter(output_dim=output_dim)
        self.criterion = SupervisedContrastiveLoss(temperature=0.07)
        self.pq = None
        
        if pq_path and os.path.exists(pq_path):
            import faiss
            self.pq = faiss.read_ProductQuantizer(pq_path)

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
        labels = data_dict["label"]
        
        if audio_batch.dim() == 3:
            audio_batch = audio_batch.squeeze(-1)
            
        results = []
        
        for i in range(audio_batch.shape[0]):
            audio = audio_batch[i] # (Time,)
            song_id = labels[i].item()
            
            if audio.shape[0] < 120000:
                continue
                
            windows = audio.unfold(0, 120000, 24000) 
            
            with torch.no_grad():
                # Get continuous embeddings instead of binary hashes
                embeddings = self.model(windows) # (N, 64)
                
            # Normalize for cosine similarity
            embeddings_norm = F.normalize(embeddings, p=2, dim=1)
            embeddings_cpu = embeddings_norm.cpu()
            
            timestamps = [j * 1.0 for j in range(len(embeddings_norm))]
            
            selected_embeddings = []
            selected_times = []
            
            for j, e in enumerate(embeddings_cpu):
                if not selected_embeddings:
                    selected_embeddings.append(e)
                    selected_times.append(timestamps[j])
                    continue
                
                # Stack & Cosine Sim
                stack = torch.stack(selected_embeddings)
                sim = torch.matmul(stack, e) # both normalized
                
                if torch.max(sim) < 0.85: # Threshold for "different enough"
                    selected_embeddings.append(e)
                    selected_times.append(timestamps[j])
                    if len(selected_embeddings) >= 15: break
            
            final_embs = torch.stack(selected_embeddings).numpy()
            
            # Apply PQ quantization if available
            if self.pq is not None:
                # final_embs is (N, 64), pq.compute_codes returns (N, 8) uint8
                final_embs = self.pq.compute_codes(final_embs.astype('float32'))
            
            results.append({
                "id": song_id,
                "embeddings": final_embs,
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
