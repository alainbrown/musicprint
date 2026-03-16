import pytorch_lightning as pl
import torch
import torch.optim as optim
from models.arcface import MusicEmbeddingSystem, contrastive_loss


CHUNK_SIZE = 32  # Process windows through MERT in chunks to avoid OOM


class MusicPrintSystem(pl.LightningModule):
    def __init__(self, num_classes=None, lr=1e-4, output_dim=768):
        super().__init__()
        self.save_hyperparameters()
        self.model = MusicEmbeddingSystem(embedding_dim=output_dim)

    def forward(self, x):
        return self.model(x)

    def _encode_windows(self, windows):
        """Encode windows in chunks to avoid OOM. Returns (num_windows, D)."""
        chunks = []
        for start in range(0, len(windows), CHUNK_SIZE):
            chunk = windows[start : start + CHUNK_SIZE]
            with torch.no_grad():
                # Frozen MERT doesn't need gradients
                pass
            chunks.append(self(chunk))
        return torch.cat(chunks, dim=0)

    def training_step(self, batch, batch_idx):
        # batch is a list of (windows_tensor, label) from collate_songs
        all_embeddings = []
        all_labels = []

        for windows, label in batch:
            windows = windows.to(self.device)
            embs = self._encode_windows(windows)  # (num_windows, D)
            all_embeddings.append(embs)
            all_labels.append(label.expand(len(embs)))

        embeddings = torch.cat(all_embeddings, dim=0)
        labels = torch.cat(all_labels, dim=0).to(self.device)

        loss = contrastive_loss(embeddings, labels)

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        all_embeddings = []
        all_labels = []

        for windows, label in batch:
            windows = windows.to(self.device)
            embs = self._encode_windows(windows)
            all_embeddings.append(embs)
            all_labels.append(label.expand(len(embs)))

        embeddings = torch.cat(all_embeddings, dim=0)
        labels = torch.cat(all_labels, dim=0).to(self.device)

        loss = contrastive_loss(embeddings, labels)

        self.log("val_loss", loss, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        params = filter(lambda p: p.requires_grad, self.model.parameters())
        return optim.AdamW(params, lr=self.hparams.lr)
