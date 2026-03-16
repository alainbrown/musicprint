import pytorch_lightning as pl
import torch
import torch.optim as optim
from models.arcface import MusicEmbeddingSystem, contrastive_loss


WINDOW_SAMPLES = 120000  # 5 seconds at 24kHz
STRIDE_SAMPLES = 24000   # 1 second stride
CHUNK_SIZE = 32          # Encode this many windows at a time


class MusicPrintSystem(pl.LightningModule):
    def __init__(self, num_classes=None, lr=1e-4, output_dim=768):
        super().__init__()
        self.save_hyperparameters()
        self.model = MusicEmbeddingSystem(embedding_dim=output_dim)

    def forward(self, x):
        return self.model(x)

    def _make_windows(self, audio):
        """Split raw audio into overlapping 5s windows."""
        if audio.shape[0] < WINDOW_SAMPLES:
            return torch.nn.functional.pad(audio, (0, WINDOW_SAMPLES - audio.shape[0])).unsqueeze(0)
        return audio.unfold(0, WINDOW_SAMPLES, STRIDE_SAMPLES)

    def _encode_windows(self, windows):
        """Encode windows in chunks to avoid OOM."""
        chunks = []
        for start in range(0, len(windows), CHUNK_SIZE):
            chunk = windows[start : start + CHUNK_SIZE]
            chunks.append(self(chunk))
        return torch.cat(chunks, dim=0)

    def _process_batch(self, batch):
        """Window, encode, and collect embeddings + labels for a batch of songs."""
        all_embeddings = []
        all_labels = []

        for audio, label in batch:
            audio = audio.to(self.device)
            windows = self._make_windows(audio)
            embs = self._encode_windows(windows)
            all_embeddings.append(embs)
            all_labels.append(label.expand(len(embs)))

        embeddings = torch.cat(all_embeddings, dim=0)
        labels = torch.cat(all_labels, dim=0).to(self.device)
        return embeddings, labels

    def training_step(self, batch, batch_idx):
        embeddings, labels = self._process_batch(batch)
        loss = contrastive_loss(embeddings, labels)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        embeddings, labels = self._process_batch(batch)
        loss = contrastive_loss(embeddings, labels)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        params = filter(lambda p: p.requires_grad, self.model.parameters())
        return optim.AdamW(params, lr=self.hparams.lr)
