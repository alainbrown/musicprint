import pytorch_lightning as pl
import torch
import torch.optim as optim
import torch.nn.functional as F
from models.arcface import MusicArcFaceSystem
import os

class MusicPrintSystem(pl.LightningModule):
    def __init__(self, num_classes, lr=1e-4, output_dim=64):
        super().__init__()
        self.save_hyperparameters()
        
        # Initialize ArcFace System (Backbone + Head)
        self.model = MusicArcFaceSystem(
            num_classes=num_classes,
            embedding_dim=output_dim
        )

    def forward(self, x):
        return self.model.backbone(x)

    def training_step(self, batch, batch_idx):
        # Handle Dual View output from DALI
        data_dict = batch[0]
        
        # ArcFace treats views as independent samples of the class
        a1 = data_dict["audio_1"]
        a2 = data_dict["audio_2"]
        labels = data_dict["label"].squeeze().long()
        
        # Concatenate: [B, T] + [B, T] -> [2*B, T]
        audio = torch.cat([a1, a2], dim=0)
        # Duplicate labels: [B] -> [2*B]
        targets = torch.cat([labels, labels], dim=0)
        
        # Squeeze channel if present
        if audio.dim() == 3:
            audio = audio.squeeze(-1)
            
        # Get embeddings from backbone
        embeddings = self(audio)
        
        # Calculate ArcFace Loss
        loss = self.model.get_loss(embeddings, targets)
        
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        # Validation is similar, but we might just use one view
        # Or use both to get better loss estimate
        data_dict = batch[0]
        
        if "audio" in data_dict:
            audio = data_dict["audio"]
        elif "audio_1" in data_dict:
            audio = data_dict["audio_1"]
        else:
            raise KeyError(f"Batch keys {data_dict.keys()} do not contain 'audio' or 'audio_1'")
            
        labels = data_dict["label"].squeeze().long()
        
        if audio.dim() == 3:
            audio = audio.squeeze(-1)
            
        embeddings = self(audio)
        loss = self.model.get_loss(embeddings, labels)
        
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        # We must optimize BOTH the backbone adapter AND the ArcFace head (loss_func parameters)
        # backbone.backbone is frozen inside MERTAdapter, so we filter for requires_grad
        params = [
            {'params': filter(lambda p: p.requires_grad, self.model.backbone.parameters())},
            {'params': self.model.loss_func.parameters()} 
        ]
        
        optimizer = optim.AdamW(params, lr=self.hparams.lr)
        return optimizer
