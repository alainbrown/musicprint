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

    def configure_optimizers(self):
        # Only optimize the adapter, not the frozen backbone
        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()), 
            lr=self.hparams.lr
        )
        return optimizer
