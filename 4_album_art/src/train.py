
import argparse
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import AlbumArtDataset, collate_skip_none
from model import VQVAE

# Configuration
MODEL_OUTPUT = "release/visual_encoder.pth"
CHECKPOINT_DIR = "/vol/checkpoints"

# Hyperparameters
LEARNING_RATE = 1e-3
NUM_HIDDENS = 128
NUM_RESIDUAL_LAYERS = 2
NUM_RESIDUAL_HIDDENS = 32
NUM_EMBEDDINGS = 1024
EMBEDDING_DIM = 64
COMMITMENT_COST = 0.25

def train(data_dir, manifest_path, epochs=10, batch_size=32, auto_batch=False, model_output_path=MODEL_OUTPUT):
    # 1. Setup Data
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    dataset = AlbumArtDataset(manifest_path, data_dir, transform=transform)
    print(f"Dataset size: {len(dataset)} albums")

    # 2. Setup Model
    model = VQVAE(NUM_HIDDENS, NUM_RESIDUAL_LAYERS, NUM_RESIDUAL_HIDDENS,
                  NUM_EMBEDDINGS, EMBEDDING_DIM, COMMITMENT_COST, LEARNING_RATE)

    # 3. Setup Trainer
    checkpoint_callback = ModelCheckpoint(
        dirpath=CHECKPOINT_DIR,
        filename='vqvae-{epoch:02d}-{train_loss:.2f}',
        save_top_k=2,
        monitor='train_loss'
    )

    trainer = pl.Trainer(
        accelerator='auto',
        devices=1,
        max_epochs=epochs,
        callbacks=[checkpoint_callback],
        enable_progress_bar=True,
        default_root_dir="/vol/logs"
    )

    # 4. Auto-Scaling Batch Size
    if auto_batch:
        print(">>> Tuning Batch Size...")
        tuner = pl.tuner.tuning.Tuner(trainer)
        
        # We need a DataModule for clean tuning
        class ArtDataModule(pl.LightningDataModule):
            def __init__(self, ds, bs):
                super().__init__()
                self.ds = ds
                self.batch_size = bs
            def train_dataloader(self):
                return DataLoader(self.ds, batch_size=self.batch_size, 
                                  shuffle=True, num_workers=4, collate_fn=collate_skip_none)

        dm = ArtDataModule(dataset, batch_size)
        tuner.scale_batch_size(model, datamodule=dm, mode="power")
        print(f">>> Found optimal batch size: {dm.batch_size}")
        batch_size = dm.batch_size
    
    # 5. Final Training
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, 
                        num_workers=4, collate_fn=collate_skip_none)
    
    print(f"Starting training with Batch Size: {batch_size}")
    trainer.fit(model, train_dataloaders=loader)
    
    # 6. Export Legacy Weights
    print(f"Exporting state dict to {model_output_path}...")
    torch.save(model.state_dict(), model_output_path)
    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="Path to covers folder")
    parser.add_argument("--manifest", type=str, required=True, help="Path to manifest CSV")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Initial batch size")
    parser.add_argument("--auto_batch", action="store_true", help="Enable auto-scaling batch size")
    
    args = parser.parse_args()
    train(args.data_dir, args.manifest, args.epochs, args.batch_size, args.auto_batch)
