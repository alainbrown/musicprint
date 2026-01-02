import argparse
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from system import MusicPrintSystem
from datamodule import MusicDataModule
import torch

def main(args):
    # Set matrix precision for H200 (Tensor Cores)
    torch.set_float32_matmul_precision('medium') # or 'high'
    
    # 1. Init Data
    dm = MusicDataModule(data_dir=args.data_dir, batch_size=args.batch_size)
    
    # 2. Init System
    system = MusicPrintSystem(lr=args.lr)
    
    # 3. Init Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=args.checkpoint_dir,
        filename='mert-adapter-{epoch:02d}-{train_loss:.2f}',
        save_top_k=3,
        monitor='train_loss',
        mode='min'
    )
    
    # 4. Init Trainer
    trainer = pl.Trainer(
        accelerator="gpu",
        devices="auto", # Will auto-detect all 8 GPUs
        strategy="ddp_find_unused_parameters_true", # Needed for frozen backbone
        precision="bf16-mixed", # H200 Sweet Spot
        max_epochs=args.epochs,
        callbacks=[checkpoint_callback, LearningRateMonitor(logging_interval='step')],
        log_every_n_steps=10,
        default_root_dir=args.checkpoint_dir
    )
    
    # 5. Train
    print(f"Starting training on {trainer.num_devices} GPUs with strategy {trainer.strategy}")
    trainer.fit(system, datamodule=dm)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="/app/data")
    parser.add_argument("--checkpoint_dir", type=str, default="/app/checkpoints")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    
    args = parser.parse_args()
    main(args)