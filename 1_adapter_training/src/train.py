import argparse
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from system import MusicPrintSystem
from data.module import MusicDataModule
import torch


def main(args):
    torch.set_float32_matmul_precision('medium')

    # 1. Init Data
    dm = MusicDataModule(data_dir=args.data_dir, batch_size=args.batch_size)
    print(f"Detected {len(dm.all_files)} songs.")

    # 2. Init System
    system = MusicPrintSystem(lr=args.lr)

    # 3. Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=args.checkpoint_dir,
        filename='encoder-{epoch:02d}-{val_loss:.4f}',
        save_top_k=3,
        save_last=True,
        monitor='val_loss',
        mode='min'
    )

    # 4. Trainer
    trainer = pl.Trainer(
        accelerator=args.accelerator,
        devices="auto",
        strategy=args.strategy,
        precision="bf16-mixed",
        max_epochs=args.epochs,
        callbacks=[checkpoint_callback],
        log_every_n_steps=1,
        default_root_dir=args.checkpoint_dir,
    )

    # 5. Train
    print(f"Starting training on {trainer.num_devices} GPUs")
    if args.resume_checkpoint:
        print(f"Resuming from: {args.resume_checkpoint}")

    trainer.fit(system, datamodule=dm, ckpt_path=args.resume_checkpoint)


def train(args):
    main(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="/vol/data")
    parser.add_argument("--checkpoint_dir", type=str, default="/vol/checkpoints")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--auto_batch_size", action="store_true")
    parser.add_argument("--accelerator", type=str, default="auto")
    parser.add_argument("--strategy", type=str, default="auto")
    parser.add_argument("--resume_checkpoint", type=str, default=None)

    args = parser.parse_args()
    train(args)
