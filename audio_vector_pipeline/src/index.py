import argparse
import pytorch_lightning as pl
from system import MusicPrintSystem
from datamodule import MusicDataModule
from writer import IndexWriter
import torch

def main(args):
    # Precision
    torch.set_float32_matmul_precision('medium')
    
    # 1. Init Data
    # For inference, we might want a simple DALI loader that scans all files sequentially
    # Our MusicDataModule defaults to 'train_dataloader'. 
    # We need to ensure we use a sequential (non-shuffled) loader for indexing if we want determinism,
    # but for a massive index, random access is fine as long as we cover everything.
    # The 'predict' mode in Lightning handles the DistributedSampler automatically to ensure 
    # each file is processed exactly once across the cluster.
    dm = MusicDataModule(data_dir=args.data_dir, batch_size=args.batch_size)
    
    # 2. Init System
    # Load from checkpoint
    system = MusicPrintSystem.load_from_checkpoint(args.checkpoint_path)
    
    if args.pq_path:
        import faiss
        print(f"Loading PQ from {args.pq_path}...")
        system.pq = faiss.read_ProductQuantizer(args.pq_path)
    
    # 3. Init Writer
    writer = IndexWriter(output_dir=args.output_dir, write_interval="batch")
    
    # 4. Init Trainer
    trainer = pl.Trainer(
        accelerator="gpu",
        devices="auto",
        strategy="ddp", # Distributed Data Parallel
        precision="bf16-mixed",
        callbacks=[writer]
    )
    
    # 5. Run Inference
    print(f"Starting distributed indexing on {trainer.num_devices} GPUs...")
    # We use the training loader for now as the source of all files
    trainer.predict(system, datamodule=dm)
    print("Indexing Complete.")

def index(args):
    """
    Main indexing entry point.
    args: Namespace or object with attributes:
        checkpoint_path, pq_path, data_dir, output_dir, batch_size
    """
    main(args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--pq_path", type=str, default=None, help="Path to trained PQ codebook (faiss file)")
    parser.add_argument("--data_dir", type=str, default="/vol/data")
    parser.add_argument("--output_dir", type=str, default="/vol/cache/index")
    parser.add_argument("--batch_size", type=int, default=16) # Smaller batch because we expand windows inside
    
    args = parser.parse_args()
    index(args)
