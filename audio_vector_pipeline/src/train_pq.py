import argparse
import os
import torch
import numpy as np
import faiss
import pytorch_lightning as pl
from system import MusicPrintSystem
from datamodule import MusicDataModule
from tqdm import tqdm

def train_pq(args):
    # 1. Load System & Data
    print(f">>> Loading model from {args.checkpoint}...")
    system = MusicPrintSystem.load_from_checkpoint(args.checkpoint)
    system.to("cuda").eval()
    
    dm = MusicDataModule(data_dir=args.data_dir, batch_size=args.batch_size)
    loader = dm.train_dataloader()
    
    # 2. Harvest Embeddings
    # We need a representative sample (e.g., 256,000 to 1,000,000 vectors)
    print(f">>> Harvesting embeddings (Target: {args.num_samples})...")
    collected = []
    
    with torch.no_grad():
        for batch in tqdm(loader):
            audio = batch[0]["audio"].to("cuda").squeeze(-1)
            # Use forward to get raw 64D vectors
            embeddings = system(audio)
            # Normalize for cosine similarity (standard for PQ)
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            collected.append(embeddings.cpu().numpy())
            
            if sum(len(x) for x in collected) >= args.num_samples:
                break
                
    embeddings_np = np.vstack(collected)[:args.num_samples].astype('float32')
    print(f"Collected {embeddings_np.shape[0]} vectors.")

    # 3. Train PQ
    # D=64, M=8 (8 bytes), nbits=8 (256 centroids per sub-space)
    print(">>> Training Faiss PQ Quantizer (M=8, nbits=8)...")
    d = embeddings_np.shape[1]
    m = 8
    nbits = 8
    
    pq = faiss.ProductQuantizer(d, m, nbits)
    pq.train(embeddings_np)
    
    # 4. Save
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    faiss.write_ProductQuantizer(pq, args.output)
    print(f"SUCCESS! PQ Codebook saved to {args.output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data_dir", type=str, default="/app/data")
    parser.add_argument("--output", type=str, default="release/audio_pq.bin")
    parser.add_argument("--num_samples", type=int, default=256000)
    parser.add_argument("--batch_size", type=int, default=128)
    args = parser.parse_args()
    train_pq(args)
