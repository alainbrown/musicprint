import argparse
import torch
from tqdm import tqdm
import os
import numpy as np
from system import MusicPrintSystem
from datamodule import MusicDataModule

def evaluate_accuracy(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating on: {device}")

    # 1. Load Model
    print(f"Loading checkpoint: {args.checkpoint_path}")
    system = MusicPrintSystem.load_from_checkpoint(args.checkpoint_path).to(device)
    system.eval()

    # 2. Build Index (Reference)
    # We use the datamodule to get clean files. 
    # In a real test, we'd ensure 'train' and 'eval' sets are split.
    dm = MusicDataModule(data_dir=args.data_dir, batch_size=args.batch_size)
    loader = dm.train_dataloader() # Using train_dataloader for simplicity in this script

    index_hashes = []
    index_labels = []
    
    print("Step 1: Building Clean Index...")
    max_eval_tracks = 1000 # Limit for speed
    
    with torch.no_grad():
        for i, batch in tqdm(enumerate(loader), total=max_eval_tracks//args.batch_size):
            data = batch[0]
            audio = data["audio"].to(device).squeeze(-1)
            labels = data["label"].to(device).squeeze()
            
            # Use deterministic first window for index
            # (In build_index.py we'd use the Adaptive method, but for simple Recall@1 this is fine)
            hashes = system.model.get_hash(audio)
            
            index_hashes.append(hashes.cpu())
            index_labels.append(labels.cpu())
            
            if len(index_labels) * args.batch_size >= max_eval_tracks:
                break

    index_hashes = torch.cat(index_hashes)
    index_labels = torch.cat(index_labels)

    # 3. Query with Noisy Audio
    # We simulate noise using the 'augment' flag if we had it in the datamodule
    # Let's assume we want to see how it performs against the same tracks with augmentations
    print("Step 2: Querying with Noisy Audio...")
    
    # We create a noisy loader
    from data.dali_loader import DALIGPULoader
    noisy_loader = DALIGPULoader(
        file_root=args.data_dir, 
        batch_size=args.batch_size, 
        augment=True, # Critical: This adds the noise/shifts
        device_id=0 if torch.cuda.is_available() else None
    )

    correct = 0
    total = 0

    with torch.no_grad():
        for i, batch in tqdm(enumerate(noisy_loader), total=max_eval_tracks//args.batch_size):
            data = batch[0]
            audio = data["audio"].to(device).squeeze(-1)
            labels = data["label"].to(device).squeeze().cpu()
            
            query_hashes = system.model.get_hash(audio).cpu()
            
            # Rank 1 Search (Cosine similarity)
            # (B, 64) @ (N, 64).T -> (B, N)
            sims = torch.matmul(query_hashes, index_hashes.T) / 64.0
            
            # Get top indices
            top_indices = torch.argmax(sims, dim=1)
            predicted_labels = index_labels[top_indices]
            
            correct += (predicted_labels == labels).sum().item()
            total += labels.size(0)
            
            if total >= max_eval_tracks:
                break

    recall = (correct / total) * 100.0
    print(f"\n✅ Evaluation Results")
    print(f"Recall@1 (Noisy vs Clean): {recall:.2f}%")
    print(f"Samples: {total}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--data_dir", type=str, default="/app/data")
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()
    evaluate_accuracy(args)
