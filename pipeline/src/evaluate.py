import argparse
import torch
import torch.nn.functional as F
from tqdm import tqdm
import os
import numpy as np
from system import MusicPrintSystem
from datamodule import MusicDataModule
from data.dali_loader import DALIGPULoader
import nvidia.dali.fn as fn
import nvidia.dali.types as types

def build_adaptive_index_memory(system, loader, device, limit=None):
    """
    Builds an in-memory index using the Adaptive Density strategy (sliding window).
    Returns: (Hashes [N, 64], SongIDs [N])
    """
    index_hashes = []
    index_ids = []
    
    # We need to manually perform sliding window on the full tracks
    # DALI loader gives us full tracks (or large chunks) if we configure it right.
    # For this eval script, we assume the loader provides the first 30s-60s of the song to index.
    
    print("Building Adaptive Index (In-Memory)...")
    with torch.no_grad():
        for i, batch in tqdm(enumerate(loader)):
            data = batch[0]
            audio = data["audio"].to(device).squeeze(-1) # (B, Time)
            labels = data["label"].to(device).squeeze()
            
            # Sliding Window Logic
            # Window=5s (120000), Stride=1s (24000)
            # Input: (B, Time) -> Unfold -> (B, N_Windows, 120000)
            if audio.shape[1] < 120000:
                continue
                
            # Unfold: (Batch, N_Windows, Window_Size)
            windows = audio.unfold(1, 120000, 24000) 
            B, N_wins, W_size = windows.shape
            
            # Flatten to (B*N, W_size) for batch inference
            flat_windows = windows.reshape(-1, W_size)
            
            # Run Model
            hashes = system.model.get_hash(flat_windows) # (B*N, 64)
            hashes = hashes.view(B, N_wins, 64)
            
            # Simple Storage (Skip deduplication logic for Eval speed - store ALL 1s windows)
            # We want to test recall, storing more hashes makes it strictly easier/better.
            # Production would dedup, but for Eval, dense index is the baseline.
            
            for b in range(B):
                song_hashes = hashes[b] # (N_wins, 64)
                song_id = labels[b].item()
                
                index_hashes.append(song_hashes.cpu())
                # Repeat ID for each hash
                index_ids.append(torch.full((N_wins,), song_id, dtype=torch.long))
            
            if limit and len(index_ids) * N_wins >= limit:
                break
                
    if not index_hashes:
        return None, None
        
    return torch.cat(index_hashes), torch.cat(index_ids)

def evaluate_real_world(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating on: {device}")

    # 1. Load System
    system = MusicPrintSystem.load_from_checkpoint(args.checkpoint_path).to(device)
    system.eval()

    # 2. Setup Data
    # We need a loader that returns LONG clips (e.g. 30s) for indexing
    # And a loader that returns RANDOM 5s clips for querying
    
    # We can hack the DALI loader parameters.
    # Note: DALI loader is designed for fixed window.
    # We'll use the MusicDataModule to get the validation files
    dm = MusicDataModule(data_dir=args.data_dir, batch_size=args.batch_size)
    val_files = dm.val_files
    
    # --- Step A: Build Index (Clean, 30s window sliding) ---
    # We init a custom loader for 30s clips
    index_loader = DALIGPULoader(
        file_list="val_list.txt",
        batch_size=16, # Smaller batch for large audio
        window_secs=30.0, # Index first 30s of each song
        augment=False,
        device_id=0 if torch.cuda.is_available() else None
    )
    
    db_hashes, db_ids = build_adaptive_index_memory(system, index_loader, device, limit=None)
    db_hashes = db_hashes.to(device)
    db_ids = db_ids.to(device)
    
    print(f"Index Size: {db_hashes.shape[0]} hashes covering {len(val_files)} songs.")

    # --- Step B: Query (Noisy, Random 5s Crop) ---
    print("Querying with Random 5s Crops + Noise...")
    query_loader = DALIGPULoader(
        file_list="val_list.txt",
        batch_size=args.batch_size,
        window_secs=5.0,
        augment=True, # Random Crop + Noise + Speed
        device_id=0 if torch.cuda.is_available() else None
    )
    
    correct = 0
    total = 0
    
    with torch.no_grad():
        for i, batch in tqdm(enumerate(query_loader)):
            data = batch[0]
            audio = data["audio"].to(device).squeeze(-1)
            labels = data["label"].to(device).squeeze().to(device)
            
            # Query Hash
            q_hashes = system.model.get_hash(audio) # (B, 64)
            
            # Search
            # (B, 64) @ (DB, 64).T -> (B, DB)
            sims = torch.matmul(q_hashes, db_hashes.T) / 64.0
            
            # Top-1
            best_idx = torch.argmax(sims, dim=1)
            pred_ids = db_ids[best_idx]
            
            correct += (pred_ids == labels).sum().item()
            total += labels.size(0)
            
            # Since validation set is small (~500), we run full pass
            
    recall = (correct / total) * 100.0
    print(f"\n🌟 Real-World Evaluation 🌟")
    print(f"Test Condition: Random 5s Crop (Unaligned) + Noise + Speed Shift")
    print(f"Recall@1: {recall:.2f}%")
    print(f"Total Queries: {total}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--data_dir", type=str, default="/app/data")
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()
    evaluate_real_world(args)
