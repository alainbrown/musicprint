import os
import torch
import struct
import glob
import argparse
from tqdm import tqdm

def build_index(args):
    print(f">>> Merging shards from {args.input_dir}...")
    
    # 1. Collect all pairs
    # Each entry: (PQ_CODE_INT64, ISRC_INT64)
    all_pairs = []
    
    shard_files = glob.glob(os.path.join(args.input_dir, "*.pt"))
    for sf in tqdm(shard_files, desc="Reading Shards"):
        data = torch.load(sf)
        # data is a list of results from predict_step
        # result: {"id": isrc, "embeddings": (N, 64), "times": [...]}
        for result in data:
            isrc = result["id"]
            embeddings = result["embeddings"] # (N, 64)
            
            # If the embeddings are already quantized to PQ codes in predict_step, 
            # they will be uint8 arrays (N, 8). 
            # For this script, we assume they are already converted to 8-byte codes.
            # If not, we would apply PQ here.
            
            for pq_code in embeddings:
                # Convert 8-byte array/tensor to uint64 for sorting
                if isinstance(pq_code, torch.Tensor):
                    pq_code = pq_code.numpy()
                
                # Pack 8 bytes into one uint64
                pq_int = struct.unpack("<Q", pq_code.tobytes())[0]
                
                # Ensure isrc is treated as unsigned 64-bit (handles potential negative signed ints)
                isrc_unsigned = isrc & 0xFFFFFFFFFFFFFFFF
                
                all_pairs.append((pq_int, isrc_unsigned))

    print(f"Total Fingerprints: {len(all_pairs):,}")
    
    # 2. Sort by PQ Code (Critical for Binary Search)
    print(">>> Sorting Index...")
    all_pairs.sort()
    
    # 3. Write Binary File
    # Format: [Header: 64 bytes][Data: 16 bytes per entry]
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "wb") as f:
        # Header (MPAF = MusicPrint Audio Fingerprints)
        f.write(struct.pack("<4sII", b"MPAF", 1, len(all_pairs)))
        f.write(b"\x00" * 52) # Padding
        
        # Data
        for pq_int, isrc in tqdm(all_pairs, desc="Writing Binary"):
            # 8 bytes PQ + 8 bytes ISRC = 16 bytes
            f.write(struct.pack("<QQ", pq_int, isrc))

    print(f"SUCCESS! Created {args.output} ({os.path.getsize(args.output)/1e6:.1f} MB)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True, help="Directory with .pt shards")
    parser.add_argument("--output", type=str, default="release/audio_index.bin")
    args = parser.parse_args()
    build_index(args)
