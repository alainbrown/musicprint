import os
import argparse
from pathlib import Path

# Import pipeline stages
from preprocess import preprocess_dataset
from index import index

def run_indexing_pipeline(args):
    print("="*60)
    print("🔍 VECTOR INDEXING PIPELINE")
    print("="*60)

    # --- Step 1: Preprocessing ---
    print("\n[1/2] Preprocessing Audio (MP3 -> FLAC)...")
    preprocess_dataset(data_dir=args.source_dir, output_dir=args.data_dir, workers=args.workers)

    # --- Step 2: Indexing ---
    print("\n[2/2] Building Index from Model...")
    
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model file not found at: {args.model_path}. Please provide a valid .pt file.")

    index_args = argparse.Namespace(
        model_path=args.model_path,
        data_dir=args.data_dir,
        output_dir=args.index_dir,
        batch_size=args.batch_size
    )
    index(index_args)

    print("\n" + "="*60)
    print(f"✅ INDEXING COMPLETE")
    print(f"🔹 Model Used: {args.model_path}")
    print(f"🔹 Index Output: {args.index_dir}")
    print("="*60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the vector indexing pipeline")
    
    # Paths
    parser.add_argument("--source_dir", type=str, default="/vol/src_music", help="Read-only source of MP3s")
    parser.add_argument("--data_dir", type=str, default="/vol/data", help="Where to store processed FLACs")
    parser.add_argument("--model_path", type=str, required=True, help="Path to encoder.pt")
    parser.add_argument("--index_dir", type=str, default="/vol/cache/index")
    
    # Params
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=None)

    args = parser.parse_args()
    
    # Ensure dirs exist
    os.makedirs(args.data_dir, exist_ok=True)
    os.makedirs(args.index_dir, exist_ok=True)

    run_indexing_pipeline(args)