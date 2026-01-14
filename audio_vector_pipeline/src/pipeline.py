import os
import argparse
import glob
from pathlib import Path

# Import pipeline stages
from download_noise import download_noise
from preprocess import preprocess_dataset
from train import train
from index import index
from export import export

def run_pipeline(args):
    print("="*60)
    print("🎹 MUSICPRINT AUDIO VECTOR PIPELINE")
    print("="*60)

    # --- Step 1: Dependencies ---
    print("\n[1/5] Checking Dependencies...")
    download_noise()

    # --- Step 2: Preprocessing ---
    print("\n[2/5] Preprocessing Audio (MP3 -> FLAC)...")
    # We mirror source to data volume
    preprocess_dataset(data_dir=args.source_dir, output_dir=args.data_dir, workers=args.workers)

    # --- Step 3: Training ---
    print("\n[3/5] Training Model...")
    
    # Check for existing checkpoint to resume
    resume_ckpt = None
    last_ckpt_path = os.path.join(args.checkpoint_dir, "last.ckpt")
    if os.path.exists(last_ckpt_path):
        print(f"🔄 Resuming from last checkpoint: {last_ckpt_path}")
        resume_ckpt = last_ckpt_path
    
    # Prepare arguments for the training module
    train_args = argparse.Namespace(
        data_dir=args.data_dir,
        checkpoint_dir=args.checkpoint_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=1e-4,
        auto_batch_size=args.auto_batch_size,
        accelerator="gpu",
        strategy="auto",
        resume_checkpoint=resume_ckpt
    )
    train(train_args)

    # --- Step 4: Locate Best Model ---
    print("\n[4/5] Identifying Best Model...")
    # Find the latest checkpoint saved by Lightning
    # Lightning saves as "name-epoch=XX-val_loss=YY.ckpt"
    # We sort by modification time to get the last one, or by name parsing if needed
    checkpoints = sorted(glob.glob(os.path.join(args.checkpoint_dir, "*.ckpt")), key=os.path.getmtime)
    
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found in {args.checkpoint_dir}. Training may have failed.")
    
    best_ckpt = checkpoints[-1]
    print(f"🏆 Best Checkpoint: {best_ckpt}")

    # --- Step 5: Indexing & Export ---
    print(f"\n[5/5] Indexing and Exporting...")
    
    # Run Indexing
    index_args = argparse.Namespace(
        checkpoint_path=best_ckpt,
        pq_path=None, # Optional PQ training could be added here
        data_dir=args.data_dir,
        output_dir=args.index_dir,
        batch_size=16 # Inference batch size
    )
    index(index_args)

    # Run Export
    export_args = argparse.Namespace(
        checkpoint_path=best_ckpt,
        output_dir=args.release_dir
    )
    export(export_args)

    print("\n" + "="*60)
    print(f"✅ PIPELINE COMPLETE")
    print(f"🔹 Model: {best_ckpt}")
    print(f"🔹 CoreML: {os.path.join(args.release_dir, 'MusicPrintEncoder.mlpackage')}")
    print(f"🔹 Index: {args.index_dir}")
    print("="*60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the end-to-end MusicPrint pipeline")
    
    # Paths
    parser.add_argument("--source_dir", type=str, default="/vol/src_music", help="Read-only source of MP3s")
    parser.add_argument("--data_dir", type=str, default="/vol/data", help="Where to store processed FLACs")
    parser.add_argument("--checkpoint_dir", type=str, default="/vol/checkpoints")
    parser.add_argument("--index_dir", type=str, default="/vol/cache/index")
    parser.add_argument("--release_dir", type=str, default="/vol/release")
    
    # Training Params
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--auto_batch_size", action="store_true")
    parser.add_argument("--workers", type=int, default=None)

    args = parser.parse_args()
    
    # Ensure dirs exist
    os.makedirs(args.data_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.index_dir, exist_ok=True)
    os.makedirs(args.release_dir, exist_ok=True)

    run_pipeline(args)
