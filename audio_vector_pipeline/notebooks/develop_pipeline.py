# %% [markdown]
# # MusicPrint Vetting Notebook
# Use this to verify the logic of the Audio Pipeline before mass-indexing.
# This notebook will:
# 1. Download sample audio for testing.
# 2. Verify ISRC Bitpacking (50-bit).
# 3. Test the "Sphere Method" (Deduplication) on REAL audio.
# 4. Test PQ Quantization accuracy.

# %%
import sys
import os
import torch
import numpy as np
import requests
from tqdm import tqdm
import argparse

# Add src to path
sys.path.append(os.path.join(os.getcwd(), '../src'))

from isrc_utils import pack_isrc, unpack_isrc
from models.mert_adapter import MERTAdapter
from system import MusicPrintSystem
# Import refactored main functions
from train import train
from index import index
from evaluate import evaluate

# %% [markdown]
# ## 0. Download Test Samples
# If no samples exist, we fetch 5 tracks from Jamendo.

# %%
DATA_DIR = "/vol/data/test_samples"
os.makedirs(DATA_DIR, exist_ok=True)

def download_samples():
    samples = [
        ("JAMENDO_001", "https://prod-1.storage.jamendo.com/?trackid=1890710&format=mp31&from=app"), # Rock
        ("JAMENDO_002", "https://prod-1.storage.jamendo.com/?trackid=1890711&format=mp31&from=app"), # Electronic
        ("JAMENDO_003", "https://prod-1.storage.jamendo.com/?trackid=1890712&format=mp31&from=app"), # Jazz
        ("JAMENDO_004", "https://prod-1.storage.jamendo.com/?trackid=1890713&format=mp31&from=app"), # Pop
        ("JAMENDO_005", "https://prod-1.storage.jamendo.com/?trackid=1890714&format=mp31&from=app")  # Classical
    ]
    
    from preprocess import preprocess_dataset

    for name, url in samples:
        target = os.path.join(DATA_DIR, f"{name}.mp3")
        # If file exists but is empty, delete it
        if os.path.exists(target) and os.path.getsize(target) == 0:
            print(f"Removing 0-byte file: {target}")
            os.remove(target)
            
        if not os.path.exists(target):
            print(f"Downloading {name}...")
            try:
                r = requests.get(url, stream=True, timeout=30)
                r.raise_for_status()
                with open(target, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
                # Final check
                if os.path.getsize(target) == 0:
                    print(f"❌ Failed to download {name} (0 bytes)")
                    os.remove(target)
            except Exception as e:
                print(f"❌ Error downloading {name}: {e}")
                
    print("Samples downloaded.")
    
    # Run Normalization (MP3 -> FLAC)
    print("\nRunning Audio Normalization (MP3 -> FLAC)...")
    preprocess_dataset(DATA_DIR)


if __name__ == "__main__":
    download_samples()

    # %% [markdown]
    # ## 1. Verify ISRC Bitpacking
    # Ensure we can round-trip ISRCs without data loss.

    # %%
    test_isrcs = ["USRC11234567", "GBAYL1200001", "FR6V81234567"]
    for original in test_isrcs:
        packed = pack_isrc(original)
        unpacked = unpack_isrc(packed)
        status = "PASS" if original == unpacked else "FAIL"
        print(f"{original} -> {packed:016x} -> {unpacked} [{status}]")

    # %% [markdown]
    # ## 2. Verify Production Inference Path (Deduplication)
    # We use the actual `MusicPrintSystem.predict_step` to verify deduplication.
    # This ensures we are testing the SAME code that runs in the massive indexer.
    # CRITICAL: We use `MusicDataModule` to verify DALI ingestion.

    # %%
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {DEVICE}")
    
    from datamodule import MusicDataModule

    # Initialize system
    system = MusicPrintSystem().to(DEVICE).eval()
    
    # Initialize Data Module with 30s windows for vetting
    print("Initializing DALI Data Pipeline (30s windows)...")
    dm = MusicDataModule(data_dir=DATA_DIR, batch_size=1, val_split=0.0, window_secs=30.0)
    loader = dm.train_dataloader()
    
    try:
        print("Fetching batch from DALI...")
        # Get first batch
        batch = next(iter(loader))
        
        # Batch from DALI is a list of dicts: [{"audio": ..., "label": ...}]
        data_dict = batch[0]
        audio = data_dict["audio"]
        label = data_dict["label"]
        print(f"Batch received. Audio Shape: {audio.shape}, Label: {label}")

        print(f"Running production predict_step...")
        with torch.no_grad():
            # Pass the batch exactly as Lightning would receive it
            results = system.predict_step(batch, 0)

        if results:
            res = results[0]
            print(f"Song ID: {res['id']}")
            print(f"Fingerprints Generated: {len(res['embeddings'])}")
            print(f"Timestamps: {res['times']}")
            
            # Verify the 15-hash limit mentioned in system.py
            if len(res['embeddings']) > 15:
                print(f"❌ FAIL: System generated {len(res['embeddings'])} hashes (limit: 15)")
            else:
                print("✅ PASS: Deduplication logic respected constraints.")
                
            embeddings = torch.from_numpy(res['embeddings'])
        else:
            print("❌ FAIL: No results returned from predict_step (Audio might be too short?)")

    except Exception as e:
        print(f"❌ DATA INGESTION FAIL: {e}")
        import traceback
        traceback.print_exc()

    # %% [markdown]
    # ## 3. Verify Writer (Disk I/O)
    # Ensure we can actually write the results to the cache volume.

    # %%
    from writer import IndexWriter
    import shutil
    
    # We use a test directory in the cache volume
    CACHE_DIR = "/vol/cache/index_test"
    if os.path.exists(CACHE_DIR):
        shutil.rmtree(CACHE_DIR)
        
    print(f"Testing IndexWriter in {CACHE_DIR}...")
    try:
        # 1. Init Writer
        writer = IndexWriter(output_dir=CACHE_DIR)
        
        # 2. Simulate Lightning 'prediction' object structure
        # The writer expects a list of dictionaries from predict_step
        dummy_prediction = [
            {
                "id": 12345,
                "embeddings": np.random.randn(5, 8).astype(np.uint8), # Mock PQ codes
                "times": [0.0, 5.0, 10.0, 15.0, 20.0]
            }
        ]
        
        # 3. Trigger write (Manually call the hook)
        # Mock the 'trainer' object for rank info
        class MockTrainer:
            global_rank = 0
            
        writer.write_on_batch_end(
            trainer=MockTrainer(), 
            pl_module=None, 
            prediction=dummy_prediction, 
            batch_indices=None, 
            batch=None, 
            batch_idx=0, 
            dataloader_idx=0
        )
        
        # 4. Verify file exists
        # IndexWriter saves as 'batch_{rank}_{batch_idx}.pt'
        expected_file = os.path.join(CACHE_DIR, "batch_0_0.pt")
        if os.path.exists(expected_file):
            print(f"✅ PASS: Successfully wrote index shard: {expected_file}")
            # Load back to verify
            data = torch.load(expected_file)
            if len(data) == 1 and data[0]["id"] == 12345:
                print(f"   - Verified data integrity in shard.")
        else:
            print(f"❌ FAIL: File not found at {expected_file}")
            
    except Exception as e:
        print(f"❌ WRITER FAIL: {e}")
        import traceback
        traceback.print_exc()

    # %% [markdown]
    # ## 4. Verify PQ Quantization Accuracy
    # Simulation of the 8-byte compression loss using Faiss.

    # %%
    import faiss
    from train_pq import train_pq_model

    def test_pq_quality(embeddings_tensor):
        # Even for this small test, we can use the logic from the pipeline
        # but since we want to measure accuracy, we'll do a quick local train/decode
        data = embeddings_tensor.cpu().numpy().astype('float32')
        if data.ndim == 1: data = data.reshape(1, -1)
        
        d = data.shape[1]
        m = 8 
        nbits = 8
        
        # Ensure we have enough data for 256 clusters
        min_points = 256
        if data.shape[0] < min_points:
            repeats = (min_points // data.shape[0]) + 1
            train_data = np.tile(data, (repeats, 1))[:min_points]
        else:
            train_data = data

        pq = faiss.ProductQuantizer(d, m, nbits)
        pq.train(train_data)
        
        codes = pq.compute_codes(data)
        recon = pq.decode(codes)
        
        orig_norm = data / (np.linalg.norm(data, axis=1, keepdims=True) + 1e-9)
        recon_norm = recon / (np.linalg.norm(recon, axis=1, keepdims=True) + 1e-9)
        cosine_sims = np.sum(orig_norm * recon_norm, axis=1)
        
        print(f"PQ (8-byte) Reconstruction Accuracy (Avg Cosine Sim): {np.mean(cosine_sims):.4f}")

    if 'embeddings' in locals():
        test_pq_quality(embeddings)

    print("\n✅ Verification Notebook Complete.")
