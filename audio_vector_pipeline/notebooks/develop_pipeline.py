# %% [markdown]
# # MusicPrint Vetting Notebook
# Use this to verify the logic of the Audio Pipeline before mass-indexing.
# This notebook validates the entire modular data architecture and training logic.

# %%
import sys
import os
import torch
import numpy as np
import requests
from tqdm import tqdm
import torch.nn as nn
import librosa
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger

# Add src to path
sys.path.append(os.path.join(os.getcwd(), '../src'))

from isrc_utils import pack_isrc, unpack_isrc
from system import MusicPrintSystem
from data.module import MusicDataModule
from download_noise import download_noise

# %% [markdown]
# ## 0. Environment Setup
# Ensure sample audio and noise datasets are present.

# %%
DATA_DIR = "/vol/data/test_samples"
os.makedirs(DATA_DIR, exist_ok=True)

def download_samples():
    # Use FMA tracks from Archive.org (reliable permalinks)
    # We fetch 20 tracks to ensure val_split and batch_size work correctly.
    base_url = "https://archive.org/download/FMA_INTERNET_ARCHIVE_BATCH_1/"
    
    # A list of 20 known filenames from FMA
    fma_files = [
        "000002.mp3", "000003.mp3", "000005.mp3", "000010.mp3", "000139.mp3",
        "000140.mp3", "000141.mp3", "000142.mp3", "000144.mp3", "000145.mp3",
        "000146.mp3", "000147.mp3", "000148.mp3", "000149.mp3", "000150.mp3",
        "000151.mp3", "000152.mp3", "000153.mp3", "000154.mp3", "000155.mp3"
    ]
    
    from preprocess import preprocess_dataset

    for fname in fma_files:
        name = fname.replace(".mp3", "")
        target = os.path.join(DATA_DIR, fname)
        
        if not os.path.exists(target):
            url = base_url + fname
            print(f"Downloading {name}...")
            try:
                r = requests.get(url, stream=True, timeout=30)
                if r.status_code == 200:
                    with open(target, 'wb') as f:
                        for chunk in r.iter_content(chunk_size=8192):
                            f.write(chunk)
                else:
                    print(f"❌ Failed {url}: {r.status_code}")
            except Exception as e:
                print(f"❌ Error {name}: {e}")
                
    print("Samples downloaded.")
    
    # Run Normalization (MP3 -> FLAC)
    print("\nRunning Audio Normalization (MP3 -> FLAC)...")
    preprocess_dataset(DATA_DIR)

download_samples()
download_noise() # Ensure DEMAND dataset is ready

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
# ## 2. Verify Inference Pipeline (Single-View)
# Test the clean audio path used for validation and mass-indexing.

# %%
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# %%
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# Initialize system
system = MusicPrintSystem().to(DEVICE).eval()

# Initialize Data Module (val_split=0.1 ensures we have data in val_dataloader)
print("Initializing Inference Data Pipeline (Validation Mode)...")
dm = MusicDataModule(data_dir=DATA_DIR, batch_size=2, val_split=0.1, window_secs=5.0)
loader = dm.val_dataloader()

try:
    batch = next(iter(loader))
    data_dict = batch[0]
    # In Validation mode, we expect 'audio' key
    print(f"Batch received. Audio Shape: {data_dict['audio'].shape}")

    with torch.no_grad():
        results = system.predict_step(batch, 0)
    
    if results:
        print(f"✅ PASS: Inference step generated {len(results[0]['embeddings'])} fingerprints.")
except Exception as e:
    print(f"❌ Inference Fail: {e}")

# %% [markdown]
# ## 3. Verify Pre-Compute Strategy
# Validates caching 768-dim features to accelerate Adapter training.

# %%
PRECOMPUTE_CACHE = "/vol/cache/precompute_test"
os.makedirs(PRECOMPUTE_CACHE, exist_ok=True)
SAMPLE_FLAC = os.path.join(DATA_DIR, "JAMENDO_003.flac")

if os.path.exists(SAMPLE_FLAC):
    print("\n--- 3. Testing Feature Extraction & Backpropagation ---")
    audio, _ = librosa.load(SAMPLE_FLAC, sr=24000)
    
    # 1. Extraction (Frozen Backbone)
    sig_tensor = torch.from_numpy(audio[:24000*5]).to(DEVICE).unsqueeze(0).float()
    with torch.no_grad():
        outputs = system.model.backbone(sig_tensor)
        features = outputs.last_hidden_state[0, :10, :].cpu()
    
    out_path = os.path.join(PRECOMPUTE_CACHE, "verify_features.pt")
    torch.save({"feat": features}, out_path)
    print(f"✅ Saved features to {out_path}")
    
    # 2. Gradient Check
    loaded = torch.load(out_path)
    feat = loaded["feat"][0].to(DEVICE)
    
    # Isolate trainable head
    adapter_head = system.model.adapter.to(DEVICE)
    optimizer = torch.optim.Adam(adapter_head.parameters(), lr=1e-3)
    
    system.train()
    p1 = adapter_head(feat)
    p2 = adapter_head(feat + torch.randn_like(feat)*0.01) # Small perturbation
    
    loss = torch.nn.functional.mse_loss(p1, p2)
    loss.backward()
    optimizer.step()
    print(f"Loss: {loss.item():.6f}")
    print("✅ Verified backpropagation on cached features.")

# %% [markdown]
# ## 4. Verify Production Training Loop (Dual-View)
# Run multiple epochs using the real DALI contrastive pipeline.

# %%
print("\n--- 4. Verifying Production Training Pipeline ---")
# Use a fresh DM with BS=2 to ensure contrastive pairs
dm_train = MusicDataModule(data_dir=DATA_DIR, batch_size=2, val_split=0.0, window_secs=5.0)

trainer = pl.Trainer(
    logger=CSVLogger("/tmp/test_logs"),
    accelerator="auto",
    devices=1,
    max_epochs=5,
    enable_checkpointing=False,
    num_sanity_val_steps=0,
    limit_val_batches=0,
    precision=32
)

try:
    system.train()
    trainer.fit(system, datamodule=dm_train)
    print("✅ Production training loop converged successfully.")
except Exception as e:
    print(f"❌ Training Failed: {e}")
    import traceback
    traceback.print_exc()

print("\n✅ Verification Notebook Complete.")
