# %% [markdown]
# # MusicPrint Vetting Notebook
# Use this to verify the logic of the Adapter Training Pipeline.
# This notebook validates data loading, contrastive training, and TorchScript export.

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
from preprocess import preprocess_dataset

# %% [markdown]
# ## 0. Environment Setup
# Ensure sample audio and noise datasets are present.

# %%
SRC_DIR = "/vol/data/test_samples_raw"
DATA_DIR = "/vol/data/test_samples_processed"
os.makedirs(SRC_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

import scipy.io.wavfile

def generate_samples():
    print("Generating synthetic samples...")
    # Generate 10 dummy files
    for i in range(10):
        # 10 seconds of random noise at 44.1kHz
        sr = 44100
        duration = 10
        # Int16 range
        audio = np.random.randint(-32768, 32767, size=sr*duration, dtype=np.int16)
        
        fname = f"synthetic_{i:03d}.wav"
        target = os.path.join(SRC_DIR, fname)
        
        if not os.path.exists(target):
            scipy.io.wavfile.write(target, sr, audio)
            
    print("Synthetic samples created.")
    
    # Run Normalization (WAV -> FLAC)
    print("\nRunning Audio Normalization (WAV -> FLAC)...")
    preprocess_dataset(SRC_DIR, DATA_DIR)

generate_samples()
download_noise() 

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
# ## 2. Verify Data Module & DALI
# Checks that the DALI contrastive pipeline produces the expected views.

# %%
print("Initializing Data Module...")
dm = MusicDataModule(data_dir=DATA_DIR, batch_size=2, val_split=0.2, window_secs=5.0)
train_loader = dm.train_dataloader()

batch = next(iter(train_loader))
data_dict = batch[0]
print(f"✅ Success: Batch keys: {data_dict.keys()}")
print(f"View 1 Shape: {data_dict['audio_1'].shape}")
print(f"View 2 Shape: {data_dict['audio_2'].shape}")

# %% [markdown]
# ## 3. Verify TorchScript Export
# This is critical! The Indexer depends on this export working correctly.

# %%
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
system = MusicPrintSystem().to(DEVICE).eval()

# 1. Trace the model
print("Tracing model to TorchScript...")
example_input = torch.randn(1, 120000).to(DEVICE)
traced_model = torch.jit.trace(system.model, example_input)

# 2. Verify Output Parity
with torch.no_grad():
    out_orig = system.model(example_input)
    out_jit = traced_model(example_input)

diff = torch.abs(out_orig - out_jit).max().item()
if diff < 1e-5:
    print(f"✅ PASS: TorchScript parity match (max diff: {diff:.2e})")
else:
    print(f"❌ FAIL: TorchScript divergence (max diff: {diff:.2e})")

# %% [markdown]
# ## 4. Verify Training Loop
# Run a mini-training session to ensure gradients flow.

# %%
print("\n--- 4. Verifying Training Loop ---")
trainer = pl.Trainer(
    logger=CSVLogger("/tmp/test_logs"),
    accelerator="auto",
    devices=1,
    max_epochs=2,
    enable_checkpointing=False,
    precision=32
)

try:
    system.train()
    trainer.fit(system, datamodule=dm)
    print("✅ Training loop execution successful.")
except Exception as e:
    print(f"❌ Training Failed: {e}")

print("\n✅ Verification Notebook Complete.")