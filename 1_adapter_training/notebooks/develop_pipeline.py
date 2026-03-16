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

from datasets import load_dataset
import scipy.io.wavfile

import shutil

import io
import soundfile as sf

def download_gtzan_samples():
    print("Fetching MINDS14 dataset (14 clips)...")
    try:
        # Load without automatic decoding to bypass torchcodec issues
        ds = load_dataset("PolyAI/minds14", name="en-US", split="train[:14]").cast_column("audio", Audio(decode=False))
        
        if len(ds) == 0:
            raise ValueError("Dataset is empty!")

        for i, item in enumerate(ds):
            # item['audio'] now contains {'path': ..., 'bytes': ...}
            audio_bytes = item['audio']['bytes']
            
            # Manually decode using soundfile
            with io.BytesIO(audio_bytes) as f:
                audio, sr = sf.read(f)
            
            # Save as WAV
            target = os.path.join(SRC_DIR, f"minds_{i:03d}.wav")
            scipy.io.wavfile.write(target, sr, audio)
            
        # Add a SILENCE sample to stress-test normalization
        print("Adding a SILENCE sample to test stability...")
        silence = np.zeros(44100 * 10, dtype=np.int16)
        scipy.io.wavfile.write(os.path.join(SRC_DIR, "silence.wav"), 44100, silence)
            
        print("✅ Real Samples + Silence Ready.")
    except Exception as e:
        print(f"❌ Failed to fetch dataset: {e}")
        raise e

    # Run Normalization
    print("\nRunning Audio Normalization...")
    preprocess_dataset(SRC_DIR, DATA_DIR)

# Add Audio to imports from datasets
from datasets import load_dataset, Audio

download_gtzan_samples()

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
dm = MusicDataModule(data_dir=DATA_DIR, batch_size=2, val_split=0.0, window_secs=5.0)
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
# Pass num_classes=15 (14 MINDS samples + 1 silence)
system = MusicPrintSystem(num_classes=15).to(DEVICE).eval()

# 1. Trace the model
print("Tracing model to TorchScript...")
example_input = torch.randn(1, 120000).to(DEVICE)
# We export the backbone (MERTAdapter), not the full ArcFace wrapper
traced_model = torch.jit.trace(system.model.backbone, example_input)

# 2. Verify Output Parity
with torch.no_grad():
    out_orig = system.model.backbone(example_input)
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
print("\n--- 4. Verifying Model Output Stability ---")
try:
    # Manual Forward Pass Check
    batch = next(iter(dm.train_dataloader()))
    
    # In ArcFace/System refactor, we use audio_1
    audio = batch[0]["audio_1"].to(DEVICE)
    labels = batch[0]["label"].to(DEVICE).squeeze().long()
    
    print(f"Input Audio Stats: Min={audio.min():.3f}, Max={audio.max():.3f}, NaN={torch.isnan(audio).any()}")
    
    system.train() # Set to train mode for loss calc
    
    # Forward Pass through backbone
    embeddings = system(audio)
    
    print(f"Output Embeddings Shape: {embeddings.shape}")
    print(f"Output Stats: Min={embeddings.min():.3f}, Max={embeddings.max():.3f}")
    
    if torch.isnan(embeddings).any():
        print("❌ MODEL FAILURE: Model produced NaNs on valid input!")
    else:
        print("✅ Model Forward Pass Successful (No NaNs).")
        
        # Now check ArcFace Loss Function
        print("Checking ArcFace Loss...")
        loss = system.model.get_loss(embeddings, labels)
        print(f"Loss Value: {loss.item()}")
        
        if torch.isnan(loss):
            print("❌ LOSS FUNCTION FAILURE: Loss returned NaN on valid embeddings!")
        else:
            print("✅ Loss Function Successful.")

except Exception as e:
    print(f"❌ Verification Failed: {e}")
    import traceback
    traceback.print_exc()

print("\n✅ Verification Notebook Complete.")

# %% [markdown]
# ## 5. Fix Remote Code Serialization
# The 'trust_remote_code=True' model breaks TorchScript serialization due to dynamic class names.
# We test a fix here by wrapping the model before export.

# %%
print("Testing Serialization Fix...")

# Get the backbone
backbone = system.model.backbone

# HACK: Manually rename the type of the underlying Hugging Face model
# The path is usually deeply nested in transformers_modules... 
hf_model = backbone.backbone # This is the dynamic MERTModel
original_class = hf_model.__class__

print(f"Original Class: {original_class}")

# Define a dummy class to mask the origin
class MERTModel(nn.Module):
    def __call__(self, *args, **kwargs):
        return super().__call__(*args, **kwargs)

def run_test():
    try:
        # Temporarily change the class to a clean local one
        # Note: This is dangerous but might fool the JIT tracer into writing a clean name
        hf_model.__class__ = MERTModel
        
        # Now Trace
        print("Tracing with patched class...")
        traced_safe = torch.jit.trace(backbone, example_input)
        
        # Save to temp
        safe_path = "/vol/release/encoder_patched.pt"
        traced_safe.save(safe_path)
        print(f"Saved patched model to {safe_path}")
        
        # Verify Load
        print("Attempting to load patched model...")
        loaded = torch.jit.load(safe_path)
        print("✅ Success: Patched model loaded!")
        
        # Verify Parity
        with torch.no_grad():
            out_safe = loaded(example_input)
            out_orig = backbone(example_input)
        
diff = torch.abs(out_safe - out_orig).max().item()
        print(f"Parity Check: {diff:.2e}")
        
    except Exception as e:
        print(f"❌ Patching Failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Restore class just in case (though we are in a notebook)
        hf_model.__class__ = original_class
        print("Restored original class.")

run_test()