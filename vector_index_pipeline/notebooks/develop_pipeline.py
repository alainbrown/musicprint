# %% [markdown]
# # MusicPrint Indexing Pipeline Vetting
# Validates the inference and indexing workflow using a frozen TorchScript model.

# %% 
import sys
import os
import torch
import numpy as np
import scipy.io.wavfile
import shutil
import io
import soundfile as sf
from datasets import load_dataset, Audio

# Add src to path
sys.path.append(os.path.join(os.getcwd(), '../src'))

from data.module import MusicDataModule
from index import main as run_index
import argparse

# %% [markdown]
# ## 0. Environment Setup
# Generate synthetic test data (10 songs)

# %% 
SRC_DIR = "/vol/data/test_samples_raw"
DATA_DIR = "/vol/data/test_samples_processed"
MODEL_DIR = "/vol/data/test_models"
INDEX_DIR = "/vol/data/test_index"

os.makedirs(SRC_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(INDEX_DIR, exist_ok=True)

from preprocess import preprocess_dataset

def setup_data():
    print("Fetching MINDS14 dataset (14 clips)...")
    try:
        ds = load_dataset("PolyAI/minds14", name="en-US", split="train[:14]").cast_column("audio", Audio(decode=False))
        
        for i, item in enumerate(ds):
            audio_bytes = item['audio']['bytes']
            with io.BytesIO(audio_bytes) as f:
                audio, sr = sf.read(f)
            target = os.path.join(SRC_DIR, f"minds_{i:03d}.wav")
            scipy.io.wavfile.write(target, sr, audio)
            
        print("✅ Real Samples Ready.")
    except Exception as e:
        print(f"❌ Failed: {e}")
        return

    print("\nRunning Audio Normalization...")
    preprocess_dataset(SRC_DIR, DATA_DIR)

setup_data()

# %% [markdown]
# ## 1. Create Dummy TorchScript Model
# We need a model to run inference. We'll create a simple one that mimics MERT input/output.

# %% 
class DummyMERT(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.head = torch.nn.Linear(10, 64) # Dummy head
        
    def forward(self, x):
        # x is (Batch, Time)
        # We just do a mean pool to fake embedding
        return torch.randn(x.shape[0], 64).to(x.device)

model = DummyMERT().eval()
scripted_model = torch.jit.script(model)
model_path = os.path.join(MODEL_DIR, "encoder.pt")
scripted_model.save(model_path)
print(f"✅ Created dummy model at {model_path}")

# %% [markdown]
# ## 2. Run Indexing Logic
# We call the 'index' function directly to verify the loop.

# %% 
print("Running Indexing...")

# Mock Args
args = argparse.Namespace(
    model_path=model_path,
    data_dir=DATA_DIR,
    output_dir=INDEX_DIR,
    batch_size=4,
    pq_path=None,
    accelerator="cpu",
    precision="bf16-mixed"
)

try:
    run_index(args)
    print("✅ Indexing function returned successfully.")
except Exception as e:
    print(f"❌ Indexing Failed: {e}")
    import traceback
    traceback.print_exc()

# %% [markdown]
# ## 3. Verify Output
# Check if files were written.

# %% 
files = os.listdir(INDEX_DIR)
print(f"Index Directory Contents: {files}")
if len(files) > 0:
    print("✅ Success: Index artifacts generated.")
else:
    print("❌ Fail: No artifacts found.")

# %% [markdown]
# ## 4. Test Release Model
# Run the pipeline with the actual trained model mounted from the training pipeline.

# %% 
print("\nTesting Release Model...")
release_model_path = "/vol/model/encoder.pt"

if not os.path.exists(release_model_path):
    print(f"⚠️ Release model not found at {release_model_path}. Skipping test.")
else:
    print(f"Found release model at {release_model_path}")
    
    # Update args to use release model
    args.model_path = release_model_path
    args.precision = "32" # Force float32 for CPU compatibility
    # Use a separate output dir to not overwrite dummy test
    args.output_dir = os.path.join(INDEX_DIR, "release_test")
    os.makedirs(args.output_dir, exist_ok=True)
    
    try:
        run_index(args)
        print("✅ Release model indexing successful.")
        
        # Verify artifacts
        files = os.listdir(args.output_dir)
        print(f"Release Index Contents: {files}")
        if len(files) > 0:
            print("✅ Success: Release index artifacts generated.")
        else:
            print("❌ Fail: No artifacts found for release model.")
            
    except Exception as e:
        print(f"❌ Release Model Failed: {e}")
