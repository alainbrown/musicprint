# %% [markdown]
# # MusicPrint Development Notebook
# This notebook is for interactive debugging of the DALI pipeline and MERT model.
# Run this in VS Code (Interactive Window) or convert to ipynb.

# %%
import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display

# Add src to path
sys.path.append(os.path.join(os.getcwd(), '../src'))

from data.dali_loader import DALIGPULoader
from models.mert_adapter import MERTAdapter

# Configuration
DATA_DIR = "../data/test_samples" # Relative to this notebook
BATCH_SIZE = 2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Running on: {DEVICE}")
print(f"Data Source: {DATA_DIR}")

# %% [markdown]
# ## 1. Initialize DALI Loader
# We create a loader that mimics the training environment.

# %%
# We need absolute path for DALI usually, but let's try relative
abs_data_dir = os.path.abspath(DATA_DIR)

loader = DALIGPULoader(
    batch_size=BATCH_SIZE,
    file_root=abs_data_dir,
    window_secs=5.0,
    augment=True, # Turn on augmentations to visualize them
    device_id=0 if torch.cuda.is_available() else None
)

print("Loader initialized.")

# %% [markdown]
# ## 2. Visualize Augmented Audio
# Fetch a batch and plot the waveforms to verify data integrity.

# %%
# Get one batch
iterator = iter(loader)
batch = next(iterator)

# Extract data
# DALI outputs [Batch, Time, 1]
audio_gpu = batch[0]["audio"]
labels = batch[0]["label"]

# Move to CPU for plotting
audio_cpu = audio_gpu.cpu().numpy().squeeze() # (Batch, Time)
print(f"Batch Shape: {audio_cpu.shape}")
print(f"Sample Rate (Target): 24000 Hz")

# Plot
plt.figure(figsize=(15, 6))
for i in range(min(BATCH_SIZE, 2)):
    plt.subplot(2, 1, i+1)
    librosa.display.waveshow(audio_cpu[i], sr=24000)
    plt.title(f"Sample {i} - Label: {labels[i].item()}")
    plt.tight_layout()

plt.show()

# %% [markdown]
# ## 3. Model Inference (MERT)
# Load the heavy model and check the forward pass.

# %%
print("Loading MERT Adapter (this may take 10-20s)...")
model = MERTAdapter().to(DEVICE)
model.eval()
print("Model loaded.")

# %%
# Run Inference
# Input to MERT: (Batch, Time) tensor
audio_tensor = audio_gpu.to(DEVICE).squeeze(-1) 

with torch.no_grad():
    # 1. Continuous Embeddings (Before Tanh/Sign)
    embeddings = model(audio_tensor)
    # 2. Binary Hashes
    hashes = model.get_hash(audio_tensor)

print(f"Embedding Shape: {embeddings.shape}")
print(f"Hash Shape: {hashes.shape}")

# %% [markdown]
# ## 4. Fingerprint Analysis
# Visualize the binary codes. If the image is solid color, the model has collapsed (bad init).
# Ideally, you see a random-looking pattern (white/black barcode).

# %%
plt.figure(figsize=(10, 4))
# Display as heatmap (0=Black, 1=White)
# Map -1 -> 0, +1 -> 1
binary_img = (hashes.cpu().numpy() > 0).astype(int)

plt.imshow(binary_img, aspect='auto', cmap='gray', interpolation='nearest')
plt.xlabel("Hash Bits (0-63)")
plt.ylabel("Batch Sample Index")
plt.title("Generated Fingerprints (Visual Check)")
plt.yticks(range(BATCH_SIZE))
plt.colorbar(label="Bit Value")
plt.show()

# Print Hex representation
for i in range(min(BATCH_SIZE, 5)):
    # Convert bits to hex string
    bits = binary_img[i]
    # Pack bits into integer
    val = int("".join(str(b) for b in bits), 2)
    print(f"Song {i} Hash: {val:016x}")
