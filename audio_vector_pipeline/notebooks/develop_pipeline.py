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
import matplotlib.pyplot as plt
import requests
from tqdm import tqdm

# Add src to path
sys.path.append(os.path.join(os.getcwd(), '../src'))

from isrc_utils import pack_isrc, unpack_isrc
from models.mert_adapter import MERTAdapter
from system import MusicPrintSystem

# %% [markdown]
# ## 0. Download Test Samples
# If no samples exist, we fetch 5 tracks from Jamendo.

# %%
DATA_DIR = "../data/test_samples"
os.makedirs(DATA_DIR, exist_ok=True)

def download_samples():
    samples = [
        ("JAMENDO_001", "https://prod-1.storage.jamendo.com/?trackid=1890710&format=mp31&from=app"), # Rock
        ("JAMENDO_002", "https://prod-1.storage.jamendo.com/?trackid=1890711&format=mp31&from=app"), # Electronic
        ("JAMENDO_003", "https://prod-1.storage.jamendo.com/?trackid=1890712&format=mp31&from=app"), # Jazz
        ("JAMENDO_004", "https://prod-1.storage.jamendo.com/?trackid=1890713&format=mp31&from=app"), # Pop
        ("JAMENDO_005", "https://prod-1.storage.jamendo.com/?trackid=1890714&format=mp31&from=app")  # Classical
    ]
    
    for name, url in samples:
        target = os.path.join(DATA_DIR, f"{name}.mp3")
        if not os.path.exists(target):
            print(f"Downloading {name}...")
            r = requests.get(url, stream=True)
            with open(target, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
    print("Samples ready.")

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
# ## 2. Verify Sphere Method on Real Audio
# We load MERT and see how it deduplicates a real track.

# %%
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

model = MERTAdapter().to(DEVICE).eval()
sample_path = os.path.join(DATA_DIR, os.listdir(DATA_DIR)[0])

# Load 30s of audio
import librosa
audio, sr = librosa.load(sample_path, sr=24000, duration=30)
audio_pt = torch.from_numpy(audio).to(DEVICE)

# Create 5s sliding windows (1s stride)
windows = audio_pt.unfold(0, 5 * 24000, 1 * 24000) # (N, 120000)

with torch.no_grad():
    embeddings = model(windows)
    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

# Apply Sphere Method
selected = []
threshold = 0.85

for e in embeddings:
    if not selected:
        selected.append(e)
        continue
    stack = torch.stack(selected)
    sim = torch.matmul(stack, e)
    if torch.max(sim) < threshold:
        selected.append(e)

print(f"Source: {os.path.basename(sample_path)}")
print(f"Total Windows: {len(embeddings)}")
print(f"Unique Fingerprints: {len(selected)} (Threshold: {threshold})")

# %% [markdown]
# ## 3. Verify PQ Quantization Accuracy
# Simulation of the 8-byte compression loss.

# %%
import faiss

def test_pq_quality(embeddings_tensor):
    data = embeddings_tensor.cpu().numpy().astype('float32')
    d = 64
    m = 8 
    nbits = 8
    
    pq = faiss.ProductQuantizer(d, m, nbits)
    # We use the small sample to train for this test
    pq.train(data)
    
    codes = pq.compute_codes(data)
    recon = pq.decode(codes)
    
    # Calculate average cosine similarity between original and reconstructed
    orig_norm = data / np.linalg.norm(data, axis=1, keepdims=True)
    recon_norm = recon / np.linalg.norm(recon, axis=1, keepdims=True)
    cosine_sims = np.sum(orig_norm * recon_norm, axis=1)
    
    print(f"PQ (8-byte) Reconstruction Accuracy (Avg Cosine Sim): {np.mean(cosine_sims):.4f}")
    print(f"Byte Code Example: {codes[0]}")

test_pq_quality(embeddings)
