# %% [markdown]
# # MusicPrint Vetting Notebook
# Use this to verify the logic of the Audio Pipeline before mass-indexing.
# Logic Vetted:
# 1. ISRC Bitpacking (50-bit)
# 2. Sphere Method (Deduplication)
# 3. PQ Quantization Accuracy

# %%
import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import struct

# Add src to path
sys.path.append(os.path.join(os.getcwd(), '../src'))

from isrc_utils import pack_isrc, unpack_isrc
from models.mert_adapter import MERTAdapter
from system import MusicPrintSystem

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
# ## 2. Verify Sphere Method (Deduplication)
# We test if the threshold correctly picks diverse embeddings.

# %%
def test_sphere_method(embeddings, threshold=0.85):
    selected = []
    # Normalize for cosine similarity
    normed = torch.nn.functional.normalize(embeddings, p=2, dim=1)
    
    for i, e in enumerate(normed):
        if not selected:
            selected.append(e)
            continue
        
        stack = torch.stack(selected)
        sim = torch.matmul(stack, e)
        if torch.max(sim) < threshold:
            selected.append(e)
            
    return len(selected)

# Simulate 100 similar vectors (walking through a song)
mock_vectors = torch.randn(100, 64)
for i in range(1, 100):
    # Make each vector 95% similar to the previous one
    mock_vectors[i] = mock_vectors[i-1] * 0.95 + mock_vectors[i] * 0.05

count = test_sphere_method(mock_vectors, threshold=0.85)
print(f"Sphere Method: Reduced 100 windows to {count} unique fingerprints.")

# %% [markdown]
# ## 3. Verify PQ Quantization
# (Note: This requires a trained audio_pq.bin to be fully tested)

# %%
import faiss

def simulate_pq_compression():
    d = 64
    m = 8  # 8 bytes
    nbits = 8
    
    # Create fake data
    data = np.random.random((1000, d)).astype('float32')
    
    # Train PQ
    pq = faiss.ProductQuantizer(d, m, nbits)
    pq.train(data)
    
    # Compress and Decompress
    codes = pq.compute_codes(data)
    rev = pq.decode(codes)
    
    # Calculate MSE / Cosine Sim loss
    mse = np.mean((data - rev)**2)
    print(f"PQ Compression MSE: {mse:.6f}")
    print(f"Code Shape: {codes.shape} (Example: {codes[0]})")

simulate_pq_compression()