# %% [markdown]
# # Album Art Pipeline: End-to-End Vetting
# 
# This notebook validates the entire Visual Layer pipeline, from fetching data to on-device simulation.
# 
# **Architecture:**
# 1. **Ingestion:** Manifest -> Cover Art Archive (CAA) [Uses `src/download_images.py`]
# 2. **Tokenizer:** 128x128 Image -> VQ-VAE -> 16x16 Tokens [Uses `src/train.py`]
# 3. **Storage:** Tokens -> `art.bin` (Fixed Layout) [Uses `src/build_index.py`]
# 4. **Decoding:** `art.bin` + Codebook -> Image (iOS Simulation)

# %% 
import os
import sys
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms

# Add src to path
sys.path.append(os.path.join(os.getcwd(), '../src'))
from model import VQVAE
import download_images
import train
import build_index

# Configuration
TEST_MANIFEST = "/tmp/test_manifest.csv"
TEST_CACHE_DIR = "/tmp/test_covers"
TEST_MODEL_PATH = "/tmp/overfit_encoder.pth"
TEST_BIN = "/tmp/test_art.bin"

# Cleanup from previous runs
if os.path.exists(TEST_MANIFEST): os.remove(TEST_MANIFEST)
# Note: we don't delete cache so we can re-run faster

# %% [markdown]
# ## 1. Ingestion: The Bridge Check
# We load a dummy manifest and verify we can fetch images using the production downloader.

# %% 
def verify_ingestion():
    print(f"--- 1. Ingestion Verification (via src/download_images.py) ---")
    
    # Create Dummy Manifest
    df = pd.DataFrame([
        {"album_index": 0, "release_uuid": "12fb801c-5903-4a99-a10c-7af599fe3986", "album_name": "Dark Side of the Moon"},
        {"album_index": 1, "release_uuid": "12fb801c-5903-4a99-a10c-7af599fe3986", "album_name": "Dark Side of the Moon (Duplicate)"}
    ])
    df.to_csv(TEST_MANIFEST, index=False)
    print(f"Created Test Manifest: {TEST_MANIFEST}")

    # Call Production Code
    download_images.main(manifest_path=TEST_MANIFEST, output_dir=TEST_CACHE_DIR)
    
    # Verify Results
    downloaded = []
    for _, row in df.iterrows():
        # Re-use the production logic to find where it put the file
        path = download_images.get_shard_path(row['release_uuid'], TEST_CACHE_DIR)
        file_path = os.path.join(path, f"{row['release_uuid']}.jpg")
        
        if os.path.exists(file_path):
            print(f"  ✅ Verified: {file_path}")
            downloaded.append((row['album_index'], file_path))
        else:
            print(f"  ❌ Missing: {file_path}")

    return downloaded

test_samples = verify_ingestion()

# %% [markdown]
# ## 2. The Tokenizer: Overfit Training Check
# We initialize the VQ-VAE model and train it on our downloaded samples using `src/train.py`.

# %% 
def verify_training():
    print(f"\n--- 2. Overfit Training (via src/train.py) ---")
    
    if not test_samples:
        print("⚠️ No samples to train on.")
        return

    # Call Production Code
    # We set a very small batch size and few epochs just to prove it runs
    train.train(
        data_dir=TEST_CACHE_DIR,
        manifest_path=TEST_MANIFEST,
        epochs=5, # Short run
        batch_size=2,
        auto_batch=False,
        model_output_path=TEST_MODEL_PATH
    )
    
    if os.path.exists(TEST_MODEL_PATH):
         print(f"  ✅ Model Saved: {TEST_MODEL_PATH}")
    else:
         print(f"  ❌ Model Generation Failed")

verify_training()

# %% [markdown]
# ## 3. Binary Packing (Simulation)
# We take our downloaded images and pack them into a mini `art.bin` using `src/build_index.py`.

# %% 
def verify_packing():
    print(f"\n--- 3. Packing Binary (via src/build_index.py) ---")
    
    if not os.path.exists(TEST_MODEL_PATH):
        print("Skipping packing (no model).")
        return

    # Call Production Code
    build_index.build_index(
        manifest_path=TEST_MANIFEST, 
        output_bin=TEST_BIN, 
        model_path=TEST_MODEL_PATH,
        covers_dir=TEST_CACHE_DIR
    )
    
    if os.path.exists(TEST_BIN):
        size = os.path.getsize(TEST_BIN)
        print(f"  ✅ Binary Generated: {TEST_BIN} ({size} bytes)")
    else:
        print(f"  ❌ Binary Generation Failed")

verify_packing()

# %% [markdown]
# ## 4. iOS Simulator (The Decoder)
# This simulates the C++ client on the iPhone.
# It reads the binary file we just created and decodes it using the model's codebook.

# %% 
def ios_decoder_simulation(target_idx):
    print(f"\n--- 4. iOS Simulation for Index {target_idx} ---")
    
    if not os.path.exists(TEST_BIN):
        print("Binary not found.")
        return

    # Load Model to get Codebook
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Hyperparams must match training
    model = VQVAE(128, 2, 32, 1024, 64, 0.25).to(device)
    model.load_state_dict(torch.load(TEST_MODEL_PATH, map_location=device))
    model.eval()

    # 1. Read Tokens from Binary
    with open(TEST_BIN, "rb") as f:
        offset = target_idx * 512 # 256 tokens * 2 bytes
        f.seek(offset)
        data = f.read(512)
        
        if not data or len(data) < 512:
            print("Index out of bounds.")
            return

        tokens = np.frombuffer(data, dtype=np.uint16)
        
        # Check for placeholder
        if tokens[0] == 65535:
            print("Found Placeholder (No Art).")
            return

    print(f"Read 256 tokens. First 10: {tokens[:10]}")

    # 2. Reconstruct Tensor from Codebook
    codebook = model._vq_vae._embedding.weight.data # [1024, 64]
    
    # Lookup: [256] -> [256, 64]
    indices_tensor = torch.from_numpy(tokens.astype(np.int64)).to(device)
    quantized_flat = torch.index_select(codebook, 0, indices_tensor)
    
    # Reshape: [1, 64, 16, 16] (CoreML Input Format)
    quantized = quantized_flat.view(1, 16, 16, 64).permute(0, 3, 1, 2)
    
    # 3. Decode
    with torch.no_grad():
        recon = model._decoder(quantized)
        
    # 4. Visualize
    recon_img = recon.squeeze(0).cpu() * 0.5 + 0.5 # Un-normalize
    recon_pil = transforms.ToPILImage()(recon_img)
    
    plt.figure(figsize=(10, 5))
    
    # Original
    # We need to find the file again
    path = download_images.get_shard_path("12fb801c-5903-4a99-a10c-7af599fe3986", TEST_CACHE_DIR)
    original_path = os.path.join(path, "12fb801c-5903-4a99-a10c-7af599fe3986.jpg")
    
    if os.path.exists(original_path):
        plt.subplot(1, 2, 1)
        plt.title("Original (250px)")
        plt.imshow(Image.open(original_path))
        plt.axis('off')
    
    # Reconstructed
    plt.subplot(1, 2, 2)
    plt.title("Decoded from art.bin (128px)")
    plt.imshow(recon_pil)
    plt.axis('off')
    plt.show()
    print("Displaying comparison...")

if test_samples:
    # Simulate decoding for the first valid sample
    ios_decoder_simulation(test_samples[0][0])