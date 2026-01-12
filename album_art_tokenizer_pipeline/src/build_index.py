
import os
import sys
import torch
import pandas as pd
import numpy as np
import struct
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

# Add src to path
sys.path.append(os.path.dirname(__file__))
from model import VQVAE

# Configuration
MANIFEST_PATH = "data/album_manifest.csv"
OUTPUT_BIN = "release/art.bin"
MODEL_PATH = "release/visual_encoder.pth"
COVERS_DIR = "/vol/data"

# Model Hyperparameters (Must match training)
NUM_HIDDENS = 128
NUM_RESIDUAL_LAYERS = 2
NUM_RESIDUAL_HIDDENS = 32
NUM_EMBEDDINGS = 1024 # 10-bit vocab
EMBEDDING_DIM = 64

def get_shard_path(uuid, covers_dir):
    if not uuid or len(uuid) < 4: return "misc"
    return os.path.join(covers_dir, uuid[:2], uuid[2:4], f"{uuid}.jpg")

def build_index(manifest_path=MANIFEST_PATH, output_bin=OUTPUT_BIN, model_path=MODEL_PATH, covers_dir=COVERS_DIR):
    print(">>> Building Album Art Binary Index (art.bin)...")
    
    # 1. Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Load Model
    model = VQVAE(NUM_HIDDENS, NUM_RESIDUAL_LAYERS, NUM_RESIDUAL_HIDDENS,
                  NUM_EMBEDDINGS, EMBEDDING_DIM).to(device)
    
    if os.path.exists(model_path):
        print(f"Loading trained model from {model_path}...")
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        print("⚠️ WARNING: No trained model found! Using random weights for testing.")
        print("   (Run src/train.py to generate a real codebook)")
    
    model.eval()

    # 3. Load Manifest
    if not os.path.exists(manifest_path):
        print("Error: Manifest not found.")
        return
    
    df = pd.read_csv(manifest_path)
    print(f"Processing {len(df):,} albums...")

    # 4. Preprocessing
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # 5. Build Binary
    # Format: Flat sequence of uint16. 
    # Each record is fixed size: 16x16 = 256 tokens.
    # Total bytes per record: 512 bytes.
    
    batch_size = 64
    total_processed = 0
    missing_count = 0
    
    with open(output_bin, "wb") as f_out:
        # We process in chunks to batch GPU inference
        for i in tqdm(range(0, len(df), batch_size)):
            chunk = df.iloc[i:i+batch_size]
            
            # Prepare Batch
            images = []
            valid_indices = [] # Which positions in the batch are valid images
            
            for local_idx, row in enumerate(chunk.itertuples()):
                path = get_shard_path(row.release_uuid, covers_dir)
                
                if os.path.exists(path):
                    try:
                        img = Image.open(path).convert('RGB')
                        images.append(transform(img))
                        valid_indices.append(local_idx)
                    except:
                        # Corrupt image, treat as missing
                        pass
                
            # Inference (only if we have images)
            tokens_batch = []
            if images:
                img_tensor = torch.stack(images).to(device)
                with torch.no_grad():
                    # Encoder pass
                    z = model._encoder(img_tensor)
                    z = model._pre_vq_conv(z)
                    # VQ pass (returns loss, quantized, perplexity, encodings, encoding_indices)
                    # We modified model.py to return encoding_indices as 4th element
                    # Wait, let's check model.py signature:
                    # return loss, quantized.permute(0, 3, 1, 2).contiguous(), encodings, encoding_indices
                    _, _, _, encoding_indices = model._vq_vae(z)
                    
                    # encoding_indices shape: [Batch, H*W, 1] -> Flatten to [Batch, H*W]
                    tokens_batch = encoding_indices.view(len(images), -1).cpu().numpy()

            # Write to disk (Re-aligning with the manifest order)
            # If an image was missing, we write a placeholder
            current_batch_tokens = []
            token_ptr = 0
            
            for local_idx in range(len(chunk)):
                if local_idx in valid_indices:
                    # Valid Art: Write the 256 tokens
                    # Cast to uint16 (0-65535)
                    data = tokens_batch[token_ptr].astype(np.uint16)
                    token_ptr += 1
                else:
                    # Missing Art: Write Placeholder (0xFFFF)
                    # This tells the decoder "Do not render"
                    data = np.full((256,), 65535, dtype=np.uint16)
                    missing_count += 1
                
                f_out.write(data.tobytes())
            
            total_processed += len(chunk)

    print(f"\n>>> Build Complete.")
    print(f"Output: {OUTPUT_BIN}")
    print(f"Total Albums: {total_processed}")
    print(f"Missing/Skipped: {missing_count} ({missing_count/total_processed*100:.1f}%)")

if __name__ == "__main__":
    build_index()
