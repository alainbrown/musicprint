
import os
import sys
import torch
import subprocess
import struct
import numpy as np

# Paths
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AUDIO_PIPE = os.path.join(ROOT, "1_adapter_training")
LIB_ROOT = os.path.join(ROOT, "libmusicprint")
CLI_BIN = os.path.join(LIB_ROOT, "build/cli_search")

# Fixture Paths
FIXTURE_DIR = os.environ.get("FIXTURE_DIR", "/tmp/musicprint_smoke")
os.makedirs(FIXTURE_DIR, exist_ok=True)

QUERY_BIN = os.path.join(FIXTURE_DIR, "query.bin")
INDEX_BIN = os.path.join(FIXTURE_DIR, "index.bin")
CENTROIDS_BIN = os.path.join(FIXTURE_DIR, "centroids.bin")

# Production Paths
META_BIN = os.path.join(ROOT, "3_meta_tokenizer/release/music_meta.bin")
VOCAB_BIN = os.path.join(ROOT, "3_meta_tokenizer/release/music_decoder.bin")

# Target
TARGET_ISRC = "GBAYE0601498" # Yellow Submarine

def pack_isrc(isrc_str):
    isrc_str = isrc_str.upper()
    c1, c2 = ord(isrc_str[0]) - ord('A'), ord(isrc_str[1]) - ord('A')
    country = (c1 * 26) + c2
    def c2i(c):
        if 'A' <= c <= 'Z': return ord(c) - ord('A')
        if '0' <= c <= '9': return ord(c) - ord('0') + 26
        return 0
    reg = (c2i(isrc_str[2])*36*36) + (c2i(isrc_str[3])*36) + c2i(isrc_str[4])
    year, desig = int(isrc_str[5:7]), int(isrc_str[7:12])
    return (country << 40) | (reg << 24) | (year << 17) | desig

def run_smoke_test(args):
    print("🚀 Starting E2E Smoke Test (Real Audio -> Metadata)...")

    # 1. Generate Query Vector from Real Audio
    print("Step 1: Running MERT Inference on sample1.wav...")
    sys.path.append(os.path.join(AUDIO_PIPE, "src"))
    from models.mert_adapter import MERTAdapter
    import librosa

    if args.checkpoint:
        print(f"Loading model from {args.checkpoint}...")
        from system import MusicPrintSystem
        system = MusicPrintSystem.load_from_checkpoint(args.checkpoint, map_location="cpu")
        model = system.model
        model.eval()
    else:
        print("Using random MERTAdapter weights...")
        model = MERTAdapter()

    audio_path = os.path.join(AUDIO_PIPE, "data/test_samples/sample1.wav")
    # librosa loads as (Time,), we need (Batch, Time)
    # sr=24000 to match MERT requirement
    audio, sr = librosa.load(audio_path, sr=24000)
    audio_tensor = torch.from_numpy(audio).unsqueeze(0) # (1, T)
    
    with torch.no_grad():
        embeddings = model(audio_tensor) # (1, 768)
    
    vector = embeddings[0].numpy()
    with open(QUERY_BIN, "wb") as f:
        f.write(vector.astype(np.float32).tobytes())

    # 2. Generate Index (Binary Hamming)
    if args.skip_index and os.path.exists(INDEX_BIN):
        print(f"✅ Using existing fixtures in {FIXTURE_DIR}")
        return

    print("Step 2: Creating temporary index fixtures...")
    
    # Binarize vector for Index
    # Matches Python: bit[i] = 1 if float[i] > 0
    bits = (vector > 0).astype(np.uint8) # (64,)
    packed = np.packbits(bits, bitorder='little') # (8,)
    pq_int = struct.unpack("<Q", packed.tobytes())[0]

    with open(INDEX_BIN, "wb") as f:
        # Header: Magic(4), Version(4), Count(4), Padding(52)
        f.write(struct.pack("<4sII", b"MPAF", 1, 1))
        f.write(b"\x00" * 52)
        
        # Entry: Hash(8) + Packed ISRC(8)
        f.write(struct.pack("<QQ", pq_int, pack_isrc(TARGET_ISRC)))

    print(f"✅ Fixtures generated at {FIXTURE_DIR}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, help="Path to trained model checkpoint")
    parser.add_argument("--skip_index", action="store_true", help="Skip index generation if fixtures exist")
    args = parser.parse_args()
    run_smoke_test(args)
