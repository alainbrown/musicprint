# %% [markdown]
# # MusicPrint End-to-End Demo
#
# This notebook trains the encoder, builds the search index, and verifies
# recall on a subset of the catalog.
#
# **Prerequisites:**
# - MP3 files in `music/` named by 12-character ISRC (e.g., `GBAYE0601498.mp3`)
# - Running inside the training pipeline container (`docker compose exec training-pipeline bash`)
#
# **Outputs:** All artifacts written to `release/`.

# %%
import os
import sys
import subprocess
import random
import time
import shutil
import struct

import numpy as np
import torch

# Add demo_app to path for search/audio modules
# In Jupyter, __file__ is not defined, so we use cwd (must be repo root)
ROOT = os.getcwd()
sys.path.insert(0, os.path.join(ROOT, "demo_app"))

from audio import load_and_resample, window_audio, binarize
from search import SearchEngine

MUSIC_DIR = os.path.join(ROOT, "music")
RELEASE_DIR = os.path.join(ROOT, "release")
DATA_DIR = "/tmp/musicprint_data"
CHECKPOINT_DIR = "/tmp/musicprint_checkpoints"

os.makedirs(RELEASE_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# List all tracks
tracks = [f for f in os.listdir(MUSIC_DIR) if f.endswith((".mp3", ".flac", ".wav"))]
isrcs = [os.path.splitext(f)[0] for f in tracks]
print(f"Found {len(tracks)} tracks in music/")

# %% [markdown]
# ## Step 1: Train Encoder

# %%
print("Training encoder (this may take a while)...")
env = os.environ.copy()
env["WANDB_MODE"] = "disabled"

result = subprocess.run(
    [
        sys.executable,
        "1_adapter_training/src/pipeline.py",
        "--source_dir", MUSIC_DIR,
        "--data_dir", DATA_DIR,
        "--checkpoint_dir", CHECKPOINT_DIR,
        "--release_dir", RELEASE_DIR,
    ],
    env=env,
    cwd=ROOT,
)
assert result.returncode == 0, f"Training failed with code {result.returncode}"
assert os.path.exists(os.path.join(RELEASE_DIR, "encoder.pt")), "encoder.pt not found"
print("Encoder trained successfully.")

# %% [markdown]
# ## Step 2: Build Index

# %%
print("Building index...")
INDEX_DIR = os.path.join(RELEASE_DIR, "index")

result = subprocess.run(
    [
        sys.executable,
        "2_vector_index/src/pipeline.py",
        "--model_path", os.path.join(RELEASE_DIR, "encoder.pt"),
        "--source_dir", MUSIC_DIR,
        "--data_dir", DATA_DIR,
        "--index_dir", INDEX_DIR,
    ],
    cwd=ROOT,
)
assert result.returncode == 0, f"Indexing failed with code {result.returncode}"

INDEX_PATH = os.path.join(RELEASE_DIR, "audio_index.bin")
assert os.path.exists(INDEX_PATH), "audio_index.bin not found"

# Report index stats
with open(INDEX_PATH, "rb") as f:
    _, _, count = struct.unpack("<4sII", f.read(12))
index_size_mb = os.path.getsize(INDEX_PATH) / 1e6
print(f"Index built: {count} entries, {index_size_mb:.1f} MB")

# %% [markdown]
# ## Step 3: Copy Metadata Artifacts

# %%
META_SRC = os.path.join(ROOT, "3_meta_tokenizer", "release")
for fname in ["music_meta.bin", "music_decoder.bin"]:
    src = os.path.join(META_SRC, fname)
    dst = os.path.join(RELEASE_DIR, fname)
    if os.path.exists(src):
        shutil.copy2(src, dst)
        print(f"Copied {fname} ({os.path.getsize(dst) / 1024:.0f} KB)")
    else:
        print(f"WARNING: {fname} not found at {src}")

# %% [markdown]
# ## Step 4 & 5: Clean and Degraded Recall Verification

# %%
# Helper: pack ISRC to uint64 (matches build_db.py)
def pack_isrc(isrc_str):
    if not isrc_str or len(isrc_str) != 12:
        return 0
    isrc_str = isrc_str.upper()
    c1, c2 = ord(isrc_str[0]) - ord('A'), ord(isrc_str[1]) - ord('A')
    country = (c1 * 26) + c2
    def c2i(c):
        if 'A' <= c <= 'Z': return ord(c) - ord('A')
        if '0' <= c <= '9': return ord(c) - ord('0') + 26
        return 0
    reg = (c2i(isrc_str[2]) * 36 * 36) + (c2i(isrc_str[3]) * 36) + c2i(isrc_str[4])
    year, desig = int(isrc_str[5:7]), int(isrc_str[7:12])
    return (country << 40) | (reg << 24) | (year << 17) | desig

def degrade_audio(audio, sr=24000):
    """Apply combined degradation: noise + volume + low-pass."""
    # Additive white noise (SNR 5-15 dB)
    snr_db = random.uniform(5, 15)
    signal_power = np.mean(audio ** 2)
    noise_power = signal_power / (10 ** (snr_db / 10))
    audio = audio + np.random.normal(0, np.sqrt(noise_power), len(audio)).astype(np.float32)

    # Volume scaling (-12 to +6 dB)
    gain_db = random.uniform(-12, 6)
    audio = audio * (10 ** (gain_db / 20))

    # Low-pass filter at 4kHz (simple FIR via FFT)
    fft = np.fft.rfft(audio)
    freqs = np.fft.rfftfreq(len(audio), 1.0 / sr)
    fft[freqs > 4000] = 0
    audio = np.fft.irfft(fft, n=len(audio)).astype(np.float32)

    return audio

def run_recall_test(tracks, isrcs, engine, encoder, device, degrade=False, indices=None):
    """Test recall on a subset of tracks. Returns (correct, total, avg_time)."""
    if indices is None:
        sample_size = max(1, len(tracks) // 20)  # 5%
        indices = random.sample(range(len(tracks)), sample_size)
    correct = 0
    total_time = 0

    for idx in indices:
        track_path = os.path.join(MUSIC_DIR, tracks[idx])
        expected_isrc = pack_isrc(isrcs[idx])

        audio = load_and_resample(track_path)

        # Random 10-second segment
        ten_sec = 24000 * 10
        if len(audio) > ten_sec:
            start = random.randint(0, len(audio) - ten_sec)
            audio = audio[start : start + ten_sec]

        if degrade:
            audio = degrade_audio(audio)

        windows = window_audio(audio)
        if not windows:
            continue

        t0 = time.time()
        best_result = None
        for win in windows:
            tensor = torch.from_numpy(win).unsqueeze(0).to(device)
            with torch.no_grad():
                embedding = encoder(tensor)
            query_hash = binarize(embedding[0].cpu().numpy())
            result = engine.search(query_hash)
            if result and (best_result is None or result["distance"] < best_result["distance"]):
                best_result = result
        total_time += time.time() - t0

        if best_result and best_result["song_id"] == expected_isrc:
            correct += 1

    avg_time = total_time / max(len(indices), 1)
    return correct, len(indices), avg_time

# Load encoder and index
device = "cuda" if torch.cuda.is_available() else "cpu"
encoder = torch.jit.load(os.path.join(RELEASE_DIR, "encoder.pt"), map_location=device)
encoder.eval()
engine = SearchEngine(os.path.join(RELEASE_DIR, "audio_index.bin"))

# Select the same 5% sample for both tests (spec: "same 5% of tracks")
sample_size = max(1, len(tracks) // 20)
test_indices = random.sample(range(len(tracks)), sample_size)

# Clean recall
print("Running clean recall test (5% of catalog, 10s clips)...")
clean_correct, clean_total, clean_time = run_recall_test(tracks, isrcs, engine, encoder, device, degrade=False, indices=test_indices)
clean_recall = clean_correct / max(clean_total, 1) * 100
print(f"Clean Recall: {clean_correct}/{clean_total} ({clean_recall:.1f}%) | Avg time: {clean_time:.3f}s")

# Degraded recall (same tracks)
print("\nRunning degraded recall test (noise + volume + low-pass)...")
deg_correct, deg_total, deg_time = run_recall_test(tracks, isrcs, engine, encoder, device, degrade=True, indices=test_indices)
deg_recall = deg_correct / max(deg_total, 1) * 100
print(f"Degraded Recall: {deg_correct}/{deg_total} ({deg_recall:.1f}%) | Avg time: {deg_time:.3f}s")

# %% [markdown]
# ## Summary

# %%
print("=" * 50)
print("MUSICPRINT VERIFICATION REPORT")
print("=" * 50)
print(f"Catalog:          {len(tracks)} songs")
print(f"Index entries:    {engine.num_entries}")
print(f"Index size:       {os.path.getsize(os.path.join(RELEASE_DIR, 'audio_index.bin')) / 1e6:.1f} MB")
print(f"Clean recall:     {clean_correct}/{clean_total} ({clean_recall:.1f}%)")
print(f"Degraded recall:  {deg_correct}/{deg_total} ({deg_recall:.1f}%)")
print(f"Avg query time:   {clean_time:.3f}s (clean), {deg_time:.3f}s (degraded)")
print("=" * 50)
