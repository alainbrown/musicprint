"""
MusicPrint end-to-end verification script.

Runs all pipelines in sequence, then tests recall on a 5% sample.
Usage: docker compose up verify
"""
import os
import sys
import subprocess
import random
import time
import shutil
import struct

import numpy as np
import torch

# Paths
ROOT = "/workspace"
sys.path.insert(0, os.path.join(ROOT, "demo_app"))

from audio import load_and_resample, window_audio, binarize
from search import SearchEngine

MUSIC_DIR = os.path.join(ROOT, "music")

# All working files go to Docker volumes — never write to /workspace
DATA_DIR = "/vol/data"
CHECKPOINT_DIR = "/vol/checkpoints"
ARTIFACTS_DIR = "/vol/artifacts"

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(ARTIFACTS_DIR, exist_ok=True)


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


def file_to_id(filepath, data_dir):
    """Convert a file path to an ID, matching the data module's logic.

    Uses pack_isrc if the stem is a 12-char ISRC, otherwise hashes
    the relative path. Must match 2_vector_index/src/data/module.py.
    """
    rel_path = os.path.relpath(filepath, data_dir)
    name = os.path.splitext(rel_path)[0]
    if len(name) == 12:
        return pack_isrc(name)
    return hash(name) & 0xFFFFFFFFFFFFFFFF


def discover_tracks(music_dir):
    """Recursively find audio files. Returns (filepaths, ids)."""
    import glob
    files = sorted(glob.glob(os.path.join(music_dir, "**/*.flac"), recursive=True))
    if not files:
        files = sorted(glob.glob(os.path.join(music_dir, "**/*.wav"), recursive=True))
    if not files:
        files = sorted(glob.glob(os.path.join(music_dir, "**/*.mp3"), recursive=True))
    ids = [file_to_id(f, music_dir) for f in files]
    return files, ids


def degrade_audio(audio, sr=24000):
    snr_db = random.uniform(5, 15)
    signal_power = np.mean(audio ** 2)
    noise_power = signal_power / (10 ** (snr_db / 10))
    audio = audio + np.random.normal(0, np.sqrt(noise_power), len(audio)).astype(np.float32)

    gain_db = random.uniform(-12, 6)
    audio = audio * (10 ** (gain_db / 20))

    fft = np.fft.rfft(audio)
    freqs = np.fft.rfftfreq(len(audio), 1.0 / sr)
    fft[freqs > 4000] = 0
    audio = np.fft.irfft(fft, n=len(audio)).astype(np.float32)

    return audio


def run_recall_test(track_paths, track_ids, engine, encoder, device, degrade=False, indices=None):
    if indices is None:
        sample_size = max(1, len(track_paths) // 20)
        indices = random.sample(range(len(track_paths)), sample_size)
    correct = 0
    total_time = 0

    for idx in indices:
        track_path = track_paths[idx]
        expected_id = track_ids[idx]

        audio = load_and_resample(track_path)

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

        if best_result and best_result["song_id"] == expected_id:
            correct += 1

    avg_time = total_time / max(len(indices), 1)
    return correct, len(indices), avg_time


def run_step(name, cmd):
    print(f"\n{'=' * 60}")
    print(f"  {name}")
    print(f"{'=' * 60}\n")
    result = subprocess.run(cmd, cwd=ROOT)
    if result.returncode != 0:
        print(f"\nFAILED: {name} (exit code {result.returncode})")
        sys.exit(1)


def create_subset(track_paths, max_songs, music_dir):
    """Create a temp directory with symlinks to a random subset of songs."""
    import tempfile
    subset_dir = tempfile.mkdtemp(prefix="musicprint_subset_")
    random.seed(42)
    selected = random.sample(track_paths, min(max_songs, len(track_paths)))
    for path in selected:
        rel = os.path.relpath(path, music_dir)
        link = os.path.join(subset_dir, rel)
        os.makedirs(os.path.dirname(link), exist_ok=True)
        os.symlink(path, link)
    return subset_dir


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_songs", type=int, default=0, help="Limit to N songs (0 = all)")
    args = parser.parse_args()

    print("=" * 60)
    print("  MUSICPRINT END-TO-END VERIFICATION")
    print("=" * 60)

    # Discover tracks (recursive, matches data module logic)
    track_paths, track_ids = discover_tracks(MUSIC_DIR)
    if not track_paths:
        print(f"ERROR: No audio files found in {MUSIC_DIR}")
        sys.exit(1)
    print(f"Found {len(track_paths)} tracks in music/")

    # Optional subset
    source_dir = MUSIC_DIR
    if args.max_songs > 0 and args.max_songs < len(track_paths):
        source_dir = create_subset(track_paths, args.max_songs, MUSIC_DIR)
        track_paths, track_ids = discover_tracks(source_dir)
        print(f"Using subset of {len(track_paths)} tracks for quick verification")

    # Step 1: Train encoder
    env = os.environ.copy()
    env["WANDB_MODE"] = "disabled"
    run_step("Step 1: Train Encoder", [
        sys.executable, "1_adapter_training/src/pipeline.py",
        "--source_dir", source_dir,
        "--data_dir", DATA_DIR,
        "--checkpoint_dir", CHECKPOINT_DIR,
        "--release_dir", ARTIFACTS_DIR,
        "--auto_batch_size",
    ])

    encoder_path = os.path.join(ARTIFACTS_DIR, "encoder.pt")
    assert os.path.exists(encoder_path), "encoder.pt not found after training"

    # Step 2: Build index
    index_dir = os.path.join(ARTIFACTS_DIR, "index")
    run_step("Step 2: Build Index", [
        sys.executable, "2_vector_index/src/pipeline.py",
        "--model_path", encoder_path,
        "--source_dir", source_dir,
        "--data_dir", DATA_DIR,
        "--index_dir", index_dir,
    ])

    index_path = os.path.join(ARTIFACTS_DIR, "audio_index.bin")
    assert os.path.exists(index_path), "audio_index.bin not found after indexing"

    with open(index_path, "rb") as f:
        _, _, count = struct.unpack("<4sII", f.read(12))
    index_size_mb = os.path.getsize(index_path) / 1e6
    print(f"Index built: {count} entries, {index_size_mb:.1f} MB")

    # Step 3: Copy metadata
    print(f"\n{'=' * 60}")
    print(f"  Step 3: Copy Metadata Artifacts")
    print(f"{'=' * 60}\n")
    meta_src = os.path.join(ROOT, "3_meta_tokenizer", "release")
    for fname in ["music_meta.bin", "music_decoder.bin"]:
        src = os.path.join(meta_src, fname)
        dst = os.path.join(ARTIFACTS_DIR, fname)
        if os.path.exists(src):
            shutil.copy2(src, dst)
            print(f"Copied {fname} ({os.path.getsize(dst) / 1024:.0f} KB)")
        else:
            print(f"WARNING: {fname} not found at {src}")

    # Step 4: Recall tests
    print(f"\n{'=' * 60}")
    print(f"  Step 4: Recall Verification")
    print(f"{'=' * 60}\n")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder = torch.jit.load(encoder_path, map_location=device)
    encoder.eval()
    engine = SearchEngine(index_path)

    sample_size = max(1, len(track_paths) // 20)
    test_indices = random.sample(range(len(track_paths)), sample_size)

    print(f"Testing {sample_size} tracks ({sample_size / len(track_paths) * 100:.0f}% of catalog)...")

    print("\nClean recall (10s clips)...")
    clean_correct, clean_total, clean_time = run_recall_test(
        track_paths, track_ids, engine, encoder, device, degrade=False, indices=test_indices
    )
    clean_recall = clean_correct / max(clean_total, 1) * 100

    print("Degraded recall (noise + volume + low-pass)...")
    deg_correct, deg_total, deg_time = run_recall_test(
        track_paths, track_ids, engine, encoder, device, degrade=True, indices=test_indices
    )
    deg_recall = deg_correct / max(deg_total, 1) * 100

    # Summary
    print(f"\n{'=' * 60}")
    print(f"  MUSICPRINT VERIFICATION REPORT")
    print(f"{'=' * 60}")
    print(f"Catalog:          {len(track_paths)} songs")
    print(f"Index entries:    {count}")
    print(f"Index size:       {index_size_mb:.1f} MB")
    print(f"Clean recall:     {clean_correct}/{clean_total} ({clean_recall:.1f}%)")
    print(f"Degraded recall:  {deg_correct}/{deg_total} ({deg_recall:.1f}%)")
    print(f"Avg query time:   {clean_time:.3f}s (clean), {deg_time:.3f}s (degraded)")
    print(f"{'=' * 60}")

    if clean_recall == 0:
        print("\nWARNING: Zero clean recall — something is likely broken.")
        sys.exit(1)

    print("\nVERIFICATION COMPLETE")


if __name__ == "__main__":
    main()
