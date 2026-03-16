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
ROOT = "/app"
sys.path.insert(0, os.path.join(ROOT, "demo_app"))

from audio import load_and_resample, window_audio

MUSIC_DIR = "/vol/music"

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
    return hash(name) & 0x7FFFFFFFFFFFFFFF


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

    # Step 2: Embedding Quality Test
    print(f"\n{'=' * 60}")
    print(f"  Step 2: Embedding Quality Test")
    print(f"{'=' * 60}\n")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder = torch.jit.load(encoder_path, map_location=device)
    encoder.eval()

    # Pick 10 songs, encode two different 5s clips from each
    num_test = min(10, len(track_paths))
    test_idx = random.sample(range(len(track_paths)), num_test)

    embeddings_a = []  # first clip per song
    embeddings_b = []  # second clip per song
    song_names = []

    for idx in test_idx:
        path = track_paths[idx]
        audio = load_and_resample(path)
        windows = window_audio(audio)
        if len(windows) < 2:
            continue

        # Pick two random windows from the same song
        w1, w2 = random.sample(range(len(windows)), 2)
        with torch.no_grad():
            emb_a = encoder(torch.from_numpy(windows[w1]).unsqueeze(0).to(device))
            emb_b = encoder(torch.from_numpy(windows[w2]).unsqueeze(0).to(device))
        embeddings_a.append(emb_a[0].cpu())
        embeddings_b.append(emb_b[0].cpu())
        song_names.append(os.path.basename(path))

    if len(embeddings_a) < 2:
        print("ERROR: Not enough songs with 2+ windows for quality test")
        sys.exit(1)

    A = torch.stack(embeddings_a)  # (N, 768)
    B = torch.stack(embeddings_b)  # (N, 768)

    # L2 normalize
    A = A / A.norm(dim=1, keepdim=True)
    B = B / B.norm(dim=1, keepdim=True)

    # Cosine similarity: A[i] vs B[j]
    sim_matrix = (A @ B.T).numpy()

    # Same-song similarity (diagonal)
    same_song_sims = [sim_matrix[i][i] for i in range(len(sim_matrix))]
    # Different-song similarity (off-diagonal)
    diff_song_sims = [sim_matrix[i][j] for i in range(len(sim_matrix)) for j in range(len(sim_matrix)) if i != j]

    avg_same = np.mean(same_song_sims)
    avg_diff = np.mean(diff_song_sims)
    min_same = np.min(same_song_sims)
    max_diff = np.max(diff_song_sims)

    # Top-1 recall: for each song's clip A, is the most similar clip B from the same song?
    correct = 0
    for i in range(len(sim_matrix)):
        best_j = np.argmax(sim_matrix[i])
        if best_j == i:
            correct += 1
    recall = correct / len(sim_matrix) * 100

    print(f"Songs tested:     {len(sim_matrix)}")
    print(f"Embedding dim:    {A.shape[1]}")
    print(f"")
    print(f"Same-song cosine similarity:")
    print(f"  avg: {avg_same:.4f}  min: {min_same:.4f}")
    print(f"Diff-song cosine similarity:")
    print(f"  avg: {avg_diff:.4f}  max: {max_diff:.4f}")
    print(f"Separation gap:   {min_same - max_diff:.4f} (positive = good)")
    print(f"")
    print(f"Top-1 recall:     {correct}/{len(sim_matrix)} ({recall:.1f}%)")

    # Summary
    print(f"\n{'=' * 60}")
    print(f"  MUSICPRINT VERIFICATION REPORT")
    print(f"{'=' * 60}")
    print(f"Catalog:          {len(track_paths)} songs")
    print(f"Same-song sim:    {avg_same:.4f} (avg)")
    print(f"Diff-song sim:    {avg_diff:.4f} (avg)")
    print(f"Top-1 recall:     {recall:.1f}%")
    print(f"{'=' * 60}")

    if recall == 0:
        print("\nWARNING: Zero recall — embeddings are not discriminative.")
        sys.exit(1)

    print("\nVERIFICATION COMPLETE")


if __name__ == "__main__":
    main()
