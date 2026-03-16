"""
MusicPrint verification script.

Usage:
  python scripts/verify.py --max_songs 100 --skip_training
  python scripts/verify.py --max_songs 100 --skip_training --stride 5
  python scripts/verify.py --max_songs 100 --skip_training --kmeans 10
"""
import os
import sys
import subprocess
import random
import glob

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio

ROOT = "/app"
MUSIC_DIR = "/vol/music"
DATA_DIR = "/vol/data"
CHECKPOINT_DIR = "/vol/checkpoints"
ARTIFACTS_DIR = "/vol/artifacts"

SAMPLE_RATE = 24000
WINDOW_SAMPLES = 120000  # 5 seconds
CHUNK_SIZE = 32

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(ARTIFACTS_DIR, exist_ok=True)


def discover_tracks(music_dir):
    for ext in ("*.flac", "*.wav", "*.mp3"):
        files = sorted(glob.glob(os.path.join(music_dir, "**", ext), recursive=True))
        if files:
            return files
    return []


def create_subset(track_paths, max_songs, music_dir):
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


def load_audio(path):
    audio, sr = torchaudio.load(path)
    if audio.shape[0] > 1:
        audio = audio.mean(dim=0, keepdim=True)
    if sr != SAMPLE_RATE:
        audio = torchaudio.transforms.Resample(sr, SAMPLE_RATE)(audio)
    return audio.squeeze(0)


def make_windows(audio, stride_seconds=1):
    stride = stride_seconds * SAMPLE_RATE
    if audio.shape[0] < WINDOW_SAMPLES:
        return [F.pad(audio, (0, WINDOW_SAMPLES - audio.shape[0]))]
    windows = []
    start = 0
    while start + WINDOW_SAMPLES <= audio.shape[0]:
        windows.append(audio[start : start + WINDOW_SAMPLES])
        start += stride
    return windows


def encode_windows(encoder, windows, device):
    all_embs = []
    for start in range(0, len(windows), CHUNK_SIZE):
        chunk = torch.stack(windows[start : start + CHUNK_SIZE]).to(device)
        with torch.no_grad():
            embs = encoder(chunk)
        all_embs.append(embs.cpu())
    embs = torch.cat(all_embs, dim=0)
    return F.normalize(embs, dim=1)


def kmeans_compress(embeddings, k):
    """Reduce embeddings to k centroids using k-means."""
    from sklearn.cluster import KMeans
    n = len(embeddings)
    if n <= k:
        return embeddings
    data = embeddings.numpy()
    km = KMeans(n_clusters=k, n_init=3, random_state=42)
    km.fit(data)
    centroids = torch.from_numpy(km.cluster_centers_).float()
    return F.normalize(centroids, dim=1)


def run_step(name, cmd):
    print(f"\n{'=' * 60}")
    print(f"  {name}")
    print(f"{'=' * 60}\n")
    result = subprocess.run(cmd, cwd=ROOT)
    if result.returncode != 0:
        print(f"\nFAILED: {name} (exit code {result.returncode})")
        sys.exit(1)


def recall_test(encoder, track_paths, device, stride_seconds=1, kmeans_k=0, query_fraction=0.1):
    """Full-index recall test with optional compression."""

    # Step 1: Build full index (all windows, 1s stride) for query selection
    print("Encoding all songs...")
    all_song_embeddings = []  # list of (song_id, embeddings) before compression
    skipped = 0

    for song_id, path in enumerate(track_paths):
        try:
            audio = load_audio(path)
            windows = make_windows(audio, stride_seconds=1)  # always 1s for full coverage
            embs = encode_windows(encoder, windows, device)
            all_song_embeddings.append((song_id, embs))
        except Exception:
            skipped += 1

    if skipped:
        print(f"Skipped {skipped} songs due to decode errors")

    # Step 2: Build compressed index (what would be stored)
    print("Building compressed index...")
    index_embeddings = []
    index_song_ids = []

    for song_id, embs in all_song_embeddings:
        if stride_seconds > 1:
            # Re-encode with wider stride
            # Subsample from existing: take every Nth window
            step = stride_seconds
            embs_compressed = embs[::step]
        else:
            embs_compressed = embs

        if kmeans_k > 0:
            embs_compressed = kmeans_compress(embs_compressed, kmeans_k)

        index_embeddings.append(embs_compressed)
        index_song_ids.extend([song_id] * len(embs_compressed))

    index = torch.cat(index_embeddings, dim=0)
    index_ids = torch.tensor(index_song_ids)

    # Step 3: Build query set from full (uncompressed) windows
    all_full_embs = []
    all_full_ids = []
    for song_id, embs in all_song_embeddings:
        all_full_embs.append(embs)
        all_full_ids.extend([song_id] * len(embs))

    full_index = torch.cat(all_full_embs, dim=0)
    full_ids = torch.tensor(all_full_ids)
    total_full = len(full_index)

    num_queries = max(1, int(total_full * query_fraction))
    query_indices = random.sample(range(total_full), num_queries)

    songs_indexed = len(all_song_embeddings)
    windows_per_song = len(index) / max(songs_indexed, 1)
    emb_dim = index.shape[1]
    bytes_per_song = windows_per_song * emb_dim * 4

    print(f"Songs: {songs_indexed}")
    print(f"Index: {len(index)} embeddings ({windows_per_song:.0f}/song, {emb_dim}-dim)")
    print(f"Storage: {bytes_per_song / 1024:.1f} KB/song")
    print(f"Queries: {num_queries} (from {total_full} full windows)")

    # Step 4: Query
    correct = 0
    for qi in query_indices:
        query = full_index[qi].unsqueeze(0)
        expected_song = full_ids[qi].item()

        sims = (query @ index.T).squeeze(0)
        best_idx = sims.argmax().item()
        if index_ids[best_idx].item() == expected_song:
            correct += 1

    recall = correct / num_queries * 100
    print(f"Top-1 recall: {correct}/{num_queries} ({recall:.1f}%)")

    return {
        "songs": songs_indexed,
        "index_size": len(index),
        "windows_per_song": windows_per_song,
        "emb_dim": emb_dim,
        "bytes_per_song": bytes_per_song,
        "queries": num_queries,
        "correct": correct,
        "recall": recall,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_songs", type=int, default=0)
    parser.add_argument("--skip_training", action="store_true")
    parser.add_argument("--stride", type=int, default=1, help="Window stride in seconds")
    parser.add_argument("--kmeans", type=int, default=0, help="K-means centroids per song (0=disabled)")
    args = parser.parse_args()

    print("=" * 60)
    print("  MUSICPRINT VERIFICATION")
    print("=" * 60)

    track_paths = discover_tracks(MUSIC_DIR)
    if not track_paths:
        print(f"ERROR: No audio files found in {MUSIC_DIR}")
        sys.exit(1)
    print(f"Found {len(track_paths)} tracks in music/")

    source_dir = MUSIC_DIR
    if args.max_songs > 0 and args.max_songs < len(track_paths):
        source_dir = create_subset(track_paths, args.max_songs, MUSIC_DIR)
        track_paths = discover_tracks(source_dir)
        print(f"Using subset of {len(track_paths)} tracks")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.skip_training:
        print(f"\n  Frozen MERT Baseline (no training)\n")
        sys.path.insert(0, os.path.join(ROOT, "1_adapter_training", "src"))
        from models.mert_adapter import MERTAdapter

        class FrozenMERT(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.backbone = MERTAdapter(output_dim=768)
                self.backbone.adapter = torch.nn.Identity()
            def forward(self, x):
                return self.backbone(x)

        encoder = FrozenMERT().to(device).eval()
    else:
        run_step("Train Encoder", [
            sys.executable, "1_adapter_training/src/pipeline.py",
            "--source_dir", source_dir,
            "--data_dir", DATA_DIR,
            "--checkpoint_dir", CHECKPOINT_DIR,
            "--release_dir", ARTIFACTS_DIR,
        ])
        encoder_path = os.path.join(ARTIFACTS_DIR, "encoder.pt")
        assert os.path.exists(encoder_path)
        encoder = torch.jit.load(encoder_path, map_location=device).eval()

    print(f"\n{'=' * 60}")
    print(f"  Recall Test")
    print(f"  stride={args.stride}s, kmeans={args.kmeans or 'off'}")
    print(f"{'=' * 60}\n")

    results = recall_test(encoder, track_paths, device,
                          stride_seconds=args.stride,
                          kmeans_k=args.kmeans)

    print(f"\n{'=' * 60}")
    print(f"  REPORT")
    print(f"{'=' * 60}")
    print(f"Songs:          {results['songs']}")
    print(f"Index:          {results['index_size']} embeddings ({results['windows_per_song']:.0f}/song)")
    print(f"Storage/song:   {results['bytes_per_song'] / 1024:.1f} KB")
    print(f"Queries:        {results['queries']}")
    print(f"Top-1 recall:   {results['recall']:.1f}%")
    print(f"Stride:         {args.stride}s")
    print(f"K-means:        {args.kmeans or 'off'}")
    print(f"Mode:           {'frozen MERT' if args.skip_training else 'trained'}")
    print(f"{'=' * 60}")

    print("\nDONE")


if __name__ == "__main__":
    main()
