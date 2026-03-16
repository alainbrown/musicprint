"""
MusicPrint end-to-end verification script.

Usage:
  docker compose up verify-quick          # train + test
  docker compose run verify-quick python scripts/verify.py --max_songs 5 --skip_training  # frozen MERT baseline
"""
import os
import sys
import subprocess
import random
import time
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
STRIDE_SAMPLES = 24000   # 1 second
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


def make_windows(audio):
    if audio.shape[0] < WINDOW_SAMPLES:
        return [F.pad(audio, (0, WINDOW_SAMPLES - audio.shape[0]))]
    windows = []
    start = 0
    while start + WINDOW_SAMPLES <= audio.shape[0]:
        windows.append(audio[start : start + WINDOW_SAMPLES])
        start += STRIDE_SAMPLES
    return windows


def encode_windows(encoder, windows, device):
    """Encode windows in chunks, return L2-normalized embeddings."""
    all_embs = []
    for start in range(0, len(windows), CHUNK_SIZE):
        chunk = torch.stack(windows[start : start + CHUNK_SIZE]).to(device)
        with torch.no_grad():
            embs = encoder(chunk)
        all_embs.append(embs.cpu())
    embs = torch.cat(all_embs, dim=0)
    return F.normalize(embs, dim=1)


def run_step(name, cmd):
    print(f"\n{'=' * 60}")
    print(f"  {name}")
    print(f"{'=' * 60}\n")
    result = subprocess.run(cmd, cwd=ROOT)
    if result.returncode != 0:
        print(f"\nFAILED: {name} (exit code {result.returncode})")
        sys.exit(1)


def recall_test(encoder, track_paths, device, query_fraction=0.1):
    """Full-index recall test.

    1. Index all windows from all songs
    2. Pick random query windows (query_fraction of total)
    3. Search nearest neighbor across full index
    4. Report top-1 recall
    """
    print("Building index...")
    index_embeddings = []  # all window embeddings
    index_song_ids = []    # which song each embedding belongs to
    song_window_counts = []

    for song_id, path in enumerate(track_paths):
        audio = load_audio(path)
        windows = make_windows(audio)
        embs = encode_windows(encoder, windows, device)
        index_embeddings.append(embs)
        index_song_ids.extend([song_id] * len(embs))
        song_window_counts.append(len(embs))

    index = torch.cat(index_embeddings, dim=0)  # (total_windows, D)
    index_ids = torch.tensor(index_song_ids)
    total_windows = len(index)

    print(f"Index: {total_windows} windows from {len(track_paths)} songs")
    print(f"Avg windows/song: {total_windows / len(track_paths):.0f}")

    # Pick random query windows
    num_queries = max(1, int(total_windows * query_fraction))
    query_indices = random.sample(range(total_windows), num_queries)

    print(f"Querying {num_queries} random windows ({query_fraction*100:.0f}% of index)...")

    correct = 0
    for qi in query_indices:
        query = index[qi].unsqueeze(0)  # (1, D)
        expected_song = index_ids[qi].item()

        # Cosine similarity against full index
        sims = (query @ index.T).squeeze(0)  # (total_windows,)

        # Exclude the query itself
        sims[qi] = -2.0

        best_idx = sims.argmax().item()
        if index_ids[best_idx].item() == expected_song:
            correct += 1

    recall = correct / num_queries * 100
    print(f"Top-1 recall: {correct}/{num_queries} ({recall:.1f}%)")
    return recall, num_queries


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_songs", type=int, default=0)
    parser.add_argument("--skip_training", action="store_true", help="Use frozen MERT, no training")
    args = parser.parse_args()

    print("=" * 60)
    print("  MUSICPRINT VERIFICATION")
    print("=" * 60)

    # Discover tracks
    track_paths = discover_tracks(MUSIC_DIR)
    if not track_paths:
        print(f"ERROR: No audio files found in {MUSIC_DIR}")
        sys.exit(1)
    print(f"Found {len(track_paths)} tracks in music/")

    # Optional subset
    source_dir = MUSIC_DIR
    if args.max_songs > 0 and args.max_songs < len(track_paths):
        source_dir = create_subset(track_paths, args.max_songs, MUSIC_DIR)
        track_paths = discover_tracks(source_dir)
        print(f"Using subset of {len(track_paths)} tracks")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.skip_training:
        # Frozen MERT baseline — no adapter, no training
        print(f"\n{'=' * 60}")
        print(f"  Frozen MERT Baseline (no training)")
        print(f"{'=' * 60}\n")

        sys.path.insert(0, os.path.join(ROOT, "1_adapter_training", "src"))
        from models.mert_adapter import MERTAdapter

        # Use Identity adapter — just MERT + mean pool
        class FrozenMERT(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.backbone = MERTAdapter(output_dim=768)
                # Override adapter with identity
                self.backbone.adapter = torch.nn.Identity()

            def forward(self, x):
                return self.backbone(x)

        encoder = FrozenMERT().to(device).eval()
    else:
        # Train encoder
        run_step("Step 1: Train Encoder", [
            sys.executable, "1_adapter_training/src/pipeline.py",
            "--source_dir", source_dir,
            "--data_dir", DATA_DIR,
            "--checkpoint_dir", CHECKPOINT_DIR,
            "--release_dir", ARTIFACTS_DIR,
        ])

        encoder_path = os.path.join(ARTIFACTS_DIR, "encoder.pt")
        assert os.path.exists(encoder_path), "encoder.pt not found"
        encoder = torch.jit.load(encoder_path, map_location=device)
        encoder.eval()

    # Recall test
    print(f"\n{'=' * 60}")
    print(f"  Recall Test (full index)")
    print(f"{'=' * 60}\n")

    recall, num_queries = recall_test(encoder, track_paths, device)

    # Summary
    print(f"\n{'=' * 60}")
    print(f"  VERIFICATION REPORT")
    print(f"{'=' * 60}")
    print(f"Songs:       {len(track_paths)}")
    print(f"Queries:     {num_queries}")
    print(f"Top-1 recall: {recall:.1f}%")
    print(f"Mode:        {'frozen MERT' if args.skip_training else 'trained adapter'}")
    print(f"{'=' * 60}")

    print("\nVERIFICATION COMPLETE")


if __name__ == "__main__":
    main()
