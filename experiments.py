# %% [markdown]
# # MusicPrint: Neural Audio Fingerprinting Experiments
#
# This notebook reproduces the findings from the MusicPrint research paper.
# It demonstrates that a frozen pretrained audio model (MERT-v1-95M) achieves
# high top-1 recall for song identification, and explores compression strategies
# to reduce per-song storage for mobile deployment.
#
# ## Prerequisites
#
# - Run inside the MusicPrint pipeline Docker container:
#   ```bash
#   docker compose build training
#   docker compose run --rm --gpus all training bash
#   # Inside container:
#   cd /app
#   jupyter lab --ip=0.0.0.0 --port=8888 --allow-root
#   ```
# - Audio files in `/vol/music/` (mounted from `music/` on host)
# - GPU with at least 8GB VRAM
#
# ## How it works
#
# 1. Load frozen MERT encoder (no training needed)
# 2. For each song: encode all 5s windows → k-means to 10 centroids + save 10 query windows
# 3. Cache results to disk (~430 MB for 7000 songs)
# 4. Run compression experiments on the cached data (seconds per test)
#
# The encoding step takes ~1.2 hours for 7000 songs on an RTX 2000 Ada.
# Subsequent runs load from cache and skip encoding entirely.

# %%
import os
import sys
import glob
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

SAMPLE_RATE = 24000
WINDOW_SAMPLES = 120000  # 5 seconds
STRIDE_SAMPLES = 24000   # 1 second
CHUNK_SIZE = 32
K_CENTROIDS = 10         # centroids per song
N_QUERIES_PER_SONG = 10  # random query windows per song

CACHE_PATH = "/vol/data/experiment_cache.pt"
MUSIC_DIR = "/vol/music"

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

# %% [markdown]
# ## 1. Load Frozen MERT Encoder
#
# MERT-v1-95M is a HuBERT-based model pretrained on music audio. It has 95M
# parameters, 12 transformer layers, and a hidden dimension of 768.
#
# We use it frozen — no fine-tuning, no adapter. The encoder is simply:
# audio → MERT → mean pool over sequence → 768-dim vector.

# %%
sys.path.insert(0, os.path.join("/app", "1_adapter_training", "src"))
from models.mert_adapter import MERTAdapter

class FrozenMERT(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mert = MERTAdapter(output_dim=768)
        self.mert.adapter = torch.nn.Identity()

    def forward(self, x):
        return self.mert(x)

encoder = FrozenMERT().to(device).eval()
print("Encoder loaded: MERT-v1-95M (frozen), output dim = 768")

# %% [markdown]
# ## 2. Encode Songs (with caching)
#
# For each song we:
# 1. Load audio, split into overlapping 5-second windows (1s stride)
# 2. Encode all windows through MERT (chunked to fit GPU memory)
# 3. K-means cluster to 10 centroids (what would be stored in the index)
# 4. Save 10 random query windows (for recall testing)
# 5. Discard the full window set to keep memory flat
#
# Results are cached to disk. On subsequent runs, encoding is skipped.

# %%
def load_audio(path):
    """Load audio, downmix to mono, resample to 24 kHz."""
    audio, sr = torchaudio.load(path)
    if audio.shape[0] > 1:
        audio = audio.mean(dim=0, keepdim=True)
    if sr != SAMPLE_RATE:
        audio = torchaudio.transforms.Resample(sr, SAMPLE_RATE)(audio)
    return audio.squeeze(0)


def make_windows(audio):
    """Split audio into overlapping 5-second windows with 1-second stride."""
    if audio.shape[0] < WINDOW_SAMPLES:
        return [F.pad(audio, (0, WINDOW_SAMPLES - audio.shape[0]))]
    windows = []
    start = 0
    while start + WINDOW_SAMPLES <= audio.shape[0]:
        windows.append(audio[start : start + WINDOW_SAMPLES])
        start += STRIDE_SAMPLES
    return windows


def encode_windows_batch(windows):
    """Encode windows in chunks through MERT. Returns L2-normalized embeddings."""
    all_embs = []
    for start in range(0, len(windows), CHUNK_SIZE):
        chunk = torch.stack(windows[start : start + CHUNK_SIZE]).to(device)
        with torch.no_grad():
            embs = encoder(chunk)
        all_embs.append(embs.cpu())
    return F.normalize(torch.cat(all_embs, dim=0), dim=1)


def kmeans_compress(embeddings, k):
    """Reduce embeddings to k centroids using k-means."""
    if len(embeddings) <= k:
        return embeddings
    km = KMeans(n_clusters=k, n_init=3, random_state=42)
    km.fit(embeddings.numpy())
    centroids = torch.from_numpy(km.cluster_centers_).float()
    return F.normalize(centroids, dim=1)

# %%
def build_cache():
    """Encode all songs, compress to centroids + query windows, save to disk."""
    all_files = sorted(glob.glob(os.path.join(MUSIC_DIR, "**/*.mp3"), recursive=True))
    if not all_files:
        all_files = sorted(glob.glob(os.path.join(MUSIC_DIR, "**/*.flac"), recursive=True))
    print(f"Found {len(all_files)} songs")

    centroids_list = []  # (song_id, centroids_tensor)
    queries_list = []    # (song_id, query_embeddings_tensor)
    song_names = []
    skipped = 0

    t0 = time.time()
    for i, path in enumerate(all_files):
        try:
            audio = load_audio(path)
            windows = make_windows(audio)
            embs = encode_windows_batch(windows)

            # K-means to centroids
            cents = kmeans_compress(embs, K_CENTROIDS)
            centroids_list.append(cents)

            # Random query windows (different from centroids)
            n_q = min(N_QUERIES_PER_SONG, len(embs))
            q_idx = random.sample(range(len(embs)), n_q)
            queries_list.append(embs[q_idx])

            song_names.append(os.path.relpath(path, MUSIC_DIR))
        except Exception as e:
            skipped += 1
            centroids_list.append(None)
            queries_list.append(None)
            song_names.append(None)

        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            remaining = (len(all_files) - i - 1) / rate
            print(f"  {i+1}/{len(all_files)} songs | {elapsed/60:.1f}m elapsed | ~{remaining/60:.1f}m remaining | {skipped} skipped")

    # Filter out failed songs
    valid = [(j, c, q, n) for j, (c, q, n) in enumerate(zip(centroids_list, queries_list, song_names)) if c is not None]
    song_ids = list(range(len(valid)))

    all_centroids = []
    all_centroid_ids = []
    all_queries = []
    all_query_ids = []
    valid_names = []

    for new_id, (_, cents, qs, name) in enumerate(valid):
        all_centroids.append(cents)
        all_centroid_ids.extend([new_id] * len(cents))
        all_queries.append(qs)
        all_query_ids.extend([new_id] * len(qs))
        valid_names.append(name)

    cache = {
        "centroids": torch.cat(all_centroids, dim=0),
        "centroid_ids": torch.tensor(all_centroid_ids),
        "queries": torch.cat(all_queries, dim=0),
        "query_ids": torch.tensor(all_query_ids),
        "song_names": valid_names,
        "n_songs": len(valid),
        "k_centroids": K_CENTROIDS,
        "n_queries_per_song": N_QUERIES_PER_SONG,
        "skipped": skipped,
    }

    torch.save(cache, CACHE_PATH)
    elapsed = time.time() - t0
    print(f"\nDone in {elapsed/60:.1f} minutes")
    print(f"Songs: {len(valid)} ({skipped} skipped)")
    print(f"Centroids: {len(cache['centroids'])} ({K_CENTROIDS}/song)")
    print(f"Queries: {len(cache['queries'])} ({N_QUERIES_PER_SONG}/song)")
    print(f"Cache saved to {CACHE_PATH} ({os.path.getsize(CACHE_PATH) / 1e6:.0f} MB)")
    return cache

# %%
# Load from cache or build
if os.path.exists(CACHE_PATH):
    print(f"Loading cache from {CACHE_PATH}...")
    cache = torch.load(CACHE_PATH, weights_only=False)
    print(f"Loaded: {cache['n_songs']} songs, {len(cache['centroids'])} centroids, {len(cache['queries'])} queries")
else:
    print("No cache found. Encoding all songs (this takes ~1-2 hours)...")
    cache = build_cache()

centroids = cache["centroids"]
centroid_ids = cache["centroid_ids"]
queries = cache["queries"]
query_ids = cache["query_ids"]
n_songs = cache["n_songs"]

print(f"\nDataset: {n_songs} songs")
print(f"Index: {len(centroids)} centroids ({K_CENTROIDS}/song, 768-dim)")
print(f"Queries: {len(queries)} windows ({N_QUERIES_PER_SONG}/song)")

# %% [markdown]
# ## 3. Recall Test Function
#
# For each query window, find the nearest centroid in the index by cosine
# similarity (or Hamming distance for binary hashes). A query is correct if
# the nearest centroid belongs to the same song.

# %%
def recall_test(index, index_ids, query_embs, query_ids):
    """Top-1 recall: for each query, is the nearest index entry from the same song?"""
    correct = 0
    total = len(query_embs)
    for i in range(total):
        query = query_embs[i].unsqueeze(0)
        expected = query_ids[i].item()
        sims = (query @ index.T).squeeze(0)
        best = sims.argmax().item()
        if index_ids[best].item() == expected:
            correct += 1
    return correct, total, correct / total * 100

# %% [markdown]
# ## 4. Baseline: K-means k=10 (No Further Compression)
#
# Based on earlier experiments with 100 songs, k-means k=10 achieved 100%
# recall. We verify this holds at full corpus scale.
#
# Note: We skip the full-index baseline (all ~175 windows/song) since it
# was validated at 100% on 100 songs and is too memory-intensive at 7000 songs.

# %%
correct, total, recall = recall_test(centroids, centroid_ids, queries, query_ids)
print(f"Baseline (k=10, 768-dim float32)")
print(f"  Recall: {correct}/{total} ({recall:.1f}%)")
print(f"  Storage: {K_CENTROIDS * 768 * 4:.0f} bytes/song ({K_CENTROIDS * 768 * 4 / 1024:.1f} KB)")
print(f"  @ 100M songs: {K_CENTROIDS * 768 * 4 * 100_000_000 / (1024**4):.1f} TB")

baseline_recall = recall

# %% [markdown]
# ## 5. Experiment A: Reduce Centroids Per Song
#
# Can we use fewer than 10 centroids and still identify songs?
# We re-cluster from the existing k=10 centroids (approximation —
# ideally would re-cluster from full windows, but centroids are all we cached).

# %%
def reduce_centroids(centroids, centroid_ids, n_songs, new_k):
    """Re-cluster per-song centroids to fewer centroids."""
    new_idx = []
    new_ids = []
    for song_id in range(n_songs):
        mask = centroid_ids == song_id
        song_cents = centroids[mask]
        reduced = kmeans_compress(song_cents, new_k)
        new_idx.append(reduced)
        new_ids.extend([song_id] * len(reduced))
    return torch.cat(new_idx, dim=0), torch.tensor(new_ids)

print("Experiment A: Reducing centroids per song")
print("=" * 70)
print(f"{'K':>5} {'Embs/song':>10} {'Storage/song':>12} {'Recall':>10}")
print("-" * 70)

results_a = []
for k in [10, 5, 3, 1]:
    if k == 10:
        idx, ids = centroids, centroid_ids
    else:
        idx, ids = reduce_centroids(centroids, centroid_ids, n_songs, k)
    correct, total, recall = recall_test(idx, ids, queries, query_ids)
    storage = k * 768 * 4
    results_a.append((k, storage, recall))
    print(f"{k:>5} {k:>10} {storage/1024:>10.1f} KB {recall:>9.1f}%")

# %% [markdown]
# ## 6. Experiment B: Dimensionality Reduction + Binary Hashing
#
# Using k=10 centroids as the base, we reduce the size of each embedding:
#
# - **PCA**: Project 768-dim → lower dims, capturing the principal axes of variation
# - **Binary hashing**: Take sign bits (positive → 1, negative → 0)
# - **Combined**: PCA then binary hash
#
# For binary search, dot product on {-1, +1} vectors is equivalent to
# Hamming distance (counts matching bits).

# %%
def apply_pca(index, queries, n_components):
    """Fit PCA on index, transform both."""
    pca = PCA(n_components=n_components, random_state=42)
    idx_r = torch.from_numpy(pca.fit_transform(index.numpy())).float()
    q_r = torch.from_numpy(pca.transform(queries.numpy())).float()
    return F.normalize(idx_r, dim=1), F.normalize(q_r, dim=1)


def apply_binary(embeddings):
    """Sign-bit binarization → {-1, +1} for dot-product search."""
    return (embeddings > 0).float() * 2 - 1

# %%
print("Experiment B: Dimensionality reduction (base: k=10 centroids)")
print("=" * 85)
print(f"{'Strategy':<30} {'Dims':>6} {'Storage/song':>12} {'Recall':>10} {'@ 100M songs':>14}")
print("-" * 85)

configs_b = [
    ("Float32 768-dim", 0, False),
    ("PCA 256", 256, False),
    ("PCA 128", 128, False),
    ("PCA 64", 64, False),
    ("Binary 768-bit", 0, True),
    ("Binary 256-bit (PCA)", 256, True),
    ("Binary 128-bit (PCA)", 128, True),
    ("Binary 64-bit (PCA)", 64, True),
]

results_b = []
for name, pca_dim, binary in configs_b:
    idx = centroids.clone()
    q = queries.clone()

    if pca_dim > 0:
        idx, q = apply_pca(idx, q, pca_dim)

    dim = idx.shape[1]

    if binary:
        idx = apply_binary(idx)
        q = apply_binary(q)
        bytes_per_emb = (dim + 7) // 8
    else:
        bytes_per_emb = dim * 4

    storage = K_CENTROIDS * bytes_per_emb
    at_100m = storage * 100_000_000 / (1024**3)

    correct, total, recall = recall_test(idx, centroid_ids, q, query_ids)
    results_b.append((name, dim, storage, recall, at_100m))
    print(f"{name:<30} {dim:>6} {storage:>10.0f} B {recall:>9.1f}% {at_100m:>12.1f} GB")

# %% [markdown]
# ## 7. Summary

# %%
print()
print("=" * 90)
print("SUMMARY: Full Compression Pipeline")
print("=" * 90)
print(f"{'Config':<40} {'Storage/song':>12} {'Recall':>10} {'@ 100M songs':>14}")
print("-" * 90)

summary = []
for name, dim, storage, recall, at_100m in results_b:
    summary.append((name, storage, recall, at_100m))
    print(f"{'k=10, ' + name:<40} {storage:>10.0f} B {recall:>9.1f}% {at_100m:>12.1f} GB")

print("-" * 90)
print(f"Target: < 3 GB for 100M songs = < 30 bytes/song")
print()
print(f"Encoder: MERT-v1-95M (frozen, no fine-tuning)")
print(f"Corpus: {n_songs} songs (Billboard Hot 100, 1920-2020s)")
print(f"Queries: {len(queries)} ({N_QUERIES_PER_SONG}/song)")
