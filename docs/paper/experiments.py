# %% [markdown]
# # MusicPrint: Neural Audio Fingerprinting Experiments
#
# This notebook reproduces the findings from the MusicPrint research paper.
# It demonstrates that a frozen pretrained audio model (MERT-v1-95M) achieves
# 100% top-1 recall for song identification, and explores compression strategies
# to reduce per-song storage for mobile deployment.
#
# ## Prerequisites
#
# - Run inside the MusicPrint pipeline Docker container:
#   ```bash
#   docker compose build training
#   docker compose run --rm training bash
#   # Inside container:
#   cd /app
#   jupyter lab --ip=0.0.0.0 --port=8888 --allow-root
#   ```
# - Audio files in `/vol/music/` (mounted from `music/` on host)
# - GPU with at least 8GB VRAM
#
# ## Overview
#
# 1. Load frozen MERT encoder (no training needed)
# 2. Encode songs into 768-dim embeddings (5s windows, 1s stride)
# 3. Measure baseline recall (full index)
# 4. Test compression: k-means clustering, PCA, binary hashing
# 5. Summary table

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

# Reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

SAMPLE_RATE = 24000
WINDOW_SAMPLES = 120000  # 5 seconds
CHUNK_SIZE = 32          # encode this many windows at a time

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
        self.mert.adapter = torch.nn.Identity()  # bypass adapter, raw MERT output

    def forward(self, x):
        return self.mert(x)

encoder = FrozenMERT().to(device).eval()
print(f"Encoder loaded: MERT-v1-95M (frozen), output dim = 768")

# %% [markdown]
# ## 2. Load and Encode Songs
#
# Each song is:
# 1. Loaded as mono audio at 24 kHz
# 2. Split into overlapping 5-second windows (1-second stride)
# 3. Each window encoded through MERT → one 768-dim embedding
# 4. Embeddings L2-normalized
#
# A typical 3-minute song produces ~175 windows.

# %%
# Discover audio files
MUSIC_DIR = "/vol/music"
MAX_SONGS = 100  # adjust for speed vs coverage

all_files = sorted(glob.glob(os.path.join(MUSIC_DIR, "**/*.mp3"), recursive=True))
if not all_files:
    all_files = sorted(glob.glob(os.path.join(MUSIC_DIR, "**/*.flac"), recursive=True))

random.seed(42)
selected = random.sample(all_files, min(MAX_SONGS, len(all_files)))
print(f"Selected {len(selected)} songs from {len(all_files)} total")

# %%
def load_audio(path):
    """Load audio, downmix to mono, resample to 24 kHz."""
    audio, sr = torchaudio.load(path)
    if audio.shape[0] > 1:
        audio = audio.mean(dim=0, keepdim=True)
    if sr != SAMPLE_RATE:
        audio = torchaudio.transforms.Resample(sr, SAMPLE_RATE)(audio)
    return audio.squeeze(0)


def make_windows(audio, stride_seconds=1):
    """Split audio into overlapping 5-second windows."""
    stride = stride_seconds * SAMPLE_RATE
    if audio.shape[0] < WINDOW_SAMPLES:
        return [F.pad(audio, (0, WINDOW_SAMPLES - audio.shape[0]))]
    windows = []
    start = 0
    while start + WINDOW_SAMPLES <= audio.shape[0]:
        windows.append(audio[start : start + WINDOW_SAMPLES])
        start += stride
    return windows


def encode_windows(windows):
    """Encode a list of audio windows through the encoder. Returns L2-normalized embeddings."""
    all_embs = []
    for start in range(0, len(windows), CHUNK_SIZE):
        chunk = torch.stack(windows[start : start + CHUNK_SIZE]).to(device)
        with torch.no_grad():
            embs = encoder(chunk)
        all_embs.append(embs.cpu())
    return F.normalize(torch.cat(all_embs, dim=0), dim=1)

# %%
# Encode all songs
print("Encoding songs...")
t0 = time.time()

song_embeddings = []  # list of (song_id, embeddings_tensor)
skipped = 0

for song_id, path in enumerate(selected):
    try:
        audio = load_audio(path)
        windows = make_windows(audio)
        embs = encode_windows(windows)
        song_embeddings.append((song_id, embs))
        if (song_id + 1) % 10 == 0:
            print(f"  {song_id + 1}/{len(selected)} songs encoded...")
    except Exception as e:
        skipped += 1

elapsed = time.time() - t0
total_windows = sum(len(embs) for _, embs in song_embeddings)
print(f"\nEncoded {len(song_embeddings)} songs ({total_windows} windows) in {elapsed:.1f}s")
print(f"Skipped {skipped} songs due to decode errors")
print(f"Avg windows/song: {total_windows / len(song_embeddings):.0f}")

# %% [markdown]
# ## 3. Recall Test Function
#
# The recall test simulates the real use case:
# 1. Build an index from (optionally compressed) embeddings
# 2. Select random query windows from the full uncompressed set
# 3. For each query, find the nearest neighbor in the index
# 4. Check if the nearest neighbor belongs to the correct song
#
# Queries are always drawn from the full 1-second-stride windows, even when
# the index uses fewer embeddings. This ensures the test reflects real conditions
# where a user's recording won't exactly match any stored embedding.

# %%
def build_full_queries(song_embeddings, fraction=0.1):
    """Build the full query pool from all song embeddings."""
    all_embs = []
    all_ids = []
    for song_id, embs in song_embeddings:
        all_embs.append(embs)
        all_ids.extend([song_id] * len(embs))
    full = torch.cat(all_embs, dim=0)
    ids = torch.tensor(all_ids)
    n = max(1, int(len(full) * fraction))
    indices = random.sample(range(len(full)), n)
    return full, ids, indices


def recall_test(index, index_ids, query_embs, query_ids, query_indices):
    """Run top-1 recall test. Returns (correct, total, recall_pct)."""
    correct = 0
    for qi in query_indices:
        query = query_embs[qi].unsqueeze(0)
        expected = query_ids[qi].item()
        sims = (query @ index.T).squeeze(0)
        best = sims.argmax().item()
        if index_ids[best].item() == expected:
            correct += 1
    total = len(query_indices)
    return correct, total, correct / total * 100


# Build query pool (used for all experiments)
query_embs, query_ids, query_indices = build_full_queries(song_embeddings)
print(f"Query pool: {len(query_indices)} queries from {len(query_embs)} total windows")

# %% [markdown]
# ## 4. Baseline: Full Index (No Compression)
#
# Every 5-second window from every song is stored in the index.
# This is the upper bound on recall — if the encoder is good, this should be 100%.

# %%
# Build full index
index_embs = []
index_ids = []
for song_id, embs in song_embeddings:
    index_embs.append(embs)
    index_ids.extend([song_id] * len(embs))

full_index = torch.cat(index_embs, dim=0)
full_index_ids = torch.tensor(index_ids)

correct, total, recall = recall_test(full_index, full_index_ids, query_embs, query_ids, query_indices)
print(f"Baseline: {correct}/{total} ({recall:.1f}%) recall")
print(f"  {len(full_index)} embeddings, 768-dim float32")
print(f"  {len(full_index) / len(song_embeddings):.0f} embeddings/song")
print(f"  {len(full_index) * 768 * 4 / len(song_embeddings) / 1024:.1f} KB/song")

# %% [markdown]
# ## 5. Experiment A: Reducing Embeddings Per Song
#
# Songs contain temporal redundancy — repeated choruses, sustained sections, and
# similar passages produce near-identical embeddings. We test two strategies:
#
# - **Wider stride**: Fewer windows by increasing the stride from 1s to 5s
# - **K-means clustering**: Cluster all windows per song, keep k centroids

# %%
def kmeans_compress(embeddings, k):
    """Reduce embeddings to k centroids using k-means."""
    if len(embeddings) <= k:
        return embeddings
    km = KMeans(n_clusters=k, n_init=3, random_state=42)
    km.fit(embeddings.numpy())
    centroids = torch.from_numpy(km.cluster_centers_).float()
    return F.normalize(centroids, dim=1)


def build_index_compressed(song_embeddings, stride=1, kmeans_k=0):
    """Build a compressed index from song embeddings."""
    idx_embs = []
    idx_ids = []
    for song_id, embs in song_embeddings:
        compressed = embs[::stride] if stride > 1 else embs
        if kmeans_k > 0:
            compressed = kmeans_compress(compressed, kmeans_k)
        idx_embs.append(compressed)
        idx_ids.extend([song_id] * len(compressed))
    return torch.cat(idx_embs, dim=0), torch.tensor(idx_ids)

# %%
print("Experiment A: Reducing embeddings per song")
print("=" * 70)
print(f"{'Strategy':<25} {'Embs/song':>10} {'Storage/song':>12} {'Recall':>10}")
print("-" * 70)

configs_a = [
    ("Baseline (1s stride)", 1, 0),
    ("5s stride", 5, 0),
    ("K-means k=10", 1, 10),
    ("K-means k=5", 1, 5),
    ("K-means k=3", 1, 3),
    ("K-means k=1", 1, 1),
]

results_a = []
for name, stride, k in configs_a:
    idx, ids = build_index_compressed(song_embeddings, stride=stride, kmeans_k=k)
    correct, total, recall = recall_test(idx, ids, query_embs, query_ids, query_indices)
    per_song = len(idx) / len(song_embeddings)
    storage = per_song * 768 * 4
    results_a.append((name, per_song, storage, recall))
    print(f"{name:<25} {per_song:>10.0f} {storage/1024:>10.1f} KB {recall:>9.1f}%")

# %% [markdown]
# ## 6. Experiment B: Dimensionality Reduction + Binary Hashing
#
# Using k-means k=10 as the base (100% recall, 10 embeddings/song), we now
# reduce the size of each embedding:
#
# - **PCA**: Project 768-dim → 128 or 64 dims
# - **Binary hashing**: Take sign bits (positive → 1, negative → 0)
# - **Combined**: PCA then binary hash
#
# Binary hashing reduces each float dimension to a single bit. Search uses
# Hamming distance (equivalent to dot product on {-1, +1} vectors).

# %%
def apply_pca(index, queries, n_components):
    """Fit PCA on index, transform both index and queries."""
    pca = PCA(n_components=n_components, random_state=42)
    idx_reduced = torch.from_numpy(pca.fit_transform(index.numpy())).float()
    q_reduced = torch.from_numpy(pca.transform(queries.numpy())).float()
    return F.normalize(idx_reduced, dim=1), F.normalize(q_reduced, dim=1)


def apply_binary(embeddings):
    """Sign-bit binarization. Returns {-1, +1} tensor for dot-product search."""
    return (embeddings > 0).float() * 2 - 1

# %%
# Base: k-means k=10
base_idx, base_ids = build_index_compressed(song_embeddings, kmeans_k=10)
per_song = len(base_idx) / len(song_embeddings)

print("Experiment B: Dimensionality reduction (base: k-means k=10)")
print("=" * 80)
print(f"{'Strategy':<30} {'Dims':>6} {'Storage/song':>12} {'Recall':>10} {'@ 100M songs':>14}")
print("-" * 80)

configs_b = [
    ("Float32 (baseline)", 0, False),
    ("PCA 128", 128, False),
    ("Binary 768-bit", 0, True),
    ("PCA 128 + binary", 128, True),
    ("PCA 64 + binary", 64, True),
]

results_b = []
for name, pca_dim, binary in configs_b:
    idx = base_idx.clone()
    q = query_embs.clone()

    if pca_dim > 0:
        idx, q = apply_pca(idx, q, pca_dim)

    dim = idx.shape[1]

    if binary:
        idx = apply_binary(idx)
        q = apply_binary(q)
        bytes_per_emb = (dim + 7) // 8
    else:
        bytes_per_emb = dim * 4

    storage = per_song * bytes_per_emb
    at_100m = storage * 100_000_000 / (1024**3)

    correct, total, recall = recall_test(idx, base_ids, q, query_ids, query_indices)
    results_b.append((name, dim, storage, recall, at_100m))
    print(f"{name:<30} {dim:>6} {storage:>10.0f} B {recall:>9.1f}% {at_100m:>12.1f} GB")

# %% [markdown]
# ## 7. Summary
#
# Combined results showing the full compression pipeline from baseline to
# most compressed configuration.

# %%
print()
print("=" * 90)
print("SUMMARY: Full Compression Pipeline")
print("=" * 90)
print(f"{'Config':<35} {'Storage/song':>12} {'Recall':>10} {'@ 100M songs':>14}")
print("-" * 90)

summary = [
    ("Full baseline (206 embs, 768 f32)", 206 * 768 * 4, results_a[0][3]),
    ("k=10, 768 f32", 10 * 768 * 4, results_a[2][3]),
    ("k=10, binary 768-bit", 10 * 96, results_b[2][3]),
    ("k=10, PCA 128 + binary", 10 * 16, results_b[3][3]),
    ("k=10, PCA 64 + binary", 10 * 8, results_b[4][3]),
]

for name, storage, recall in summary:
    at_100m = storage * 100_000_000 / (1024**3)
    print(f"{name:<35} {storage:>10.0f} B {recall:>9.1f}% {at_100m:>12.1f} GB")

print("-" * 90)
print(f"Target: < 3 GB for 100M songs = < 30 bytes/song")
print()
print("Encoder: MERT-v1-95M (frozen, no fine-tuning)")
print(f"Test corpus: {len(song_embeddings)} songs, {len(query_indices)} queries")
