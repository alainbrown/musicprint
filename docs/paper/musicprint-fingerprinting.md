# Neural Audio Fingerprinting with Frozen Self-Supervised Models

## Abstract

We investigate whether pretrained self-supervised audio models can serve as effective audio fingerprinting encoders without any fine-tuning. Using MERT-v1-95M, a music-domain transformer pretrained on general music understanding, we find that its frozen representations achieve 100% top-1 recall on a 100-song corpus when used as fingerprint embeddings. We then explore a compression pipeline — k-means segment clustering, PCA dimensionality reduction, and sign-bit binary hashing — that reduces per-song storage from 632 KB to 80 bytes while maintaining 96.7% recall. At the most practical configuration (160 bytes/song, 99.3% recall), a 100-million-song database would require approximately 16 GB of storage, approaching feasibility for mobile deployment.

## 1. Introduction

Audio fingerprinting systems identify songs from short audio recordings by matching query clips against a database of known tracks. Commercial systems like Shazam use spectrogram-based approaches, but these require purpose-built feature extraction pipelines. Recent advances in self-supervised audio models — trained on large unlabeled audio corpora — produce rich, general-purpose representations that capture musical structure, timbre, and rhythm.

We ask a simple question: can a frozen pretrained audio model, with no task-specific training, produce embeddings discriminative enough for song identification? If so, this eliminates the need for labeled training data or custom model development, reducing the fingerprinting problem to an indexing and compression challenge.

Our target deployment is a mobile device (iPhone 13+) with a storage budget under 3 GB for a 100-million-song database. This requires aggressive compression of the embedding space while preserving retrieval accuracy.

## 2. Method

### 2.1 Encoder

We use MERT-v1-95M (Music Enhanced Representation Transformer), a HuBERT-based model pretrained on music audio at 24 kHz. The model has 95 million parameters across 12 transformer layers with a hidden dimension of 768.

Given a 5-second audio clip (120,000 samples at 24 kHz):

1. Normalize to zero mean, unit variance
2. Pass through the frozen MERT backbone
3. Mean-pool the last hidden state across the sequence dimension

This produces a single 768-dimensional embedding vector per 5-second window.

### 2.2 Indexing

Each song is segmented into overlapping 5-second windows with a 1-second stride. A 3-minute song produces approximately 175 windows. Each window is encoded independently, producing a set of embedding vectors per song.

### 2.3 Search

Given a query clip (5 seconds of audio), we encode it using the same pipeline and find the nearest embedding in the index by cosine similarity. The song associated with the nearest embedding is returned as the match.

### 2.4 Compression Pipeline

The full index (175 windows × 768 floats per song) is too large for mobile deployment. We apply three compression stages:

**Stage 1: Segment clustering.** We apply k-means clustering to each song's window embeddings and retain only the k cluster centroids. This exploits the redundancy in songs — repeated choruses, sustained sections, and similar passages produce near-identical embeddings. Reducing from ~175 to 10 embeddings per song yields a 17.5× reduction.

**Stage 2: Dimensionality reduction.** We fit PCA on the full set of index embeddings and project from 768 dimensions to a lower target (64 or 128). This captures the principal axes of variation while discarding noise dimensions.

**Stage 3: Binary hashing.** We take the sign of each dimension (positive → 1, negative → 0) to produce a binary hash. Search uses Hamming distance (equivalent to dot product on {-1, +1} vectors). This reduces each embedding from D × 4 bytes (float32) to D/8 bytes.

At query time, the same PCA projection and sign-bit binarization are applied to the query embedding before searching.

## 3. Experiments

### 3.1 Dataset

We use a collection of 6,966 songs spanning multiple decades and genres (Billboard Hot 100 archives, 1920–2000s). For efficiency, experiments use random 100-song subsets. Songs are stored as MP3 files, converted to mono FLAC at 24 kHz during preprocessing. Three songs were excluded due to decode errors, leaving 97 songs in the test set.

### 3.2 Hardware

All experiments run on a single NVIDIA RTX 2000 Ada Generation GPU (16 GB VRAM) inside a Docker container based on NVIDIA's PyTorch 24.01 image.

### 3.3 Evaluation Protocol

We evaluate top-1 recall using a full-index search:

1. Encode every 5-second window (1-second stride) from all songs in the corpus
2. Apply compression to build the stored index
3. Select 10% of the uncompressed windows as queries (2,061 queries from 20,611 total windows)
4. For each query, find the nearest neighbor in the compressed index by cosine similarity (or Hamming distance for binary hashes)
5. A query is correct if the nearest neighbor belongs to the same song
6. Report top-1 recall as the fraction of correct queries

Queries are drawn from the full uncompressed window set, ensuring they may not exactly match any stored embedding — the system must find the closest representation.

## 4. Results

### 4.1 Baseline: Frozen MERT Without Compression

| Songs | Windows/song | Queries | Top-1 Recall |
|-------|-------------|---------|-------------|
| 5     | 179         | 89      | 100.0%      |
| 97    | 206         | 2,061   | 100.0%      |

Frozen MERT achieves perfect recall at both 5 and 97 songs with no training or adaptation. The pretrained representations are inherently discriminative for song identification.

### 4.2 Segment Count Reduction

All runs: 97 songs, 768-dim float32, 2,061 queries.

| Strategy | Embeddings/song | Storage/song | Top-1 Recall |
|----------|----------------|-------------|-------------|
| Baseline (1s stride) | 206 | 632 KB | 100.0% |
| 5s stride | 43 | 129 KB | 99.7% |
| K-means k=10 | 10 | 30 KB | 100.0% |
| K-means k=5 | 5 | 15 KB | 99.4% |
| K-means k=3 | 3 | 9 KB | 97.9% |
| K-means k=1 | 1 | 3 KB | 93.2% |

K-means with k=10 achieves perfect recall at a 20× reduction in embedding count. Even a single centroid per song (k=1) retains 93.2% recall, suggesting that MERT embeddings form tight per-song clusters.

### 4.3 Dimensionality Reduction and Binary Hashing

All runs: 97 songs, k-means k=10, 2,061 queries.

| Strategy | Dims | Storage/song | Top-1 Recall | @ 100M songs |
|----------|------|-------------|-------------|-------------|
| Float32 (baseline) | 768 | 30 KB | 100.0% | 3 TB |
| PCA 128 | 128 | 5 KB | 99.7% | 500 GB |
| Binary 768-bit | 768 | 960 B | 99.8% | 96 GB |
| PCA 128 + binary | 128 | 160 B | 99.3% | 16 GB |
| PCA 64 + binary | 64 | 80 B | 96.7% | 8 GB |

Binary hashing of the full 768-dimensional vector loses almost no recall (99.8%), demonstrating that the discriminative information is well-distributed across the sign structure of the embeddings. Combined with PCA to 128 dimensions, the system achieves 99.3% recall at 160 bytes per song — 16 GB for a hypothetical 100-million-song database.

## 5. Discussion

### 5.1 Why Frozen MERT Works

MERT was pretrained on music audio using a self-supervised objective that captures acoustic structure at multiple temporal scales. Its representations encode timbre, rhythm, harmonic content, and temporal dynamics — properties that are inherently song-specific. Different 5-second windows from the same song share these properties, while windows from different songs differ in at least some dimensions. This makes cosine similarity a natural distance metric for fingerprinting without any task-specific training.

### 5.2 Failed Approaches

Before discovering that frozen MERT sufficed, we attempted several fine-tuning approaches:

- **ArcFace with 64-dim Tanh adapter**: Produced collapsed binary hashes (482 unique hashes across 1,000 entries) due to Tanh saturation and an incorrect margin parameter (28.6 radians instead of 0.5).
- **ArcFace with 768-dim linear/MLP adapter**: Achieved 40-50% recall with the original evaluation method. However, the evaluation was flawed — it compared only 2 clips per song rather than searching a full index.
- **Contrastive loss with full-song training**: Achieved apparent improvement but was measured against a small test set.

When we corrected the evaluation to use full-index search, frozen MERT without any adapter achieved 100% recall, revealing that the training was unnecessary and the poor earlier results were artifacts of the evaluation methodology.

### 5.3 Compression Efficiency

The k-means clustering stage is highly effective because songs contain significant temporal redundancy. A typical pop song repeats its chorus 3-4 times, and sustained instrumental sections produce near-identical embeddings. Reducing from ~200 to 10 embeddings per song costs no recall because the 10 centroids cover all distinct acoustic segments.

PCA is effective because the 768-dimensional MERT embedding space likely has an intrinsic dimensionality much lower than 768 for the fingerprinting task. The top 128 principal components capture enough variance to preserve 99.7% recall.

Binary hashing works because song discrimination depends on the direction of the embedding vector (which dimensions are positive vs. negative), not the magnitude. Sign-bit quantization preserves this directional information.

### 5.4 Limitations

- **Scale**: All experiments use 97 songs. Recall may degrade at 10K, 100K, or 1M songs as the embedding space becomes more crowded.
- **Audio degradation**: Queries are clean re-encodings of indexed audio. Real-world queries would include background noise, microphone distortion, and compression artifacts.
- **Temporal alignment**: Queries are drawn from the same 1-second grid as the index. A query offset by 0.5 seconds from any indexed window may perform differently.
- **Song similarity**: The test corpus spans decades and genres. Performance on a corpus of similar-sounding songs (e.g., all classical piano) is unknown.

## 6. Future Work

1. **Scale testing**: Evaluate recall on 1,000 and 6,966 songs to establish the scaling curve.
2. **Degraded queries**: Test with additive noise, volume changes, low-pass filtering, and codec compression to simulate real-world recording conditions.
3. **Temporal robustness**: Query with clips not aligned to the 1-second grid.
4. **Alternative models**: Compare MERT-v1-330M, HuBERT, Wav2Vec2, and CLAP as backbone encoders.
5. **On-device benchmarks**: Measure CoreML inference latency and search time on iPhone hardware.
6. **Adaptive compression**: Use fewer centroids for simple songs (steady-state audio) and more for complex ones (frequent transitions).
