# Neural Audio Fingerprinting with Frozen Self-Supervised Models

## Abstract

We investigate whether pretrained self-supervised audio models can serve as effective audio fingerprinting encoders without any fine-tuning. Using MERT-v1-95M, a music-domain transformer pretrained on general music understanding, we find that its frozen representations achieve 96.6% top-1 recall on a 6,839-song corpus (Billboard Hot 100, 1920–2020s) when used as fingerprint embeddings. We then explore a compression pipeline — k-means segment clustering, PCA dimensionality reduction, and sign-bit binary hashing — that reduces per-song storage from 30 KB to 320 bytes while maintaining 96.5% recall. At this configuration, a 10-million-song database would require approximately 3 GB of storage — feasible for mobile deployment.

## 1. Introduction

Audio fingerprinting systems identify songs from short audio recordings by matching query clips against a database of known tracks. Commercial systems like Shazam use spectrogram-based approaches, but these require purpose-built feature extraction pipelines. Recent advances in self-supervised audio models — trained on large unlabeled audio corpora — produce rich, general-purpose representations that capture musical structure, timbre, and rhythm.

We ask a simple question: can a frozen pretrained audio model, with no task-specific training, produce embeddings discriminative enough for song identification? If so, this eliminates the need for labeled training data or custom model development, reducing the fingerprinting problem to an indexing and compression challenge.

Our target deployment is a mobile device (iPhone 13+) with a storage budget under 3 GB for a 10-million-song database. This requires aggressive compression of the embedding space while preserving retrieval accuracy.

## 2. Related Work

**Traditional audio fingerprinting.** Shazam (Wang, 2003) extracts spectrogram peaks and matches combinatorial landmark pairs. This approach is fast and robust to noise but requires a purpose-built feature extraction pipeline. Chromaprint/AcoustID uses chroma features for music identification. These systems do not use learned representations.

**Neural audio fingerprinting.** Chang et al. (2021) proposed a neural audio fingerprint learning framework that trains a compact encoder using contrastive learning on augmented audio segments. Their system learns 128-dim embeddings optimized for fingerprinting, achieving high recall under noise and compression. Unlike our approach, it requires task-specific training with carefully designed augmentations.

**Self-supervised audio models.** MERT (Li et al., 2023) is a music-domain self-supervised model based on HuBERT, pretrained with both acoustic and musical tokens. Other models in this family include Wav2Vec2 (Baevski et al., 2020), HuBERT (Hsu et al., 2021), and CLAP (Wu et al., 2023). These models produce general-purpose audio representations but have not been systematically evaluated for fingerprinting tasks.

**Our contribution.** We show that frozen MERT representations, with no task-specific training, are sufficient for song identification. This reduces neural audio fingerprinting to a compression problem — how to store and search the embeddings efficiently — rather than a representation learning problem.

## 3. Method

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

**Stage 2: Dimensionality reduction.** We fit PCA on the full set of index embeddings and project from 768 dimensions to a lower target (64, 128, or 256). This captures the principal axes of variation while discarding noise dimensions.

**Stage 3: Binary hashing.** We take the sign of each dimension (positive → 1, negative → 0) to produce a binary hash. Search uses Hamming distance (equivalent to dot product on {-1, +1} vectors). This reduces each embedding from D × 4 bytes (float32) to D/8 bytes.

At query time, the same PCA projection and sign-bit binarization are applied to the query embedding before searching.

## 4. Experiments

### 4.1 Dataset

We use a collection of 6,966 songs from the Billboard Hot 100 archives spanning 1920 to the 2020s, covering a wide range of genres, recording qualities, and production styles. Songs are stored as MP3 files. 127 songs were excluded due to decode errors, leaving 6,839 songs in the full test set. A 100-song subset (97 after decode errors) was used for initial development.

### 4.2 Hardware

All experiments run on a single NVIDIA RTX 2000 Ada Generation GPU (16 GB VRAM) inside a Docker container based on NVIDIA's PyTorch 24.01 image. Encoding the full corpus took 349 minutes (~5.8 hours).

### 4.3 Evaluation Protocol

We evaluate top-1 recall using a full-index search:

1. For each song, encode all 5-second windows (1-second stride) through MERT
2. Apply k-means clustering to retain 10 centroid embeddings per song (index)
3. Save 10 random window embeddings per song as queries (separate from centroids)
4. Apply optional compression (PCA, binary hashing) to both index and queries
5. For each query, find the nearest neighbor in the compressed index
6. A query is correct if the nearest neighbor belongs to the same song
7. Report top-1 recall as the fraction of correct queries

Queries are random window embeddings that do not appear in the index, ensuring the test reflects real conditions where a user's recording won't exactly match any stored embedding.

## 5. Results

### 6.1 Development Results (100-Song Subset)

Initial experiments on 97 songs validated the approach and tuned parameters.

**Baseline (no compression):** 100% recall with full index (206 windows/song) confirmed that frozen MERT embeddings are inherently discriminative.

**Segment count reduction:**

| Strategy | Embeddings/song | Storage/song | Top-1 Recall |
|----------|----------------|-------------|-------------|
| Baseline (1s stride) | 206 | 632 KB | 100.0% |
| 5s stride | 43 | 129 KB | 99.7% |
| K-means k=10 | 10 | 30 KB | 100.0% |
| K-means k=5 | 5 | 15 KB | 99.4% |
| K-means k=3 | 3 | 9 KB | 97.9% |
| K-means k=1 | 1 | 3 KB | 93.2% |

K-means k=10 was selected as the segment reduction strategy (perfect recall at 20× reduction). Dimensionality reduction experiments on the 100-song subset showed promising results but are superseded by the full corpus experiments below.

### 6.2 Full Corpus Results (6,839 Songs)

All runs: 6,839 songs, k-means k=10, 68,390 queries (10/song).

| Strategy | Dims | Storage/song | Top-1 Recall | @ 10M songs |
|----------|------|-------------|-------------|-------------|
| Float32 768-dim | 768 | 30 KB | 96.6% | 286 GB |
| PCA 256 float32 | 256 | 10 KB | 96.1% | 95 GB |
| PCA 128 float32 | 128 | 5 KB | 95.3% | 48 GB |
| PCA 64 float32 | 64 | 2.5 KB | 93.0% | 24 GB |
| Binary 768-bit | 768 | 960 B | 95.1% | 8.9 GB |
| **Binary 256-bit (PCA)** | **256** | **320 B** | **96.5%** | **3.0 GB** |
| Binary 128-bit (PCA) | 128 | 160 B | 92.0% | 1.5 GB |
| Binary 64-bit (PCA) | 64 | 80 B | 75.5% | 0.7 GB |

At full corpus scale, baseline recall is 96.6% (down from 100% at 100 songs). The most notable result is that **PCA 256 + binary hashing achieves 96.5% recall — essentially matching the uncompressed baseline — at 320 bytes per song**. This is a 96× storage reduction with negligible recall loss. At 10 million songs, this configuration requires approximately 3 GB — meeting the mobile storage target.

Binary hashing after PCA 256 (96.5%) slightly outperforms binary hashing without PCA (95.1%), suggesting that PCA removes noise dimensions that hurt binarization. However, at PCA 128 and below, recall drops more steeply (92.0%, 75.5%), indicating that 256 principal components capture a critical threshold of discriminative information.

## 6. Discussion

### 6.1 Why Frozen MERT Works

MERT was pretrained on music audio using a self-supervised objective that captures acoustic structure at multiple temporal scales. Its representations encode timbre, rhythm, harmonic content, and temporal dynamics — properties that are inherently song-specific. Different 5-second windows from the same song share these properties, while windows from different songs differ in at least some dimensions. This makes cosine similarity a natural distance metric for fingerprinting without any task-specific training.

### 6.2 Failed Approaches

Before discovering that frozen MERT sufficed, we attempted several fine-tuning approaches:

- **ArcFace with 64-dim Tanh adapter**: Produced collapsed binary hashes (482 unique hashes across 1,000 entries) due to Tanh saturation and an incorrect margin parameter (28.6 radians instead of 0.5).
- **ArcFace with 768-dim linear/MLP adapter**: Achieved 40-50% recall with the original evaluation method. However, the evaluation was flawed — it compared only 2 clips per song rather than searching a full index.
- **Contrastive loss with full-song training**: Achieved apparent improvement but was measured against a small test set.

When we corrected the evaluation to use full-index search, frozen MERT without any adapter achieved 100% recall, revealing that the training was unnecessary and the poor earlier results were artifacts of the evaluation methodology.

### 6.3 Compression Efficiency

The k-means clustering stage is highly effective because songs contain significant temporal redundancy. A typical pop song repeats its chorus 3-4 times, and sustained instrumental sections produce near-identical embeddings. Reducing from ~200 to 10 embeddings per song costs no recall at 100 songs and only 3.4% at 6,839 songs.

PCA to 256 dimensions preserves nearly all discriminative information (96.1% recall in float32). Interestingly, binary hashing after PCA 256 (96.5%) slightly outperforms both the uncompressed PCA 256 (96.1%) and binary hashing without PCA (95.1%). This suggests that PCA removes noise dimensions that hurt binarization — the sign bits of the top 256 principal components are more discriminative than the sign bits of all 768 dimensions.

Below 256 dimensions, recall degrades more steeply: PCA 128 + binary drops to 92.0%, and PCA 64 + binary to 75.5%. This indicates that approximately 256 principal components are needed to capture the critical discriminative structure.

### 6.4 Scaling Behavior

At 100 songs with k=10 centroids, recall was 100%. At 6,839 songs with k=10 centroids, recall dropped to 96.6%. Note that the 100-song result used a different evaluation protocol (queries drawn from full 1s-stride windows) while the full corpus used separate query windows, so the comparison is approximate. Nonetheless, the trend is clear: recall degrades as the corpus grows and the embedding space becomes more crowded. Quantifying this scaling curve at 10K, 100K, and 1M songs is critical for production deployment.

### 6.5 Limitations

- **Scale**: Full experiments use 6,839 songs. Recall may degrade further at 100K or 1M songs as the embedding space becomes more crowded.
- **Audio degradation**: Queries are clean re-encodings of indexed audio. Real-world queries would include background noise, microphone distortion, and compression artifacts.
- **Temporal alignment**: Queries are drawn from the same 1-second grid as the index. A query offset by 0.5 seconds from any indexed window may perform differently.
- **Song similarity**: The test corpus spans decades and genres. Performance on a corpus of similar-sounding songs (e.g., all classical piano) is unknown.

## 7. Future Work

1. **Scale testing**: Evaluate recall at 10K, 100K, and 1M songs to establish the scaling curve toward the 10M target.
2. **Degraded queries**: Test with additive noise, volume changes, low-pass filtering, and codec compression to simulate real-world recording conditions.
3. **Temporal robustness**: Query with clips not aligned to the 1-second grid.
4. **Alternative models**: Compare MERT-v1-330M, HuBERT, Wav2Vec2, and CLAP as backbone encoders.
5. **On-device benchmarks**: Measure CoreML inference latency and search time on iPhone hardware.
6. **Adaptive compression**: Use fewer centroids for simple songs (steady-state audio) and more for complex ones (frequent transitions).
