# Training Experiments Log

## Goal

Train an encoder that maps any 5-second audio clip to an embedding where clips from the same song are close and clips from different songs are far apart. The encoder should generalize to unseen songs.

## Architecture

```
Song (full) → sliding 5s windows → encoder → set of embeddings (all same song label)
                                                      ↓
                                               Contrastive loss (same-song close, different-song far)
                                                      ↓
                                               Trained encoder (loss discarded)

Indexing:  song → all window embeddings → store all (deduplicate later with k-means)
Search:   5s clip → encoder → embedding → nearest neighbor search → song ID
```

## Training Approach

Each training step processes a full song:
1. Load entire song audio (raw, via DataLoader)
2. Split into all overlapping 5-second windows (1s stride) in the training step
3. Encode every window through the model (chunked to fit GPU memory)
4. All window embeddings get the same song label
5. Contrastive loss across all embeddings in the batch

This replaces the previous approach of picking two random 5-second clips per song, which only showed the model fragments rather than the full song structure.

## Recall Test Method

The proper recall test indexes all songs and searches against the full index:
1. Encode every 5-second window from all songs → full index (e.g., 175 windows × 5 songs = 875 vectors)
2. Pick random 5-second clips (~10% of total windows, e.g., ~88 clips)
3. For each clip, find the nearest vector in the full index
4. Check if the nearest vector belongs to the correct song
5. Report top-1 recall

This replaces the previous test which only compared 2 clips per song (10 vectors total for 5 songs). The proper test reflects the actual use case: searching a real index.

## Experimental Variables

### Base Model
| Option | Size | Notes |
|--------|------|-------|
| MERT-v1-95M | 95M params | Current choice. Pretrained on music, 24kHz input. |
| MERT-v1-330M | 330M params | Larger MERT variant, may produce richer features. |
| HuBERT-base | 90M params | Pretrained on speech, may transfer to music. |
| Wav2Vec2-base | 95M params | Similar architecture to MERT. |
| CLAP (audio encoder) | ~80M params | Pretrained on audio-text pairs, may have better general audio features. |

### Adapter Architecture
| Option | Trainable Params | Notes |
|--------|-----------------|-------|
| Linear(768, D) | 768×D | Single linear projection. |
| MLP: Linear-ReLU-Linear | ~1.2M at D=768 | Current choice. Nonlinear feature combination. |
| Deeper MLP (3+ layers) | More | More expressive but risk overfitting. |
| Fine-tune top N MERT layers | Millions | Most expressive, slowest, risk overfitting on small data. |

### Loss Function
| Option | Notes |
|--------|-------|
| ArcFace (margin=0.5, scale=64) | Current choice. Angular margin for discrimination. |
| ArcFace with different margin/scale | margin and scale are tunable. |
| Triplet loss | Anchor-positive-negative. Simpler, doesn't need class count. |
| NT-Xent (SimCLR-style) | Contrastive within batch. Doesn't need class count. |
| ProxyNCA++ | Proxy-based metric learning. |

### Embedding Dimension
| Option | Storage per window | Notes |
|--------|-------------------|-------|
| 768 | 3KB | Current. Same as MERT hidden size, no compression. |
| 512 | 2KB | Slight compression. |
| 256 | 1KB | Moderate compression. |
| 128 | 512B | Aggressive. Test after recall is good at higher dims. |

### Other Knobs
- **Window size**: 5s currently. Could try 3s or 10s.
- **Window stride**: 1s currently. Affects number of training examples per song.
- **Backbone freezing**: Currently fully frozen. Could unfreeze top N layers.
- **Learning rate**: 1e-4 currently. Standard for adapter training.
- **Batch size**: Auto-tuned. Affects how many songs ArcFace sees per step.
- **Epochs**: 50 currently. May need more with full-song training.
- **Augmentation**: Noise addition (0-0.3 gain), volume perturbation (0.5-1.5x). Could add time stretching, pitch shifting.

## Experiment Results

### Control: Baseline Configuration
- **Base model**: MERT-v1-95M (frozen)
- **Adapter**: MLP Linear(768,768)-ReLU-Linear(768,768)
- **Loss**: Contrastive loss (simple pairs — same-song close, different-song far)
- **Embedding dim**: 768
- **Window**: 5s, 1s stride
- **Training**: Full-song window processing (all windows per song per step)
- **Data**: TBD (need to run with new full-song approach)

Rationale for contrastive loss as baseline: ArcFace adds complexity (stateful weight matrix, num_classes upfront, margin/scale hyperparameters). Starting with the simplest loss function isolates the encoder and training approach as variables. If contrastive loss doesn't work, the problem is the encoder or data, not the loss. ArcFace, NT-Xent, and triplet loss can be tried later as improvements.

### Run 1: 64-dim, Tanh, margin=28.6 (broken)
- 100 songs, 50 epochs, two random clips per song
- Hash collapse: 482 unique hashes out of 1000 entries
- 0% recall
- **Cause**: Tanh saturation + wrong margin units + low dimensionality

### Run 2: 768-dim, nn.Identity, margin=0.5
- 100 songs, 50 epochs, two random clips per song
- Same-song sim: 0.86, diff-song sim: 0.81
- 50% recall (10 songs)
- **Note**: No trainable parameters persisted — this was frozen MERT's raw representations

### Run 3: 768-dim, Linear(768,768), margin=0.5
- 100 songs, 50 epochs, two random clips per song
- Same-song sim: 0.88, diff-song sim: 0.84
- 40% recall (10 songs)
- Essentially same as run 2 — single linear layer didn't help

### Run 4: 768-dim, MLP (Linear-ReLU-Linear), margin=0.5
- 100 songs, 50 epochs, two random clips per song
- Same-song sim: 0.92, diff-song sim: 0.90
- 40% recall (10 songs)
- Higher similarities but gap still small. MLP alone doesn't fix it on 100 songs.

### Run 5: Full-song training + contrastive loss (5 songs, old test)
- 5 songs, 50 epochs, full-song window processing, contrastive loss
- Same-song sim: 0.92, diff-song sim: -0.18
- Separation gap: +0.08 (first positive gap)
- 100% recall (5/5) — but using old test method (2 clips per song)
- **Key result**: Massive improvement in diff-song separation (-0.18 vs 0.81-0.90 in previous runs)
- **Caveat**: Old test method, only 5 songs. Need proper full-index recall test to validate.

### Run 6: Frozen MERT baseline (no adapter, proper recall test)
- 5 songs, **no training**, frozen MERT + mean pool, no adapter
- Proper full-index recall test:
  - Indexed 897 windows from 5 songs (avg 179 windows/song)
  - Queried 89 random windows (10% of index)
  - Nearest neighbor search across full index
- **Top-1 recall: 89/89 (100.0%)**
- **Key finding**: MERT's raw pretrained representations are already discriminative enough for song fingerprinting on 5 songs with zero fine-tuning.
- **Implication**: The adapter training in Runs 1-5 may have been unnecessary. The poor recall in Runs 2-4 was due to the flawed test method (2 clips per song), not the encoder quality.
- **Open question**: Does this hold at scale (100, 1000, 6966 songs)?

### Run 7: Frozen MERT at 100 songs
- 100 songs (3 skipped due to decode errors), **no training**, frozen MERT + mean pool
- Indexed 20,611 windows (avg 206 windows/song)
- Queried 2,061 random windows (10%)
- **Top-1 recall: 2,061/2,061 (100.0%)**
- **Conclusion**: Frozen MERT maintains perfect recall at 100-song scale. Adapter training is not needed for basic fingerprinting.

---

## Phase 2: Embedding Compression

The encoder works. Now the question is: how much can we compress the index while preserving recall?

### Baseline
- 100 songs, ~206 windows/song, 768-dim float32 per window
- Storage: 206 × 768 × 4 bytes = ~632KB per song
- At 100M songs: ~63TB (not feasible for mobile)
- **Top-1 recall: 100%**

### Compression Strategies to Test

Each strategy will be tested on 100 songs with the same proper recall test (10% query). Goal: find the best recall at the smallest storage.

#### A. Reduce embeddings per song

| Strategy | Description | Target |
|----------|-------------|--------|
| **Wider stride** | Increase window stride (e.g., 5s instead of 1s) — fewer windows, no computation change | ~35 windows/song |
| **K-means clustering** | Cluster the ~206 windows per song, keep k centroids | 10 embeddings/song |
| **Temporal pooling** | Average groups of adjacent windows | 10-20 embeddings/song |

#### B. Reduce embedding dimensionality

| Strategy | Description | Target |
|----------|-------------|--------|
| **PCA** | Project 768-dim → lower dim using principal components | 128-256 dim |
| **Trained linear projection** | Learn a Linear(768, D) that preserves discrimination | 128-256 dim |
| **Binary hashing** | Sign bits of embedding → uint64/uint128 | 8-16 bytes/embedding |

#### C. Reduce precision

| Strategy | Description | Target |
|----------|-------------|--------|
| **float16** | Half precision | 50% size reduction |
| **Product quantization** | Quantize sub-vectors to codebook indices | 8-16 bytes/embedding |

#### D. Combined

The final system will likely combine strategies from A + B or A + C. For example:
- K-means (206 → 10 windows) + binary hashing (768 floats → 16 bytes) = 160 bytes/song
- At 100M songs: ~16GB (feasible for mobile)

### Compression Experiment Results

All runs: frozen MERT, 100 songs (~97 after decode errors), 2,061 queries (10% of full 1s-stride windows).

#### A. Reducing embeddings per song

| Run | Strategy | Windows/song | Storage/song | Top-1 Recall | Notes |
|-----|----------|-------------|-------------|--------------|-------|
| 7   | Baseline (1s stride) | 206 | 632 KB | 100.0% | Full index, no compression |
| 8   | Wider stride (5s) | 43 | 129 KB | 99.7% | 6 queries missed |
| 9   | K-means k=10 | 10 | 30 KB | 100.0% | Perfect recall at 20x reduction |
| 9b  | K-means k=5 | 5 | 15 KB | 99.4% | 13 queries missed |
| 9c  | K-means k=3 | 3 | 9 KB | 97.9% | Graceful degradation |
| 9d  | K-means k=1 | 1 | 3 KB | 93.2% | Single centroid per song still viable |

**Finding**: K-means k=10 is the sweet spot — 100% recall with 10 embeddings per song (30 KB/song at 768-dim float32). Even k=1 (one embedding per song) retains 93% recall.

At 100M songs with k=10 and 768-dim float32: 30 KB × 100M = ~3 TB. Still too large for mobile. Need to also reduce embedding dimensionality.

#### B. Reducing embedding dimensionality (k-means k=10 base)

| Run | Strategy | Windows/song | Storage/song | Top-1 Recall | @ 100M songs |
|-----|----------|-------------|-------------|--------------|-------------|
| 10  | + PCA 128 | 10 | 5 KB | 99.7% | 500 GB |
| 11  | + binary 768-bit | 10 | 960 bytes | 99.8% | 96 GB |
| 12  | + PCA 128 + binary | 10 | 160 bytes | 99.3% | 16 GB |
| 13  | + PCA 64 + binary | 10 | 80 bytes | 96.7% | 8 GB |

**Finding**: Binary hashing of the full 768-dim vector (Run 11) loses almost nothing — 99.8% recall at 960 bytes/song. PCA 128 + binary (Run 12) achieves 99.3% at just 160 bytes/song — that's 16 GB for 100M songs.

#### Summary: Full compression pipeline

| Config | Storage/song | Recall | @ 100M songs | Feasible for mobile? |
|--------|-------------|--------|-------------|---------------------|
| Full baseline | 632 KB | 100.0% | 63 TB | No |
| k=10 | 30 KB | 100.0% | 3 TB | No |
| k=10 + binary 768 | 960 B | 99.8% | 96 GB | Borderline |
| k=10 + PCA128 + binary | 160 B | 99.3% | 16 GB | Possible |
| k=10 + PCA64 + binary | 80 B | 96.7% | 8 GB | Yes |

The target was <3 GB. At 80 bytes/song (k=10, PCA 64, binary), 100M songs = 8 GB — still above target. Options:
- Reduce k from 10 to 5 (halves storage): k=5 + PCA64 + binary = ~40 bytes/song = 4 GB
- Reduce k to 3: ~24 bytes/song = 2.4 GB (within target)
- Accept fewer songs or slightly larger storage
