# Training Experiments Log

## Goal

Train an encoder that maps any 5-second audio clip to an embedding where clips from the same song are close and clips from different songs are far apart. The encoder should generalize to unseen songs.

## Architecture

```
Song (full) → sliding 5s windows → encoder → set of embeddings (all same song label)
                                                      ↓
                                               ArcFace loss (pushes same-song together, different-song apart)
                                                      ↓
                                               Trained encoder (ArcFace discarded)

Indexing:  song → all window embeddings → store all (deduplicate later with k-means)
Search:   5s clip → encoder → embedding → nearest neighbor search → song ID
```

## Training Approach

Each training step processes a full song:
1. Load entire song audio
2. Split into all overlapping 5-second windows (1s stride)
3. Encode every window through the model (chunked to fit GPU memory)
4. All window embeddings get the same song label
5. ArcFace loss across all embeddings in the batch

This replaces the previous approach of picking two random 5-second clips per song, which only showed the model fragments rather than the full song structure.

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

### Run 5: Control (full-song training + contrastive loss)
- **Status**: Not yet run
- **Changes from Run 4**:
  - Full-song window processing (all windows per song, not two random clips)
  - Contrastive loss replaces ArcFace (simplest possible loss, no state)
- This is the baseline for all future experiments
