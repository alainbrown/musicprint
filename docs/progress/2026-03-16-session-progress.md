# Session Progress: 2026-03-16

## What We Built

### Repository Restructure
- Numbered pipeline folders for clarity: `1_adapter_training`, `2_vector_index`, `3_meta_tokenizer`, `4_album_art`
- Root `release/` directory with symlinks to each pipeline's artifacts
- Consolidated 5 per-pipeline Dockerfiles into `docker/Dockerfile.pipeline` (GPU) and `docker/Dockerfile.demo` (CPU)
- Consolidated 5 per-pipeline docker-compose files into one root `docker-compose.yml`
- Removed stale `ios_app/` placeholder
- Removed old smoke test (replaced by verify service)
- Added `requirements.txt` to each pipeline
- Added `.dockerignore` for clean builds

### Demo App (`demo_app/`)
- **search.py** — Hamming distance search over mmap'd binary index (14 unit tests passing)
- **audio.py** — Binarize embeddings, window audio into 5s chunks, load/resample via pydub
- **metadata.py** — Binary ISRC lookup + BPE decoder for music_meta.bin and music_decoder.bin
- **app.py** — Flask backend: POST /identify accepts audio, returns {title, artist, isrc, distance}
- **static/index.html** — Browser UI with mic recording (MediaRecorder) and file upload
- **Dockerfile + docker-compose** — CPU-only container mounting release/ artifacts

### Verification Service
- `scripts/verify.py` — End-to-end pipeline runner: train, then test embedding quality
- `docker compose up verify` — Full catalog run
- `docker compose up verify-quick` — 100-song subset for fast iteration
- `--max_songs` flag creates temp directory with symlinks to random subset
- All working files go to Docker volumes, workspace mounted read-only
- Embedding quality test: encodes two clips per song, measures cosine similarity, reports top-1 recall

### DALI to PyTorch DataLoader Migration
- Replaced DALI data loading with PyTorch DataLoader + torchaudio in both pipelines
- Training: `ContrastiveAudioDataset` loads audio, crops two random 5s windows, applies augmentation
- Indexing: `InferenceAudioDataset` loads full tracks, variable-length collation, chunked inference (32 windows at a time to avoid OOM)
- Added torchaudio + torchcodec to pipeline image

### Model Architecture Changes (Phase 1)
- Removed Tanh activation from adapter (was causing hash collapse)
- Fixed ArcFace margin from 28.6 (wrong units, was treated as radians) to 0.5 radians
- Embedding dim: 64 to 768
- Replaced adapter with `Linear(768, 768)` — initial attempt used `nn.Identity()` which meant no trainable parameters persisted into the exported encoder (bug)

## Bugs Fixed Along the Way
- `system.py` `.squeeze()` to `.reshape(-1)` for batch_size=1 compatibility
- `build_index.py` `torch.load()` needs `weights_only=False` for numpy arrays in newer PyTorch
- `index.py` OOM on long tracks: process windows in chunks of 32
- Hash IDs masked to `0x7FFFFFFFFFFFFFFF` (signed int64) instead of `0xFFFFFFFFFFFFFFFF` (unsigned)
- CoreML export symlink handling
- `ModelCheckpoint` `save_last=True` for training resume across container restarts
- Added pydub/flask to pipeline image for verify recall tests
- `nn.Identity()` adapter meant no trainable params persisted — replaced with `Linear(768, 768)`

## Results

### Old Model (64-dim, Tanh, margin=28.6)
- 100 songs, 50 epochs
- 482 unique hashes out of 1000 index entries (hash collapse)
- 0% top-1 recall: model could not distinguish any songs

### Run 2: nn.Identity (768-dim, no trainable adapter, margin=0.5 rad)
- 100 songs, 50 epochs
- Same-song cosine similarity: 0.86 (avg)
- Different-song cosine similarity: 0.81 (avg)
- 50% top-1 recall on 10 test songs
- Note: nn.Identity meant no trainable parameters persisted — this was testing frozen MERT's raw representations

### Run 3: Linear(768, 768) (trainable adapter, margin=0.5 rad)
- 100 songs, 50 epochs
- Same-song cosine similarity: 0.88 (avg)
- Different-song cosine similarity: 0.84 (avg)
- 40% top-1 recall on 10 test songs
- Separation gap still negative (-0.28)
- Performance similar to frozen MERT — the trainable layer didn't meaningfully improve discrimination on 100 songs

### Key Observations
- All embeddings are too similar to each other (diff-song sim 0.81-0.84 is very high)
- 10-song test with 4/10 vs 5/10 is within noise — runs 2 and 3 are essentially equivalent
- 100 songs may be too few for ArcFace to learn meaningful discrimination
- Unclear whether more data will fix it or if the approach needs fundamental changes

## Open Questions
1. Does recall improve with more songs (1000+)? Would confirm data scale vs architecture issue
2. Is the contrastive training objective correct? Two random crops from the same song could sound completely different — is ArcFace strong enough to handle this?
3. Should we try augmented copies of the same clip instead of random crops?

## Next Steps
1. Try 1000 songs to see if recall improves with more training data
2. If recall plateaus, reconsider training objective (same-clip augmentation vs random crops)
3. Once recall is high, move to Phase 2: compress embeddings for mobile deployment
4. Update indexing pipeline for float vector search (faiss or cosine similarity)
5. Update demo app to work with float vectors
