# MusicPrint

> Offline music recognition on a phone. 10 million songs, under 3GB, no server.

MusicPrint is an experiment to see if a complete Shazam-like system can run entirely on a mobile device. The key finding: a frozen pretrained audio model (MERT-v1-95M) produces embeddings discriminative enough for song identification with no fine-tuning. Combined with k-means clustering, PCA, and binary hashing, the index compresses to 320 bytes per song — 3 GB for 10 million songs.

See [PAPER.md](PAPER.md) for the full write-up and results.

## Results

On a corpus of 6,839 songs (Billboard Hot 100, 1920–2020s):

| Config | Storage/song | Recall | @ 10M songs |
|--------|-------------|--------|-------------|
| Frozen MERT, k=10 centroids, float32 | 30 KB | 96.6% | 286 GB |
| + PCA 256 + binary hashing | 320 B | 96.5% | 3.0 GB |
| + PCA 128 + binary hashing | 160 B | 92.0% | 1.5 GB |

## How It Works

1. **Encode**: MERT-v1-95M (frozen, 95M params) takes a 5-second audio clip at 24kHz and produces a 768-dim embedding via mean pooling.
2. **Index**: Each song is split into overlapping 5-second windows. The ~175 window embeddings are clustered to 10 centroids via k-means.
3. **Compress**: PCA reduces 768 dims to 256. Sign-bit binarization produces a 256-bit hash per centroid.
4. **Search**: Query clip → encode → PCA → binarize → nearest neighbor by Hamming distance.

## Running the Experiments

Requires Docker and an NVIDIA GPU.

**1. Build the pipeline image**
```bash
docker compose build training
```

**2. Run experiments interactively**
```bash
docker compose up training
# Open http://localhost:8888 (token: musicprint)
# Open experiments.py as a notebook (jupytext format)
```

**3. Or run experiments as a script**
```bash
docker compose run --rm training python experiments.py
```

The first run encodes all songs through MERT (~6 hours on RTX 2000 Ada) and caches results to disk. Subsequent runs load the cache and run compression experiments in seconds.

Audio files go in `music/` (MP3, FLAC, or WAV in any subdirectory structure).

## Repository Structure

```
experiments.py          # Reproducible experiments notebook (jupytext)
PAPER.md                # Research paper
demo_app/               # Web app for interactive song identification
  app.py                # Flask backend
  static/index.html     # Mic recording + file upload UI
  search.py             # Hamming distance search
  metadata.py           # Binary metadata reader
  audio.py              # Audio loading and binarization
docker/
  Dockerfile.pipeline   # GPU image (MERT, PyTorch, torchaudio)
  Dockerfile.demo       # CPU image (Flask, pydub)
docker-compose.yml      # All services
1_adapter_training/     # Encoder training pipeline (experimental)
2_vector_index/         # Binary index builder
3_meta_tokenizer/       # BPE metadata compression
4_album_art/            # VQ-VAE album art compression
libmusicprint/          # C++ search library (for iOS deployment)
```

## License

MIT
