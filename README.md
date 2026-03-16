# MusicPrint

> Offline music recognition on a phone. 100 million songs, under 3GB, no server.

MusicPrint is an experiment to see if a complete Shazam-like system can run entirely on a mobile device with no network connection. It uses a self-supervised audio model (MERT-v1) to learn acoustic fingerprints, compresses them down to single bits, and searches with a brute-force Hamming scan fast enough for real-time use on an iPhone.

## How It Works

### Fingerprinting

MERT-v1-95M is a pretrained audio transformer — like BERT, but for music. We freeze it and train a small adapter head on top using ArcFace loss, which pushes embeddings for the same song together and different songs apart on a hypersphere. The adapter projects MERT's 768-dim output down to 64 floats.

### Binarization

Each of those 64 floats gets reduced to a single bit: positive → 1, negative → 0. One song becomes one `uint64`. This works because ArcFace training makes same-song embeddings land in similar regions of the space, so their sign patterns stay consistent even after this aggressive compression.

### Search

The index file is a flat array of 16-byte entries: 8 bytes for the binary hash, 8 bytes for a packed ISRC identifier. The C++ searcher memory-maps this file and does a linear scan.

For each entry, it XORs the query hash against the stored hash — the result has a 1 bit everywhere the two disagree. `__builtin_popcountll` counts those bits. That's the Hamming distance. Fewer differing bits means a more similar song. A max-heap of size k tracks the best matches.

At 100M songs the index is ~1.6GB. A linear scan over it is ~800M popcount operations — achievable in under 150ms on an Apple A15.

### Metadata

Song titles and artist names are compressed with a BPE tokenizer (65k vocab, 16-bit tokens) into a binary database with clustered range tables for O(1) ISRC lookup. Album art is encoded through a VQ-VAE into 320-byte records. Both ship alongside the index.

### On the Phone

The device gets five files: a CoreML encoder model, the binary hash index, the metadata database, the BPE vocabulary, and the album art index. Record 10 seconds of audio → CoreML inference → binarize → Hamming scan → look up metadata → display result. No network required.

## Pipelines

The system is built from four independent pipelines with no shared code between them. They communicate only through file artifacts.

**Adapter Training** (`1_adapter_training/`) — Trains the MERT adapter with ArcFace loss. Takes MP3s, outputs `encoder.pt` (TorchScript) and `MusicPrintEncoder.mlpackage` (CoreML).

**Vector Indexing** (`2_vector_index/`) — Loads the frozen encoder as a black box, runs inference on every song, binarizes the embeddings, and writes `audio_index.bin`.

**Meta Tokenizer** (`3_meta_tokenizer/`) — Imports a MusicBrainz dump, trains a BPE tokenizer, encodes all metadata, and builds the binary lookup tables.

**Album Art Tokenizer** (`4_album_art/`) — Trains a VQ-VAE on album covers and serializes them as 320-byte records.

```
music/ (MP3s)
  │
  ├──► Training Pipeline ──► encoder.pt ──► Index Pipeline ──► audio_index.bin
  │                       └► CoreML model
  │
  ├──► Meta Tokenizer ──► music_meta.bin + music_decoder.bin
  │
  └──► Art Tokenizer ──► art.bin
```

**C++ Search Library** (`libmusicprint/`) — Static library (C++17, no external dependencies) that loads all the `.bin` files via mmap and runs the search. Also builds a `cli_search` tool for testing.

## Quick Start

Requires Docker, Docker Compose, and an NVIDIA GPU. Everything runs from the root `docker-compose.yml`.

**1. Build the pipeline image**
```bash
docker compose build training
```

**2. Train the encoder**
```bash
docker compose run training python src/pipeline.py
```

**3. Build the index**
```bash
docker compose run indexing
```

**4. Build the C++ library** (optional, for iOS)
```bash
cd libmusicprint
make
```

## Demo Notebook

The `demo.py` notebook trains, indexes, and verifies the full system from a `music/` folder. Run it inside the training container:

```bash
docker compose up training
# Open http://localhost:8888 (token: musicprint)
# Open demo.py as a notebook
```

It reports clean and degraded recall numbers for a 5% sample of your catalog.

## Web App Demo

A browser-based demo that identifies songs from mic input or file upload.

```bash
docker compose up demo
# Open http://localhost:5000
```

Requires pre-built artifacts in `release/` (produced by the notebook or by running the pipelines individually).

## Status

The build-time infrastructure is done — all four pipelines run, the C++ search works end-to-end, and CoreML export is in place. What's left is training on a real catalog and validating on-device performance at scale.

## License

MIT
