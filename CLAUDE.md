# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MusicPrint is an offline acoustic fingerprinting system for iOS. It compresses a 100M-song database into <3GB using MERT transformers, Product Quantization, and a custom C++ Hamming-distance search engine (<150ms queries on mobile).

## Build & Run Commands

### C++ Library (libmusicprint)
```bash
cd libmusicprint
make          # mkdir build, cmake, make
make test     # runs test_art, test_meta, test_search
make clean    # rm -rf build
```
Individual test binaries after build: `build/test_art`, `build/test_meta`, `build/test_search`.
CLI tool: `build/cli_search <query.bin> <index.bin> <music_meta.bin> <music_decoder.bin>`

### Training Pipeline (adapter_training_pipeline)
```bash
cd adapter_training_pipeline
docker compose up --build -d
docker compose exec training-pipeline python src/pipeline.py
```
Outputs: `release/encoder.pt` (TorchScript), `release/MusicPrintEncoder.mlpackage` (CoreML)

### Indexing Pipeline (vector_index_pipeline)
```bash
cd vector_index_pipeline
docker compose up --build -d
docker compose exec index-pipeline python src/pipeline.py --model_path /vol/model/encoder.pt
```
Mounts the training pipeline's `release/` as read-only at `/vol/model`.

### End-to-End Smoke Test
```bash
docker compose -f docker-compose.test.yml up --build
```
Two-stage: `smoke-gen` (Python/GPU) generates fixtures → `smoke-run` (GCC 12) builds and runs `cli_search` against them.

## Architecture

### Split Pipeline Design
The system has **four independent pipelines** with no shared code between them:

1. **adapter_training_pipeline/** — "Teacher": Trains a MERT-v1-95M adapter with ArcFace loss. Entry point: `src/pipeline.py` → preprocess → train (Lightning) → export. Uses NVIDIA DALI for GPU data loading.

2. **vector_index_pipeline/** — "Librarian": Consumes frozen `encoder.pt` as a black box. Entry point: `src/pipeline.py` → preprocess → index (inference) → build_index (merge shards into `audio_index.bin`).

3. **meta_tokenizer_pipeline/** — BPE tokenizer (HuggingFace Tokenizers, 65k vocab, 16-bit tokens) for compressing song metadata. Imports from MusicBrainz, builds clustered range tables. Release artifacts are git-tracked.

4. **album_art_tokenizer_pipeline/** — VQ-VAE (1024 codebook, 10-bit) for visual hashing of album art into 320-byte records.

### C++ Search Library (libmusicprint/)
Static library (`musicprint_core.a`) targeting iOS, built with CMake/C++17. No external dependencies.

Key components in `musicprint` namespace:
- **Searcher** — Binarized query search via XOR + `__builtin_popcountll`. Loads mmap'd `audio_index.bin` (16 bytes per entry: 8B hash + 8B packed ISRC).
- **MetadataDatabase** — Dual-level range tables mapping ISRC → tokenized metadata.
- **BPEDecoder** — Decodes 16-bit token sequences to human-readable strings.
- **ArtDatabase** — O(1) album art retrieval by Album ID (320-byte records).
- **BinaryReader** — `mmap`-based file reader shared by all components.

### Binary Format Convention
All binary files are little-endian, memory-mapped. Index header is 64 bytes followed by fixed-size entries.

### Docker Volume Wiring
- Training pipeline writes to `musicprint_training_processed` (Docker volume)
- Indexing pipeline mounts that volume as external + mounts `adapter_training_pipeline/release` read-only at `/vol/model`
- Smoke tests use a `shared-fixtures` volume for fixture exchange between Python generator and C++ runner
- Source audio lives in `music/` (gitignored), mounted read-only at `/vol/src_music`

## Key Technical Details

- All Python pipelines run inside `nvidia/pytorch:24.01-py3` containers (except meta_tokenizer which uses `python:3.10-slim`)
- MERT model requires `trust_remote_code=True` for HuggingFace loading
- ArcFace loss is in `adapter_training_pipeline/src/models/loss.py`
- C++ tests are plain assertion-based (no framework), conditionally compiled when `NOT IOS`
- `cli_search` is both a test harness and the reference implementation for the search algorithm
