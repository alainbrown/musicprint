# Demo Notebook & Web App Design

## Problem

MusicPrint has four independent pipelines and a C++ library but no way to run the full system end-to-end and demonstrate it working. We need two things: a reproducible verification workflow for technical users, and a visual demo for anyone.

## Deliverables

### 1. Top-Level Notebook (`demo.ipynb`)

A single Jupyter notebook at the repo root that takes a `music/` folder of MP3s and builds + verifies the entire system.

Runs inside the existing `musicprint-training` container (the training pipeline image), which already has PyTorch, DALI, HuggingFace, and all ML dependencies. No new Docker image needed.

**Filename convention:** MP3 files in `music/` must be named with their 12-character ISRC (e.g., `GBAYE0601498.mp3`). This is how the existing data pipeline maps files to identifiers.

**Step 1 — Train Encoder**

Runs the adapter training pipeline as a subprocess (`python adapter_training_pipeline/src/pipeline.py`), not via direct import. This avoids conflicts with WandB initialization, Lightning trainer event loops, and DALI GPU contexts inside the notebook kernel. The notebook sets `WANDB_MODE=disabled` in the environment before invoking.

Output: `release/encoder.pt` (TorchScript).

**Step 2 — Build Index**

Runs the vector index pipeline as a subprocess (`python vector_index_pipeline/src/pipeline.py --model_path release/encoder.pt`). This uses the existing windowing logic: each song is split into 5-second windows (120,000 samples at 24kHz) with 1-second stride, each window produces a separate binary hash, and all windows are written as entries in the index. A single song has multiple entries.

Output: `release/audio_index.bin` (64-byte header + 16-byte entries: 8B hash + 8B packed ISRC).

**Step 3 — Copy Metadata Artifacts**

The metadata DB requires a MusicBrainz PostgreSQL instance, which is outside the scope of this demo. Instead, the notebook copies the pre-built artifacts from `meta_tokenizer_pipeline/release/` (which are git-tracked) into `release/`. These files are: `music_meta.bin`, `music_decoder.bin`, `music_encoder.json`.

**Step 4 — Clean Verification**

Randomly selects 5% of tracks. For each, takes a random 10-second segment, splits it into 5-second windows (matching the encoder's training window), encodes each window, binarizes, and searches the index. A song is "identified" if any window's top-1 match returns the correct ISRC (majority vote across windows).

Reports top-1 recall (% of test tracks correctly identified).

**Step 5 — Degraded Verification**

Same 5% of tracks, same random 10-second segments, but with degradation applied before windowing and encoding:
- Additive background noise (white noise, SNR 5-15 dB random)
- Volume scaling (random gain -12 to +6 dB)
- Low-pass filter at 4kHz (simulates phone speaker)
- Degradations are combined (all three applied to each clip)

Reports degraded top-1 recall.

**Step 6 — Summary**

Prints a report: catalog size (number of songs), number of index entries (windows), index size on disk, clean recall, degraded recall, average query time per track.

### 2. Web App (`demo_app/`)

A lightweight Flask app with its own Docker container. Consumes pre-built artifacts from the notebook. No GPU required.

**Backend (`demo_app/app.py`)**

- Loads `encoder.pt` on startup. Auto-detects GPU, falls back to CPU.
- Single endpoint `POST /identify`: accepts audio (uploaded file or recorded blob in any common format — WAV, MP3, WebM, OGG). Uses ffmpeg (via pydub) to decode and resample to 24kHz mono. Splits into 5-second windows, encodes each, binarizes, runs Hamming search against index (Python: XOR + popcount), takes best match across windows. Looks up metadata and returns JSON `{title, artist, isrc, distance}`.
- Validates that artifact files exist on startup; exits with a clear error if missing.
- Serves static frontend files.

**Metadata lookup module (`demo_app/metadata.py`)**

A Python module that reads the binary metadata files. This is non-trivial and needs its own module:
- `music_meta.bin` reader: parses the 128-byte header, binary-searches the ISRC index (12-byte entries: 8B packed ISRC + 4B internal ID), reads title tokens from offset table, reads artist from clustered range table.
- `music_decoder.bin` reader: maps 16-bit BPE token IDs back to strings using the binary vocabulary format (header + offset array + UTF-8 blob). This avoids depending on the `tokenizers` library.
- Exposes a simple `lookup(packed_isrc) -> {title, artist}` interface.

**Frontend (`demo_app/static/index.html`)**

- Single page with two input modes:
  - Mic record button: uses browser MediaRecorder API, sends recorded blob to `/identify`
  - File upload: sends audio file to `/identify`
- Displays result card: song title, artist, Hamming distance.

**Docker (`demo_app/Dockerfile`)**

- Python slim image with PyTorch (CPU wheel), Flask, pydub, and ffmpeg.
- No NVIDIA runtime required.

**Compose (`demo_app/docker-compose.yml`)**

- Mounts `release/` artifacts read-only: `encoder.pt`, `audio_index.bin`, `music_meta.bin`, `music_decoder.bin`
- Exposes port 5000.

## How The Pieces Connect

```
music/ (MP3s, named by ISRC)
  |
  +---> demo.ipynb (runs inside training pipeline container, GPU)
  |       |-- Train encoder       -> release/encoder.pt
  |       |-- Build index         -> release/audio_index.bin
  |       |-- Copy metadata       -> release/music_meta.bin, music_decoder.bin
  |       |-- Clean recall (5%, 10s clips, 5s windows)
  |       +-- Degraded recall (5%, 10s clips + noise, 5s windows)
  |
  +---> demo_app/ (standalone, no GPU required)
          |-- Mounts release/ read-only
          |-- Flask + Python Hamming search + metadata reader
          +-- Browser frontend (mic / file upload -> result card)
```

The notebook produces the artifacts. The web app consumes them. They are independent.

## What We Are Not Building

- No changes to libmusicprint (stays as-is for the iOS story)
- No album art pipeline integration (title + artist is sufficient)
- No new Docker image for the notebook (reuses training pipeline image)
- No MusicBrainz PostgreSQL setup (pre-built metadata artifacts are git-tracked)
- No server-side infrastructure — everything runs locally

## Technical Notes

**Windowing:** The encoder was trained on 5-second windows (120,000 samples at 24kHz). The indexer creates multiple hashes per song using `audio.unfold(0, 120000, 24000)` — 5-second windows with 1-second stride. Both the notebook verification and the web app must use the same 5-second windowing for queries, with the best match across windows as the result.

**Terminology:** The existing index pipeline code uses variable names like `codes_` and comments referencing "PQ codes," but no Product Quantization codebook is involved. The actual operation is sign-bit binarization (threshold at 0, packbits). The naming is a leftover from an earlier PQ-based design.

**Python Hamming search:** Both the notebook and web app implement the search in Python rather than calling the C++ library. For the demo catalog sizes this is fast enough. The C++ library remains the path for mobile deployment.

## Decisions Made

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Holdout test method | Index all songs, query 5% with random 10s segments windowed to 5s | Tests audio recall with the same windowing used at index time |
| Query degradation | Clean pass + degraded pass (noise + volume + low-pass combined) | Clean catches pipeline bugs, degraded measures real-world robustness |
| Web app audio input | Mic recording + file upload | Mic is the showpiece, file upload is practical for testing |
| Web app GPU requirement | Auto-detect (GPU if available, CPU fallback) | Runs anywhere without configuration |
| Notebook container | Reuse training pipeline image | Already has all dependencies, no duplication |
| Web app container | Separate lightweight image | No GPU runtime, minimal dependencies |
| Pipeline invocation | Subprocess, not direct import | Avoids WandB/Lightning/DALI conflicts in notebook kernel |
| Metadata DB | Copy pre-built git-tracked artifacts | MusicBrainz PostgreSQL is out of scope for demo |
| Audio format handling | pydub + ffmpeg in web app | Handles WebM/OGG from MediaRecorder and common upload formats |
