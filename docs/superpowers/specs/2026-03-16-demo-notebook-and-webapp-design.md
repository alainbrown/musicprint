# Demo Notebook & Web App Design

## Problem

MusicPrint has four independent pipelines and a C++ library but no way to run the full system end-to-end and demonstrate it working. We need two things: a reproducible verification workflow for technical users, and a visual demo for anyone.

## Deliverables

### 1. Top-Level Notebook (`demo.ipynb`)

A single Jupyter notebook at the repo root that takes a `music/` folder of MP3s and builds + verifies the entire system.

Runs inside the existing `musicprint-pipeline` container (the training pipeline image), which already has PyTorch, DALI, HuggingFace, and all ML dependencies. No new Docker image needed.

**Step 1 — Train Encoder**

Imports and calls the adapter training pipeline modules directly (not subprocess). Preprocesses MP3s to FLAC, trains the MERT-v1-95M adapter with ArcFace loss, exports `encoder.pt` (TorchScript). Uses existing code from `adapter_training_pipeline/src/`.

**Step 2 — Build Index**

Runs inference on all songs (full tracks) using the frozen encoder. Binarizes embeddings (sign bits of 64 floats to uint64). Writes `audio_index.bin` with the standard binary format (64-byte header + 16-byte entries). Also builds the metadata DB using meta tokenizer pipeline modules, producing `music_meta.bin` and `music_decoder.bin`.

All artifacts written to a `release/` directory that the web app can mount.

**Step 3 — Clean Verification**

Randomly selects 5% of tracks. For each, takes a random 10-second clip, encodes it through the encoder, binarizes, and runs a Hamming distance search against the full index. The song is in the index — the test is whether a short clip produces a fingerprint close enough for a correct top-1 match.

Reports top-1 recall (% of clips that correctly identify their song).

**Step 4 — Degraded Verification**

Same 5% of tracks, same random 10-second clips, but with realistic degradation applied before encoding:
- Background noise (additive)
- Volume scaling (random gain)
- Low-pass filtering (simulate phone speaker)

Reports degraded top-1 recall.

**Step 5 — Summary**

Prints a report: catalog size (number of songs), index size on disk, clean recall, degraded recall, average query time.

### 2. Web App (`demo_app/`)

A lightweight Flask app with its own Docker container. Consumes pre-built artifacts from the notebook. No GPU required.

**Backend (`demo_app/app.py`)**

- Loads `encoder.pt` on startup. Auto-detects GPU, falls back to CPU.
- Single endpoint `POST /identify`: accepts audio (uploaded file or recorded WAV blob), resamples to 24kHz, takes 10 seconds, runs through encoder, binarizes to uint64, Hamming search against index (Python — XOR + popcount + argmin), looks up metadata via the binary DB, returns JSON `{title, artist, isrc, distance}`.
- Serves static frontend files.

**Frontend (`demo_app/static/index.html`)**

- Single page with two input modes:
  - Mic record button: uses browser MediaRecorder API, sends recorded blob to `/identify`
  - File upload: sends audio file to `/identify`
- Displays result card: song title, artist, confidence (Hamming distance).

**Docker (`demo_app/Dockerfile`)**

- Lightweight Python image with PyTorch (CPU-capable), Flask, and audio processing (librosa or torchaudio for resampling).
- No NVIDIA runtime required.

**Compose (`demo_app/docker-compose.yml`)**

- Mounts `release/` artifacts read-only: `encoder.pt`, `audio_index.bin`, `music_meta.bin`, `music_decoder.bin`
- Exposes port 5000.

## How The Pieces Connect

```
music/ (MP3s)
  |
  +---> demo.ipynb (runs inside training pipeline container, GPU)
  |       |-- Train encoder  -> release/encoder.pt
  |       |-- Build index    -> release/audio_index.bin
  |       |-- Build meta DB  -> release/music_meta.bin, release/music_decoder.bin
  |       |-- Clean recall test (5%, 10s clips)
  |       +-- Degraded recall test (5%, 10s clips + noise)
  |
  +---> demo_app/ (standalone, no GPU required)
          |-- Mounts release/ read-only
          |-- Flask backend (encoder + Hamming search in Python)
          +-- Browser frontend (mic record / file upload -> result card)
```

The notebook produces the artifacts. The web app consumes them. They are independent — you can run the notebook without the web app, and the web app without re-running the notebook (as long as artifacts exist).

## What We Are Not Building

- No changes to libmusicprint (stays as-is for the iOS story)
- No album art pipeline integration (title + artist is sufficient for the demo)
- No new Docker image for the notebook (reuses the training pipeline image)
- No server-side infrastructure — everything runs locally

## Decisions Made

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Holdout test method | Index all songs, query 5% with random 10s clips | Tests audio recall — impossible to recall something not in the index |
| Query degradation | Clean pass + degraded pass | Clean catches pipeline bugs, degraded measures real-world robustness |
| Web app audio input | Mic recording + file upload | Mic is the showpiece, file upload is practical for testing |
| Web app GPU requirement | Auto-detect (GPU if available, CPU fallback) | Runs anywhere without configuration |
| Notebook container | Reuse training pipeline image | Already has all dependencies, no duplication |
| Web app container | Separate lightweight image | No GPU runtime, minimal dependencies |
