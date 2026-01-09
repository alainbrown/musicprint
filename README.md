# MusicPrint

> **High-scale acoustic fingerprinting system using MERT transformers, Product Quantization, and optimized C++ search. Scales to 100M tracks locally on mobile devices.**

MusicPrint is an offline-first music recognition engine designed to run efficiently on consumer hardware (e.g., iPhone 13+). It leverages state-of-the-art self-supervised audio models (MERT-v1) and vector quantization techniques to compress a 100 million song database into under 3GB.

## 🏗 Architecture

The system is composed of a symmetrical **Encoder-Database-Decoder** pipeline:

1.  **Audio Understanding:**  
    Input audio (10s @ 24kHz) is processed by a **MERT-v1 adapter** (PyTorch) to produce robust 768-dim embeddings, which are projected to a compact **64-dim latent space**.
2.  **Indexing (Training):**  
    These vectors are compressed using **Product Quantization (PQ)** (8 sub-vectors x 8 bits) into an optimized binary index.
3.  **Search (Inference):**  
    A custom **C++ Searcher** uses Asymmetric Distance Calculation (ADC) to perform nearest-neighbor search over 100M vectors in < 150ms on a mobile CPU.

## 📂 Repository Structure

*   **`audio_vector_pipeline/`**: PyTorch Lightning pipeline for training the MERT adapter and PQ codebooks. Uses NVIDIA DALI for high-performance ingestion.
*   **`libmusicprint/`**: High-performance C++ library for loading the index and performing search. (Targeting iOS/CoreML).
*   **`meta_tokenizer_pipeline/`**: NLP pipeline for training BPE tokenizers to compress song metadata (Artist/Title).
*   **`album_art_tokenizer_pipeline/`**: (In Progress) VQ-VAE pipeline for compressing album artwork.
*   **`tests/`**: End-to-End smoke tests and verification scripts.

## 🚀 Quick Start

### Prerequisites
*   **Docker** & **Docker Compose**
*   **NVIDIA GPU** + **NVIDIA Container Toolkit** (Required for training/indexing)

### Running the End-to-End Smoke Test
The easiest way to verify the system is to run the containerized smoke test. This orchestrates the full lifecycle:
1.  Ingests a sample audio file.
2.  Trains the MERT adapter and PQ codebook.
3.  Builds a binary index.
4.  Compiles the C++ searcher and runs a query.

```bash
# Run the full E2E pipeline (requires GPU)
docker compose -f docker-compose.test.yml up --build
```

### Manual Development

**1. Audio Pipeline (Python/PyTorch)**
```bash
cd audio_vector_pipeline
docker compose up -d # Starts JupyterLab & Training container
```

**2. Core Library (C++)**
```bash
cd libmusicprint
mkdir build && cd build
cmake .. && make
./cli_search <query.bin> <index.bin> <centroids.bin> ...
```

## 🛠 Technology Stack

*   **ML Framework:** PyTorch Lightning, HuggingFace Transformers
*   **Audio Model:** MERT-v1-95M
*   **Data Loading:** NVIDIA DALI
*   **Indexing:** Faiss (Training), Custom C++ (Inference)
*   **Target Platform:** iOS (Swift/C++ Interop)
