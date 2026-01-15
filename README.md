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

*   **`adapter_training_pipeline/`**: The "Teacher". Trains the MERT adapter using **ArcFace Loss** to create robust acoustic fingerprints. Exports `encoder.pt` (TorchScript) and `MusicPrintEncoder.mlpackage` (CoreML).
*   **`vector_index_pipeline/`**: The "Librarian". Consumes the frozen `encoder.pt` model to index millions of songs into a compressed binary format. **Zero shared code** with the training pipeline.
*   **`libmusicprint/`**: High-performance C++ library for loading the index and performing search on iOS.
*   **`meta_tokenizer_pipeline/`**: NLP pipeline for training BPE tokenizers to compress song metadata.
*   **`tests/`**: End-to-End smoke tests and verification scripts.

## 🚀 Quick Start

### Prerequisites
*   **Docker** & **Docker Compose**
*   **NVIDIA GPU** (Required for training/indexing)

### Manual Development

**1. Train the Adapter (ArcFace)**
```bash
cd adapter_training_pipeline
docker compose up --build -d
docker compose exec training-pipeline python src/pipeline.py
```
*Output: `release/encoder.pt`, `release/MusicPrintEncoder.mlpackage`*

**2. Build the Index**
```bash
cd vector_index_pipeline
docker compose up --build -d
# Mounts the model from the training pipeline automatically
docker compose exec index-pipeline python src/pipeline.py --model_path /vol/model/encoder.pt
```
*Output: `cache/index/*.bin`*

**3. Core Library (C++)**
```bash
cd libmusicprint
mkdir build && cd build
cmake .. && make
./cli_search <query.bin> <index.bin> ...
```

## 🛠 Technology Stack

*   **ML Framework:** PyTorch Lightning, HuggingFace Transformers
*   **Audio Model:** MERT-v1-95M
*   **Data Loading:** NVIDIA DALI
*   **Indexing:** Faiss (Training), Custom C++ (Inference)
*   **Target Platform:** iOS (Swift/C++ Interop)
