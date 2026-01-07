# Album Art Tokenizer Pipeline

A dedicated pipeline for compressing millions of album covers into a tiny "Visual Vocabulary" using **VQ-VAE** (Vector Quantized Variational AutoEncoder).

## 🎯 Goal
Fit **500,000+ Album Covers** (representing 5M+ songs) into **< 200 MB** of storage for mobile offline access.

## 🧠 Architecture

### 1. The Concept
Just as we tokenize text into "subwords" (BPE), we tokenize images into "visual patches".
*   **Standard Image:** 128x128 pixels x 3 colors = 49,152 values.
*   **Tokenized Image:** 16x16 grid of integers = 256 values.
*   **Compression Ratio:** ~192:1 (Lossy).

### 2. The Model (VQ-VAE)
*   **Encoder:** CNN that downsamples the image 8x (128 -> 16).
*   **Quantizer:** Snaps each pixel vector to the nearest "Codebook Vector" (Vocabulary size: 1024).
*   **Decoder:** CNN that reconstructs the image from the codebook vectors.

## 📂 Directory Structure
*   `src/`: Training and inference scripts.
    *   `model.py`: PyTorch VQ-VAE definition.
    *   `train.py`: Main training loop.
    *   `dataset.py`: Cover Art Archive downloader/loader.
*   `data/`: Raw images (cache).
*   `release/`: Trained models and binary codebooks.
    *   `visual_encoder.pth`: The VQ-VAE weights.
    *   `visual_codebook.bin`: The learned vocabulary (exported for iOS).

## 🚀 Usage

### 1. Train the Tokenizer
```bash
python src/train.py --data_dir ./data --vocab_size 1024
```

### 2. Build the Index
Compress all your album art into the binary database:
```bash
python src/build_index.py --input_csv albums.csv --output release/art.bin
```

### 3. Export for Mobile
Convert the Decoder to CoreML for on-device inference:
```bash
python src/export_coreml.py
```
