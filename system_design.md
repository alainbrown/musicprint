# Comprehensive Architecture Review: 100M Track Offline Music Identification

**Goal:** Identify audio clips (10s) locally on iPhone against a **100M track database**.
**Constraint:** Storage < 3.0 GB | Latency < 150ms | iPhone 13+ (A15 Bionic)
**Core Innovation:** **Unified Tokenization (Codebooks)** for both Audio and Text.

---

## 1. The Core Technique: Codebook-Based Compression
To address the **Information Theoretic Limit** of storing 100M tracks on a mobile device (which exceeds 4GB for raw text alone), we adopt a **Codebook-First Architecture**. We do not store raw data; we store sequences of integers that reference learned domain-specific vocabularies.

### The Unified Storage Model
The system is divided into three highly optimized layers. Based on empirical analysis of 45M tracks, we project a **~2.3 GB Total** footprint for 100M tracks.

| Layer | Content | Technique | Size Est. (100M) |
| :--- | :--- | :--- | :--- |
| **1. Audio (Index)** | 100M Fingerprints | **Product Quantization (PQ)** | **~400 MB** |
| **2. Structure** | Artist/Album Link | **Clustered Range Index** | **~100 MB** |
| **3. Text (Metadata)** | Titles/Artists/Albums| **Domain-Specific BPE** | **~1.74 GB** |
| **4. Visual (Art)** | Album Covers | **VQ-VAE Tokenizer** | **~200 MB** |

### Symmetrical Pipeline Architecture
To maintain a unified deployment model, each data stream (Audio, Text, Visual) follows a symmetrical **Encoder-Database-Decoder** pattern. This ensures the mobile client uses a consistent `mmap`-based access pattern across all modules.

| Data Stream | Encoder (Cloud/Training) | Database (Binary Artifact) | Decoder (On-Device) |
| :--- | :--- | :--- | :--- |
| **Audio** | MERT + PQ | `audio_index.bin` | Asymmetric Distance (ADC) |
| **Text** | BPE Tokenizer | `music_meta.bin` | BPE Decoder (16-bit) |
| **Visual** | VQ-VAE Encoder | `art.bin` | VQ-VAE Decoder (Neural) |

---

## 2. Architecture Specs

### Option A+: MERT + PQ + BPE + VQ-VAE (The "Full Experience" Path)

#### A. The Backbone (Audio Understanding)
*   **Model:** `m-a-p/MERT-v1-95M` (HuggingFace).
*   **Execution:** Quantized to Float16 on Apple Neural Engine (ANE).
*   **Adapter:** Projects 768-dimensional MERT embeddings to a compact **64-dimensional** latent space.

#### B. The Audio Index (Search)
*   **Technique:** **Product Quantization (PQ)**.
    *   The 64-dim embedding is split into **8 sub-vectors** (each of dimension 8).
    *   Each sub-vector is quantised to a centroid ID (0-255).
*   **Storage:** 8 bytes per song (8 x 1 byte indices).
*   **Lookup:** Look up pre-computed distance tables for centroids (Asymmetric Distance Calculation).

#### C. The Metadata Database (Text)
We replace standard SQLite with a custom **Memory-Mapped Binary Blob** (`music_meta.bin`).

1.  **Artist/Album Layer (Structural Deduplication):**
    *   **Method:** Physical clustering (sort by Artist ID then Album ID).
    *   **Structure:** Dual-level Range Tables (`[Start_ID] -> Offset`).
    *   **Benefit:** Removes redundant storage of Artist/Album names across 100M rows.

2.  **Title Layer (BPE Tokenization):**
    *   **Decision:** **65,535 Vocabulary** using 16-bit Fixed-Width integers (`uint16`).
    *   **Truncation (UX-Optimized):** 
        *   Artist: **15** tokens max.
        *   Album: **20** tokens max.
        *   Title: **25** tokens max.
    *   **Encoding:** Standard HuggingFace Byte-Level BPE (Zero-loss for all scripts).

#### D. The Visual Layer (Album Art)
*   **Goal:** Store ~500k unique album covers in < 200MB.
*   **Technique:** **VQ-VAE (Vector Quantized VAE)**.
    *   **Resolution:** 128x128 input -> 16x16 token grid.
    *   **Vocabulary:** 1024 learned visual tokens (10-bit).
*   **Storage (art.bin):** 
    *   **Flat Array:** Indexed directly by `Album ID` for $O(1)$ access.
    *   **Payload:** ~320 bytes per record.
*   **UX Flow:** Decoded on-demand via the neural decoder only upon a successful match.

---

## 3. Current Benchmarks (Phase 1 Build)
The following metrics were verified during the initialization of the MusicBrainz (45.4M records) database:

*   **Total tracks indexed:** 53,650,719
*   **Metadata DB size:** **934.06 MB**
*   **Vocabulary size:** 689 KB (Binary)
*   **Average bytes per track:** 18.26 bytes (Total Metadata)
*   **Total App Footprint (Current):** **~1.26 GB** (Model + Index + 53M Meta)

---

## 3. Evaluation Plan

### A. The "Golden Dataset" Strategy
To rigorously test the system without encoding 100M tracks immediately, we construct a **Micro-Universe** of 10,000 tracks (0.01% scale) that statistically represents the full distribution.

*   **Source:** FMA (Free Music Archive) "Large" Subset + GTZAN.
*   **Composition:**
    *   **The "Head" (Pop/Rock):** 60% of tracks. Simple structures.
    *   **The "Tail" (Jazz/Classical/Experimental):** 20% of tracks. Complex, dynamic structures.
    *   **The "Noise" (Speech/Silence/White Noise):** 20% negative samples.

### B. Key Performance Indicators (KPIs)

#### Intelligence Metrics (Recall & Robustness)
*Target: We assume "Success" if the correct Song ID is Rank 1.*

| Metric | Condition | Target | Rationale |
| :--- | :--- | :--- | :--- |
| **Recall@1 (Clean)** | Direct digital crop (10s). | **> 99.0%** | Baseline functionality. |
| **Recall@1 (Noisy)** | Added "Cafe/Bar" noise @ 5dB SNR. | **> 92.0%** | The core "Shazam" use case. |
| **Recall@1 (Distorted)** | Mic impulse response + compression (GSM). | **> 90.0%** | Simulates phone mic hardware. |
| **Recall@1 (Speed)** | Speed/Pitch shift ±4%. | **> 85.0%** | Radio speed-up / Turntable variation. |
| **False Positive Rate** | Input = Speech, Silence, Traffic. | **< 0.1%** | User trust. Better to say "No Match" than wrong match. |

#### Physics Metrics (Hardware Constraints)
*Target: iPhone 13 (A15 Bionic) Limits.*

| Metric | Definition | Target | Failure Threshold |
| :--- | :--- | :--- | :--- |
| **Storage (Total)** | Index + Metadata + App. | **< 1.5 GB** | > 3.0 GB. |
| **End-to-End Latency** | `Audio Buffer` $\to$ `Song Title` (P99). | **< 150 ms** | > 300 ms. |
| **Peak RAM (Cold)** | Memory spike during 1st query. | **< 350 MB** | > 1.2 GB. |

### C. End-to-End Validation
To ensure system integrity across the Python training pipeline and C++ runtime, we employ a containerized "Smoke Test" that:
1.  **Generates** a micro-universe (1 track) from real audio.
2.  **Trains** the MERT adapter and PQ codebook on this sample.
3.  **Indexes** the audio using the generated artifacts.
4.  **Searches** using the C++ CLI tool to verify the correct ISRC is retrieved.
This ensures parity between the training (PyTorch) and inference (C++) environments.

---

## 4. Development & Scaling Strategy

### A. The Two-Stage Trainer
The pipeline now requires two distinct training phases before indexing:

1.  **Audio Codebook Trainer:**
    *   Input: 1M MERT vectors.
    *   Output: PQ Centroids (for audio compression).
2.  **Text Codebook Trainer:**
    *   Input: 10M Song Titles.
    *   Output: BPE Vocabulary (for text compression).

### B. Incremental Rollout (Production)
Scaling from 1 to 100 million tracks will be done in phases.

1.  **Phase 1: Alpha (1M Tracks):** Top global hits.
    *   **Index Size:** ~15 MB.
2.  **Phase 2: Beta (10M Tracks):** Full mainstream discographies.
    *   **Index Size:** ~150 MB.
3.  **Phase 3: Gold (100M Tracks):** Comprehensive global catalog.
    *   **Index Size:** ~1.2 GB.

---

## 5. Technology Stack

### A. Development (Cloud/Vast.ai)
*   **Language:** Python 3.10+
*   **Audio/ML:** **PyTorch Lightning** + **Faiss** (for training PQ centroids and VQ-VAE).
*   **Tokenizer:** **HuggingFace Tokenizers** (Rust-based BPE).
*   **Data Processing:** **NVIDIA DALI**.

### B. Deployment (iPhone 13+)
*   **Interface:** **Swift / SwiftUI**.
*   **Neural Inference:** **CoreML** (MERT).
*   **Search Engine:** **Custom C++ Searcher**.
    *   Implements Asymmetric Distance Calculation (ADC) for PQ.
    *   Implements BPE Decoder for text.
    *   Memory-mapped binary reader.

---

## 6. Citations

*   **MERT Model:** Li, Y., et al. (2024). *"MERT: Acoustic Music Understanding Model with Large-Scale Self-supervised Training"*. International Conference on Learning Representations (ICLR). [arXiv:2306.01075]
*   **Product Quantization:** Jegou, H., et al. (2011). *"Product Quantization for Nearest Neighbor Search"*. IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI).
*   **Faiss:** Johnson, J., et al. (2019). *"Billion-scale similarity search with GPUs"*. IEEE Transactions on Big Data. [arXiv:1702.08734]
*   **BPE:** Sennrich, R., et al. (2016). *"Neural Machine Translation of Rare Words with Subword Units"*. [arXiv:1508.07909](https://arxiv.org/abs/1508.07909)
*   **VQ-VAE:** Van den Oord, A., et al. (2017). *"Neural Discrete Representation Learning"*. Neural Information Processing Systems (NeurIPS). [arXiv:1711.00937](https://arxiv.org/abs/1711.00937)
