# Comprehensive Architecture Review: 100M Track Offline Music Identification

**Goal:** Identify audio clips (10s) locally on iPhone against a **100M track database**.
**Constraint:** Storage < 3.0 GB | Latency < 150ms | iPhone 13+ (A15 Bionic)
**Core Innovation:** **Unified Tokenization (Codebooks)** for both Audio and Text.

---

## 1. The Core Technique: "DeepSeek-Style" Compression
To solve the "Brick Wall" problem of storing 100M tracks on a phone (which mathematically requires >4GB for raw text alone), we adopt a **Codebook-First Architecture**. We do not store raw data; we store sequences of integers that reference learned vocabularies.

### The Unified Storage Model
The system is divided into three highly optimized layers that fit into **~1.2 GB Total**.

| Layer | Content | Technique | Size Est. |
| :--- | :--- | :--- | :--- |
| **1. Audio (Index)** | 100M Fingerprints | **Product Quantization (PQ)** | **~400 MB** |
| **2. Structure** | Artist <-> Song Link | **Clustered Range Index** | **~200 MB** |
| **3. Text (Metadata)** | Song Titles | **Domain-Specific BPE** | **~600 MB** |

---

## 2. Architecture Specs

### Option A+: MERT + PQ + BPE (The "Offline 100M" Path)

#### A. The Backbone (Audio Understanding)
*   **Model:** `m-a-p/MERT-v1-95M` (HuggingFace).
*   **Execution:** Quantized to Float16 on Apple Neural Engine (ANE).
*   **Output:** 768-dimensional embeddings.

#### B. The Audio Index (Search)
*   **Technique:** **Product Quantization (PQ)**.
    *   Instead of raw vectors or binary hashes, we split the 768-dim vector into 8 sub-vectors.
    *   Each sub-vector is mapped to a centroid ID (0-255).
*   **Storage:** 8 bytes per song (8 x 1 byte indices).
*   **Lookup:** Look up pre-computed distance tables for centroids (Asymmetric Distance Calculation).

#### C. The Metadata Database (Text)
We replace standard SQLite with a custom **Memory-Mapped Binary Blob**.

1.  **Artist Layer (Structural Deduplication):**
    *   **Insight:** "Taylor Swift" appears on 300 tracks.
    *   **Method:** Sort the database by Artist. Store `ArtistID` only once per range.
    *   **Structure:** `[Start_Song_ID, End_Song_ID] -> ArtistID`.
    *   **Cost:** Effectively **0 bytes per song**.

2.  **Title Layer (BPE Tokenization):**
    *   **Insight:** Titles are repetitive ("Symphony", "Remix", "feat.").
    *   **Method:** Train a custom **Byte Pair Encoding (BPE)** tokenizer on 10M music titles.
    *   **Vocabulary:** Static `vocab.json` (~500KB) shipped with the app.
    *   **Storage:** Variable Length Integers (VarInts). Common words = 1 byte.
    *   **Example:** "Bohemian Rhapsody" $\to$ `[4821, 931]` (3 bytes).

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
*   **Audio/ML:** **PyTorch** + **Faiss** (for training PQ centroids).
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
*   **BPE:** Sennrich, R., et al. (2016). *"Neural Machine Translation of Rare Words with Subword Units"*.
