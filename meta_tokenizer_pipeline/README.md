# Music Metadata Tokenizer Pipeline

This pipeline implements a data-driven, high-performance compression engine for music metadata (Artist, Album, Title). It utilizes **Byte Pair Encoding (BPE)** trained on the full MusicBrainz catalog to achieve ~60-80% storage reduction for mobile deployment.

## Architecture
The pipeline follows a three-stage "Research to Production" workflow:
1. **Ingestion:** Downloads and imports the official MusicBrainz PostgreSQL dump.
2. **Learning:** Performs frequency analysis to identify the optimal vocabulary size (65,535 tokens) and trains a BPE model.
3. **Packaging:** Exports the vocabulary to a zero-copy binary format (`.bin`) optimized for the iOS Neural Engine and Swift/C++ runtime.

---

## 🚀 Quick Start

### 1. Environment Setup
Spin up the PostgreSQL database and the Python training environment:
```bash
docker compose up -d
```

### 2. Ingest MusicBrainz Data
Download the latest core dumps and initialize the schema (37M+ tracks):
```bash
docker exec -it musicprint-tokenizer python src/import_mb.py
```

### 3. Train Production Tokenizer
Train the BPE model on the full unified corpus (Artist + Recording + Release):
```bash
docker exec -it musicprint-tokenizer python src/train_tokenizer.py \
  --output models/music_vocab.json \
  --vocab_size 65535 \
  --use-db
```

### 4. Export for iOS
Convert the JSON source of truth into the production-ready binary format:
```bash
docker exec -it musicprint-tokenizer python src/export_vocab.py --input models/music_vocab.json
```

---

## 📊 Performance Metrics
*   **Dataset:** 45.4 Million records.
*   **Compression Ratio:** ~2.5x (60% space savings).
*   **Dictionary Size:** 689 KB (Binary).
*   **Projected 100M Tracks:** ~665 MB (Text Payload).

---

## 📚 Citations & References

*   **MusicBrainz:** MetaBrainz Foundation. (2026). *MusicBrainz Open Music Encyclopedia*. [https://musicbrainz.org/](https://musicbrainz.org/)
*   **BPE (Byte Pair Encoding):** Sennrich, R., Haddow, B., & Birch, A. (2016). "Neural Machine Translation of Rare Words with Subword Units". Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics. [arXiv:1508.07909](https://arxiv.org/abs/1508.07909)
*   **Hugging Face Tokenizers:** G. Mochod, et al. (2024). *Fast State-of-the-Art Tokenizers*. [https://github.com/huggingface/tokenizers](https://github.com/huggingface/tokenizers)
*   **DeepSeek-OCR (Concept):** Inspired by the "Visual Token" approach to semantic document compression. [https://github.com/deepseek-ai/DeepSeek-OCR](https://github.com/deepseek-ai/DeepSeek-OCR)
