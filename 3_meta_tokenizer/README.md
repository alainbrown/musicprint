# Music Metadata Pipeline

A high-performance compression engine for music metadata (Artist, Album, Title) using **Byte Pair Encoding (BPE)** and **Clustered Storage**. Designed to fit 100M+ tracks on mobile devices within a 3GB total app budget.

## 🏗️ Architecture
1. **Ingestion:** Automated PostgreSQL mirror of the [MusicBrainz](https://musicbrainz.org/) catalog (~45M tracks).
2. **Learning:** Data-driven BPE training using a "Scientific Back-Solver" to identify the 65,535 token optimum.
3. **Storage:** Physically clusters data by Artist/Album to remove redundancy, reducing footprint by ~3x.

---

## 🚀 Execution Flow

### 1. Ingest Data
Setup the Postgres DB and import core metadata:
```bash
docker compose up -d
docker exec -it musicprint-tokenizer python src/import_mb.py
```

### 2. Train Tokenizer
Learn the vocabulary from the full unified corpus (Artist + Recording + Release):
```bash
docker exec -it musicprint-tokenizer python src/train_tokenizer.py \
  --output release/music_encoder.json \
  --vocab_size 65535 \
  --use-db
```

### 3. Build Production Database
Pack the 45M+ tracks into the clustered binary format for iOS:
```bash
docker exec -it musicprint-tokenizer python src/build_db.py
```

### 4. Export Artifacts
Generate the optimized binary vocabulary lookup for the app:
```bash
docker exec -it musicprint-tokenizer python src/export_vocab.py \
  --input release/music_encoder.json \
  --output release/music_decoder.bin
```

---

## 📊 Performance (Current Scale: 53.6M Tracks)
*   **Metadata DB:** 934 MB (`release/music_meta.bin`).
*   **Encoder:** `release/music_encoder.json` (Full BPE model).
*   **Decoder:** `release/music_decoder.bin` (Optimized binary lookup).
*   **Compression:** ~18.2 bytes per track (Total Metadata).
*   **iOS Access:** Zero-copy memory mapping (`mmap`).

---

## 📚 Citations
* **MusicBrainz:** MetaBrainz Foundation. (2026). [https://musicbrainz.org/](https://musicbrainz.org/)
* **BPE:** Sennrich et al. (2016). *Neural Machine Translation of Rare Words with Subword Units*. [arXiv:1508.07909](https://arxiv.org/abs/1508.07909)
* **Tokenizers:** Hugging Face. (2024). [https://github.com/huggingface/tokenizers](https://github.com/huggingface/tokenizers)