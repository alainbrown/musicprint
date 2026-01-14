# MusicPrint Pipeline Development Guide

This guide outlines the workflow for training, evaluating, and exporting the MusicPrint fingerprinting model.

## 1. Environment Setup (H200 Cluster) 

We use a GPU-Direct architecture (NVIDIA DALI + PyTorch Lightning).

```bash
cd audio_vector_pipeline
# Build and start the container
docker compose up --build -d

# Enter the container for interactive commands
docker compose exec pipeline bash
```

## 2. Training (The Learning Phase)

We train the `MERTAdapter` to compress 768d audio features into 64-bit binary hashes.
We use **Supervised Contrastive Loss (SupCon)** to ensure robustness to noise.

```bash
# Run training on all 8 GPUs
# --auto_batch_size: Automatically finds the max batch size for your H200s (likely 2048+)
python src/train.py \
    --data_dir /app/data \
    --checkpoint_dir /app/checkpoints \
    --epochs 100 \
    --auto_batch_size
```

*   **Artifacts:** Checkpoints are saved to `audio_vector_pipeline/checkpoints/`.
*   **Best Model:** The trainer automatically saves the top 3 models based on `val_loss`.

## 3. Evaluation (The Validation Phase)

Before building the massive index, we verify the model's "Brain" using a **Mini-Universe Simulation**.
This script builds a temporary in-memory index of the validation set and queries it with noisy, random crops.

```bash
# Test a specific checkpoint
python src/evaluate.py --checkpoint_path /app/checkpoints/mert-adapter-epoch=42-val_loss=0.03.ckpt
```

*   **Success Metric:** Look for `Recall@1 > 90%`.
*   **What it proves:** The model is robust to Noise, Speed Shifts, and Temporal Misalignment.

## 4. Production Release

Once you have a model with >90% Recall, you prepare it for the iOS app.

### A. Export to CoreML
Converts the PyTorch model to an Apple `.mlpackage` for the iPhone Neural Engine.

```bash
python src/export.py --checkpoint_path /app/checkpoints/best.ckpt
```
*   **Output:** `audio_vector_pipeline/release/MusicPrintEncoder.mlpackage`
*   **Sync:** This file is automatically symlinked to `ios_app/Models/` for Xcode.

### B. Build Full Index (Optional Server-Side)
If you are deploying a server-side search (instead of on-device), use this to index 100M songs.

```bash
python src/index.py --checkpoint_path /app/checkpoints/best.ckpt
```
*   **Output:** `/app/cache/index/` (Sharded `.pt` files).

## 5. Development Tips

*   **Jupyter Lab:** Available at `http://localhost:8888` (Password: `musicprint`). Use this for visualizing embeddings or debugging DALI pipelines.
*   **Monitoring:** Use TensorBoard (if configured) or watch the CLI progress bar for `val_loss`.
*   **Git LFS:** Large model files (`.pt`, `.mlpackage`) are tracked by LFS. Do not force-commit them to standard git.

```