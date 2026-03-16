# Replace DALI with PyTorch DataLoader

## Problem

The DALI-based data loading pipeline runs audio decoding and augmentation on CPU, causing GPU starvation during training (~9 min/epoch with low GPU utilization). DALI's audio decoder is CPU-only, and the downstream DALI ops (slicing, noise mixing) also run on CPU. The GPU sits idle between short forward/backward pass bursts.

## Solution

Replace DALI with PyTorch DataLoader + torchaudio. DataLoader workers decode and augment audio on CPU in parallel, with `pin_memory=True` for async GPU transfer. This is simpler, removes the DALI dependency, and should keep the GPU better fed through proper prefetching.

## Files to Delete

- `1_adapter_training/src/data/` (entire directory: `__init__.py`, `loader.py`, `module.py`, `pipelines/__init__.py`, `pipelines/contrastive.py`, `pipelines/inference.py`, `pipelines/ops.py`)
- `2_vector_index/src/data/` (entire directory: `__init__.py`, `loader.py`, `module.py`, `pipelines/__init__.py`, `pipelines/inference.py`, `pipelines/ops.py`)

## Files to Create

### `1_adapter_training/src/data/dataset.py`

Two Dataset classes:

**`ContrastiveAudioDataset`** — Used for training.
- `__init__`: receives list of `(filepath, label)` pairs and noise directory path
- `__getitem__`: loads audio with `torchaudio.load()`, resamples to 24kHz with `torchaudio.transforms.Resample` if needed, crops two random 5-second windows (120,000 samples), applies augmentation (additive noise from noise directory, random gain 0.5-1.5), returns `(view_1, view_2, label)` as tensors
- Loads noise files lazily on first access, caches in memory

**`AudioDataset`** — Used for validation.
- `__getitem__`: loads audio, resamples, takes first 5 seconds (deterministic crop), returns `(audio, label)`

### `1_adapter_training/src/data/module.py`

**`MusicDataModule`** (Lightning DataModule):
- Discovers files recursively (`**/*.flac`, fallback to `**/*.wav`, `**/*.mp3`)
- Maps filenames to integer IDs: 12-char stem uses `pack_isrc()`, otherwise `hash(name) & 0xFFFFFFFFFFFFFFFF`
- Train/val split (95/5, seeded)
- Creates DataLoaders with `num_workers=os.cpu_count()`, `pin_memory=True`, `persistent_workers=True`, `prefetch_factor=2`

### `2_vector_index/src/data/dataset.py`

**`InferenceAudioDataset`** — loads full track (no crop), resamples to 24kHz, returns `(audio, label)`. Windowing happens in `index.py:predict_step`.

### `2_vector_index/src/data/module.py`

**`MusicDataModule`** — same file discovery and ID mapping as training, single predict DataLoader.

## Files to Modify

### `1_adapter_training/src/system.py`

Batch format changes from DALI dict to tuples:

```python
# Before (DALI):
data_dict = batch[0]
a1 = data_dict["audio_1"]
a2 = data_dict["audio_2"]
labels = data_dict["label"].reshape(-1).long()

# After (DataLoader):
a1, a2, labels = batch
```

Same change in `validation_step`.

### `2_vector_index/src/index.py`

```python
# Before (DALI):
data_dict = batch[0]
audio_batch = data_dict["audio"]
labels = data_dict["label"]

# After (DataLoader):
audio_batch, labels = batch
```

### `docker/Dockerfile.pipeline`

Remove DALI-related pip installs. Add `torchaudio` if not already present in the base image.

### Requirements files

- `1_adapter_training/requirements.txt` — remove DALI reference, add `torchaudio`
- `2_vector_index/requirements.txt` — remove DALI reference, add `torchaudio`

## What Stays the Same

- `pipeline.py` in both pipelines
- `models/` (MERTAdapter, ArcFace)
- `export.py`, `preprocess.py`, `isrc_utils.py`
- `train.py` (calls `trainer.fit`, doesn't touch data directly)
- All binary formats, search, downstream code

## DataLoader Configuration

```python
DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True,  # training only
    num_workers=os.cpu_count(),
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=2,
)
```

## Decisions Made

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Decoder | torchaudio | Native PyTorch tensors, no numpy conversion, GPU resampling support |
| Scope | Both pipelines | Same DALI code in both, removes dependency entirely |
| File discovery | In MusicDataModule | Module owns split + ID mapping, Dataset loads individual samples |
| Augmentation | In Dataset (CPU workers) | Runs in parallel across workers, GPU only sees ready tensors |
