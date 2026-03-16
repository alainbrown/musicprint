# Replace DALI with PyTorch DataLoader Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace DALI data loading with PyTorch DataLoader + torchaudio in both pipelines to fix GPU starvation during training.

**Architecture:** Standard PyTorch Datasets handle audio loading/augmentation in parallel CPU workers. DataLoader with `pin_memory=True` handles async GPU transfer. Lightning DataModule provides the same interface to `train.py` and `index.py`.

**Tech Stack:** PyTorch, torchaudio, pytorch-lightning

**Spec:** `docs/superpowers/specs/2026-03-16-replace-dali-with-pytorch-dataloader.md`

---

## File Structure

```
1_adapter_training/src/data/
├── __init__.py          # empty
├── dataset.py           # ContrastiveAudioDataset, AudioDataset
└── module.py            # MusicDataModule (Lightning)

2_vector_index/src/data/
├── __init__.py          # empty
├── dataset.py           # InferenceAudioDataset
└── module.py            # MusicDataModule (Lightning)
```

**Deleted:** `loader.py`, `pipelines/` (entire subdirectory) in both pipelines.

---

## Chunk 1: Training Pipeline Data Loading

### Task 1: Add torchaudio to Dockerfile

**Files:**
- Modify: `docker/Dockerfile.pipeline`
- Modify: `1_adapter_training/requirements.txt`
- Modify: `2_vector_index/requirements.txt`

- [ ] **Step 1: Add torchaudio to Dockerfile**

```dockerfile
# docker/Dockerfile.pipeline — add torchaudio to the pip install block
RUN pip install --no-cache-dir \
    transformers==4.36.2 \
    jupyterlab \
    ipywidgets \
    jupytext \
    pytorch-lightning \
    coremltools \
    wandb \
    faiss-gpu \
    scipy \
    librosa \
    datasets \
    pytorch-metric-learning \
    torchvision \
    torchaudio \
    pillow \
    matplotlib \
    pandas \
    requests \
    tokenizers \
    psycopg2-binary \
    sqlalchemy \
    tqdm
```

- [ ] **Step 2: Update requirements.txt files**

`1_adapter_training/requirements.txt`:
```
transformers==4.36.2
pytorch-lightning
pytorch-metric-learning
coremltools
wandb
faiss-gpu
scipy
librosa
torchaudio
requests
tqdm
numpy
```

`2_vector_index/requirements.txt`:
```
pytorch-lightning
faiss-gpu
wandb
torchaudio
tqdm
numpy
```

- [ ] **Step 3: Rebuild image and verify torchaudio works**

Run: `docker compose build training && docker run --rm musicprint-pipeline:latest python -c "import torchaudio; print(torchaudio.__version__)"`
Expected: version string printed

- [ ] **Step 4: Commit**

```bash
git add docker/Dockerfile.pipeline 1_adapter_training/requirements.txt 2_vector_index/requirements.txt
git commit -m "chore: add torchaudio to pipeline image"
```

---

### Task 2: Training Pipeline Dataset

**Files:**
- Create: `1_adapter_training/src/data/dataset.py`

- [ ] **Step 1: Write dataset.py**

```python
# 1_adapter_training/src/data/dataset.py
import os
import glob
import random
import torch
import torchaudio
from torch.utils.data import Dataset

SAMPLE_RATE = 24000
WINDOW_SAMPLES = 120000  # 5 seconds


class ContrastiveAudioDataset(Dataset):
    """Loads audio, crops two random 5s windows, applies augmentation."""

    def __init__(self, file_label_pairs, noise_dir=None):
        self.files = file_label_pairs  # list of (filepath, label_int)
        self.noise_files = []
        if noise_dir and os.path.isdir(noise_dir):
            for ext in ("*.wav", "*.flac", "*.mp3"):
                self.noise_files.extend(glob.glob(os.path.join(noise_dir, "**", ext), recursive=True))

    def _load_audio(self, path):
        audio, sr = torchaudio.load(path)
        if audio.shape[0] > 1:
            audio = audio.mean(dim=0, keepdim=True)  # downmix
        if sr != SAMPLE_RATE:
            audio = torchaudio.transforms.Resample(sr, SAMPLE_RATE)(audio)
        return audio.squeeze(0)  # (Time,)

    def _random_crop(self, audio):
        if audio.shape[0] < WINDOW_SAMPLES:
            return torch.nn.functional.pad(audio, (0, WINDOW_SAMPLES - audio.shape[0]))
        start = random.randint(0, audio.shape[0] - WINDOW_SAMPLES)
        return audio[start : start + WINDOW_SAMPLES]

    def _augment(self, audio):
        # Additive noise
        if self.noise_files:
            noise_path = random.choice(self.noise_files)
            try:
                noise = self._load_audio(noise_path)
                if noise.shape[0] >= audio.shape[0]:
                    start = random.randint(0, noise.shape[0] - audio.shape[0])
                    noise = noise[start : start + audio.shape[0]]
                else:
                    noise = torch.nn.functional.pad(noise, (0, audio.shape[0] - noise.shape[0]))
                gain = random.uniform(0.0, 0.3)
                audio = audio + noise * gain
            except Exception:
                pass  # skip noise on decode error

        # Volume perturbation
        audio = audio * random.uniform(0.5, 1.5)
        return audio

    def __getitem__(self, idx):
        path, label = self.files[idx]
        audio = self._load_audio(path)
        view_1 = self._augment(self._random_crop(audio))
        view_2 = self._augment(self._random_crop(audio))
        return view_1, view_2, torch.tensor(label, dtype=torch.long)

    def __len__(self):
        return len(self.files)


class AudioDataset(Dataset):
    """Loads audio, deterministic crop from start. For validation."""

    def __init__(self, file_label_pairs):
        self.files = file_label_pairs

    def _load_audio(self, path):
        audio, sr = torchaudio.load(path)
        if audio.shape[0] > 1:
            audio = audio.mean(dim=0, keepdim=True)
        if sr != SAMPLE_RATE:
            audio = torchaudio.transforms.Resample(sr, SAMPLE_RATE)(audio)
        return audio.squeeze(0)

    def __getitem__(self, idx):
        path, label = self.files[idx]
        audio = self._load_audio(path)
        if audio.shape[0] < WINDOW_SAMPLES:
            audio = torch.nn.functional.pad(audio, (0, WINDOW_SAMPLES - audio.shape[0]))
        else:
            audio = audio[:WINDOW_SAMPLES]
        return audio, torch.tensor(label, dtype=torch.long)

    def __len__(self):
        return len(self.files)
```

- [ ] **Step 2: Commit**

```bash
git add 1_adapter_training/src/data/dataset.py
git commit -m "feat(training): add PyTorch audio datasets"
```

---

### Task 3: Training Pipeline DataModule

**Files:**
- Rewrite: `1_adapter_training/src/data/module.py`
- Keep: `1_adapter_training/src/data/__init__.py` (empty, already exists)
- Delete: `1_adapter_training/src/data/loader.py`
- Delete: `1_adapter_training/src/data/pipelines/` (entire directory)

- [ ] **Step 1: Rewrite module.py**

```python
# 1_adapter_training/src/data/module.py
import pytorch_lightning as pl
import os
import glob
import random
from torch.utils.data import DataLoader
from isrc_utils import pack_isrc
from .dataset import ContrastiveAudioDataset, AudioDataset


def discover_files(data_dir):
    """Recursively find audio files, matching DALI's priority order."""
    for ext in ("*.flac", "*.wav", "*.mp3"):
        files = sorted(glob.glob(os.path.join(data_dir, "**", ext), recursive=True))
        if files:
            return files
    return []


def build_file_label_pairs(files, data_dir):
    """Map files to integer labels. ArcFace needs contiguous [0, N-1] IDs."""
    pairs = []
    for idx, fp in enumerate(sorted(files)):
        pairs.append((fp, idx))
    return pairs


class MusicDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size=32, val_split=0.05, window_secs=5.0, noise_dir=None):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.val_split = val_split
        self.noise_dir = noise_dir or os.path.join(data_dir, "noise")

        self.all_files = discover_files(data_dir)

        random.seed(42)
        shuffled = list(self.all_files)
        random.shuffle(shuffled)

        split_idx = int(len(shuffled) * (1 - val_split))
        self.train_pairs = build_file_label_pairs(shuffled[:split_idx], data_dir)
        self.val_pairs = build_file_label_pairs(shuffled[split_idx:], data_dir)

    def train_dataloader(self):
        ds = ContrastiveAudioDataset(self.train_pairs, noise_dir=self.noise_dir)
        return DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=os.cpu_count(),
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=2,
        )

    def val_dataloader(self):
        if not self.val_pairs:
            return None
        ds = AudioDataset(self.val_pairs)
        return DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=os.cpu_count(),
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=2,
        )
```

- [ ] **Step 2: Delete old DALI files**

```bash
rm 1_adapter_training/src/data/loader.py
rm -r 1_adapter_training/src/data/pipelines/
```

- [ ] **Step 3: Commit**

```bash
git add -A 1_adapter_training/src/data/
git commit -m "feat(training): replace DALI DataModule with PyTorch DataLoader"
```

---

### Task 4: Update system.py for DataLoader batch format

**Files:**
- Modify: `1_adapter_training/src/system.py`

The DataLoader returns `(view_1, view_2, label)` tuples instead of DALI's `[{"audio_1": ..., "audio_2": ..., "label": ...}]`.

- [ ] **Step 1: Update system.py**

```python
# 1_adapter_training/src/system.py
import pytorch_lightning as pl
import torch
import torch.optim as optim
from models.arcface import MusicArcFaceSystem


class MusicPrintSystem(pl.LightningModule):
    def __init__(self, num_classes, lr=1e-4, output_dim=64):
        super().__init__()
        self.save_hyperparameters()
        self.model = MusicArcFaceSystem(
            num_classes=num_classes,
            embedding_dim=output_dim
        )

    def forward(self, x):
        return self.model.backbone(x)

    def training_step(self, batch, batch_idx):
        a1, a2, labels = batch

        audio = torch.cat([a1, a2], dim=0)
        targets = torch.cat([labels, labels], dim=0)

        embeddings = self(audio)
        loss = self.model.get_loss(embeddings, targets)

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        audio, labels = batch

        embeddings = self(audio)
        loss = self.model.get_loss(embeddings, labels)

        self.log("val_loss", loss, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        params = [
            {'params': filter(lambda p: p.requires_grad, self.model.backbone.parameters())},
            {'params': self.model.loss_func.parameters()}
        ]
        return optim.AdamW(params, lr=self.hparams.lr)
```

- [ ] **Step 2: Commit**

```bash
git add 1_adapter_training/src/system.py
git commit -m "feat(training): update system.py for DataLoader batch format"
```

---

## Chunk 2: Indexing Pipeline Data Loading

### Task 5: Indexing Pipeline Dataset and DataModule

**Files:**
- Create: `2_vector_index/src/data/dataset.py`
- Rewrite: `2_vector_index/src/data/module.py`
- Delete: `2_vector_index/src/data/loader.py`
- Delete: `2_vector_index/src/data/pipelines/`

- [ ] **Step 1: Write dataset.py**

```python
# 2_vector_index/src/data/dataset.py
import torch
import torchaudio
from torch.utils.data import Dataset

SAMPLE_RATE = 24000


class InferenceAudioDataset(Dataset):
    """Loads full audio track for indexing. No cropping — windowing happens in predict_step."""

    def __init__(self, file_label_pairs):
        self.files = file_label_pairs

    def __getitem__(self, idx):
        path, label = self.files[idx]
        audio, sr = torchaudio.load(path)
        if audio.shape[0] > 1:
            audio = audio.mean(dim=0, keepdim=True)
        if sr != SAMPLE_RATE:
            audio = torchaudio.transforms.Resample(sr, SAMPLE_RATE)(audio)
        return audio.squeeze(0), torch.tensor(label, dtype=torch.long)

    def __len__(self):
        return len(self.files)


def collate_variable_length(batch):
    """Pad audio to longest in batch since tracks have different lengths."""
    audios, labels = zip(*batch)
    max_len = max(a.shape[0] for a in audios)
    padded = torch.stack([
        torch.nn.functional.pad(a, (0, max_len - a.shape[0])) for a in audios
    ])
    labels = torch.stack(list(labels))
    return padded, labels
```

- [ ] **Step 2: Rewrite module.py**

```python
# 2_vector_index/src/data/module.py
import pytorch_lightning as pl
import os
import glob
from torch.utils.data import DataLoader
from isrc_utils import pack_isrc
from .dataset import InferenceAudioDataset, collate_variable_length


def discover_files(data_dir):
    for ext in ("*.flac", "*.wav", "*.mp3"):
        files = sorted(glob.glob(os.path.join(data_dir, "**", ext), recursive=True))
        if files:
            return files
    return []


def build_file_label_pairs(files, data_dir):
    pairs = []
    for fp in files:
        rel_path = os.path.relpath(fp, data_dir)
        name = os.path.splitext(rel_path)[0]
        label = pack_isrc(name) if len(name) == 12 else hash(name) & 0xFFFFFFFFFFFFFFFF
        pairs.append((fp, label))
    return pairs


class MusicDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size=1):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.all_files = discover_files(data_dir)
        self.pairs = build_file_label_pairs(self.all_files, data_dir)

    def predict_dataloader(self):
        if not self.pairs:
            return None
        ds = InferenceAudioDataset(self.pairs)
        return DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=os.cpu_count(),
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=2,
            collate_fn=collate_variable_length,
        )
```

- [ ] **Step 3: Delete old DALI files**

```bash
rm 2_vector_index/src/data/loader.py
rm -r 2_vector_index/src/data/pipelines/
```

- [ ] **Step 4: Commit**

```bash
git add -A 2_vector_index/src/data/
git commit -m "feat(indexing): replace DALI DataModule with PyTorch DataLoader"
```

---

### Task 6: Update index.py for DataLoader batch format

**Files:**
- Modify: `2_vector_index/src/index.py`

- [ ] **Step 1: Update predict_step**

Change lines 19-24 from DALI dict unpacking to tuple unpacking:

```python
    def predict_step(self, batch, batch_idx):
        audio_batch, labels = batch

        if audio_batch.dim() == 3:
            audio_batch = audio_batch.squeeze(-1)
```

Everything else in predict_step stays the same (windowing, binarization, result dict).

- [ ] **Step 2: Commit**

```bash
git add 2_vector_index/src/index.py
git commit -m "feat(indexing): update index.py for DataLoader batch format"
```

---

## Chunk 3: Verify and Clean Up

### Task 7: Test the training pipeline starts

- [ ] **Step 1: Rebuild the Docker image**

```bash
docker compose build training
```

- [ ] **Step 2: Run a quick training smoke test**

```bash
docker compose run --rm training python -c "
import sys
sys.path.insert(0, 'src')
from data.module import MusicDataModule
dm = MusicDataModule(data_dir='/vol/data', batch_size=4)
print(f'Found {len(dm.all_files)} files')
print(f'Train: {len(dm.train_pairs)}, Val: {len(dm.val_pairs)}')
dl = dm.train_dataloader()
batch = next(iter(dl))
v1, v2, labels = batch
print(f'view_1: {v1.shape}, view_2: {v2.shape}, labels: {labels.shape}')
print('SUCCESS')
"
```

Expected: shapes `(4, 120000)` for views, `(4,)` for labels.

- [ ] **Step 3: Commit if any fixes needed**

---

### Task 8: Run full verify

- [ ] **Step 1: Run verify service**

```bash
docker compose up verify
```

- [ ] **Step 2: Monitor GPU utilization**

```bash
# In another terminal, during training:
watch -n 1 nvidia-smi
```

Expected: sustained GPU utilization (not 0% between spikes).

- [ ] **Step 3: Commit any fixes**
