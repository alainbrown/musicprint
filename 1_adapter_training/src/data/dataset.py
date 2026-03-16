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
        self.files = file_label_pairs
        self.noise_files = []
        if noise_dir and os.path.isdir(noise_dir):
            for ext in ("*.wav", "*.flac", "*.mp3"):
                self.noise_files.extend(glob.glob(os.path.join(noise_dir, "**", ext), recursive=True))

    def _load_audio(self, path):
        audio, sr = torchaudio.load(path)
        if audio.shape[0] > 1:
            audio = audio.mean(dim=0, keepdim=True)
        if sr != SAMPLE_RATE:
            audio = torchaudio.transforms.Resample(sr, SAMPLE_RATE)(audio)
        return audio.squeeze(0)

    def _random_crop(self, audio):
        if audio.shape[0] < WINDOW_SAMPLES:
            return torch.nn.functional.pad(audio, (0, WINDOW_SAMPLES - audio.shape[0]))
        start = random.randint(0, audio.shape[0] - WINDOW_SAMPLES)
        return audio[start : start + WINDOW_SAMPLES]

    def _augment(self, audio):
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
                pass

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
