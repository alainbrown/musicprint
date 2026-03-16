import os
import glob
import random
import torch
from torch.utils.data import Dataset

SAMPLE_RATE = 24000
WINDOW_SAMPLES = 120000  # 5 seconds


class ContrastiveAudioDataset(Dataset):
    """Loads pre-sliced 5s windows and applies augmentation.

    Each file is a .pt tensor of shape (120000,) — already decoded,
    resampled, and cropped during preprocessing.

    For contrastive training, __getitem__ picks two random windows
    from the same song and augments each independently.
    """

    def __init__(self, song_windows, noise_dir=None):
        """
        Args:
            song_windows: list of (label, [path1, path2, ...]) per song
            noise_dir: directory of noise .pt files for augmentation
        """
        self.songs = song_windows  # [(label, [window_paths]), ...]
        self.noise_files = []
        if noise_dir and os.path.isdir(noise_dir):
            self.noise_files = sorted(glob.glob(os.path.join(noise_dir, "**", "*.pt"), recursive=True))

    def _augment(self, audio):
        if self.noise_files:
            try:
                noise = torch.load(random.choice(self.noise_files), weights_only=True)
                if noise.shape[0] >= audio.shape[0]:
                    start = random.randint(0, noise.shape[0] - audio.shape[0])
                    noise = noise[start : start + audio.shape[0]]
                else:
                    noise = torch.nn.functional.pad(noise, (0, audio.shape[0] - noise.shape[0]))
                audio = audio + noise * random.uniform(0.0, 0.3)
            except Exception:
                pass

        audio = audio * random.uniform(0.5, 1.5)
        return audio

    def __getitem__(self, idx):
        label, window_paths = self.songs[idx]
        # Pick two random windows from this song
        p1, p2 = random.choices(window_paths, k=2)
        view_1 = self._augment(torch.load(p1, weights_only=True))
        view_2 = self._augment(torch.load(p2, weights_only=True))
        return view_1, view_2, torch.tensor(label, dtype=torch.long)

    def __len__(self):
        return len(self.songs)


class AudioDataset(Dataset):
    """Loads a single pre-sliced 5s window per song. For validation."""

    def __init__(self, song_windows):
        # Use first window of each song
        self.items = [(label, paths[0]) for label, paths in song_windows if paths]

    def __getitem__(self, idx):
        label, path = self.items[idx]
        audio = torch.load(path, weights_only=True)
        return audio, torch.tensor(label, dtype=torch.long)

    def __len__(self):
        return len(self.items)
