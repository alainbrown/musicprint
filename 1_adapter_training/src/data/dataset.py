import torch
import torchaudio
from torch.utils.data import Dataset

SAMPLE_RATE = 24000


class SongDataset(Dataset):
    """Returns raw audio for a song. Windowing happens in training step."""

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


def collate_songs(batch):
    """Songs have different lengths, return as list."""
    return batch
