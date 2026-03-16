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
