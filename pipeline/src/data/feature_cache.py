import numpy as np
import os
import torch

class FeatureCache:
    def __init__(self, cache_dir="/app/cache/features"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    def get_path(self, song_id):
        return os.path.join(self.cache_dir, f"{song_id}.npy")

    def exists(self, song_id):
        return os.path.exists(self.get_path(song_id))

    def save(self, song_id, feature_tensor):
        """
        Saves a MERT embedding (768d) to disk.
        """
        path = self.get_path(song_id)
        # Convert to numpy and save
        np.save(path, feature_tensor.cpu().detach().numpy())

    def load(self, song_id):
        path = self.get_path(song_id)
        data = np.load(path)
        return torch.from_numpy(data)

    def batch_load(self, song_ids):
        features = [self.load(sid) for sid in song_ids]
        return torch.stack(features)
