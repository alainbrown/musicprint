
import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class AlbumArtDataset(Dataset):
    def __init__(self, manifest_path, covers_dir, transform=None):
        self.covers_dir = covers_dir
        self.transform = transform
        
        # Load Manifest
        if os.path.exists(manifest_path):
            self.df = pd.read_csv(manifest_path)
            # Filter for valid UUIDs (simple check)
            self.df = self.df[self.df['release_uuid'].astype(str).str.len() > 10]
        else:
            print("Manifest not found, creating empty dataset.")
            self.df = pd.DataFrame(columns=['release_uuid'])

    def _get_path(self, uuid):
        # Sharded path: covers/aa/bb/uuid.jpg
        return os.path.join(self.covers_dir, uuid[:2], uuid[2:4], f"{uuid}.jpg")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if idx >= len(self.df): return None
        
        row = self.df.iloc[idx]
        uuid = row['release_uuid']
        path = self._get_path(uuid)
        
        # If image missing or corrupt, we must handle it
        # For training, we usually just skip or return a placeholder.
        # But PyTorch DataLoader doesn't like 'None'.
        # Strategy: Return a blank image (black) and a valid=False flag
        
        try:
            image = Image.open(path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image
        except:
            # If the specific image for this index is missing, 
            # we return a tensor of zeros. The trainer should filter this.
            # (In a real large-scale loop, we might use a custom collate_fn to drop these)
            return None 

def collate_skip_none(batch):
    """Custom collator to skip missing images in a batch"""
    batch = [x for x in batch if x is not None]
    if len(batch) == 0: return None
    from torch.utils.data.dataloader import default_collate
    return default_collate(batch)
