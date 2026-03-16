from data.module import MusicDataModule
import torch
import os

def check_dali_range():
    print("Checking DALI Output Range...")
    data_dir = "/vol/data/test_samples_processed"
    if not os.path.exists(data_dir):
        print(f"Directory {data_dir} does not exist.")
        return

    dm = MusicDataModule(data_dir=data_dir, batch_size=4, val_split=0.0)
    loader = dm.train_dataloader()
    
    try:
        batch = next(iter(loader))
        audio = batch[0]["audio_1"]
        
        print(f"Min: {audio.min().item()}")
        print(f"Max: {audio.max().item()}")
        print(f"Mean: {audio.mean().item()}")
        print(f"Std: {audio.std().item()}")
        
        if audio.abs().max() > 1.5:
            print("🚨 ERROR: Audio is NOT normalized to [-1, 1]. This breaks MERT.")
        else:
            print("✅ Audio is normalized.")
            
    except Exception as e:
        print(f"Failed: {e}")

if __name__ == "__main__":
    check_dali_range()