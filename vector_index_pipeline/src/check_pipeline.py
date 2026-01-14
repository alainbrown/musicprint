import torch
from models.mert_adapter import MERTAdapter
from data.dali_loader import DALIGPULoader
import os

def run_pipeline_check():
    print("Initializing GPU-Direct Pipeline...")
    
    # In a real scenario, this would point to your mounted audio volume
    data_path = "/app/data"
    
    if not os.path.exists(data_path) or not os.listdir(data_path):
        print(f"Warning: {data_path} is empty. Please mount audio files to run a full test.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Initialize Loader
    loader = DALIGPULoader(file_root=data_path, batch_size=4)
    
    # 2. Initialize Model
    model = MERTAdapter().to(device)
    model.eval()

    print("Processing first batch...")
    for i, batch in enumerate(loader):
        audio = batch[0]["audio"].to(device)
        # DALI output is (Batch, Time, 1) usually, MERT wants (Batch, Time)
        if audio.dim() == 3:
            audio = audio.squeeze(-1)
            
        with torch.no_grad():
            hashes = model.get_hash(audio)
            
        print(f"Batch {i} processed. Output shape: {hashes.shape}")
        print(f"Sample Hash (first 8 bits): {hashes[0, :8]}")
        break 

if __name__ == "__main__":
    run_pipeline_check()
