import argparse
import os
import torch
import torch.optim as optim
from tqdm import tqdm
import time

from models.mert_adapter import MERTAdapter
from models.loss import SupervisedContrastiveLoss
from data.dali_loader import DALIGPULoader

def train(args):
    # 1. Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    # 2. Setup Data Loader (GPU-Direct)
    print(f"Initializing DALI Pipeline from {args.data_dir}...")
    # DALI handles the epoch logic internally if we set the size correctly
    # For now, we assume infinite streaming or fixed epoch size
    loader = DALIGPULoader(
        file_root=args.data_dir,
        batch_size=args.batch_size,
        device_id=0 if torch.cuda.is_available() else None
    )

    # 3. Setup Model
    print("Loading MERT Adapter...")
    model = MERTAdapter(output_dim=64).to(device)
    
    # Optimizer - Only optimize adapter parameters (backbone is frozen in model __init__)
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    
    # Loss Function
    criterion = SupervisedContrastiveLoss(temperature=0.07).to(device)

    # 4. Training Loop
    model.train()
    print("Starting Training...")
    
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    step = 0
    
    # DALI iterators are infinite by default unless we define size. 
    # We will loop based on a fixed number of steps per epoch for this prototype.
    steps_per_epoch = 100 # Adjust based on dataset size: num_files // batch_size
    
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        start_time = time.time()
        
        # progress bar
        pbar = tqdm(enumerate(loader), total=steps_per_epoch, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for i, batch in pbar:
            if i >= steps_per_epoch:
                break
                
            # DALI output: [{'audio': tensor, 'label': tensor}]
            audio = batch[0]["audio"].to(device) # (B, Time, 1)
            labels = batch[0]["label"].to(device).long().squeeze() # (B,)
            
            # Squeeze channel dim if present
            if audio.dim() == 3:
                audio = audio.squeeze(-1)

            # Zero Gradients
            optimizer.zero_grad()
            
            # Forward Pass
            # Result: (Batch, 64) normalized vectors
            embeddings = model(audio)
            
            # Compute Loss
            # SupCon loss pushes same-label embeddings close, diff-label far
            loss = criterion(embeddings, labels)
            
            # Backward Pass
            loss.backward()
            optimizer.step()
            
            # Logging
            loss_val = loss.item()
            epoch_loss += loss_val
            pbar.set_postfix({"loss": f"{loss_val:.4f}"})
            
            step += 1
            
        avg_loss = epoch_loss / steps_per_epoch
        print(f"Epoch {epoch+1} Complete. Avg Loss: {avg_loss:.4f}. Time: {time.time() - start_time:.2f}s")
        
        # Save Checkpoint
        save_path = os.path.join(args.checkpoint_dir, f"mert_adapter_ep{epoch+1}.pt")
        torch.save(model.state_dict(), save_path)
        print(f"Saved checkpoint to {save_path}")

    print("Training Complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MERT Adapter with SupCon Loss")
    parser.add_argument("--data_dir", type=str, default="/app/data", help="Path to audio files")
    parser.add_argument("--checkpoint_dir", type=str, default="/app/checkpoints", help="Save path")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    
    args = parser.parse_args()
    train(args)
