
import os
import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from dataset import AlbumArtDataset, collate_skip_none
from model import VQVAE

# Configuration
MODEL_OUTPUT = "release/visual_encoder.pth"

# Hyperparameters
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
NUM_HIDDENS = 128
NUM_RESIDUAL_LAYERS = 2
NUM_RESIDUAL_HIDDENS = 32
NUM_EMBEDDINGS = 1024
EMBEDDING_DIM = 64
COMMITMENT_COST = 0.25

def train(data_dir, manifest_path, epochs=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}...")

    # 1. Setup Data
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    dataset = AlbumArtDataset(manifest_path, data_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, 
                        num_workers=4, collate_fn=collate_skip_none)
    
    print(f"Dataset size: {len(dataset)} albums")

    # 2. Setup Model
    model = VQVAE(NUM_HIDDENS, NUM_RESIDUAL_LAYERS, NUM_RESIDUAL_HIDDENS,
                  NUM_EMBEDDINGS, EMBEDDING_DIM, COMMITMENT_COST).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, amsgrad=False)
    
    # 3. Training Loop
    model.train()
    
    for epoch in range(epochs):
        total_loss = 0
        steps = 0
        
        with tqdm(loader, unit="batch") as pbar:
            pbar.set_description(f"Epoch {epoch+1}/{epochs}")
            
            for batch in pbar:
                if batch is None: continue # Skip empty batches
                
                batch = batch.to(device)
                optimizer.zero_grad()
                
                # Forward Pass
                vq_loss, data_recon, perplexity = model(batch)
                recon_error = torch.mean((data_recon - batch)**2)
                loss = recon_error + vq_loss
                
                # Backward Pass
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                steps += 1
                
                pbar.set_postfix(loss=total_loss/steps, recon=recon_error.item(), perplx=perplexity.item())

        # Save Checkpoint every epoch
        torch.save(model.state_dict(), MODEL_OUTPUT)
    
    print(f"Training Complete. Model saved to {MODEL_OUTPUT}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="Path to covers folder")
    parser.add_argument("--manifest", type=str, required=True, help="Path to manifest CSV")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    
    args = parser.parse_args()
    train(args.data_dir, args.manifest, args.epochs)
