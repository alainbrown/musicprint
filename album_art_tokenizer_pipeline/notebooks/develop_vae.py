
import os
import sys
import torch
import torch.optim as optim
import torchvision.transforms as transforms
from PIL import Image
import urllib.request

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))
from model import VQVAE

# Configuration
BATCH_SIZE = 32
NUM_TRAINING_UPDATES = 100
NUM_HIDDENS = 128
NUM_RESIDUAL_HIDDENS = 32
NUM_RESIDUAL_LAYERS = 2
EMBEDDING_DIM = 64
NUM_EMBEDDINGS = 512 # Codebook size
COMMITMENT_COST = 0.25
LEARNING_RATE = 1e-3

def download_sample_images(data_dir):
    """Download a few sample album covers for testing."""
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    
    samples = [
        ("https://archive.org/download/mbid-e014041d-3843-4e8c-859a-f33c373a0c5c/mbid-e014041d-3843-4e8c-859a-f33c373a0c5c-29672658197.jpg", "dark_side.jpg"),
        ("https://archive.org/download/mbid-5eecaf18-02ec-4708-a423-78f9f05b199c/mbid-5eecaf18-02ec-4708-a423-78f9f05b199c-32360293144.jpg", "abbey_road.jpg")
    ]
    
    local_paths = []
    for url, filename in samples:
        path = os.path.join(data_dir, filename)
        if not os.path.exists(path):
            print(f"Downloading {filename}...")
            try:
                urllib.request.urlretrieve(url, path)
            except Exception as e:
                print(f"Failed to download {filename}: {e}")
                continue
        local_paths.append(path)
    return local_paths

def main():
    print("--- Developing VQ-VAE Tokenizer ---")
    
    # 1. Setup Data
    data_dir = os.path.join(os.path.dirname(__file__), '../data/sample_art')
    image_paths = download_sample_images(data_dir)
    
    if not image_paths:
        print("No images found. Creating random noise for testing.")
        # Create dummy image
        dummy = Image.fromarray(torch.randint(0, 255, (128, 128, 3), dtype=torch.uint8).numpy())
        dummy_path = os.path.join(data_dir, "random.jpg")
        dummy.save(dummy_path)
        image_paths = [dummy_path]

    # Preprocessing
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Load images into a batch
    batch_tensors = []
    for p in image_paths:
        img = Image.open(p).convert('RGB')
        batch_tensors.append(transform(img))
    
    # Stack and repeat to simulate a batch
    training_data = torch.stack(batch_tensors)
    if training_data.shape[0] < BATCH_SIZE:
        repeat_factor = (BATCH_SIZE // training_data.shape[0]) + 1
        training_data = training_data.repeat(repeat_factor, 1, 1, 1)[:BATCH_SIZE]
    
    print(f"Input Batch Shape: {training_data.shape}")

    # 2. Initialize Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VQVAE(NUM_HIDDENS, NUM_RESIDUAL_LAYERS, NUM_RESIDUAL_HIDDENS,
                  NUM_EMBEDDINGS, EMBEDDING_DIM, COMMITMENT_COST).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, amsgrad=False)
    training_data = training_data.to(device)
    
    print(f"Model initialized on {device}")
    print(f"Compression: 128x128 pixels -> {128//8}x{128//8} tokens (factor 8x)")

    # 3. Training Loop
    model.train()
    print("\nStarting Training...")
    
    for i in range(NUM_TRAINING_UPDATES):
        optimizer.zero_grad()
        
        vq_loss, data_recon, perplexity = model(training_data)
        recon_error = torch.mean((data_recon - training_data)**2)
        loss = recon_error + vq_loss
        
        loss.backward()
        optimizer.step()
        
        if (i+1) % 10 == 0:
            print(f'Step {i+1}: Loss: {loss.item():.4f} | Recon Error: {recon_error.item():.4f} | Perplexity: {perplexity.item():.4f}')

    # 4. Visualization / Validation
    model.eval()
    with torch.no_grad():
        _, reconstructed, _, indices = model(training_data[:1])
        
        # Verify output shape
        print("\nValidation:")
        print(f"Original Shape: {training_data[:1].shape}")
        print(f"Reconstructed Shape: {reconstructed.shape}")
        print(f"Token Grid Shape: {indices.shape}")
        
        # Save reconstruction
        # Un-normalize
        reconstructed = reconstructed.cpu() * 0.5 + 0.5
        save_path = os.path.join(data_dir, "reconstruction_test.jpg")
        transforms.ToPILImage()(reconstructed[0]).save(save_path)
        print(f"Saved reconstruction to: {save_path}")

if __name__ == "__main__":
    main()
