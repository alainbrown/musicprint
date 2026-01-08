
import os
import sys
import torch
import coremltools as ct
from model import VQVAE

# Configuration
MODEL_PATH = "release/visual_encoder.pth"
OUTPUT_MLMODEL = "release/MusicArtDecoder.mlpackage"

# Hyperparameters (Must match training)
NUM_HIDDENS = 128
NUM_RESIDUAL_LAYERS = 2
NUM_RESIDUAL_HIDDENS = 32
NUM_EMBEDDINGS = 1024
EMBEDDING_DIM = 64

def export_coreml():
    print(">>> Exporting VQ-VAE Decoder to CoreML...")
    
    # 1. Load Model
    device = torch.device("cpu") # Export on CPU for stability
    model = VQVAE(NUM_HIDDENS, NUM_RESIDUAL_LAYERS, NUM_RESIDUAL_HIDDENS,
                  NUM_EMBEDDINGS, EMBEDDING_DIM)
    
    if os.path.exists(MODEL_PATH):
        print(f"Loading weights from {MODEL_PATH}...")
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    else:
        print("⚠️ WARNING: No trained model found. Exporting random weights.")
    
    model.eval()

    # 2. Isolate the Decoder
    # On the iPhone, we will:
    #   A. Read 256 tokens from art.bin
    #   B. Look up 256 vectors (64-dim) from the codebook
    #   C. Reshape to (1, 64, 16, 16)
    #   D. Feed to this CoreML model
    decoder = model._decoder
    
    # 3. Create Dummy Input (1, 64, 16, 16)
    # 64 = EMBEDDING_DIM
    # 16x16 = Token Grid
    example_input = torch.randn(1, EMBEDDING_DIM, 16, 16)
    
    # 4. Trace the Model
    traced_model = torch.jit.trace(decoder, example_input)
    
    # 5. Convert to CoreML
    mlmodel = ct.convert(
        traced_model,
        inputs=[ct.TensorType(name="quantized_input", shape=example_input.shape)],
        outputs=[ct.TensorType(name="reconstructed_image")],
        minimum_deployment_target=ct.target.iOS16, # Optimized for ANE
        convert_to="mlprogram"
    )
    
    # 6. Save
    mlmodel.save(OUTPUT_MLMODEL)
    print(f"✅ SUCCESS: CoreML model saved to {OUTPUT_MLMODEL}")
    
    # 7. Export Codebook (Vocabulary) for C++ Lookup
    # This is just a flat array of floats: [1024 indices][64 dims]
    codebook = model._vq_vae._embedding.weight.data.numpy()
    codebook_path = "release/visual_codebook.bin"
    with open(codebook_path, "wb") as f:
        f.write(codebook.astype('float32').tobytes())
    print(f"✅ SUCCESS: Codebook binary saved to {codebook_path}")

if __name__ == "__main__":
    export_coreml()
