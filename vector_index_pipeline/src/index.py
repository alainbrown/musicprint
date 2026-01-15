import argparse
import pytorch_lightning as pl
from data.module import MusicDataModule
from writer import IndexWriter
import torch
import os

class TorchScriptWrapper(pl.LightningModule):
    """Wraps JIT model to satisfy Lightning's predict expectations"""
    def __init__(self, model_path):
        super().__init__()
        self.model = torch.jit.load(model_path)
        self.pq = None

    def forward(self, x):
        return self.model(x)

    def predict_step(self, batch, batch_idx):
        # Implementation using JIT model
        # Batch is (B, Time, 1)
        data_dict = batch[0]
        audio_batch = data_dict["audio"]
        labels = data_dict["label"]
        
        if audio_batch.dim() == 3:
            audio_batch = audio_batch.squeeze(-1)
            
        results = []
        for i in range(audio_batch.shape[0]):
            audio = audio_batch[i]
            song_id = labels[i].item()
            
            # Simple windowing (5s @ 24kHz)
            # DALI might pad, so we just take valid windows
            if audio.shape[0] < 120000: continue
            windows = audio.unfold(0, 120000, 24000) 
            
            with torch.no_grad():
                embeddings = self.model(windows)
            
            # Normalize
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1).cpu().numpy()
            
            results.append({
                "id": song_id,
                "embeddings": embeddings,
                "times": [j * 1.0 for j in range(len(embeddings))]
            })
        return results

def main(args):
    torch.set_float32_matmul_precision('medium')
    
    # 1. Init Data
    dm = MusicDataModule(data_dir=args.data_dir, batch_size=args.batch_size)
    
    # 2. Init JIT Model
    print(f"Loading TorchScript model from {args.model_path}...")
    model = TorchScriptWrapper(args.model_path)
    
    # 3. Init Writer
    writer = IndexWriter(output_dir=args.output_dir, write_interval="batch")
    
    # 4. Init Trainer
    trainer = pl.Trainer(
        accelerator=args.accelerator,
        devices=1,
        precision=args.precision,
        callbacks=[writer]
    )
    
    # 5. Run Inference
    trainer.predict(model, datamodule=dm)
    print("Indexing Complete.")

def index(args):
    main(args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to encoder.pt")
    parser.add_argument("--data_dir", type=str, default="/vol/data")
    parser.add_argument("--output_dir", type=str, default="/vol/cache/index")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--accelerator", type=str, default="gpu")
    parser.add_argument("--precision", type=str, default="bf16-mixed")
    
    args = parser.parse_args()
    index(args)
