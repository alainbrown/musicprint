import torch
import torch.nn as nn
from transformers import AutoModel

class MERTAdapter(nn.Module):
    def __init__(self, model_name="m-a-p/MERT-v1-95M", output_dim=64):
        super().__init__()
        # Load backbone (weights will be cached in /app/cache/huggingface via env var)
        self.backbone = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        
        # Freeze backbone
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        # MERT-v1-95M hidden size is 768
        self.adapter = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim),
            nn.Tanh()
        )
        
    def forward(self, audio_tensors):
        """
        Input: (Batch, Time) audio tensor resampled to 24kHz
        """
        # MERT expects inputs in a specific dict format
        # In a real training loop, we'd handle padding/masks here
        outputs = self.backbone(audio_tensors)
        
        # Mean pooling over the sequence dimension
        hidden_states = outputs.last_hidden_state # (Batch, Seq, 768)
        pooled_output = torch.mean(hidden_states, dim=1)
        
        return self.adapter(pooled_output)

    def get_hash(self, audio_tensors):
        with torch.no_grad():
            continuous = self.forward(audio_tensors)
            return torch.sign(continuous)
