import torch
import torch.nn as nn
from transformers import AutoModel

class MERTAdapter(nn.Module):
    def __init__(self, model_name="m-a-p/MERT-v1-95M", output_dim=768):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name, trust_remote_code=True)

        # Freeze backbone
        for param in self.backbone.parameters():
            param.requires_grad = False

        # MERT-v1-95M hidden size is 768
        self.adapter = nn.Sequential(
            nn.Linear(768, 768),
            nn.ReLU(),
            nn.Linear(768, output_dim),
        )

    def forward(self, audio_tensors):
        """
        Input: (Batch, Time) audio tensor resampled to 24kHz
        """
        with torch.no_grad():
            mean = audio_tensors.mean(dim=-1, keepdim=True)
            std = audio_tensors.std(dim=-1, keepdim=True)
            audio_tensors = (audio_tensors - mean) / (std + 1e-7)

        outputs = self.backbone(audio_tensors)

        # Mean pooling over the sequence dimension
        hidden_states = outputs.last_hidden_state  # (Batch, Seq, 768)
        pooled_output = torch.mean(hidden_states, dim=1)

        return self.adapter(pooled_output)
