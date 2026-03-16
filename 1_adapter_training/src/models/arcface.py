import torch
import torch.nn as nn
from pytorch_metric_learning import losses, miners
from .mert_adapter import MERTAdapter

class MusicArcFaceSystem(nn.Module):
    def __init__(self, num_classes, embedding_dim=768, margin=0.5, scale=64):
        """
        ArcFace System for Music Fingerprinting.
        
        Args:
            num_classes (int): Total number of unique songs in the training set.
            embedding_dim (int): Output dimension of the adapter (default 64).
            margin (float): Angular margin in degrees (default 28.6 degrees / 0.5 rad).
            scale (float): Scaling factor s (default 64).
        """
        super().__init__()
        self.backbone = MERTAdapter(output_dim=embedding_dim)
        
        # ArcFace Loss expects a classifier layer (The "Metric Head")
        # We use the official library which handles the W matrix internally within the loss function
        # But wait, pytorch-metric-learning separates the "Loss" from the "Classifier Weights" usually?
        # Actually, ArcFaceLoss in this lib maintains the weights itself.
        
        self.loss_func = losses.ArcFaceLoss(
            num_classes=num_classes,
            embedding_size=embedding_dim,
            margin=margin,
            scale=scale
        )
        
    def forward(self, x):
        return self.backbone(x)
        
    def get_loss(self, embeddings, labels):
        # ArcFaceLoss.forward takes (embeddings, labels)
        return self.loss_func(embeddings, labels)
