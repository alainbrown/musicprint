import torch
import torch.nn as nn
import torch.nn.functional as F
from .mert_adapter import MERTAdapter


class MusicEmbeddingSystem(nn.Module):
    def __init__(self, embedding_dim=768):
        super().__init__()
        self.backbone = MERTAdapter(output_dim=embedding_dim)

    def forward(self, x):
        return self.backbone(x)


def contrastive_loss(embeddings, labels, margin=1.0):
    """Simple pairwise contrastive loss.

    For all pairs in the batch:
    - Same label: minimize distance
    - Different label: push apart up to margin

    Args:
        embeddings: (N, D) L2-normalized embeddings
        labels: (N,) integer song labels
        margin: minimum distance for different-song pairs
    """
    # L2 normalize
    embeddings = F.normalize(embeddings, dim=1)

    # Pairwise cosine similarity
    sim = embeddings @ embeddings.T  # (N, N)

    # Same-song mask
    labels = labels.unsqueeze(0)
    same = (labels == labels.T).float()

    # Loss: same-song pairs should have high similarity (target 1.0)
    #        diff-song pairs should have low similarity (target <= -margin)
    pos_loss = same * (1 - sim)  # pull together
    neg_loss = (1 - same) * F.relu(sim + margin)  # push apart

    # Average over all pairs (excluding diagonal for pos)
    n = embeddings.shape[0]
    mask = 1 - torch.eye(n, device=embeddings.device)
    loss = (pos_loss * mask + neg_loss * mask).sum() / mask.sum()

    return loss
