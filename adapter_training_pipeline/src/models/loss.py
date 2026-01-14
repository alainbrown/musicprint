import torch
import torch.nn as nn
import torch.nn.functional as F

class SupervisedContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        """
        Args:
            features: [batch_size, dim] - normalized embeddings
            labels: [batch_size] - ground truth song IDs
        """
        device = features.device
        batch_size = features.shape[0]
        
        # 1. Normalize features robustly
        norm = torch.norm(features, p=2, dim=1, keepdim=True)
        features = features / (norm + 1e-7)
        
        # 2. Similarity matrix
        logits = torch.div(torch.matmul(features, features.T), self.temperature)
        
        # 3. Mask for self-similarity
        mask = torch.eye(batch_size, device=device).bool()
        logits.masked_fill_(mask, -float('inf'))
        
        # 4. Target matrix (where labels match)
        labels = labels.contiguous().view(-1, 1)
        target_mask = torch.eq(labels, labels.T).float().to(device)
        # Remove self-matching from target mask
        target_mask = target_mask * (~mask).float()
        
        # 5. Compute Log-Softmax
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach()
        
        exp_logits = torch.exp(logits)
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        
        # 6. Mean log-likelihood over positives
        # We handle cases where a sample might have no positives in the batch
        mean_log_prob_pos = (target_mask * log_prob).sum(1) / torch.clamp(target_mask.sum(1), min=1.0)
        
        loss = -mean_log_prob_pos.mean()
        return loss
