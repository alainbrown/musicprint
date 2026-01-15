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
        logits = torch.matmul(features, features.T) / self.temperature
        
        # 3. Mask for self-similarity
        mask = torch.eye(batch_size, device=device).bool()
        logits.masked_fill_(mask, -1e9) # Use large negative number instead of -inf to avoid NaN math
        
        # 4. Target matrix (where labels match)
        labels = labels.contiguous().view(-1, 1)
        target_mask = torch.eq(labels, labels.T).float().to(device)
        # Remove self-matching from target mask
        target_mask = target_mask * (~mask).float()
        
        # 5. Compute Log-Softmax
        # Subtract max for stability
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits_stable = logits - logits_max.detach()
        
        exp_logits = torch.exp(logits_stable)
        log_prob = logits_stable - torch.log(exp_logits.sum(1, keepdim=True) + 1e-7)
        
        # 6. Mean log-likelihood over positives
        # Sum of log-probs for positive pairs
        mean_log_prob_pos = (target_mask * log_prob).sum(1)
        
        # Count of positive pairs per anchor
        num_positives = target_mask.sum(1)
        
        # Handle cases with no positives:
        # If num_positives is 0, we want loss to be 0 (masked out)
        # We divide by (num_positives + epsilon) to avoid div-by-zero
        mean_log_prob_pos = mean_log_prob_pos / (num_positives + 1e-7)
        
        # Only average over anchors that HAVE positives
        # Create a mask for anchors with >0 positives
        valid_anchors = (num_positives > 0).float()
        
        if valid_anchors.sum() == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)
            
        loss = - (mean_log_prob_pos * valid_anchors).sum() / valid_anchors.sum()
        
        if torch.isnan(loss):
            print("NaN Loss Detected!")
            print(f"Logits range: {logits.min()} - {logits.max()}")
            print(f"Features range: {features.min()} - {features.max()}")
            print(f"Num positives: {num_positives}")
            
        return loss
