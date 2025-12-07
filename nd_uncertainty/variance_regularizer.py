"""
Patch Variance Regularizer

Implements spatial-temporal consistency regularization for uncertainty predictions.
Based on the paper's equations (2) and (3).

Paper formula:
- Neighbor set: N(r) = {r' | cos(f, f') > η}
- Average uncertainty: β̄(r) = (1/|N(r)|) * Σ_{r'∈N(r)} β(r')
- Regularization: L_reg(r) = (1/|N(r)|) * Σ_{r'∈N(r)} (β̄(r) - β(r'))²
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchVarianceRegularizer(nn.Module):
    """
    Regularizes uncertainty using spatial-temporal consistency.
    
    Implements the paper's variance regularization (Eq. 2 & 3):
    1. For each ray r, find neighbors N(r) with cosine similarity > η
    2. Compute average uncertainty: β̄(r) = (1/|N(r)|) * Σ β(r')
    3. Penalize variance: L_reg(r) = (1/|N(r)|) * Σ (β̄(r) - β(r'))²
    
    This encourages similar feature vectors to have similar uncertainty predictions.
    """
    
    def __init__(self, top_k=128, similarity_threshold=0.75, weight=0.1):
        """
        Args:
            top_k: Number of top similar patches to consider (default 128, matches NeRF-OTG)
            similarity_threshold: Minimum similarity to include (default 0.75, matches NeRF-OTG)
            weight: Weight for variance regularization term (default 0.1)
        """
        super().__init__()
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold
        self.weight = weight
    
    def forward(self, patch_features, beta):
        """
        Compute variance regularization loss per paper's Eq. 2 & 3.
        
        For each ray r:
        1. Find neighbors: N(r) = {r' | cos(f, f') > η}
        2. Compute average: β̄(r) = (1/|N(r)|) * Σ_{r'∈N(r)} β(r')
        3. Compute variance: L_reg(r) = (1/|N(r)|) * Σ_{r'∈N(r)} (β̄(r) - β(r'))²
        
        Args:
            patch_features: (B, R, C_patch) DINO patch feature vectors f
            beta: (B, R, 1) predicted uncertainty values β(r)
        
        Returns:
            loss: scalar variance regularization loss (averaged over all rays)
        """
        B, R, C = patch_features.shape
        
        # Flatten to (B*R, C) to treat all rays in batch together
        feat_flatten = patch_features.reshape(-1, C)  # (B*R, C) - feature vectors f
        beta_flatten = beta.reshape(-1, 1)  # (B*R, 1) - uncertainty values β(r)
        
        # Normalize features for cosine similarity computation
        feat_norm = F.normalize(feat_flatten, p=2, dim=-1)  # (B*R, C)
        
        # Compute cosine similarity matrix: cos(f, f')
        affinity = torch.matmul(feat_norm, feat_norm.t())  # (B*R, B*R)
        
        # Find top-k similar rays for each ray r
        # This finds candidates for N(r) = {r' | cos(f, f') > η}
        topk_values, topk_indices = torch.topk(affinity, k=min(self.top_k, B*R), dim=-1)  # (B*R, top_k)
        value_mask = (topk_values > self.similarity_threshold)  # (B*R, top_k) - binary mask for N(r)
        
        # Get uncertainty values for neighbors: β(r') for r' ∈ N(r)
        uncer_nn = beta_flatten[topk_indices] * value_mask.float().unsqueeze(-1)  # (B*R, top_k, 1)
        uncer_nn = uncer_nn.squeeze(-1)  # (B*R, top_k)
        
        # Paper Eq. 2: Compute average uncertainty for each ray's neighbors
        # β̄(r) = (1/|N(r)|) * Σ_{r'∈N(r)} β(r')
        sums = (uncer_nn * value_mask.float()).sum(dim=-1)  # (B*R,) - sum of β(r') for r' ∈ N(r)
        counts = value_mask.float().sum(dim=-1) + 1e-6  # (B*R,) - |N(r)|, add epsilon to avoid division by zero
        uncer_means = sums / counts  # (B*R,) - β̄(r) for each ray r
        
        # Paper Eq. 3: Compute variance regularization for each ray
        # L_reg(r) = (1/|N(r)|) * Σ_{r'∈N(r)} (β̄(r) - β(r'))²
        uncer_means_expanded = uncer_means.unsqueeze(-1)  # (B*R, 1) - β̄(r) expanded to match shape
        squared_diffs = (uncer_nn - uncer_means_expanded) ** 2  # (B*R, top_k) - (β̄(r) - β(r'))²
        uncer_variances = (squared_diffs * value_mask.float()).sum(dim=-1) / counts  # (B*R,) - L_reg(r)
        
        # Average L_reg(r) across all rays to get final loss
        variance_loss = uncer_variances.mean()  # scalar
        
        # Apply weight
        return self.weight * variance_loss
