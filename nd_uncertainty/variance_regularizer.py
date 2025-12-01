"""
Patch Variance Regularizer

Implements DINO patch similarity variance regularization from NeRF-on-the-Go.
Based on train_utils.py lines 81-85 and models.py lines 238-262.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchVarianceRegularizer(nn.Module):
    """
    Regularizes uncertainty using DINO patch similarity variance.
    
    From NeRF-on-the-Go (models.py lines 238-262):
    - Computes affinity matrix from normalized DINO patch features
    - Finds top-k similar patches (similarity > threshold)
    - Computes variance of uncertainty values among similar patches
    - Penalizes high variance (encourages similar patches to have similar uncertainty)
    
    This implements equations (2) and (3) from the NeRF-on-the-Go paper.
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
        Compute patch variance regularization loss.
        
        Args:
            patch_features: (B, R, C_patch) DINO patch embeddings
            beta: (B, R, 1) predicted uncertainty values
        
        Returns:
            loss: scalar variance regularization loss
        """
        B, R, C = patch_features.shape
        
        # Flatten to (B*R, C) for batch processing
        feat_flatten = patch_features.reshape(-1, C)  # (B*R, C)
        beta_flatten = beta.reshape(-1, 1)  # (B*R, 1)
        
        # Normalize features for cosine similarity
        # Matches NeRF-on-the-Go line 245
        feat_norm = F.normalize(feat_flatten, p=2, dim=-1)  # (B*R, C)
        
        # Compute affinity matrix (cosine similarity)
        # Matches NeRF-on-the-Go line 246
        affinity = torch.matmul(feat_norm, feat_norm.t())  # (B*R, B*R)
        
        # Find top-k similar patches for each patch
        # Matches NeRF-on-the-Go lines 249-250
        topk_values, topk_indices = torch.topk(affinity, k=min(self.top_k, B*R), dim=-1)  # (B*R, top_k)
        value_mask = (topk_values > self.similarity_threshold)  # (B*R, top_k)
        
        # Get uncertainty values for similar patches
        # Matches NeRF-on-the-Go line 254
        uncer_nn = beta_flatten[topk_indices] * value_mask.float().unsqueeze(-1)  # (B*R, top_k, 1)
        uncer_nn = uncer_nn.squeeze(-1)  # (B*R, top_k)
        
        # Compute mean uncertainty for each patch's neighbors
        # Matches NeRF-on-the-Go lines 255-257
        sums = (uncer_nn * value_mask.float()).sum(dim=-1)  # (B*R,)
        counts = value_mask.float().sum(dim=-1) + 1e-6  # (B*R,)
        uncer_means = sums / counts  # (B*R,)
        
        # Compute variance of uncertainty among similar patches
        # Matches NeRF-on-the-Go lines 258-259
        uncer_means_expanded = uncer_means.unsqueeze(-1)  # (B*R, 1)
        squared_diffs = (uncer_nn - uncer_means_expanded) ** 2  # (B*R, top_k)
        uncer_variances = (squared_diffs * value_mask.float()).sum(dim=-1) / counts  # (B*R,)
        
        # Average variance across all patches
        # Matches NeRF-on-the-Go line 84: jnp.mean(rendering['uncer_var'])
        variance_loss = uncer_variances.mean()
        
        # Apply weight
        return self.weight * variance_loss
