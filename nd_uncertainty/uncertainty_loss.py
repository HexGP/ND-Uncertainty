"""
Uncertainty Loss Implementation

Implements the uncertainty-weighted color loss from NeRF in the Wild,
with extensions from NeRF-on-the-Go (clipping, regularization, SSIM).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from nd_uncertainty.ssim_utils import compute_ssim_components, bias_function


class UncertaintyColorLoss(nn.Module):
    """
    Implements the uncertainty-weighted color loss from NeRF in the Wild,
    with optional SSIM-based uncertainty loss from NeRF-on-the-Go.
    
    Core formula:
        L = ||C - Ĉ||² / (2 β(r)²) + λ * log β(r)
    
    Optional SSIM extension (from NeRF-on-the-Go lines 162-173):
        L_ssim = (rate * (1-l)*(1-s)*(1-c)) / β² + λ * log β
    
    Where:
        C   = ground truth RGB
        Ĉ   = rendered RGB  
        β(r) = uncertainty predicted by the MLP
        l, c, s = SSIM components (luminance, contrast, structure)
    
    Based on NeRF-on-the-Go's implementation (lines 159-179 in train_utils.py):
    - Clips uncertainty for numerical stability
    - Uses squared L2 residual
    - Adds log regularization to prevent β → ∞
    - Optional SSIM-based uncertainty loss
    """
    
    def __init__(
        self,
        lambda_reg=0.5,
        uncer_clip_min=0.1,
        eps=1e-3,
        use_ssim=False,
        ssim_mult=0.5,
        ssim_anneal=0.8,
        ssim_clip_max=5.0,
        ssim_window_size=5,
        stop_ssim_gradient=False,
    ):
        """
        Args:
            lambda_reg: Weight for log regularization term (default 0.5, matches NeRF-OTG's reg_mult)
            uncer_clip_min: Minimum uncertainty value for clipping (default 0.1, matches NeRF-OTG)
            eps: Small epsilon added after clipping for stability (default 1e-3, matches NeRF-OTG)
            use_ssim: If True, add SSIM-based uncertainty loss (default False, optional extension)
            ssim_mult: Weight for SSIM loss term (default 0.5, matches NeRF-OTG config)
            ssim_anneal: Annealing parameter for SSIM (default 0.8, matches NeRF-OTG)
            ssim_clip_max: Maximum value to clip SSIM loss (default 5.0, matches NeRF-OTG)
            ssim_window_size: Window size for SSIM (default 5, matches NeRF-OTG)
            stop_ssim_gradient: If True, stop gradient from SSIM to RGB (default False)
        """
        super().__init__()
        self.lambda_reg = lambda_reg
        self.uncer_clip_min = uncer_clip_min
        self.eps = eps
        self.use_ssim = use_ssim
        self.ssim_mult = ssim_mult
        self.ssim_anneal = ssim_anneal
        self.ssim_clip_max = ssim_clip_max
        self.ssim_window_size = ssim_window_size
        self.stop_ssim_gradient = stop_ssim_gradient
    
    def forward(self, rgb_pred, rgb_gt, beta, mask=None, train_frac=None):
        """
        Compute uncertainty-weighted color loss with optional SSIM extension.
        
        Args:
            rgb_pred: (B, R, 3) predicted RGB from ND-SDF
            rgb_gt:   (B, R, 3) ground truth RGB
            beta:     (B, R, 1) predicted uncertainty β(r)
            mask:     (B, R, 1) optional mask to apply (e.g., foreground mask)
            train_frac: float in [0, 1], training progress (for SSIM annealing)
        
        Returns:
            loss: scalar uncertainty-weighted color loss
        """
        # Clip uncertainty for numerical stability (matches NeRF-on-the-Go line 161)
        beta = beta.clamp(min=self.uncer_clip_min) + self.eps
        
        # Squared L2 residual: ||C - Ĉ||²
        # Sum over RGB channels to get per-ray error
        residual_sq = (rgb_pred - rgb_gt).pow(2).sum(dim=-1, keepdim=True)  # (B, R, 1)
        
        # Uncertainty-weighted color loss: 0.5 * ||C - Ĉ||² / β²
        # Matches NeRF-on-the-Go line 178: data_loss = 0.5 * resid_sq / (uncer) ** 2
        weighted_term = 0.5 * residual_sq / (beta ** 2)  # (B, R, 1)
        
        # Regularizer to prevent β → ∞ collapse: λ * log β
        # Matches NeRF-on-the-Go line 173: reg_mult * jnp.log(uncer)
        reg_term = self.lambda_reg * torch.log(beta)  # (B, R, 1)
        
        # Combine base terms
        loss_per_ray = weighted_term + reg_term  # (B, R, 1)
        
        # Optional SSIM-based uncertainty loss (from NeRF-on-the-Go lines 162-173)
        if self.use_ssim:
            # Compute SSIM components
            if self.stop_ssim_gradient:
                rgb_pred_ssim = rgb_pred.detach()
            else:
                rgb_pred_ssim = rgb_pred
            
            l, c, s = compute_ssim_components(rgb_pred_ssim, rgb_gt, self.ssim_window_size)
            
            # Calculate SSIM loss rate (starts at 100, scales up to 1000)
            # Matches NeRF-on-the-Go lines 170-171
            if train_frac is not None:
                train_frac_tensor = torch.full_like(beta, train_frac)  # (B, R, 1)
                rate = 100 + bias_function(train_frac_tensor, self.ssim_anneal) * 900  # (B, R, 1)
            else:
                rate = torch.full_like(beta, 100.0)  # Default rate
            
            # SSIM loss: rate * (1-l)*(1-s)*(1-c)
            # Matches NeRF-on-the-Go line 172
            my_ssim_loss = rate * (1 - l) * (1 - s) * (1 - c)  # (B, R, 1)
            my_ssim_loss = torch.clamp(my_ssim_loss, max=self.ssim_clip_max)
            
            # SSIM uncertainty loss: my_ssim_loss / β² + λ * log β
            # Matches NeRF-on-the-Go line 173
            ssim_loss = my_ssim_loss / (beta ** 2) + self.lambda_reg * torch.log(beta)  # (B, R, 1)
            
            # Add SSIM loss term
            # Matches NeRF-on-the-Go line 179: data_loss += config.ssim_mult * ssim_loss
            loss_per_ray = loss_per_ray + self.ssim_mult * ssim_loss
        
        # Apply mask if provided
        if mask is not None:
            # mask: (B, R, 1) or (B, R)
            if mask.dim() == 2:
                mask = mask.unsqueeze(-1)  # (B, R) -> (B, R, 1)
            loss_per_ray = loss_per_ray * mask.float()
            # Average over masked rays
            if mask.float().sum() > 0:
                loss = loss_per_ray.sum() / mask.float().sum()
            else:
                loss = torch.tensor(0.0, device=loss_per_ray.device)
        else:
            # Average over all rays
            loss = loss_per_ray.mean()
        
        return loss
