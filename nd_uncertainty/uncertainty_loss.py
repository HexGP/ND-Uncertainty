"""
Uncertainty Loss Implementation (Nerf-on-the-go Paper Approach)

Implements decoupled SSIM-based uncertainty loss from Nerf-on-the-go paper:
- Eq. 8: LSSIM = (1-L)·(1-C)·(1-S)
- Eq. 9: Luncer(r) = LSSIM / (2β(r)²) + λ1 log β(r)

Key principle: Uncertainty MLP training is DECOUPLED from ND-SDF training.
- Uncertainty loss trains ONLY the uncertainty MLP
- ND-SDF trains with standard RGB loss (separate)
- Gradients are stopped to prevent coupling
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from nd_uncertainty.ssim_utils import compute_ssim_components


class UncertaintyColorLoss(nn.Module):
    """
    Implements SSIM-based uncertainty loss from ND-SDF paper (Eq. 9).
    
    Paper approach:
    - Uses SSIM components to detect distractors (Eq. 8)
    - Trains uncertainty MLP separately from ND-SDF (decoupled)
    - Stops gradients from SSIM to RGB to prevent coupling
    
    Formula:
        Luncer(r) = LSSIM / (2β(r)²) + λ1 log β(r)
    where:
        LSSIM = (1-L)·(1-C)·(1-S)  (Eq. 8)
        L, C, S = SSIM components (luminance, contrast, structure)
        β(r) = uncertainty predicted by MLP
    """
    
    def __init__(
        self,
        lambda_reg=0.5,
        uncer_clip_min=0.1,
        eps=1e-3,
        use_ssim=True,  # Default to True per paper
        ssim_window_size=5,
        stop_ssim_gradient=True,  # Default to True for decoupling
        ssim_anneal=0.8,  # Annealing parameter for SSIM rate scaling (default 0.8)
        ssim_clip_max=5.0,  # Maximum value to clip scaled SSIM loss (default 5.0)
    ):
        """
        Args:
            lambda_reg: Weight for log regularization term (λ1 in paper, default 0.5)
            uncer_clip_min: Minimum uncertainty value for clipping (default 0.1)
            eps: Small epsilon added after clipping for stability (default 1e-3)
            use_ssim: If True, use SSIM-based uncertainty loss (default True, per paper)
            ssim_window_size: Window size for SSIM computation (default 5)
            stop_ssim_gradient: If True, stop gradient from SSIM to RGB (default True, required for decoupling)
            ssim_anneal: Annealing parameter for SSIM rate scaling (default 0.8, matches NeRF-on-the-Go)
            ssim_clip_max: Maximum value to clip scaled SSIM loss (default 5.0, matches NeRF-on-the-Go)
        """
        super().__init__()
        self.lambda_reg = lambda_reg
        self.uncer_clip_min = uncer_clip_min
        self.eps = eps
        self.use_ssim = use_ssim
        self.ssim_window_size = ssim_window_size
        self.stop_ssim_gradient = stop_ssim_gradient
        self.ssim_anneal = ssim_anneal
        self.ssim_clip_max = ssim_clip_max
    
    def forward(self, rgb_pred, rgb_gt, beta, mask=None, train_frac=None):
        """
        Compute SSIM-based uncertainty loss (ND-SDF paper Eq. 9).
        
        This loss trains ONLY the uncertainty MLP, not the ND-SDF model.
        Gradients are stopped from flowing back to RGB predictions to ensure decoupling.
        
        Args:
            rgb_pred: (B, R, 3) predicted RGB from ND-SDF
            rgb_gt:   (B, R, 3) ground truth RGB
            beta:     (B, R, 1) predicted uncertainty β(r)
            mask:     (B, R, 1) optional mask to apply (e.g., foreground mask)
            train_frac: float in [0, 1], training progress (unused, kept for compatibility)
        
        Returns:
            loss: scalar SSIM-based uncertainty loss
            l_ssim_mean: scalar mean LSSIM value (for logging)
        """
        # Clip uncertainty for numerical stability
        beta = beta.clamp(min=self.uncer_clip_min) + self.eps
        
        if self.use_ssim:
            # Nerf-on-the-go Paper Approach: SSIM-based uncertainty loss (Eq. 8 & 9)
            # Stop gradient from SSIM to RGB to decouple uncertainty training from ND-SDF
            # This ensures uncertainty MLP training doesn't affect ND-SDF color rendering
            if self.stop_ssim_gradient:
                rgb_pred_ssim = rgb_pred.detach()  # Stop gradients to RGB
            else:
                rgb_pred_ssim = rgb_pred
            
            # Compute SSIM components: luminance (l), contrast (c), structure (s)
            # These detect structural differences (distractors) better than L2 error
            l, c, s = compute_ssim_components(rgb_pred_ssim, rgb_gt, self.ssim_window_size)
            
            # Paper Eq. 8: LSSIM = (1-L)·(1-C)·(1-S)
            # This emphasizes differences between dynamic and static elements
            # Higher values indicate more structural differences (distractors)
            l_ssim = (1 - l) * (1 - c) * (1 - s)  # (B, R, 1)
            
            # CRITICAL: NeRF-on-the-Go scales SSIM loss by 100-1000x (not in paper, but in code)
            # This prevents β from collapsing to minimum by providing stronger gradient signal
            # Without this scaling, LSSIM is too small (~0.001-0.01), making gradients too weak
            if train_frac is not None:
                # Compute annealing bias function (matches NeRF-on-the-Go train_utils.py line 170)
                # bias(x, s) = x / (1 + (1 - x)*(1 / s - 2))
                train_frac_tensor = torch.tensor(train_frac, device=l_ssim.device, dtype=l_ssim.dtype)
                bias = train_frac_tensor / (1 + (1 - train_frac_tensor) * (1 / self.ssim_anneal - 2))
                # Rate scales from 100 to 1000 based on training progress
                rate = 100 + bias * 900
            else:
                # Default to maximum rate if train_frac not provided
                rate = torch.tensor(1000.0, device=l_ssim.device, dtype=l_ssim.dtype)
            
            # Scale LSSIM by rate and clip to maximum (matches NeRF-on-the-Go train_utils.py line 172)
            my_ssim_loss = torch.clamp(rate * l_ssim, max=self.ssim_clip_max)  # (B, R, 1)
            
            # Paper Eq. 9: Luncer = LSSIM_scaled / (2β²) + λ log β
            # This is the SSIM-based uncertainty loss for training uncertainty MLP
            # The uncertainty β should be proportional to LSSIM (structural differences)
            ssim_weighted = 0.5 * my_ssim_loss / (beta ** 2)  # (B, R, 1)
            ssim_reg = self.lambda_reg * torch.log(beta)  # (B, R, 1)
            loss_per_ray = ssim_weighted + ssim_reg  # (B, R, 1)
            
        else:
            # Legacy: L2-based uncertainty loss (not recommended per paper)
            # Only use if SSIM is explicitly disabled
            residual_sq = (rgb_pred - rgb_gt).pow(2).sum(dim=-1, keepdim=True)  # (B, R, 1)
            weighted_term = 0.5 * residual_sq / (beta ** 2)  # (B, R, 1)
            reg_term = self.lambda_reg * torch.log(beta)  # (B, R, 1)
            loss_per_ray = weighted_term + reg_term  # (B, R, 1)
        
        # Compute mean LSSIM for logging (before masking)
        if self.use_ssim:
            l_ssim_mean = l_ssim.mean()
        else:
            l_ssim_mean = torch.tensor(0.0, device=loss_per_ray.device)
        
        # Apply mask if provided
        if mask is not None:
            # mask: (B, R, 1) or (B, R)
            if mask.dim() == 2:
                mask = mask.unsqueeze(-1)  # (B, R) -> (B, R, 1)
            loss_per_ray = loss_per_ray * mask.float()
            # Average over masked rays
            if mask.float().sum() > 0:
                loss = loss_per_ray.sum() / mask.float().sum()
                if self.use_ssim:
                    l_ssim_mean = (l_ssim * mask.float()).sum() / mask.float().sum()
                else:
                    l_ssim_mean = torch.tensor(0.0, device=loss_per_ray.device)
            else:
                loss = torch.tensor(0.0, device=loss_per_ray.device)
                l_ssim_mean = torch.tensor(0.0, device=loss_per_ray.device)
        else:
            # Average over all rays
            loss = loss_per_ray.mean()
            # l_ssim_mean already computed above
        
        return loss, l_ssim_mean
