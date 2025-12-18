"""
Uncertainty Loss Implementation

Implements heteroscedastic uncertainty for color loss per ND-SDF principles:
- Heteroscedastic color loss: L_color(r) = (1/(2σ²)) * ||C - Ĉ||² + (1/2) * log(σ²)
- Uncertainty regularizer: R(σ) = (1/N) * Σ_r (log σ_c(r) - log σ_0)²

Key principle: Uncertainty ONLY for color/photometric loss, NOT for SDF/eikonal/normal.
- SDF, eikonal, normal losses remain deterministic (no uncertainty)
- Only color loss uses heteroscedastic uncertainty weighting
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


def heteroscedastic_color_loss(rgb_pred, rgb_gt, sigma, mask=None):
    """
    Heteroscedastic uncertainty-weighted color loss per ND-SDF principles.
    
    Formula: L_color(r) = (1 / (2 * σ_c(r)^2)) * ||C(r) - Ĉ(r)||^2 + (1/2) * log(σ_c(r)^2)
    
    This replaces the standard RGB L1 loss when uncertainty is enabled.
    Uncertainty σ should be proportional to color prediction errors.
    
    Args:
        rgb_pred: (B, R, 3) predicted RGB from ND-SDF
        rgb_gt:   (B, R, 3) ground truth RGB
        sigma:    (B, R, 1) predicted uncertainty σ(r) (from uncertainty MLP)
        mask:     (B, R, 1) optional mask to apply
    
    Returns:
        loss: scalar heteroscedastic color loss
    """
    # Ensure sigma is positive and clamped to reasonable range for numerical stability
    # Default clamping: [1e-6, inf] - but should be clamped to [sigma_min, sigma_max] by caller
    # This is a safety check in case caller forgets to clamp
    sigma = sigma.clamp(min=1e-6)
    
    # Compute squared error: ||C(r) - Ĉ(r)||^2
    # Sum over RGB channels: (B, R, 3) -> (B, R, 1)
    residual_sq = (rgb_pred - rgb_gt).pow(2).sum(dim=-1, keepdim=True)  # (B, R, 1)
    
    # First term: (1 / (2 * σ²)) * ||C - Ĉ||²
    # This down-weights errors when uncertainty is high
    weighted_term = 0.5 * residual_sq / (sigma ** 2)  # (B, R, 1)
    
    # Second term: (1/2) * log(σ²) = log(σ)
    # This prevents σ from collapsing to zero (regularization)
    log_term = 0.5 * torch.log(sigma ** 2)  # (B, R, 1)
    
    # Per-ray loss: L_color(r) = weighted_term + log_term
    loss_per_ray = weighted_term + log_term  # (B, R, 1)
    
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
    
    # CRITICAL FIX: Prevent negative loss to avoid numerical instability
    # When sigma saturates at max clamp, log_term can dominate and make loss negative
    # Negative loss → NaN → training collapse
    loss = torch.clamp(loss, min=0.0)
    
    return loss


def uncertainty_regularizer(sigma, sigma_0=0.1, mask=None):
    """
    Regularizer on uncertainty σ to avoid trivial inflation.
    
    Formula: R(σ) = (1/N) * Σ_r (log σ_c(r) - log σ_0)^2
    
    This encourages σ to stay close to baseline σ_0 in log-space.
    
    Args:
        sigma:    (B, R, 1) predicted uncertainty σ(r)
        sigma_0:  float, baseline uncertainty (default 0.1)
        mask:     (B, R, 1) optional mask to apply
    
    Returns:
        loss: scalar regularizer loss
    """
    # Ensure sigma is positive
    sigma = sigma.clamp(min=1e-6)
    
    # Compute log differences: (log σ - log σ_0) = log(σ / σ_0)
    log_sigma = torch.log(sigma)  # (B, R, 1)
    log_sigma_0 = torch.log(torch.tensor(sigma_0, device=sigma.device, dtype=sigma.dtype))
    log_diff = log_sigma - log_sigma_0  # (B, R, 1)
    
    # Square the log differences: (log σ - log σ_0)^2
    reg_per_ray = log_diff ** 2  # (B, R, 1)
    
    # Apply mask if provided
    if mask is not None:
        # mask: (B, R, 1) or (B, R)
        if mask.dim() == 2:
            mask = mask.unsqueeze(-1)  # (B, R) -> (B, R, 1)
        reg_per_ray = reg_per_ray * mask.float()
        # Average over masked rays: (1/N) * Σ_r
        if mask.float().sum() > 0:
            loss = reg_per_ray.sum() / mask.float().sum()
        else:
            loss = torch.tensor(0.0, device=reg_per_ray.device)
    else:
        # Average over all rays: (1/N) * Σ_r
        loss = reg_per_ray.mean()
    
    return loss
