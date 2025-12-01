"""
SSIM Computation Utilities

Adapted from NeRF-on-the-Go's SSIM implementation for uncertainty loss.
Matches the compute_ssim function from train_utils.py lines 102-135.
"""

import torch
import torch.nn.functional as F


def compute_ssim_components(img1, img2, window_size=5):
    """
    Compute SSIM components (luminance, contrast, structure) per pixel.
    
    Adapted from NeRF-on-the-Go's compute_ssim (train_utils.py lines 102-135).
    Returns per-pixel SSIM components instead of a single scalar.
    
    Args:
        img1: (B, R, 3) or (B, H, W, 3) predicted RGB
        img2: (B, R, 3) or (B, H, W, 3) ground truth RGB
        window_size: Size of the SSIM window (default 5, matches NeRF-OTG)
    
    Returns:
        l: (B, R, 1) or (B, H, W, 1) luminance component
        c: (B, R, 1) or (B, H, W, 1) contrast component  
        s: (B, R, 1) or (B, H, W, 1) structure component
    """
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    C3 = C2 / 2
    
    # For ray-based inputs (B, R, 3), we compute SSIM per-ray using a simple approach
    # NeRF-on-the-Go uses spatial convolution, but for rays we use per-ray statistics
    if img1.dim() == 3:  # (B, R, 3) - ray-based
        # Compute per-ray statistics (mean, variance, covariance)
        mu1 = img1.mean(dim=-1, keepdim=True)  # (B, R, 1)
        mu2 = img2.mean(dim=-1, keepdim=True)  # (B, R, 1)
        
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = ((img1 - mu1) ** 2).mean(dim=-1, keepdim=True)  # (B, R, 1)
        sigma2_sq = ((img2 - mu2) ** 2).mean(dim=-1, keepdim=True)  # (B, R, 1)
        sigma12 = ((img1 - mu1) * (img2 - mu2)).mean(dim=-1, keepdim=True)  # (B, R, 1)
        
        # Clip variances for numerical stability
        eps = torch.finfo(torch.float32).eps ** 2
        sigma1_sq = torch.clamp(sigma1_sq, min=eps)
        sigma2_sq = torch.clamp(sigma2_sq, min=eps)
        sigma12 = torch.sign(sigma12) * torch.minimum(
            torch.sqrt(sigma1_sq * sigma2_sq),
            torch.abs(sigma12)
        )
        
        # SSIM components
        l = (2 * mu1_mu2 + C1) / (mu1_sq + mu2_sq + C1)  # (B, R, 1)
        c = (2 * torch.sqrt(sigma1_sq) * torch.sqrt(sigma2_sq) + C2) / (sigma1_sq + sigma2_sq + C2)  # (B, R, 1)
        s = (sigma12 + C3) / (torch.sqrt(sigma1_sq) * torch.sqrt(sigma2_sq) + C3)  # (B, R, 1)
        
        # Clip as in NeRF-on-the-Go (line 132-133)
        c = torch.clamp(c, max=0.98)
        s = torch.clamp(s, max=0.98)
        
    else:  # (B, H, W, 3) - spatial image
        # Use spatial convolution (more accurate but slower)
        # This matches NeRF-on-the-Go's approach more closely
        # For now, we'll use the simpler per-ray approach above
        # Full spatial SSIM can be added later if needed
        raise NotImplementedError("Spatial SSIM not yet implemented for ray-based rendering")
    
    return l, c, s


def bias_function(x, s):
    """
    Bias function from NeRF-on-the-Go (line 170):
    bias = lambda x, s: x / (1 + (1 - x)*(1 / s - 2))
    
    Used for training progress-based annealing.
    
    Args:
        x: float or tensor, value in [0, 1] (training progress)
        s: float, annealing parameter (typically 0.8)
    
    Returns:
        Biased value
    """
    if isinstance(x, torch.Tensor):
        return x / (1 + (1 - x) * (1 / s - 2))
    else:
        return x / (1 + (1 - x) * (1 / s - 2))
