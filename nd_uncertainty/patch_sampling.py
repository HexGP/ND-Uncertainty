"""
Dilated Patch Sampling

Implements dilated patch sampling from DINO feature maps, following
NeRF-on-the-Go's approach. Extracts a neighborhood of patches around
each ray's pixel location.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


def dilated_pixel_coordinates(
    width: int,
    height: int,
    dilated_factor: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate dilated pixel coordinates for patch sampling.
    
    Adapted from NeRF-on-the-Go's camera_utils.dilated_pixel_coordinates.
    
    Args:
        width: Patch width
        height: Patch height
        dilated_factor: Dilation factor (spacing between sampled pixels)
    
    Returns:
        x_coords: (width, height) x coordinates
        y_coords: (width, height) y coordinates
    """
    x_coords, y_coords = torch.meshgrid(
        torch.arange(0, width, dilated_factor, dtype=torch.long),
        torch.arange(0, height, dilated_factor, dtype=torch.long),
        indexing='xy'
    )
    return x_coords, y_coords


class DilatedPatchSampler(nn.Module):
    """
    Dilated Patch Sampling as in NeRF-on-the-Go.

    Responsibility:
      - From DINO feature maps (B, C, H_feat, W_feat)
      - And a set of ray / pixel indices
      - Extract a per-ray patch embedding (B, R, C_patch)

    The exact dilation / patch size mirrors NeRF-on-the-Go.
    """

    def __init__(self, patch_size: int = 7, dilation: int = 2):
        """
        Args:
            patch_size: Size of the patch neighborhood (e.g., 7x7).
            dilation: Dilation factor for sampling (spacing between patches).
        """
        super().__init__()
        self.patch_size = patch_size
        self.dilation = dilation

    def forward(
        self,
        feature_maps: torch.Tensor,
        sampling_idx: torch.Tensor,
        heights: torch.Tensor,
        widths: torch.Tensor,
    ) -> torch.Tensor:
        """
        Extract dilated patches from DINO feature maps for each ray.

        Args:
            feature_maps: (B, C, H_feat, W_feat) from DinoV2Encoder.
                         C is the DINO feature dimension (384, 768, or 1024).
            sampling_idx: (B, R) 1D indices into flattened image.
                         Can be converted to (y, x) via:
                         y = sampling_idx // w, x = sampling_idx % w
            heights: (B,) image heights in pixels (original image resolution).
            widths: (B,) image widths in pixels (original image resolution).

        Returns:
            patches: (B, R, C_patch)
                DINO patch features for each ray. This is the flattened
                neighborhood around the downsampled feature position.
                C_patch = C * patch_size * patch_size (flattened patch)
        """
        B, C, H_feat, W_feat = feature_maps.shape
        R = sampling_idx.shape[1]

        # Convert 1D sampling indices to (y, x) pixel coordinates
        # sampling_idx is in [0, h*w) range
        # We need to convert to feature map coordinates (H_feat, W_feat)
        device = feature_maps.device
        dtype = feature_maps.dtype

        # Get image dimensions
        # Handle both tensor and list/array inputs
        if isinstance(heights, torch.Tensor):
            h = heights[0].item() if heights.dim() > 0 else heights.item()
        else:
            h = heights[0] if hasattr(heights, '__getitem__') else heights
            
        if isinstance(widths, torch.Tensor):
            w = widths[0].item() if widths.dim() > 0 else widths.item()
        else:
            w = widths[0] if hasattr(widths, '__getitem__') else widths

        # Convert 1D indices to 2D pixel coordinates
        # sampling_idx is in [0, h*w) range
        w_tensor = torch.tensor(w, device=device, dtype=torch.float32)
        y_pix = (sampling_idx.float() // w_tensor)  # (B, R)
        x_pix = (sampling_idx.float() % w_tensor)   # (B, R)

        # Map pixel coordinates to feature map coordinates
        # DINO patch size is 14, so feature map is downsampled by 14
        y_feat = (y_pix / 14.0).clamp(0, H_feat - 1)  # (B, R)
        x_feat = (x_pix / 14.0).clamp(0, W_feat - 1)  # (B, R)

        # Generate dilated patch offsets
        # Following NeRF-on-the-Go: create a patch_size x patch_size grid
        # with dilation spacing
        patch_offsets_x, patch_offsets_y = dilated_pixel_coordinates(
            width=self.patch_size,
            height=self.patch_size,
            dilated_factor=self.dilation,
        )
        # patch_offsets_x, patch_offsets_y: (patch_size, patch_size)

        # Center the patch around the feature location
        patch_center = self.patch_size // 2
        patch_offsets_x = patch_offsets_x - patch_center  # (patch_size, patch_size)
        patch_offsets_y = patch_offsets_y - patch_center  # (patch_size, patch_size)

        # Flatten offsets for easier indexing
        patch_offsets_x = patch_offsets_x.flatten()  # (patch_size^2,)
        patch_offsets_y = patch_offsets_y.flatten()  # (patch_size^2,)

        # Vectorized patch sampling
        # Expand coordinates for all rays and all patch offsets
        # y_feat: (B, R) -> (B, R, 1) -> (B, R, patch_size^2)
        y_centers = y_feat.unsqueeze(-1)  # (B, R, 1)
        x_centers = x_feat.unsqueeze(-1)  # (B, R, 1)

        # Broadcast patch offsets: (patch_size^2,) -> (1, 1, patch_size^2)
        patch_offsets_y_expanded = patch_offsets_y.to(device).float().unsqueeze(0).unsqueeze(0)  # (1, 1, patch_size^2)
        patch_offsets_x_expanded = patch_offsets_x.to(device).float().unsqueeze(0).unsqueeze(0)  # (1, 1, patch_size^2)

        # Compute patch coordinates for all rays
        y_coords = (y_centers + patch_offsets_y_expanded).clamp(0, H_feat - 1)  # (B, R, patch_size^2)
        x_coords = (x_centers + patch_offsets_x_expanded).clamp(0, W_feat - 1)  # (B, R, patch_size^2)

        # Round to nearest integer for indexing
        y_coords = y_coords.round().long()  # (B, R, patch_size^2)
        x_coords = x_coords.round().long()  # (B, R, patch_size^2)

        # Sample features using advanced indexing
        # We need to sample from feature_maps: (B, C, H_feat, W_feat)
        # Strategy: Use gather or manual indexing with proper broadcasting
        
        # Create batch indices for each sample
        B_idx = torch.arange(B, device=device).view(B, 1, 1).expand(B, R, patch_size * patch_size)  # (B, R, patch_size^2)
        
        # Flatten all indices
        B_idx_flat = B_idx.flatten()  # (B * R * patch_size^2,)
        y_coords_flat = y_coords.flatten()  # (B * R * patch_size^2,)
        x_coords_flat = x_coords.flatten()  # (B * R * patch_size^2,)
        
        # Sample features: feature_maps[B_idx_flat, :, y_coords_flat, x_coords_flat]
        # This gives us (B * R * patch_size^2, C)
        # Note: PyTorch advanced indexing requires all indices to be broadcastable
        sampled_features = feature_maps[
            B_idx_flat,
            :,
            y_coords_flat,
            x_coords_flat
        ]  # (B * R * patch_size^2, C)
        
        # Reshape to (B, R, patch_size^2, C) and flatten patches
        patches = sampled_features.view(B, R, patch_size * patch_size, C)  # (B, R, patch_size^2, C)
        patches = patches.flatten(start_dim=2)  # (B, R, C * patch_size^2)

        return patches
