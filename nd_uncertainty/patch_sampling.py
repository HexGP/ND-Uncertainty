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
        # sampling_idx is in [0, h*w) range (original image dimensions)
        w_tensor = torch.tensor(w, device=device, dtype=torch.float32)
        y_pix = (sampling_idx.float() // w_tensor)  # (B, R) - original pixel y
        x_pix = (sampling_idx.float() % w_tensor)   # (B, R) - original pixel x

        # Map pixel coordinates to feature map coordinates
        # DINO patch size is 14, so feature map is downsampled by 14
        # Note: Images are padded to multiples of 14, so feature map dimensions
        # are based on padded dimensions. Pixel coordinates from original image
        # are still valid since padding is on bottom/right edges.
        y_feat = (y_pix / 14.0).clamp(0, H_feat - 1)  # (B, R)
        x_feat = (x_pix / 14.0).clamp(0, W_feat - 1)  # (B, R)

        # Generate patch offsets
        # We want a full patch_size x patch_size grid of offsets
        # The dilation controls spacing, but we still sample all patch_size^2 positions
        # Create symmetric offsets: for patch_size=7: [-3, -2, -1, 0, 1, 2, 3] = 7 elements
        # Use (patch_size-1)//2 to get symmetric range for both odd and even sizes
        patch_size_val = int(self.patch_size)  # Ensure it's an int
        half_size = (patch_size_val - 1) // 2  # For 7: (7-1)//2 = 3, for 8: (8-1)//2 = 3
        offset_range = torch.arange(
            -half_size,
            half_size + 1,
            dtype=torch.float32,
            device=device
        )
        
        # Verify we have exactly patch_size elements
        assert len(offset_range) == patch_size_val, \
            f"offset_range has {len(offset_range)} elements, expected {patch_size_val} (patch_size={self.patch_size})"
        
        # Create a full grid: all combinations of offsets
        patch_offsets_x, patch_offsets_y = torch.meshgrid(
            offset_range, offset_range, indexing='xy'
        )
        # patch_offsets_x, patch_offsets_y: (patch_size, patch_size)
        
        # Verify grid shape
        assert patch_offsets_x.shape == (patch_size_val, patch_size_val), \
            f"patch_offsets_x shape {patch_offsets_x.shape} != ({patch_size_val}, {patch_size_val}) (patch_size={self.patch_size})"

        # Apply dilation: multiply offsets by dilation factor
        # This controls the spacing between sampled feature locations
        patch_offsets_x = patch_offsets_x * self.dilation
        patch_offsets_y = patch_offsets_y * self.dilation

        # Flatten offsets for easier indexing
        patch_offsets_x = patch_offsets_x.flatten()  # (patch_size^2,)
        patch_offsets_y = patch_offsets_y.flatten()  # (patch_size^2,)
        
        # Verify flattened size
        num_patches = patch_size_val * patch_size_val
        assert len(patch_offsets_x) == num_patches, \
            f"Flattened patch_offsets_x has {len(patch_offsets_x)} elements, expected {num_patches} (patch_size={self.patch_size})"
        assert len(patch_offsets_y) == num_patches, \
            f"Flattened patch_offsets_y has {len(patch_offsets_y)} elements, expected {num_patches} (patch_size={self.patch_size})"

        # Vectorized patch sampling
        # Expand coordinates for all rays and all patch offsets
        
        # y_feat: (B, R) -> (B, R, 1) -> (B, R, patch_size^2)
        y_centers = y_feat.unsqueeze(-1)  # (B, R, 1)
        x_centers = x_feat.unsqueeze(-1)  # (B, R, 1)

        # Broadcast patch offsets: (patch_size^2,) -> (1, 1, patch_size^2)
        patch_offsets_y_expanded = patch_offsets_y.to(device).float().unsqueeze(0).unsqueeze(0)  # (1, 1, patch_size^2)
        patch_offsets_x_expanded = patch_offsets_x.to(device).float().unsqueeze(0).unsqueeze(0)  # (1, 1, patch_size^2)

        # Compute patch coordinates for all rays
        # Explicitly expand to ensure correct shapes
        # y_centers: (B, R, 1), patch_offsets_y_expanded: (1, 1, num_patches)
        # We want: (B, R, num_patches)
        y_centers_expanded = y_centers.expand(B, R, num_patches)  # (B, R, num_patches)
        x_centers_expanded = x_centers.expand(B, R, num_patches)  # (B, R, num_patches)
        patch_offsets_y_exp = patch_offsets_y_expanded.expand(B, R, num_patches)  # (B, R, num_patches)
        patch_offsets_x_exp = patch_offsets_x_expanded.expand(B, R, num_patches)  # (B, R, num_patches)
        
        y_coords = (y_centers_expanded + patch_offsets_y_exp).clamp(0, H_feat - 1)  # (B, R, num_patches)
        x_coords = (x_centers_expanded + patch_offsets_x_exp).clamp(0, W_feat - 1)  # (B, R, num_patches)

        # Round to nearest integer for indexing
        y_coords = y_coords.round().long()  # (B, R, num_patches)
        x_coords = x_coords.round().long()  # (B, R, num_patches)

        # Sample features using memory-efficient advanced indexing
        # We avoid creating any large expanded tensors.
        #
        # feature_maps: (B, C, H_feat, W_feat)
        # y_coords, x_coords: (B, R, num_patches)
        #
        # For each batch b we index:
        #   feat_map_b[:, y_coords_b, x_coords_b]  -> (C, R, num_patches)
        # which is handled efficiently by PyTorch's advanced indexing.
        patches_list = []
        for b in range(B):
            feat_map_b = feature_maps[b]              # (C, H_feat, W_feat)
            y_coords_b = y_coords[b]                  # (R, num_patches)
            x_coords_b = x_coords[b]                  # (R, num_patches)

            # Advanced indexing over spatial dims: result (C, R, num_patches)
            sampled_b = feat_map_b[:, y_coords_b, x_coords_b]  # (C, R, num_patches)

            # Permute to (R, num_patches, C) and collect
            sampled_b = sampled_b.permute(1, 2, 0)   # (R, num_patches, C)
            patches_list.append(sampled_b)

        # Stack all batches and flatten spatial patch dimensions
        patches = torch.stack(patches_list, dim=0)   # (B, R, num_patches, C)
        patches = patches.flatten(start_dim=2)       # (B, R, C * num_patches)

        return patches
