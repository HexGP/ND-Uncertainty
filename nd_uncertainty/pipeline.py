"""
Uncertainty Pipeline

Orchestrates the full uncertainty prediction branch:
    RGB Input → DINOv2 Features → Dilated Patch Sampling →
    DINO Patch → Uncertainty MLP → β(r)

This module will be called in the trainer BEFORE ND-SDF's forward()
to attach β(r) to the `sample` dict.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple

from .dino_encoder import DinoV2Encoder
from .patch_sampling import DilatedPatchSampler
from .uncertainty_mlp import UncertaintyMLP


class UncertaintyPipeline(nn.Module):
    """
    Full branch for:

        RGB Input → DINOv2 Features → Dilated Patch Sampling →
        DINO Patch → Uncertainty MLP → β(r)

    This module will be called in the trainer BEFORE ND-SDF's forward()
    to attach β(r) to the `sample` dict.
    """

    def __init__(
        self,
        patch_size: int = 7,
        dilation: int = 2,
        dino_model_name: str = "dinov2_vitb14",
        patch_hidden_dim: int = 64,
        dropout_rate: float = 0.25,
        device: Optional[torch.device] = None,
    ):
        """
        Args:
            patch_size: Size of the patch neighborhood (default 7x7).
            dilation: Dilation factor for patch sampling (default 2).
            dino_model_name: DINOv2 model variant (default "dinov2_vitb14").
            patch_hidden_dim: Hidden dimension for Uncertainty MLP (default 64).
            dropout_rate: Dropout rate for Uncertainty MLP (default 0.25).
            device: Device to run on. If None, uses CUDA if available.
        """
        super().__init__()

        self.patch_size = patch_size
        self.dilation = dilation
        self.patch_hidden_dim = patch_hidden_dim

        # Initialize DINO encoder
        self.dino_encoder = DinoV2Encoder(
            model_name=dino_model_name,
            frozen=True,
            device=device,
        )

        # Initialize patch sampler
        self.patch_sampler = DilatedPatchSampler(
            patch_size=patch_size,
            dilation=dilation,
        )

        # Uncertainty MLP will be created lazily on first forward()
        # because we don't know the patch embedding dimension until we see features
        self.uncertainty_mlp: Optional[UncertaintyMLP] = None
        self.patch_embedding_dim: Optional[int] = None
        self.dropout_rate = dropout_rate

    def _build_uncertainty_mlp_if_needed(self, patches: torch.Tensor):
        """
        Lazy initialization of Uncertainty MLP.

        Args:
            patches: (B, R, C_patch) patch embeddings from which we infer C_patch.
        """
        if self.uncertainty_mlp is None:
            _, _, c_patch = patches.shape
            self.patch_embedding_dim = c_patch

            self.uncertainty_mlp = UncertaintyMLP(
                in_dim=c_patch,
                hidden_dim=self.patch_hidden_dim,
                dropout_rate=self.dropout_rate,
            ).to(patches.device)

            # Register as a submodule so it's included in state_dict
            self.add_module('uncertainty_mlp', self.uncertainty_mlp)

    def forward(
        self,
        rgb_full: torch.Tensor,
        sampling_idx: torch.Tensor,
        heights: torch.Tensor,
        widths: torch.Tensor,
        camera_indices: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Run the full uncertainty prediction pipeline.

        Args:
            rgb_full: (B, 3, H, W) full-resolution input RGB frames
                      (same images ND-SDF uses for supervision).
                      Expected to be in [0, 1] range.
            sampling_idx: (B, R) 1D indices into flattened image.
                          ND-SDF uses this format: indices in [0, h*w).
            heights: (B,) tensor of image heights in pixels.
            widths: (B,) tensor of image widths in pixels.
            camera_indices: (B,) optional indices for frames / cameras.
                           Currently unused but kept for future precomputed features.

        Returns:
            {
              "beta": (B, R, 1)      # per-ray β(r)
              "patch_features": (B, R, C_patch)  # for regularization / debugging
            }
        """
        # 1) DINOv2 feature extraction
        feature_maps = self.dino_encoder(rgb_full, camera_indices=camera_indices)
        # feature_maps: (B, C, H_feat, W_feat)

        # 2) Dilated Patch Sampling → DINO Patch
        patches = self.patch_sampler(
            feature_maps=feature_maps,
            sampling_idx=sampling_idx,
            heights=heights,
            widths=widths,
        )  # (B, R, C_patch)

        # 3) Build Uncertainty MLP if not yet initialized
        self._build_uncertainty_mlp_if_needed(patches)

        # 4) Uncertainty MLP → β(r)
        beta = self.uncertainty_mlp(patches)  # (B, R, 1)

        return {
            "beta": beta,
            "patch_features": patches,
        }
    
    def render_beta_map(
        self,
        feature_maps: torch.Tensor,
        H: int,
        W: int,
        device: torch.device,
        max_chunk_rays: int = 16384,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute per-pixel β map and basic statistics.
        
        Args:
            feature_maps: (B, C, Hf, Wf) DINO feature maps; assume B=1 here.
            H, W: output image resolution (pixels).
            device: torch device.
            max_chunk_rays: chunk size to avoid OOM.
        
        Returns:
            beta_image: (H, W) tensor on CPU (float32).
            stats: dict with mean/median/std/min/max/percentiles.
        """
        import torch.nn.functional as F
        B, C, Hf, Wf = feature_maps.shape
        assert B == 1, "render_beta_map currently assumes batch size 1."
        
        # All pixel indices in raster order
        num_rays = H * W
        all_idx = torch.arange(num_rays, device=device, dtype=torch.long)
        
        # Height/width tensors as expected by DilatedPatchSampler
        heights_tensor = torch.tensor([H], device=device, dtype=torch.long)
        widths_tensor = torch.tensor([W], device=device, dtype=torch.long)
        
        beta_flat = torch.empty(num_rays, device=device, dtype=torch.float32)
        
        # Lazily build MLP if needed (same as training)
        # We call it once on a dummy batch
        if self.uncertainty_mlp is None:
            with torch.no_grad():
                dummy_idx = all_idx[:min(max_chunk_rays, num_rays)].unsqueeze(0)  # (1, N)
                dummy_patches = self.patch_sampler(
                    feature_maps=feature_maps,
                    sampling_idx=dummy_idx,
                    heights=heights_tensor,
                    widths=widths_tensor,
                )
                self._build_uncertainty_mlp_if_needed(dummy_patches)
        
        # Chunked pass over all pixels
        start = 0
        while start < num_rays:
            end = min(start + max_chunk_rays, num_rays)
            idx_chunk = all_idx[start:end].unsqueeze(0)  # (1, N_chunk)
            patches = self.patch_sampler(
                feature_maps=feature_maps,
                sampling_idx=idx_chunk,
                heights=heights_tensor,
                widths=widths_tensor,
            )  # (1, N_chunk, C_patch)
            beta_chunk = self.uncertainty_mlp(patches)  # (1, N_chunk, 1)
            beta_chunk = beta_chunk.squeeze(0).squeeze(-1)  # (N_chunk,)
            beta_flat[start:end] = beta_chunk
            start = end
        
        # Reshape to (H, W) and move to CPU
        beta_image = beta_flat.view(H, W).detach().cpu()
        
        # Clamp to positive for stats (same spirit as training)
        beta_for_stats = torch.clamp(beta_image, min=1e-6)
        
        # Compute stats
        stats = {
            "mean": float(beta_for_stats.mean().item()),
            "median": float(beta_for_stats.median().item()),
            "std": float(beta_for_stats.std(unbiased=False).item()),
            "min": float(beta_for_stats.min().item()),
            "max": float(beta_for_stats.max().item()),
            "p25": float(torch.quantile(beta_for_stats, 0.25).item()),
            "p50": float(torch.quantile(beta_for_stats, 0.50).item()),
            "p75": float(torch.quantile(beta_for_stats, 0.75).item()),
            "p95": float(torch.quantile(beta_for_stats, 0.95).item()),
        }
        
        return beta_image, stats
