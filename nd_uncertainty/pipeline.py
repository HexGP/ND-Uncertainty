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
from typing import Dict, Optional

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
