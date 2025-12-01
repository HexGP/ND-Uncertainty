"""
DINOv2 Feature Encoder

Extracts DINOv2 features from RGB images, following NeRF-on-the-Go's approach.
Can either run DINOv2 on-the-fly or load precomputed features.
"""

import torch
import torch.nn as nn
import torchvision.transforms as T
from typing import Optional


class DinoV2Encoder(nn.Module):
    """
    Wrapper around a DINOv2 backbone.

    Responsibility:
      - Take full RGB images from the ND-SDF dataset.
      - Produce a per-image feature map (C, H_feat, W_feat) suitable for
        dilated patch sampling as in NeRF-on-the-Go.

    This module internally loads a pretrained DINOv2 model via torch.hub.
    In NeRF-on-the-Go they use ViT-S/14 or ViT-B/14; we default to ViT-B/14.
    """

    def __init__(
        self,
        model_name: str = "dinov2_vitb14",
        frozen: bool = True,
        device: Optional[torch.device] = None,
    ):
        """
        Args:
            model_name: DINOv2 model variant. Options:
                - "dinov2_vits14" (ViT-S/14, 384-dim features)
                - "dinov2_vitb14" (ViT-B/14, 768-dim features)
                - "dinov2_vitl14" (ViT-L/14, 1024-dim features)
            frozen: If True, freeze DINOv2 parameters (default for feature extraction).
            device: Device to load model on. If None, uses CUDA if available.
        """
        super().__init__()

        self.model_name = model_name
        self.device = device if device is not None else torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )

        # Load DINOv2 model from torch.hub (same as NeRF-on-the-Go)
        # NeRF-on-the-Go uses dinov2_vits14, but we allow other variants
        try:
            self.backbone = torch.hub.load('facebookresearch/dinov2', model_name)
            self.backbone.to(self.device)
            self.backbone.eval()
        except Exception as e:
            raise RuntimeError(
                f"Failed to load DINOv2 model '{model_name}' from torch.hub. "
                f"Error: {e}\n"
                f"Make sure you have internet connection or the model cached."
            )

        if frozen:
            for p in self.backbone.parameters():
                p.requires_grad_(False)

        # ImageNet normalization (same as NeRF-on-the-Go)
        self.normalize = T.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        )

        # Determine feature dimension based on model
        if 'vits' in model_name:
            self.feat_dim = 384
        elif 'vitb' in model_name:
            self.feat_dim = 768
        elif 'vitl' in model_name:
            self.feat_dim = 1024
        else:
            # Default to ViT-B/14 dimension
            self.feat_dim = 768

    @torch.no_grad()
    def forward(
        self,
        images: torch.Tensor,
        camera_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Extract DINOv2 features from RGB images.

        Args:
            images: (B, 3, H, W) float tensor in [0, 1] range (RGB images).
            camera_indices: optional (B,) tensor. Currently unused, but kept for
                    future support of precomputed features per camera/frame.

        Returns:
            features: (B, C, H_feat, W_feat) DINO feature map.
                     C = 384 (ViT-S), 768 (ViT-B), or 1024 (ViT-L)
                     H_feat = H // 14, W_feat = W // 14 (patch size is 14)
        """
        B, C, H, W = images.shape
        assert C == 3, f"Expected RGB images (3 channels), got {C} channels"

        # Normalize images to ImageNet stats
        images_norm = self.normalize(images)

        # Forward through DINOv2
        # DINOv2 returns a dict with 'x_norm_patchtokens' containing patch features
        features_dict = self.backbone.forward_features(images_norm)

        # Extract patch tokens (normalized patch embeddings)
        # Shape: (B, N_patches, feat_dim) where N_patches = (H//14) * (W//14)
        patch_tokens = features_dict['x_norm_patchtokens']

        # Reshape to spatial feature map: (B, H_feat, W_feat, feat_dim)
        H_feat = H // 14
        W_feat = W // 14
        features = patch_tokens.view(B, H_feat, W_feat, self.feat_dim)

        # Convert to (B, C, H_feat, W_feat) format for easier indexing
        features = features.permute(0, 3, 1, 2)  # (B, feat_dim, H_feat, W_feat)

        return features
