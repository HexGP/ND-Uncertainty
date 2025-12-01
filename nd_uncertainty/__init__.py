"""
ND-SDF Uncertainty Package

This package implements the uncertainty prediction branch from NeRF-on-the-Go,
integrated into ND-SDF. The pipeline extracts DINO features, performs dilated
patch sampling, and predicts per-ray uncertainty β(r).
"""

from .dino_encoder import DinoV2Encoder
from .patch_sampling import DilatedPatchSampler
from .uncertainty_mlp import UncertaintyMLP
from .pipeline import UncertaintyPipeline
from .uncertainty_loss import UncertaintyColorLoss
from .variance_regularizer import PatchVarianceRegularizer
from .loss_wrapper import UncertaintyAwareLoss
from .trainer import UncertaintyTrainer
from .ssim_utils import compute_ssim_components, bias_function

__all__ = [
    'DinoV2Encoder',
    'DilatedPatchSampler',
    'UncertaintyMLP',
    'UncertaintyPipeline',
    'UncertaintyColorLoss',
    'PatchVarianceRegularizer',
    'UncertaintyAwareLoss',
    'UncertaintyTrainer',
    'compute_ssim_components',
    'bias_function',
]
