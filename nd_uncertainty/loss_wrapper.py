"""
Loss Wrapper for ND-SDF with Uncertainty

Wraps ND-SDF's ImplicitReconLoss to add uncertainty-weighted color loss
without modifying ND-SDF's core loss implementation.
"""

import torch
import torch.nn as nn
from omegaconf import OmegaConf

from models.loss import ImplicitReconLoss
from nd_uncertainty.uncertainty_loss import UncertaintyColorLoss
from nd_uncertainty.variance_regularizer import PatchVarianceRegularizer
from nd_uncertainty.ssim_utils import bias_function


class UncertaintyAwareLoss(nn.Module):
    """
    Wraps ND-SDF's loss aggregator and adds:
        + L_uncertainty(C, Ĉ, β)
        + (optional) patch variance regularization (future)
    
    This allows us to use the original ND-SDF model unchanged.
    """
    
    def __init__(self, conf):
        """
        Args:
            conf: OmegaConf configuration object (ND-SDF's config)
        """
        super().__init__()
        
        # Initialize base ND-SDF loss with all its native components
        # Filter out uncertainty-related parameters that ImplicitReconLoss doesn't know about
        loss_conf = dict(conf.loss)
        uncertainty_params = [
            'use_uncertainty', 'use_ssim_uncertainty', 'use_variance_regularizer',
            'weight_unc', 'unc_lambda_reg', 'unc_clip_min', 'unc_eps',
            'ssim_weight', 'ssim_anneal', 'ssim_clip_max', 'ssim_window_size', 'stop_ssim_gradient',
            'variance_weight', 'use_uncertainty_annealing', 'uncertainty_anneal_param', 'weight_unc_sched'
        ]
        for param in uncertainty_params:
            loss_conf.pop(param, None)  # Remove if exists, ignore if doesn't
        
        self.base_loss = ImplicitReconLoss(**loss_conf, optim_conf=conf.optim)
        
        # Check if uncertainty is enabled
        use_uncertainty = getattr(conf.loss, 'use_uncertainty', True)
        
        if not use_uncertainty:
            # Uncertainty disabled - just use base loss
            self.unc_loss = None
            self.variance_reg = None
            self.weight_unc_fn = lambda prog: 0.0
            self.use_uncertainty_annealing = False
            return
        
        # Initialize uncertainty loss
        # Get config values with defaults matching NeRF-on-the-Go
        lambda_reg = getattr(conf.loss, 'unc_lambda_reg', 0.5)  # reg_mult in NeRF-OTG
        uncer_clip_min = getattr(conf.loss, 'unc_clip_min', 0.1)  # uncer_clip_min in NeRF-OTG
        use_ssim = getattr(conf.loss, 'use_ssim_uncertainty', False)
        ssim_mult = getattr(conf.loss, 'ssim_weight', 0.5)  # ssim_mult in NeRF-OTG
        ssim_anneal = getattr(conf.loss, 'ssim_anneal', 0.8)  # ssim_anneal in NeRF-OTG
        ssim_clip_max = getattr(conf.loss, 'ssim_clip_max', 5.0)  # ssim_clip_max in NeRF-OTG
        stop_ssim_gradient = getattr(conf.loss, 'stop_ssim_gradient', False)
        
        self.unc_loss = UncertaintyColorLoss(
            lambda_reg=lambda_reg,
            uncer_clip_min=uncer_clip_min,
            use_ssim=use_ssim,
            ssim_mult=ssim_mult,
            ssim_anneal=ssim_anneal,
            ssim_clip_max=ssim_clip_max,
            stop_ssim_gradient=stop_ssim_gradient,
        )
        
        # Initialize patch variance regularizer (optional)
        use_variance_reg = getattr(conf.loss, 'use_variance_regularizer', False)
        variance_weight = getattr(conf.loss, 'variance_weight', 0.1)  # dino_var_mult in NeRF-OTG
        if use_variance_reg:
            self.variance_reg = PatchVarianceRegularizer(
                top_k=128,
                similarity_threshold=0.75,
                weight=variance_weight,
            )
        else:
            self.variance_reg = None
        
        # Weight for uncertainty loss (can be scheduled with training progress)
        self.weight_unc = getattr(conf.loss, 'weight_unc', 1.0)
        self.use_uncertainty_annealing = getattr(conf.loss, 'use_uncertainty_annealing', False)
        self.uncertainty_anneal_param = getattr(conf.loss, 'uncertainty_anneal_param', 0.8)
        
        # Optional: sequential learning rate for uncertainty loss
        if hasattr(conf.loss, 'weight_unc_sched'):
            # Support sequential LR scheduling like ND-SDF's other losses
            from functools import partial
            def sequential_lr(progress, lr):
                if isinstance(lr, (int, float)):
                    return lr
                return lr[2] + min(1.0, max(0.0, (progress - lr[0]) / (lr[1] - lr[0]))) * (lr[3] - lr[2])
            self.weight_unc_fn = partial(sequential_lr, lr=conf.loss.weight_unc_sched)
        else:
            self.weight_unc_fn = lambda prog: self.weight_unc
    
    def set_patch_size(self, num_rays):
        """Forward set_patch_size to base_loss (called by trainer)."""
        if hasattr(self.base_loss, 'set_patch_size'):
            self.base_loss.set_patch_size(num_rays)
    
    def set_curvature_weight(self, cur_step, anneal_levels, grow_rate):
        """Forward set_curvature_weight to base_loss (called by trainer)."""
        if hasattr(self.base_loss, 'set_curvature_weight'):
            self.base_loss.set_curvature_weight(cur_step, anneal_levels, grow_rate)
    
    def __getattr__(self, name):
        """
        Forward attribute access to base_loss for attributes that don't exist
        on UncertaintyAwareLoss (e.g., lambda_curvature, set_curvature_weight, etc.)
        This allows the trainer to access base_loss attributes directly.
        """
        # Don't forward if trying to access base_loss itself
        if name == 'base_loss':
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
        
        # Forward to base_loss if it exists (check __dict__ directly to avoid recursion)
        if 'base_loss' in self.__dict__:
            return getattr(self.__dict__['base_loss'], name)
        
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
    
    def forward(self, output, sample, prog):
        """
        Compute combined ND-SDF + uncertainty loss.
        
        Args:
            output: dict from ND-SDF forward pass
                - 'rgb': (B, R, 3) rendered RGB
                - 'outside': (B, R, 1) outside mask
            sample: dict containing:
                - 'rgb': (B, R, 3) ground truth RGB
                - 'beta': (B, R, 1) predicted uncertainty β(r)
                - 'mask': (B, R, 1) optional foreground mask
            prog: training progress [0, 1]
        
        Returns:
            losses: dict with all ND-SDF losses + uncertainty_loss + updated total
        """
        # Run all ND-SDF native losses (eikonal, RGB, depth, normal, etc.)
        losses = self.base_loss(output, sample, prog)
        
        # Check if uncertainty is enabled
        if self.unc_loss is None or 'beta' not in sample:
            # Uncertainty disabled - return base losses only
            return losses
        
        # --- Uncertainty loss ---
        # Extract required tensors
        rgb_pred = output['rgb']          # (B, R, 3)
        rgb_gt = sample['rgb']             # (B, R, 3)
        beta = sample['beta']              # (B, R, 1)
        
        # Get mask (foreground + not outside)
        outside = output.get('outside', None)  # (B, R, 1)
        foreground_mask = sample.get('mask', None)  # (B, R, 1)
        
        # Combine masks: foreground AND not outside
        if outside is not None and foreground_mask is not None:
            mask = (~outside) & foreground_mask  # (B, R, 1)
        elif outside is not None:
            mask = ~outside  # (B, R, 1)
        elif foreground_mask is not None:
            mask = foreground_mask  # (B, R, 1)
        else:
            mask = None
        
        # Compute uncertainty-weighted color loss
        # Pass train_frac for SSIM annealing if SSIM is enabled
        train_frac_for_ssim = prog if (hasattr(self.unc_loss, 'use_ssim') and self.unc_loss.use_ssim) else None
        L_unc = self.unc_loss(rgb_pred, rgb_gt, beta, mask=mask, train_frac=train_frac_for_ssim)
        
        # Get uncertainty loss weight (may be scheduled)
        weight_unc = self.weight_unc_fn(prog)
        
        # Apply training progress-based uncertainty annealing (from NeRF-on-the-Go lines 176-177)
        if self.use_uncertainty_annealing:
            # Adjust uncertainty weight based on training progress
            # Matches NeRF-on-the-Go: uncer_rate = 1 + 1 * bias(train_frac, ssim_anneal)
            uncer_rate = 1.0 + bias_function(prog, self.uncertainty_anneal_param)
            weight_unc = weight_unc * uncer_rate
        
        # Add uncertainty loss to total
        losses['uncertainty_loss'] = L_unc
        losses['total'] = losses['total'] + weight_unc * L_unc
        
        # Add patch variance regularization if enabled
        if self.variance_reg is not None and 'uncertainty_features' in sample:
            patch_features = sample['uncertainty_features']  # (B, R, C_patch)
            L_var = self.variance_reg(patch_features, beta)
            losses['variance_regularizer'] = L_var
            losses['total'] = losses['total'] + L_var
        
        return losses
